from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import output_root
from .modeling import (
    load_tokenizer,
    quantization_kwargs,
    render_chat,
    supported_kwargs,
    training_precision,
    upcast_trainable_parameters,
    validate_model_runtime,
)
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .utils import read_jsonl, save_run_metadata, set_seed, write_json

LOGGER = logging.getLogger(__name__)


def build_reward_records(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int | None = None,
) -> list[dict[str, str]]:
    records = []
    for row in rows:
        review_text = row["text"]
        if max_length is not None:
            empty_prompt = render_chat(
                tokenizer,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": analysis_user_prompt("")},
                ],
                add_generation_prompt=True,
            )
            overhead = len(
                tokenizer(empty_prompt, add_special_tokens=False)["input_ids"]
            )
            completion_tokens = max(
                len(tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]),
                len(tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]),
            )
            review_budget = max(16, max_length - overhead - completion_tokens - 8)
            review_ids = tokenizer(
                review_text,
                add_special_tokens=False,
                truncation=True,
                max_length=review_budget,
            )["input_ids"]
            review_text = tokenizer.decode(review_ids, skip_special_tokens=True)
        prompt = render_chat(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": analysis_user_prompt(review_text)},
            ],
            add_generation_prompt=True,
        )
        records.append(
            {
                "chosen": prompt + row["chosen"],
                "rejected": prompt + row["rejected"],
            }
        )
    return records


def _score_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[float]:
    import torch

    scores: list[float] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            scores.extend(float(value) for value in logits.detach().cpu())
    return scores


def evaluate_reward_pairs(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
) -> dict[str, float | int]:
    if not rows:
        return {
            "examples": 0,
            "preference_accuracy": 0.0,
            "mean_reward_margin": 0.0,
        }
    records = build_reward_records(rows, tokenizer, max_length)
    chosen_scores = _score_texts(
        model,
        tokenizer,
        [row["chosen"] for row in records],
        max_length,
        batch_size,
    )
    rejected_scores = _score_texts(
        model,
        tokenizer,
        [row["rejected"] for row in records],
        max_length,
        batch_size,
    )
    margins = np.asarray(chosen_scores) - np.asarray(rejected_scores)
    return {
        "examples": len(rows),
        "preference_accuracy": float((margins > 0).mean()),
        "mean_reward_margin": float(margins.mean()),
    }


def train_reward(config: dict[str, Any]) -> Path:
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    set_seed(int(config["project"]["seed"]))
    rlhf_dir = output_root(config) / "rlhf"
    train_rows = read_jsonl(rlhf_dir / "rm_train.jsonl")
    human_eval_rows = read_jsonl(rlhf_dir / "rm_human_eval.jsonl")
    ai_eval_rows = read_jsonl(rlhf_dir / "rm_ai_validation.jsonl")
    if not train_rows:
        raise RuntimeError("RLHF data is missing. Run build-rlhf-data first.")

    reward_config = config["rlhf"]["reward"]
    validate_model_runtime(config["model"])
    merged_path = Path(config["rlhf"]["sft_merged_dir"]).resolve()
    if not merged_path.exists():
        raise RuntimeError(f"Merged SFT model does not exist: {merged_path}")
    tokenizer = load_tokenizer(str(merged_path), padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(merged_path),
        num_labels=1,
        trust_remote_code=True,
        **quantization_kwargs(config["model"]),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    max_length = int(reward_config["max_sequence_length"])
    train_dataset = Dataset.from_list(
        build_reward_records(train_rows, tokenizer, max_length)
    )
    eval_dataset = Dataset.from_list(
        build_reward_records(ai_eval_rows, tokenizer, max_length)
    )
    output_dir = Path(reward_config["output_dir"]).resolve()
    values = {
        "output_dir": str(output_dir),
        "learning_rate": float(reward_config["learning_rate"]),
        "num_train_epochs": float(reward_config["epochs"]),
        "per_device_train_batch_size": int(reward_config["per_device_batch_size"]),
        "per_device_eval_batch_size": int(reward_config["per_device_batch_size"]),
        "gradient_accumulation_steps": int(
            reward_config["gradient_accumulation_steps"]
        ),
        "warmup_ratio": float(reward_config["warmup_ratio"]),
        "logging_steps": int(reward_config["logging_steps"]),
        "eval_steps": int(reward_config["eval_steps"]),
        "save_steps": int(reward_config["save_steps"]),
        "max_steps": int(reward_config.get("max_steps", -1)),
        "max_length": int(reward_config["max_sequence_length"]),
        "gradient_checkpointing": True,
        **training_precision(reward_config),
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "report_to": "none",
        "seed": int(config["project"]["seed"]),
        "remove_unused_columns": False,
    }
    args = RewardConfig(**supported_kwargs(RewardConfig, values))
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(reward_config["lora_r"]),
        lora_alpha=int(reward_config["lora_alpha"]),
        lora_dropout=float(reward_config["lora_dropout"]),
        target_modules=list(reward_config["lora_target_modules"]),
        modules_to_save=["score"],
        bias="none",
    )
    trainer_values = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
    }
    trainer = RewardTrainer(**supported_kwargs(RewardTrainer, trainer_values))
    if bool(reward_config.get("fp16")):
        precision = upcast_trainable_parameters(trainer.model)
        LOGGER.info("Upcast Reward Model trainable parameters to FP32: %s", precision)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    batch_size = int(reward_config["per_device_batch_size"])
    metrics = {
        "ai_validation": evaluate_reward_pairs(
            trainer.model,
            tokenizer,
            ai_eval_rows,
            max_length,
            batch_size,
        ),
        "human_held_out": evaluate_reward_pairs(
            trainer.model,
            tokenizer,
            human_eval_rows,
            max_length,
            batch_size,
        ),
    }
    write_json(rlhf_dir / "reward_metrics.json", metrics)
    save_run_metadata(
        output_dir,
        config,
        "train-reward",
        {
            "train_examples": len(train_rows),
            "ai_validation_examples": len(ai_eval_rows),
            "human_eval_examples": len(human_eval_rows),
            "metrics": metrics,
        },
    )
    LOGGER.info("Saved Reward Model adapter to %s", output_dir)
    return output_dir
