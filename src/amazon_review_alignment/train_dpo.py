from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import output_root
from .modeling import (
    load_base_model,
    load_tokenizer,
    lora_config,
    render_chat,
    sft_merged_path,
    supported_kwargs,
    training_precision,
    upcast_trainable_parameters,
)
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .utils import read_jsonl, save_run_metadata, set_seed

LOGGER = logging.getLogger(__name__)


def build_dpo_records(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
    records = []
    for row in rows:
        prompt = render_chat(
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": analysis_user_prompt(row["text"])},
            ],
            add_generation_prompt=True,
        )
        records.append(
            {
                "prompt": prompt,
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    return records


def train_dpo(config: dict[str, Any]) -> Path:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    set_seed(int(config["project"]["seed"]))
    model_config = config["model"]
    training_config = config["training"]["dpo"]
    teacher_dir = output_root(config) / "teacher"
    train_rows = read_jsonl(teacher_dir / "preferences_train.jsonl")
    validation_rows = read_jsonl(teacher_dir / "preferences_validation.jsonl")
    if not train_rows or not validation_rows:
        raise RuntimeError(
            "Validated teacher preferences are missing. Complete teacher-batch first."
        )

    merged_path = sft_merged_path(config)
    if not merged_path.exists():
        raise RuntimeError(
            f"Merged SFT model does not exist: {merged_path}. Run merge-sft first."
        )
    dpo_model_config = deepcopy(model_config)
    dpo_model_config["base_model"] = str(merged_path)
    tokenizer = load_tokenizer(str(merged_path), padding_side="left")
    train_dataset = Dataset.from_list(build_dpo_records(train_rows, tokenizer))
    eval_dataset = Dataset.from_list(build_dpo_records(validation_rows, tokenizer))
    model = load_base_model(dpo_model_config, for_training=True)

    output_dir = Path(training_config["output_dir"]).resolve()
    common_args = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_config["learning_rate"]),
        "num_train_epochs": float(training_config["epochs"]),
        "beta": float(training_config["beta"]),
        "per_device_train_batch_size": int(training_config["per_device_batch_size"]),
        "per_device_eval_batch_size": int(training_config["per_device_batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "logging_steps": int(training_config["logging_steps"]),
        "eval_steps": int(training_config["eval_steps"]),
        "save_steps": int(training_config["save_steps"]),
        "max_steps": int(training_config["max_steps"]),
        "max_length": int(model_config["max_sequence_length"]),
        "max_prompt_length": int(model_config["max_sequence_length"]) // 2,
        "gradient_checkpointing": True,
        **training_precision(training_config),
        "report_to": "none",
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "seed": int(config["project"]["seed"]),
    }
    args = DPOConfig(**supported_kwargs(DPOConfig, common_args))
    trainer_values = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": lora_config(model_config),
    }
    trainer = DPOTrainer(**supported_kwargs(DPOTrainer, trainer_values))
    if bool(training_config.get("fp16")):
        precision = upcast_trainable_parameters(trainer.model)
        LOGGER.info("Upcast DPO trainable parameters to FP32: %s", precision)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_run_metadata(
        output_dir,
        config,
        "train-dpo",
        {
            "base_model": model_config["base_model"],
            "sft_merged_model": str(merged_path),
            "reference_policy": "merged SFT policy with DPO adapter disabled",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
        },
    )
    LOGGER.info("Saved DPO adapter to %s", output_dir)
    return output_dir
