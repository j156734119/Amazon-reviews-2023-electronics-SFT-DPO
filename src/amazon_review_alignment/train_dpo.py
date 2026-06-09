from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import output_root
from .modeling import (
    load_policy_model,
    load_tokenizer,
    render_chat,
    supported_kwargs,
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

    sft_dir = Path(config["training"]["sft"]["output_dir"]).resolve()
    if not sft_dir.exists():
        raise RuntimeError(f"SFT adapter does not exist: {sft_dir}")
    tokenizer = load_tokenizer(model_config["base_model"], padding_side="left")
    train_dataset = Dataset.from_list(build_dpo_records(train_rows, tokenizer))
    eval_dataset = Dataset.from_list(build_dpo_records(validation_rows, tokenizer))
    model = load_policy_model(model_config, sft_dir, for_training=True)
    ref_model = load_policy_model(model_config, sft_dir, for_training=False)

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
        "report_to": "none",
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "seed": int(config["project"]["seed"]),
    }
    args = DPOConfig(**supported_kwargs(DPOConfig, common_args))
    trainer_values = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = DPOTrainer(**supported_kwargs(DPOTrainer, trainer_values))
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_run_metadata(
        output_dir,
        config,
        "train-dpo",
        {
            "base_model": model_config["base_model"],
            "sft_adapter": str(sft_dir),
            "reference_policy": "independently loaded and frozen SFT policy",
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
        },
    )
    LOGGER.info("Saved DPO adapter to %s", output_dir)
    return output_dir
