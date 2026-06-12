from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import output_root
from .modeling import (
    load_base_model,
    load_tokenizer,
    lora_config,
    render_chat,
    supported_kwargs,
    training_precision,
    upcast_trainable_parameters,
)
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .utils import read_jsonl, save_run_metadata, set_seed

LOGGER = logging.getLogger(__name__)


def build_sft_records(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, str]]:
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
                "completion": row["chosen"],
            }
        )
    return records


def train_sft(config: dict[str, Any]) -> Path:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    set_seed(int(config["project"]["seed"]))
    model_config = config["model"]
    training_config = config["training"]["sft"]
    teacher_dir = output_root(config) / "teacher"
    train_rows = read_jsonl(teacher_dir / "preferences_train.jsonl")
    validation_rows = read_jsonl(teacher_dir / "preferences_validation.jsonl")
    if not train_rows or not validation_rows:
        raise RuntimeError(
            "Validated teacher preferences are missing. Complete teacher-batch first."
        )

    tokenizer = load_tokenizer(model_config["base_model"], padding_side="right")
    train_dataset = Dataset.from_list(build_sft_records(train_rows, tokenizer))
    eval_dataset = Dataset.from_list(build_sft_records(validation_rows, tokenizer))
    model = load_base_model(model_config, for_training=True)

    output_dir = Path(training_config["output_dir"]).resolve()
    common_args = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_config["learning_rate"]),
        "num_train_epochs": float(training_config["epochs"]),
        "per_device_train_batch_size": int(training_config["per_device_batch_size"]),
        "per_device_eval_batch_size": int(training_config["per_device_batch_size"]),
        "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
        "warmup_ratio": float(training_config["warmup_ratio"]),
        "logging_steps": int(training_config["logging_steps"]),
        "eval_steps": int(training_config["eval_steps"]),
        "save_steps": int(training_config["save_steps"]),
        "max_steps": int(training_config["max_steps"]),
        "completion_only_loss": True,
        "max_length": int(model_config["max_sequence_length"]),
        "max_seq_length": int(model_config["max_sequence_length"]),
        "gradient_checkpointing": True,
        **training_precision(training_config),
        "report_to": "none",
        "eval_strategy": "steps",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "load_best_model_at_end": False,
        "seed": int(config["project"]["seed"]),
    }
    args = SFTConfig(**supported_kwargs(SFTConfig, common_args))
    trainer_values = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "peft_config": lora_config(model_config),
    }
    trainer = SFTTrainer(**supported_kwargs(SFTTrainer, trainer_values))
    if bool(training_config.get("fp16")):
        precision = upcast_trainable_parameters(trainer.model)
        LOGGER.info("Upcast SFT trainable parameters to FP32: %s", precision)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_run_metadata(
        output_dir,
        config,
        "train-sft",
        {
            "base_model": model_config["base_model"],
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
        },
    )
    LOGGER.info("Saved SFT adapter to %s", output_dir)
    return output_dir
