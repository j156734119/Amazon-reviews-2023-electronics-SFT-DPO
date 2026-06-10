from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from .config import output_root
from .modeling import (
    load_tokenizer,
    quantization_kwargs,
    render_chat,
    supported_kwargs,
)
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .utils import read_jsonl, save_run_metadata, set_seed, write_json

LOGGER = logging.getLogger(__name__)


def build_ppo_records(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_prompt_length: int | None = None,
) -> list[dict[str, list[int]]]:
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
        input_ids = tokenizer(prompt, padding=False)["input_ids"]
        if max_prompt_length is not None and len(input_ids) > max_prompt_length:
            tail_length = min(32, max_prompt_length // 4)
            input_ids = [
                *input_ids[: max_prompt_length - tail_length],
                *input_ids[-tail_length:],
            ]
        records.append({"input_ids": input_ids})
    return records


def _load_reward_adapter(
    merged_path: Path,
    reward_path: Path,
    model_config: dict[str, Any],
    trainable: bool,
    pad_token_id: int,
) -> Any:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        str(merged_path),
        num_labels=1,
        trust_remote_code=True,
        **quantization_kwargs(model_config),
    )
    model.config.pad_token_id = pad_token_id
    if trainable:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    model = PeftModel.from_pretrained(
        model,
        str(reward_path),
        is_trainable=trainable,
    )
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model


def _finite_ppo_metrics(log_history: list[dict[str, Any]]) -> dict[str, float]:
    wanted = {
        "objective/rlhf_reward",
        "objective/scores",
        "objective/kl",
        "loss/policy_avg",
        "loss/value_avg",
    }
    result: dict[str, float] = {}
    for entry in log_history:
        for raw_key, value in entry.items():
            key = raw_key.removeprefix("train/")
            if key not in wanted:
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"PPO produced a non-finite metric: {key}={value}")
                result[key] = float(value)
    return result


def train_ppo(config: dict[str, Any]) -> Path:
    import torch
    from datasets import Dataset
    from packaging.version import Version
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    from trl.experimental.ppo import PPOConfig, PPOTrainer

    set_seed(int(config["project"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("PPO training requires a CUDA GPU.")
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version < Version("2.4"):
        raise RuntimeError(
            f"PPO training requires torch>=2.4, but found {torch.__version__}. "
            "Use a clean T4/Colab environment before training."
        )
    merged_path = Path(config["rlhf"]["sft_merged_dir"]).resolve()
    reward_path = Path(config["rlhf"]["reward"]["output_dir"]).resolve()
    if not merged_path.exists():
        raise RuntimeError(f"Merged SFT model does not exist: {merged_path}")
    if not reward_path.exists():
        raise RuntimeError(f"Reward Model adapter does not exist: {reward_path}")

    prompt_rows = read_jsonl(output_root(config) / "rlhf" / "ppo_prompts.jsonl")
    if not prompt_rows:
        raise RuntimeError("PPO prompts are missing. Run build-rlhf-data first.")
    tokenizer = load_tokenizer(str(merged_path), padding_side="left")
    ppo_config = config["rlhf"]["ppo"]
    dataset = Dataset.from_list(
        build_ppo_records(
            prompt_rows,
            tokenizer,
            int(ppo_config["max_prompt_length"]),
        )
    )
    eval_count = min(max(1, len(dataset) // 10), 16)
    eval_dataset = dataset.select(range(eval_count))

    policy = AutoModelForCausalLM.from_pretrained(
        str(merged_path),
        trust_remote_code=True,
        **quantization_kwargs(config["model"]),
    )
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy = prepare_model_for_kbit_training(
        policy,
        use_gradient_checkpointing=True,
    )
    reward_model = _load_reward_adapter(
        merged_path,
        reward_path,
        config["model"],
        trainable=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    value_model = _load_reward_adapter(
        merged_path,
        reward_path,
        config["model"],
        trainable=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    output_dir = Path(ppo_config["output_dir"]).resolve()
    values = {
        "output_dir": str(output_dir),
        "total_episodes": int(ppo_config["total_episodes"]),
        "response_length": int(ppo_config["response_length"]),
        "per_device_train_batch_size": int(
            ppo_config["per_device_train_batch_size"]
        ),
        "per_device_eval_batch_size": int(
            ppo_config["per_device_eval_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            ppo_config["gradient_accumulation_steps"]
        ),
        "local_rollout_forward_batch_size": int(
            ppo_config["local_rollout_forward_batch_size"]
        ),
        "num_mini_batches": int(ppo_config["num_mini_batches"]),
        "num_ppo_epochs": int(ppo_config["num_ppo_epochs"]),
        "learning_rate": float(ppo_config["learning_rate"]),
        "temperature": float(ppo_config["temperature"]),
        "kl_coef": float(ppo_config["kl_coef"]),
        "cliprange": float(ppo_config["cliprange"]),
        "missing_eos_penalty": float(ppo_config["missing_eos_penalty"]),
        "logging_steps": int(ppo_config["logging_steps"]),
        "save_steps": int(ppo_config["save_steps"]),
        "fp16": bool(ppo_config["fp16"]),
        "bf16": False,
        "gradient_checkpointing": True,
        "stop_token": "eos",
        "report_to": "none",
        "seed": int(config["project"]["seed"]),
        "remove_unused_columns": False,
        "num_sample_generations": 0,
    }
    args = PPOConfig(**supported_kwargs(PPOConfig, values))
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(ppo_config["lora_r"]),
        lora_alpha=int(ppo_config["lora_alpha"]),
        lora_dropout=float(ppo_config["lora_dropout"]),
        target_modules=list(ppo_config["lora_target_modules"]),
        bias="none",
    )
    trainer_values = {
        "args": args,
        "processing_class": tokenizer,
        "model": policy,
        "ref_model": None,
        "reward_model": reward_model,
        "value_model": value_model,
        "train_dataset": dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
    }
    trainer = PPOTrainer(**supported_kwargs(PPOTrainer, trainer_values))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    runtime_seconds = time.perf_counter() - start
    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    final_metrics = _finite_ppo_metrics(trainer.state.log_history)
    required_metrics = {
        "objective/rlhf_reward",
        "objective/kl",
        "loss/policy_avg",
        "loss/value_avg",
    }
    missing_metrics = required_metrics - set(final_metrics)
    if missing_metrics:
        raise RuntimeError(
            f"PPO completed without required metrics: {sorted(missing_metrics)}"
        )
    write_json(
        output_root(config) / "rlhf" / "ppo_log_history.json",
        trainer.state.log_history,
    )
    summary = {
        "episodes": int(ppo_config["total_episodes"]),
        "unique_prompts": len(prompt_rows),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_memory_allocated_gb": peak_allocated_gb,
        "peak_cuda_memory_reserved_gb": peak_reserved_gb,
        "final_logged_metrics": final_metrics,
        "reference_policy": "merged SFT policy with PPO adapter disabled",
    }
    rlhf_dir = output_root(config) / "rlhf"
    write_json(rlhf_dir / "ppo_metrics.json", summary)
    save_run_metadata(output_dir, config, "train-ppo", summary)
    LOGGER.info(
        "Saved PPO adapter to %s (peak allocated/reserved CUDA memory %.2f/%.2f GiB)",
        output_dir,
        peak_allocated_gb,
        peak_reserved_gb,
    )
    max_peak_memory_gb = ppo_config.get("max_peak_memory_gb")
    if max_peak_memory_gb is not None and peak_reserved_gb > float(max_peak_memory_gb):
        raise RuntimeError(
            "PPO completed, but peak reserved CUDA memory "
            f"{peak_reserved_gb:.2f} GiB exceeded the configured "
            f"{float(max_peak_memory_gb):.2f} GiB limit. Outputs were preserved."
        )
    return output_dir
