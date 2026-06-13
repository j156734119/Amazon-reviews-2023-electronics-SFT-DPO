from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from .config import output_root
from .grpo_rewards import evidence_reward, length_reward, schema_reward
from .modeling import (
    align_conv1d_dtype,
    load_tokenizer,
    quantization_kwargs,
    render_chat,
    sft_merged_path,
    supported_kwargs,
    training_precision,
    upcast_trainable_parameters,
)
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .train_ppo import _load_reward_adapter
from .utils import read_jsonl, save_run_metadata, set_seed, write_json

LOGGER = logging.getLogger(__name__)


def build_grpo_records(
    rows: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, str]]:
    records = []
    for row in rows:
        records.append(
            {
                "id": str(row["id"]),
                "text": row["text"],
                "prompt": render_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": analysis_user_prompt(row["text"])},
                    ],
                    add_generation_prompt=True,
                ),
            }
        )
    return records


def _finite_grpo_metrics(log_history: list[dict[str, Any]]) -> dict[str, float]:
    exact = {
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "entropy",
        "clip_ratio/region_mean",
        "completions/mean_length",
        "completions/clipped_ratio",
    }
    result: dict[str, float] = {}
    for entry in log_history:
        for raw_key, value in entry.items():
            key = raw_key.removeprefix("train/")
            if key not in exact and not key.startswith("rewards/"):
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    raise RuntimeError(f"GRPO produced a non-finite metric: {key}={value}")
                result[key] = float(value)
    return result


def grpo_reward_weights(grpo_config: dict[str, Any]) -> list[float]:
    weights = grpo_config["reward_weights"]
    return [
        float(weights["reward_model"]),
        float(weights["schema"]),
        float(weights["evidence"]),
        float(weights["length"]),
    ]


def grpo_argument_values(
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    grpo_config = config["rlhf"]["grpo"]
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": float(grpo_config["epochs"]),
        "max_steps": int(grpo_config.get("max_steps", -1)),
        "learning_rate": float(grpo_config["learning_rate"]),
        "per_device_train_batch_size": int(
            grpo_config["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            grpo_config["gradient_accumulation_steps"]
        ),
        "generation_batch_size": int(grpo_config["generation_batch_size"]),
        "num_generations": int(grpo_config["num_generations"]),
        "num_iterations": int(grpo_config["num_iterations"]),
        "max_prompt_length": int(grpo_config["max_prompt_length"]),
        "max_completion_length": int(grpo_config["max_completion_length"]),
        "temperature": float(grpo_config["temperature"]),
        "beta": float(grpo_config["beta"]),
        "epsilon": float(grpo_config["epsilon"]),
        "loss_type": str(grpo_config["loss_type"]),
        "scale_rewards": str(grpo_config["scale_rewards"]),
        "use_vllm": bool(grpo_config["use_vllm"]),
        "logging_steps": int(grpo_config["logging_steps"]),
        "save_steps": int(grpo_config["save_steps"]),
        "save_strategy": "steps",
        **training_precision(grpo_config),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "report_to": "none",
        "seed": int(config["project"]["seed"]),
        "remove_unused_columns": False,
        "reward_weights": grpo_reward_weights(grpo_config),
    }


def train_grpo(config: dict[str, Any]) -> Path:
    import torch
    from datasets import Dataset
    from packaging.version import Version
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    set_seed(int(config["project"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO training requires a CUDA GPU.")
    if Version(torch.__version__.split("+")[0]) < Version("2.4"):
        raise RuntimeError(
            f"GRPO training requires torch>=2.4, but found {torch.__version__}."
        )

    merged_path = sft_merged_path(config)
    reward_path = Path(config["rlhf"]["reward"]["output_dir"]).resolve()
    if not merged_path.exists():
        raise RuntimeError(f"Merged SFT model does not exist: {merged_path}")
    if not reward_path.exists():
        raise RuntimeError(f"Reward Model adapter does not exist: {reward_path}")

    ppo_rows = read_jsonl(output_root(config) / "rlhf" / "ppo_prompts.jsonl")
    grpo_rows = read_jsonl(output_root(config) / "rlhf" / "grpo_prompts.jsonl")
    if not ppo_rows or not grpo_rows:
        raise RuntimeError("Shared PPO/GRPO prompts are missing. Run build-rlhf-data first.")
    ppo_ids = [str(row["id"]) for row in ppo_rows]
    grpo_ids = [str(row["id"]) for row in grpo_rows]
    if ppo_ids != grpo_ids:
        raise RuntimeError("PPO and GRPO must use the same ordered prompt IDs.")

    grpo_config = config["rlhf"]["grpo"]
    prompt_count = int(grpo_config["prompt_count"])
    rows = grpo_rows[:prompt_count]
    tokenizer = load_tokenizer(str(merged_path), padding_side="left")
    dataset = Dataset.from_list(build_grpo_records(rows, tokenizer))

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
    policy_dtype = (
        torch.bfloat16 if bool(grpo_config.get("bf16")) else torch.float16
    )
    for name, model, conv_dtype in (
        ("policy", policy, policy_dtype),
        ("reward", reward_model, torch.float32),
    ):
        aligned = align_conv1d_dtype(model, conv_dtype)
        LOGGER.info("Aligned %s Conv1d modules for GRPO generation: %s", name, aligned)

    output_dir = Path(grpo_config["output_dir"]).resolve()
    values = grpo_argument_values(config, output_dir)
    args = GRPOConfig(**supported_kwargs(GRPOConfig, values))
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(grpo_config["lora_r"]),
        lora_alpha=int(grpo_config["lora_alpha"]),
        lora_dropout=float(grpo_config["lora_dropout"]),
        target_modules=grpo_config["lora_target_modules"],
        bias="none",
    )
    trainer_values = {
        "model": policy,
        "args": args,
        "reward_funcs": [
            reward_model,
            schema_reward,
            evidence_reward,
            length_reward,
        ],
        "reward_processing_classes": [tokenizer, None, None, None],
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }
    trainer = GRPOTrainer(**supported_kwargs(GRPOTrainer, trainer_values))
    if bool(grpo_config.get("fp16")):
        precision = upcast_trainable_parameters(trainer.model)
        LOGGER.info("Upcast GRPO trainable parameters to FP32: %s", precision)

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    runtime_seconds = time.perf_counter() - started
    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    final_metrics = _finite_grpo_metrics(trainer.state.log_history)
    required = {
        "reward",
        "reward_std",
        "frac_reward_zero_std",
        "kl",
        "entropy",
        "clip_ratio/region_mean",
        "completions/mean_length",
    }
    missing = required - set(final_metrics)
    if missing:
        raise RuntimeError(f"GRPO completed without required metrics: {sorted(missing)}")

    rlhf_dir = output_root(config) / "rlhf"
    write_json(rlhf_dir / "grpo_log_history.json", trainer.state.log_history)
    summary = {
        "unique_prompts": len(rows),
        "num_generations": int(grpo_config["num_generations"]),
        "expected_completions_per_epoch": (
            len(rows) * int(grpo_config["num_generations"])
        ),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_memory_allocated_gb": peak_allocated_gb,
        "peak_cuda_memory_reserved_gb": peak_reserved_gb,
        "final_logged_metrics": final_metrics,
        "reference_policy": "merged SFT policy with GRPO adapter disabled",
        "reward_weights": grpo_config["reward_weights"],
    }
    write_json(rlhf_dir / "grpo_metrics.json", summary)
    save_run_metadata(output_dir, config, "train-grpo", summary)
    LOGGER.info(
        "Saved GRPO adapter to %s (peak allocated/reserved %.2f/%.2f GiB)",
        output_dir,
        peak_allocated_gb,
        peak_reserved_gb,
    )
    max_peak_memory_gb = grpo_config.get("max_peak_memory_gb")
    if max_peak_memory_gb is not None and peak_reserved_gb > float(max_peak_memory_gb):
        raise RuntimeError(
            "GRPO completed, but peak reserved CUDA memory "
            f"{peak_reserved_gb:.2f} GiB exceeded the configured "
            f"{float(max_peak_memory_gb):.2f} GiB limit. Outputs were preserved."
        )
    return output_dir
