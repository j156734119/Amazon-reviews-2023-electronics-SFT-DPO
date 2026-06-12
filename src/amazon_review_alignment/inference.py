from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import output_root
from .modeling import load_policy_model, load_tokenizer, render_chat, sft_merged_path
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .utils import read_jsonl, save_run_metadata, set_seed, write_jsonl

LOGGER = logging.getLogger(__name__)


def generation_prompt(tokenizer: Any, review_text: str) -> str:
    return render_chat(
        tokenizer,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": analysis_user_prompt(review_text)},
        ],
        add_generation_prompt=True,
    )


def generate_one(
    model: Any,
    tokenizer: Any,
    review_text: str,
    model_config: dict[str, Any],
) -> str:
    import torch

    prompt = generation_prompt(tokenizer, review_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = float(model_config["temperature"]) > 0
    generation_kwargs = {
        "max_new_tokens": int(model_config["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=float(model_config["temperature"]),
            top_p=float(model_config["top_p"]),
        )
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    completion = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def adapter_for_variant(config: dict[str, Any], variant: str) -> Path | None:
    if variant == "base":
        return None
    if variant in {"sft", "dpo"}:
        return Path(config["training"][variant]["output_dir"]).resolve()
    if variant in {"ppo", "grpo"}:
        return Path(config["rlhf"][variant]["output_dir"]).resolve()
    raise ValueError(f"Unknown model variant: {variant}")


def model_config_for_variant(
    config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    model_config = deepcopy(config["model"])
    if variant in {"dpo", "ppo", "grpo"}:
        model_config["base_model"] = str(sft_merged_path(config))
    elif variant not in {"base", "sft"}:
        raise ValueError(f"Unknown model variant: {variant}")
    return model_config


def run_inference(
    config: dict[str, Any],
    variant: str,
    force: bool = False,
) -> Path:
    set_seed(int(config["project"]["seed"]))
    prediction_dir = output_root(config) / "evaluation" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_path = prediction_dir / f"{variant}.jsonl"
    if output_path.exists() and not force:
        LOGGER.info("Reusing existing predictions: %s", output_path)
        return output_path

    test_rows = read_jsonl(output_root(config) / "data" / "test.jsonl")
    limit = int(config["evaluation"]["max_test_samples"])
    test_rows = test_rows[:limit]
    variant_model_config = model_config_for_variant(config, variant)
    tokenizer = load_tokenizer(
        variant_model_config["base_model"],
        padding_side="left",
    )
    adapter_path = adapter_for_variant(config, variant)
    if adapter_path and not adapter_path.exists():
        raise RuntimeError(f"{variant.upper()} adapter does not exist: {adapter_path}")
    model = load_policy_model(
        variant_model_config,
        adapter_path,
        for_training=False,
    )

    predictions = []
    for index, row in enumerate(test_rows, start=1):
        raw = generate_one(model, tokenizer, row["text"], config["model"])
        predictions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "rating": row.get("rating"),
                "variant": variant,
                "raw_output": raw,
            }
        )
        if index % 25 == 0:
            LOGGER.info("%s inference: %s/%s", variant, index, len(test_rows))
    write_jsonl(output_path, predictions)
    save_run_metadata(
        prediction_dir / f"{variant}_run",
        config,
        f"inference-{variant}",
        {
            "variant": variant,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "examples": len(predictions),
        },
    )
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return output_path
