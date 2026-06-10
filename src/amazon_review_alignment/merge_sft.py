from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .utils import save_run_metadata

LOGGER = logging.getLogger(__name__)


def merge_sft(config: dict[str, Any]) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = config["model"]["base_model"]
    adapter_path = Path(config["training"]["sft"]["output_dir"]).resolve()
    output_dir = Path(config["rlhf"]["sft_merged_dir"]).resolve()
    if not adapter_path.exists():
        raise RuntimeError(f"SFT adapter does not exist: {adapter_path}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)
    save_run_metadata(
        output_dir,
        config,
        "merge-sft",
        {
            "base_model": base_model,
            "sft_adapter": str(adapter_path),
            "dtype": str(dtype),
        },
    )
    LOGGER.info("Saved merged SFT model to %s", output_dir)
    return output_dir
