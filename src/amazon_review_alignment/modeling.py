from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    return {key: value for key, value in values.items() if key in parameters}


def load_tokenizer(model_name: str, padding_side: str = "right") -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def quantization_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    import torch

    if not bool(model_config.get("load_in_4bit")) or not torch.cuda.is_available():
        return {"torch_dtype": torch.float32 if not torch.cuda.is_available() else "auto"}
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        "device_map": "auto",
    }


def load_base_model(model_config: dict[str, Any], for_training: bool) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model"],
        trust_remote_code=True,
        **quantization_kwargs(model_config),
    )
    model.config.use_cache = not for_training
    if for_training and bool(model_config.get("load_in_4bit")):
        try:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
        except ImportError:
            pass
    return model


def load_policy_model(
    model_config: dict[str, Any],
    adapter_path: str | Path | None,
    for_training: bool = False,
) -> Any:
    model = load_base_model(model_config, for_training=for_training)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=for_training,
        )
    return model


def lora_config(model_config: dict[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_dropout=float(model_config["lora_dropout"]),
        target_modules=list(model_config["lora_target_modules"]),
        bias="none",
    )


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)
