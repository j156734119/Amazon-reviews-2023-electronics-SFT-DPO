from __future__ import annotations

import inspect
from importlib.metadata import version
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
        return {"dtype": torch.float32 if not torch.cuda.is_available() else "auto"}
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


def training_precision(training_config: dict[str, Any]) -> dict[str, bool]:
    fp16 = bool(training_config.get("fp16", False))
    bf16 = bool(training_config.get("bf16", False))
    if fp16 and bf16:
        raise ValueError("fp16 and bf16 cannot both be enabled.")
    return {"fp16": fp16, "bf16": bf16}


def align_conv1d_dtype(
    model: Any,
    dtype: Any,
) -> dict[str, int | str]:
    import torch

    modules = 0
    parameters = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Conv1d):
            continue
        modules += 1
        for parameter in module.parameters(recurse=False):
            parameters += parameter.numel()
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)
    return {
        "conv1d_modules": modules,
        "conv1d_parameters": parameters,
        "dtype": str(dtype),
    }


def install_conv1d_runtime_dtype_hooks(model: Any) -> dict[str, int]:
    import torch

    installed = 0
    for module in model.modules():
        if not isinstance(module, torch.nn.Conv1d):
            continue
        if getattr(module, "_review_align_dtype_hooks", False):
            continue

        module._review_align_input_dtypes = []

        def align_input(conv: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not args or not isinstance(args[0], torch.Tensor):
                conv._review_align_input_dtypes.append(None)
                return None
            input_tensor = args[0]
            conv._review_align_input_dtypes.append(input_tensor.dtype)
            target_dtype = conv.weight.dtype
            if input_tensor.dtype == target_dtype:
                return None
            return (input_tensor.to(dtype=target_dtype), *args[1:])

        def restore_output(
            conv: Any,
            _args: tuple[Any, ...],
            output: Any,
        ) -> Any:
            original_dtype = conv._review_align_input_dtypes.pop()
            if (
                original_dtype is not None
                and isinstance(output, torch.Tensor)
                and output.dtype != original_dtype
            ):
                return output.to(dtype=original_dtype)
            return output

        module.register_forward_pre_hook(align_input)
        module.register_forward_hook(restore_output)
        module._review_align_dtype_hooks = True
        installed += 1
    return {"conv1d_runtime_hooks": installed}


def model_device_report(model: Any) -> dict[str, Any]:
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    embedding = model.get_input_embeddings()
    score = getattr(model, "score", None)
    if score is None and hasattr(model, "get_base_model"):
        score = getattr(model.get_base_model(), "score", None)
    score_parameters = list(score.parameters()) if score is not None else []
    return {
        "parameter_devices": parameter_devices,
        "parameter_dtypes": parameter_dtypes,
        "buffer_devices": buffer_devices,
        "embedding_device": str(embedding.weight.device),
        "embedding_dtype": str(embedding.weight.dtype),
        "score_device": (
            str(score_parameters[0].device) if score_parameters else None
        ),
        "score_dtype": str(score_parameters[0].dtype) if score_parameters else None,
    }


def assert_model_on_device(model: Any, expected_device: Any, name: str) -> dict[str, Any]:
    import torch

    expected = torch.device(expected_device)
    mismatches = []
    for parameter_name, parameter in model.named_parameters():
        if parameter.device != expected:
            mismatches.append(f"parameter {parameter_name}={parameter.device}")
    for buffer_name, buffer in model.named_buffers():
        if buffer.device != expected:
            mismatches.append(f"buffer {buffer_name}={buffer.device}")
    if mismatches:
        details = ", ".join(mismatches[:8])
        if len(mismatches) > 8:
            details += f", ... ({len(mismatches)} total)"
        raise RuntimeError(
            f"{name} is not fully placed on {expected}: {details}"
        )
    return model_device_report(model)


def place_auxiliary_model(
    model: Any,
    device: Any,
    dtype: Any,
    load_in_4bit: bool,
    name: str,
) -> dict[str, Any]:
    if load_in_4bit:
        return model_device_report(model)
    model.to(device=device, dtype=dtype)
    return assert_model_on_device(model, device, name)


def validate_reward_model_forward(
    model: Any,
    tokenizer: Any,
    text: str,
    device: Any,
    max_length: int,
    name: str,
) -> dict[str, Any]:
    import torch

    expected_device = torch.device(device)
    assert_model_on_device(model, expected_device, name)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(expected_device) for key, value in inputs.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            logits = model(**inputs).logits
    finally:
        model.train(was_training)
    if logits.shape != (1, 1):
        raise RuntimeError(
            f"{name} preflight expected logits shape (1, 1), got {tuple(logits.shape)}."
        )
    if logits.device != expected_device:
        raise RuntimeError(
            f"{name} preflight logits are on {logits.device}, expected {expected_device}."
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{name} preflight produced non-finite logits.")
    return {
        "logits_shape": list(logits.shape),
        "logits_device": str(logits.device),
        "logits_dtype": str(logits.dtype),
    }


def validate_model_runtime(model_config: dict[str, Any]) -> None:
    minimum = model_config.get("min_transformers_version")
    if not minimum:
        return

    from packaging.version import Version

    installed = version("transformers")
    if Version(installed) < Version(str(minimum)):
        raise RuntimeError(
            f"{model_config['base_model']} requires transformers>={minimum}; "
            f"found {installed}. Reinstall the training dependencies."
        )


def load_base_model(model_config: dict[str, Any], for_training: bool) -> Any:
    from transformers import AutoModelForCausalLM

    validate_model_runtime(model_config)
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

    target_modules = model_config["lora_target_modules"]
    if not isinstance(target_modules, str):
        target_modules = list(target_modules)
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_dropout=float(model_config["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
    )


def sft_merged_path(config: dict[str, Any]) -> Path:
    configured = config.get("rlhf", {}).get(
        "sft_merged_dir",
        config["model"].get("sft_merged_dir"),
    )
    if not configured:
        raise ValueError("A merged SFT output path is required in the configuration.")
    return Path(configured).resolve()


def upcast_trainable_parameters(model: Any) -> dict[str, int]:
    import torch

    tensors = 0
    parameters = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        tensors += 1
        parameters += parameter.numel()
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    return {
        "trainable_tensors": tensors,
        "trainable_parameters": parameters,
    }


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
