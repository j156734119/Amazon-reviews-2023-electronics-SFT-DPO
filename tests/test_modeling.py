import pytest
import torch

from amazon_review_alignment.modeling import (
    align_conv1d_dtype,
    training_precision,
    upcast_trainable_parameters,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        self.frozen = torch.nn.Parameter(
            torch.ones(2, dtype=torch.float16),
            requires_grad=False,
        )


def test_upcast_trainable_parameters_only_changes_trainable_tensors() -> None:
    model = TinyModel()

    result = upcast_trainable_parameters(model)

    assert model.trainable.dtype == torch.float32
    assert model.frozen.dtype == torch.float16
    assert result == {"trainable_tensors": 1, "trainable_parameters": 3}


def test_training_precision_supports_a100_bf16() -> None:
    assert training_precision({"fp16": False, "bf16": True}) == {
        "fp16": False,
        "bf16": True,
    }


def test_training_precision_rejects_two_mixed_precision_modes() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        training_precision({"fp16": True, "bf16": True})


def test_align_conv1d_dtype_preserves_other_modules() -> None:
    model = torch.nn.Sequential(
        torch.nn.Conv1d(2, 2, 3, dtype=torch.float32),
        torch.nn.Linear(2, 2, dtype=torch.float32),
    )

    result = align_conv1d_dtype(model, torch.bfloat16)

    assert model[0].weight.dtype == torch.bfloat16
    assert model[0].bias.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float32
    assert result["conv1d_modules"] == 1
    assert result["dtype"] == "torch.bfloat16"


def test_align_conv1d_dtype_supports_reward_float32() -> None:
    model = torch.nn.Conv1d(2, 2, 3, dtype=torch.bfloat16)

    result = align_conv1d_dtype(model, torch.float32)

    assert model.weight.dtype == torch.float32
    assert model.bias.dtype == torch.float32
    assert result["dtype"] == "torch.float32"
