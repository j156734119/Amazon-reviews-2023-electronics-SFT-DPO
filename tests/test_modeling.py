import pytest
import torch

from amazon_review_alignment.modeling import (
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
