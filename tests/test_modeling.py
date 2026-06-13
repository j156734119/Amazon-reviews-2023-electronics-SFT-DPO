import pytest
import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from amazon_review_alignment.modeling import (
    align_conv1d_dtype,
    assert_model_on_device,
    install_conv1d_runtime_dtype_hooks,
    model_device_report,
    place_auxiliary_model,
    training_precision,
    upcast_trainable_parameters,
    validate_reward_model_forward,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
        self.frozen = torch.nn.Parameter(
            torch.ones(2, dtype=torch.float16),
            requires_grad=False,
        )


class TinyRewardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.lora_adapter = torch.nn.Linear(4, 4, bias=False)
        self.score = torch.nn.Linear(4, 1)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        del attention_mask
        hidden = self.lora_adapter(self.embedding(input_ids)).mean(dim=1)
        return SequenceClassifierOutput(logits=self.score(hidden))


class TinyTokenizer:
    def __call__(self, *_args, **_kwargs) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }


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


@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
)
def test_conv1d_runtime_hooks_bridge_mixed_dtypes(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
) -> None:
    model = torch.nn.Conv1d(2, 2, 3, dtype=weight_dtype)
    result = install_conv1d_runtime_dtype_hooks(model)
    inputs = torch.ones(1, 2, 5, dtype=input_dtype, requires_grad=True)

    output = model(inputs)
    output.float().sum().backward()

    assert output.dtype == input_dtype
    assert inputs.grad is not None
    assert result == {"conv1d_runtime_hooks": 1}
    assert install_conv1d_runtime_dtype_hooks(model) == {
        "conv1d_runtime_hooks": 0
    }


def test_place_auxiliary_model_moves_embedding_and_score() -> None:
    model = TinyRewardModel()

    report = place_auxiliary_model(
        model,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        load_in_4bit=False,
        name="tiny reward",
    )

    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.lora_adapter.weight.dtype == torch.bfloat16
    assert model.score.weight.dtype == torch.bfloat16
    assert report["parameter_devices"] == ["cpu"]
    assert report["embedding_device"] == "cpu"
    assert report["score_device"] == "cpu"
    assert report["score_dtype"] == "torch.bfloat16"


def test_assert_model_on_device_names_mismatched_parameter() -> None:
    model = TinyRewardModel()
    model.off_device = torch.nn.Parameter(torch.empty(1, device="meta"))

    with pytest.raises(RuntimeError, match=r"off_device=meta"):
        assert_model_on_device(model, torch.device("cpu"), "tiny reward")


def test_reward_model_forward_preflight() -> None:
    model = TinyRewardModel()

    result = validate_reward_model_forward(
        model,
        TinyTokenizer(),
        "Works well.",
        device=torch.device("cpu"),
        max_length=32,
        name="tiny reward",
    )

    assert result["logits_shape"] == [1, 1]
    assert result["logits_device"] == "cpu"
    assert torch.isfinite(model(torch.tensor([[1, 2]])).logits).all()


def test_model_device_report_includes_embedding_and_score_dtype() -> None:
    report = model_device_report(TinyRewardModel())

    assert report["embedding_dtype"] == "torch.float32"
    assert report["score_dtype"] == "torch.float32"
