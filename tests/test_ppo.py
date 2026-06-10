import pytest

from amazon_review_alignment.inference import model_config_for_variant
from amazon_review_alignment.train_ppo import _finite_ppo_metrics


def test_ppo_variant_uses_merged_sft_base(tmp_path) -> None:
    config = {
        "model": {"base_model": "Qwen/Qwen3-0.6B", "temperature": 0.0},
        "rlhf": {
            "sft_merged_dir": str(tmp_path / "sft-merged"),
            "ppo": {"output_dir": str(tmp_path / "ppo")},
        },
    }

    ppo = model_config_for_variant(config, "ppo")
    base = model_config_for_variant(config, "base")

    assert ppo["base_model"] == str((tmp_path / "sft-merged").resolve())
    assert base["base_model"] == "Qwen/Qwen3-0.6B"


def test_ppo_metrics_reject_non_finite_values() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        _finite_ppo_metrics([{"objective/kl": float("nan")}])


def test_ppo_metrics_extract_final_values() -> None:
    metrics = _finite_ppo_metrics(
        [
            {"objective/kl": 0.1, "loss/policy_avg": 0.2},
            {"objective/kl": 0.05},
        ]
    )

    assert metrics["objective/kl"] == 0.05
    assert metrics["loss/policy_avg"] == 0.2
