import json
from pathlib import Path

import pytest

from amazon_review_alignment.grpo_rewards import (
    evidence_reward,
    length_reward,
    schema_reward,
)
from amazon_review_alignment.train_grpo import (
    _finite_grpo_metrics,
    grpo_argument_values,
    grpo_reward_weights,
)


def _response(evidence: str = "Works well.", words: int = 4) -> str:
    return json.dumps(
        {
            "sentiment": "positive",
            "evidence": [evidence],
            "analysis": " ".join(["helpful"] * words),
        }
    )


def test_grpo_rule_rewards_are_independent() -> None:
    completions = [
        _response(),
        _response("invented evidence"),
        _response(words=81),
        "not-json",
    ]
    reviews = ["Works well."] * len(completions)

    assert schema_reward(completions) == [1.0, 1.0, 1.0, 0.0]
    assert evidence_reward(completions, text=reviews) == [1.0, 0.0, 1.0, 0.0]
    assert length_reward(completions) == [1.0, 1.0, 0.0, 0.0]


def test_grpo_argument_wiring() -> None:
    grpo = {
        "epochs": 1,
        "max_steps": 1,
        "learning_rate": 3e-6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "generation_batch_size": 2,
        "num_generations": 2,
        "num_iterations": 1,
        "max_prompt_length": 192,
        "max_completion_length": 64,
        "temperature": 0.7,
        "beta": 0.02,
        "epsilon": 0.2,
        "loss_type": "dapo",
        "scale_rewards": "group",
        "use_vllm": False,
        "logging_steps": 1,
        "save_steps": 1,
        "fp16": True,
        "reward_weights": {
            "reward_model": 1.0,
            "schema": 1.0,
            "evidence": 1.5,
            "length": 0.25,
        },
    }
    config = {"project": {"seed": 42}, "rlhf": {"grpo": grpo}}
    values = grpo_argument_values(config, Path("outputs/grpo"))

    assert values["num_generations"] == 2
    assert values["generation_batch_size"] == 2
    assert values["beta"] == 0.02
    assert values["use_vllm"] is False
    assert grpo_reward_weights(grpo) == [1.0, 1.0, 1.5, 0.25]


def test_grpo_metrics_reject_non_finite_values() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        _finite_grpo_metrics([{"reward": float("nan")}])


def test_grpo_metrics_extract_components() -> None:
    metrics = _finite_grpo_metrics(
        [
            {
                "reward": 1.5,
                "kl": 0.02,
                "entropy": 2.0,
                "rewards/schema_reward/mean": 0.75,
            }
        ]
    )

    assert metrics["reward"] == 1.5
    assert metrics["kl"] == 0.02
    assert metrics["rewards/schema_reward/mean"] == 0.75
