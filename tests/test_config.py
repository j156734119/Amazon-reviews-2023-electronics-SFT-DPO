from pathlib import Path

import pytest

from amazon_review_alignment.cli import build_parser
from amazon_review_alignment.config import load_config


def test_config_inheritance_deep_merges_relative_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "project:\n  seed: 42\n  output_dir: outputs\n"
        "training:\n  ppo:\n    batch_size: 1\n    epochs: 4\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: parent.yaml\ntraining:\n  ppo:\n    epochs: 1\n",
        encoding="utf-8",
    )

    config = load_config(child)

    assert config["project"]["seed"] == 42
    assert config["training"]["ppo"]["batch_size"] == 1
    assert config["training"]["ppo"]["epochs"] == 1
    assert config["_config_path"] == str(child.resolve())


def test_config_inheritance_rejects_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\nvalue: 1\n", encoding="utf-8")
    second.write_text("extends: first.yaml\nvalue: 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        load_config(first)


def test_model_profiles_separate_smoke_and_a100_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke = load_config(root / "configs" / "rlhf_smoke.yaml")
    a100 = load_config(root / "configs" / "rlhf_a100.yaml")

    assert smoke["model"]["base_model"] == "Qwen/Qwen3-0.6B"
    assert smoke["training"]["sft"]["fp16"] is True
    assert smoke["training"]["sft"]["bf16"] is False
    assert a100["model"]["base_model"] == "Qwen/Qwen3.5-2B"
    assert a100["training"]["sft"]["fp16"] is False
    assert a100["training"]["sft"]["bf16"] is True
    assert a100["rlhf"]["human_calibration_samples"] == 0
    assert a100["rlhf"]["ppo_prompt_count"] == 128
    assert a100["rlhf"]["grpo"]["prompt_count"] == 128
    assert smoke["project"]["output_dir"] != a100["project"]["output_dir"]


def test_evaluate_cli_accepts_baseline_only() -> None:
    args = build_parser().parse_args(
        ["evaluate", "--config", "configs/rlhf_smoke.yaml", "--variants", "base"]
    )

    assert args.command == "evaluate"
    assert args.variants == ["base"]
