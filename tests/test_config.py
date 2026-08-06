from pathlib import Path

import pytest

from amazon_review_alignment.cli import build_parser
from amazon_review_alignment.config import load_config
from amazon_review_alignment.inference import prediction_cache_matches
from amazon_review_alignment.utils import write_json, write_jsonl


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


def test_current_a100_profile_uses_expected_base_and_rlhf_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    a100 = load_config(root / "configs" / "rlhf_a100.yaml")

    assert a100["model"]["base_model"] == "Qwen/Qwen3.5-2B"
    assert a100["training"]["sft"]["fp16"] is False
    assert a100["training"]["sft"]["bf16"] is True
    assert a100["rlhf"]["human_calibration_samples"] == 0
    assert a100["rlhf"]["ppo_prompt_count"] == 128
    assert a100["rlhf"]["ppo"]["auxiliary_model_load_in_4bit"] is False
    assert a100["rlhf"]["grpo"]["prompt_count"] == 128
    assert a100["rlhf"]["grpo"]["auxiliary_model_load_in_4bit"] is False


def test_expanded_online_profile_uses_shared_1024_prompt_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "rlhf_a100_online_v2.yaml")

    assert config["rlhf"]["ppo_prompt_count"] == 1024
    assert config["rlhf"]["ppo"]["total_episodes"] == 1024
    assert config["rlhf"]["grpo"]["prompt_count"] == 1024
    assert config["rlhf"]["ppo"]["output_dir"].endswith("models/ppo-v2")
    assert config["rlhf"]["grpo"]["output_dir"].endswith("models/grpo-v2")
    assert len(config["evaluation"]["judge_pairs"]) == 20
    assert "qwen35_2b_fewshot" in config["evaluation"]["variants"]
    assert "deepseek_v4_pro_fewshot" in config["evaluation"]["variants"]


def test_evaluate_cli_accepts_baseline_only() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "configs/rlhf_a100_online_v2.yaml",
            "--variants",
            "base",
            "qwen35_2b_fewshot",
        ]
    )

    assert args.command == "evaluate"
    assert args.variants == ["base", "qwen35_2b_fewshot"]


def test_evaluate_cli_accepts_ai_judge_sample_override() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "configs/rlhf_a100_dpo_v2.yaml",
            "--llm-judge",
            "--judge-samples-per-pair",
            "50",
        ]
    )

    assert args.llm_judge is True
    assert args.judge_samples_per_pair == 50


def test_prediction_cache_requires_matching_adapter_metadata(tmp_path: Path) -> None:
    prediction_path = tmp_path / "ppo.jsonl"
    metadata_dir = tmp_path / "ppo_run"
    target_ids = ["r1"]
    expected = {
        "variant": "ppo",
        "base_model": "outputs/a100-qwen3.5-2b/models/sft-merged",
        "adapter_path": "outputs/a100-qwen3.5-2b/models/ppo-v2",
        "examples": 1,
        "target_ids_sha256": "abc",
    }
    write_jsonl(
        prediction_path,
        [
            {
                "id": "r1",
                "text": "review",
                "variant": "ppo",
                "raw_output": "{}",
            }
        ],
    )
    write_json(
        metadata_dir / "run_metadata.yaml",
        {
            **expected,
            "adapter_path": "outputs/a100-qwen3.5-2b/models/ppo",
        },
    )

    assert not prediction_cache_matches(
        prediction_path,
        metadata_dir,
        expected,
        target_ids,
    )

    write_json(metadata_dir / "run_metadata.yaml", expected)

    assert prediction_cache_matches(
        prediction_path,
        metadata_dir,
        expected,
        target_ids,
    )
