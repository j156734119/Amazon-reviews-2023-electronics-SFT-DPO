from pathlib import Path

import pandas as pd

from amazon_review_alignment.rlhf_data import (
    build_rlhf_data,
    prepare_rm_human_eval,
)
from amazon_review_alignment.utils import read_jsonl, write_jsonl


def _preference(index: int, split: str = "train") -> dict:
    return {
        "id": f"{split}-{index}",
        "split": split,
        "text": f"Review text {index}.",
        "prompt": f"Analyze review {index}.",
        "chosen": f"chosen response {index}",
        "rejected": f"rejected response {index}",
    }


def _raw_train(index: int) -> dict:
    return {
        "id": f"raw-{index}",
        "text": f"Raw training review {index}.",
    }


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"seed": 42, "output_dir": str(tmp_path)},
        "rlhf": {
            "human_calibration_samples": 4,
            "human_train_fraction": 0.5,
            "ai_reward_train_pairs": 4,
            "ai_reward_validation_pairs": 3,
            "ppo_prompt_count": 4,
        },
    }


def test_human_calibration_builds_disjoint_rlhf_data(tmp_path: Path) -> None:
    teacher_dir = tmp_path / "teacher"
    data_dir = tmp_path / "data"
    write_jsonl(
        teacher_dir / "preferences_train.jsonl",
        [_preference(index) for index in range(16)],
    )
    write_jsonl(
        teacher_dir / "preferences_validation.jsonl",
        [_preference(index, "validation") for index in range(4)],
    )
    write_jsonl(
        data_dir / "train.jsonl",
        [_raw_train(index) for index in range(12)],
    )
    write_jsonl(
        data_dir / "test.jsonl",
        [{"id": f"test-{index}", "text": "held out"} for index in range(3)],
    )
    config = _config(tmp_path)
    responses_path = prepare_rm_human_eval(config)
    frame = pd.read_csv(responses_path, keep_default_na=False)
    frame["choice"] = ["A", "B", "tie", "A"]
    frame.to_csv(responses_path, index=False)

    manifest = build_rlhf_data(config, responses_path)

    assert manifest["human_non_tie_rows"] == 3
    assert manifest["human_ties_dropped"] == 1
    assert manifest["ai_reward_train_pairs"] == 4
    assert manifest["ppo_prompts"] == 4
    rm_train = read_jsonl(tmp_path / "rlhf" / "rm_train.jsonl")
    human_eval = read_jsonl(tmp_path / "rlhf" / "rm_human_eval.jsonl")
    ppo = read_jsonl(tmp_path / "rlhf" / "ppo_prompts.jsonl")
    grpo = read_jsonl(tmp_path / "rlhf" / "grpo_prompts.jsonl")
    ids = [
        {row["id"] for row in rm_train},
        {row["id"] for row in human_eval},
        {row["id"] for row in ppo},
    ]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert all(row["source"] == "raw_train_prompt_excluded_from_reward_model" for row in ppo)
    assert [row["id"] for row in ppo] == [row["id"] for row in grpo]
    assert manifest["ppo_grpo_shared_prompt_ids"] is True
    assert manifest["online_prompt_source"] == "raw_train_excluded_from_reward_model"


def test_human_choice_controls_preference_direction(tmp_path: Path) -> None:
    teacher_dir = tmp_path / "teacher"
    data_dir = tmp_path / "data"
    write_jsonl(
        teacher_dir / "preferences_train.jsonl",
        [_preference(index) for index in range(12)],
    )
    write_jsonl(
        teacher_dir / "preferences_validation.jsonl",
        [_preference(index, "validation") for index in range(3)],
    )
    write_jsonl(
        data_dir / "train.jsonl",
        [_raw_train(index) for index in range(12)],
    )
    write_jsonl(data_dir / "test.jsonl", [{"id": "test-1", "text": "held out"}])
    config = _config(tmp_path)
    responses_path = prepare_rm_human_eval(config)
    frame = pd.read_csv(responses_path, keep_default_na=False)
    expected = {}
    for index, row in frame.iterrows():
        choice = "A" if index % 2 == 0 else "B"
        frame.loc[index, "choice"] = choice
        expected[str(row["id"])] = (
            row["response_a"] if choice == "A" else row["response_b"]
        )
    frame.to_csv(responses_path, index=False)

    build_rlhf_data(config, responses_path)

    calibrated = [
        *read_jsonl(tmp_path / "rlhf" / "rm_train.jsonl"),
        *read_jsonl(tmp_path / "rlhf" / "rm_human_eval.jsonl"),
    ]
    calibrated = [row for row in calibrated if row["source"] == "human_calibrated"]
    assert {row["id"]: row["chosen"] for row in calibrated} == expected


def test_pure_rlaif_builds_without_human_response_file(tmp_path: Path) -> None:
    teacher_dir = tmp_path / "teacher"
    data_dir = tmp_path / "data"
    write_jsonl(
        teacher_dir / "preferences_train.jsonl",
        [_preference(index) for index in range(12)],
    )
    write_jsonl(
        teacher_dir / "preferences_validation.jsonl",
        [_preference(index, "validation") for index in range(3)],
    )
    write_jsonl(
        data_dir / "train.jsonl",
        [_raw_train(index) for index in range(12)],
    )
    write_jsonl(data_dir / "test.jsonl", [{"id": "test-1", "text": "held out"}])
    config = _config(tmp_path)
    config["rlhf"]["human_calibration_samples"] = 0

    manifest = build_rlhf_data(config)

    assert manifest["alignment_method"] == "rlaif"
    assert manifest["preference_source"] == "ai_teacher"
    assert manifest["human_total_rows"] == 0
    assert manifest["human_train_pairs"] == 0
    assert manifest["human_eval_pairs"] == 0
    assert manifest["ai_reward_train_pairs"] == 4
    assert manifest["ppo_prompts"] == 4
    assert read_jsonl(tmp_path / "rlhf" / "rm_human_eval.jsonl") == []
