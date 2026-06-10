from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_root
from .utils import read_jsonl, save_run_metadata, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)
VALID_CHOICES = {"a", "b", "tie"}


def _rlhf_dir(config: dict[str, Any]) -> Path:
    path = output_root(config) / "rlhf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_rm_human_eval(
    config: dict[str, Any],
    samples: int | None = None,
) -> Path:
    preferences = read_jsonl(output_root(config) / "teacher" / "preferences_train.jsonl")
    if not preferences:
        raise RuntimeError("Training preferences are missing. Complete teacher-batch first.")
    requested = int(samples or config["rlhf"]["human_calibration_samples"])
    if requested > len(preferences):
        raise ValueError(
            f"Requested {requested} human pairs but only {len(preferences)} are available."
        )

    rng = random.Random(int(config["project"]["seed"]))
    selected = rng.sample(preferences, requested)
    responses: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in selected:
        swapped = bool(rng.getrandbits(1))
        if swapped:
            response_a, response_b = row["rejected"], row["chosen"]
            origin_a, origin_b = "rejected", "chosen"
        else:
            response_a, response_b = row["chosen"], row["rejected"]
            origin_a, origin_b = "chosen", "rejected"
        responses.append(
            {
                "id": row["id"],
                "review_text": row["text"],
                "response_a": response_a,
                "response_b": response_b,
                "choice": "",
                "notes": "",
            }
        )
        key.append(
            {
                "id": row["id"],
                "origin_a": origin_a,
                "origin_b": origin_b,
            }
        )

    directory = _rlhf_dir(config)
    responses_path = directory / "rm_human_responses.csv"
    pd.DataFrame(responses).to_csv(responses_path, index=False)
    write_jsonl(directory / "rm_human_key.jsonl", key)
    save_run_metadata(
        directory / "human_calibration_run",
        config,
        "prepare-rm-human-eval",
        {
            "samples": requested,
            "instructions": (
                "Choose A, B, or tie using faithfulness, evidence grounding, "
                "format compliance, concision, and usefulness."
            ),
        },
    )
    LOGGER.info("Prepared %s blinded RM calibration pairs at %s", requested, responses_path)
    return responses_path


def _load_human_preferences(
    config: dict[str, Any],
    responses_path: str | Path,
    source_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    frame = pd.read_csv(responses_path, keep_default_na=False, dtype={"id": str})
    required = {"id", "response_a", "response_b", "choice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Human calibration CSV is missing columns: {sorted(missing)}")
    choices = frame["choice"].astype(str).str.strip().str.lower()
    invalid = ~choices.isin(VALID_CHOICES)
    if invalid.any():
        raise ValueError(
            "All human calibration rows must be A, B, or tie. "
            f"Invalid rows: {frame.index[invalid].tolist()[:10]}"
        )
    frame["choice"] = choices

    key_path = _rlhf_dir(config) / "rm_human_key.jsonl"
    if not key_path.exists():
        raise RuntimeError(f"Human calibration key is missing: {key_path}")
    key_frame = pd.DataFrame(read_jsonl(key_path))
    key_frame["id"] = key_frame["id"].astype(str)
    frame = frame.merge(key_frame, on="id", how="left", validate="one_to_one")
    if frame[["origin_a", "origin_b"]].isna().any().any():
        raise ValueError("Some human calibration rows do not match the blind key.")

    calibrated = []
    ties = 0
    for _, row in frame.iterrows():
        review_id = str(row["id"])
        if review_id not in source_by_id:
            raise ValueError(f"Unknown preference id in human calibration: {review_id}")
        if row["choice"] == "tie":
            ties += 1
            continue
        preferred = row["response_a"] if row["choice"] == "a" else row["response_b"]
        rejected = row["response_b"] if row["choice"] == "a" else row["response_a"]
        source = source_by_id[review_id]
        calibrated.append(
            {
                "id": review_id,
                "text": source["text"],
                "prompt": source["prompt"],
                "chosen": preferred,
                "rejected": rejected,
                "source": "human_calibrated",
                "human_choice": row["choice"],
            }
        )
    return calibrated, ties


def _preference_record(row: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "text": row["text"],
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
        "source": source,
    }


def _assert_disjoint(named_ids: dict[str, set[str]]) -> None:
    names = list(named_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = named_ids[left] & named_ids[right]
            if overlap:
                raise ValueError(
                    f"RLHF data leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )


def build_rlhf_data(
    config: dict[str, Any],
    responses_path: str | Path,
) -> dict[str, Any]:
    seed = int(config["project"]["seed"])
    rng = random.Random(seed)
    root = output_root(config)
    train_preferences = read_jsonl(root / "teacher" / "preferences_train.jsonl")
    validation_preferences = read_jsonl(
        root / "teacher" / "preferences_validation.jsonl"
    )
    test_rows = read_jsonl(root / "data" / "test.jsonl")
    if not train_preferences or not validation_preferences:
        raise RuntimeError("Teacher preference data is missing.")

    source_by_id = {str(row["id"]): row for row in train_preferences}
    human_pairs, ties = _load_human_preferences(
        config,
        responses_path,
        source_by_id,
    )
    if len(human_pairs) < 2:
        raise ValueError("At least two non-tie human preferences are required.")
    rng.shuffle(human_pairs)
    train_fraction = float(config["rlhf"]["human_train_fraction"])
    split_index = max(1, min(len(human_pairs) - 1, round(len(human_pairs) * train_fraction)))
    human_train = human_pairs[:split_index]
    human_eval = human_pairs[split_index:]

    human_ids = {row["id"] for row in human_pairs}
    ai_candidates = [
        row for row in train_preferences if str(row["id"]) not in human_ids
    ]
    rng.shuffle(ai_candidates)
    ai_count = int(config["rlhf"]["ai_reward_train_pairs"])
    ppo_count = int(config["rlhf"]["ppo_prompt_count"])
    if len(ai_candidates) < ai_count + ppo_count:
        raise ValueError(
            "Not enough unused training preferences for Reward Model and PPO data: "
            f"need {ai_count + ppo_count}, have {len(ai_candidates)}."
        )

    ai_train_raw = ai_candidates[:ai_count]
    ppo_raw = ai_candidates[ai_count : ai_count + ppo_count]
    validation_count = min(
        int(config["rlhf"]["ai_reward_validation_pairs"]),
        len(validation_preferences),
    )
    ai_validation_raw = list(validation_preferences)
    rng.shuffle(ai_validation_raw)
    ai_validation_raw = ai_validation_raw[:validation_count]

    rm_train = human_train + [
        _preference_record(row, "ai_teacher") for row in ai_train_raw
    ]
    rng.shuffle(rm_train)
    rm_ai_validation = [
        _preference_record(row, "ai_teacher_validation")
        for row in ai_validation_raw
    ]
    ppo_prompts = [
        {
            "id": row["id"],
            "text": row["text"],
            "prompt": row["prompt"],
            "source": "unused_teacher_train_prompt",
        }
        for row in ppo_raw
    ]

    sets = {
        "human_train": {row["id"] for row in human_train},
        "human_eval": {row["id"] for row in human_eval},
        "ai_rm_train": {str(row["id"]) for row in ai_train_raw},
        "ppo_prompts": {str(row["id"]) for row in ppo_raw},
        "ai_rm_validation": {str(row["id"]) for row in ai_validation_raw},
        "test": {str(row["id"]) for row in test_rows},
    }
    _assert_disjoint(sets)

    directory = _rlhf_dir(config)
    write_jsonl(directory / "rm_train.jsonl", rm_train)
    write_jsonl(directory / "rm_human_eval.jsonl", human_eval)
    write_jsonl(directory / "rm_ai_validation.jsonl", rm_ai_validation)
    write_jsonl(directory / "ppo_prompts.jsonl", ppo_prompts)
    manifest = {
        "human_total_rows": len(human_pairs) + ties,
        "human_non_tie_rows": len(human_pairs),
        "human_ties_dropped": ties,
        "human_train_pairs": len(human_train),
        "human_eval_pairs": len(human_eval),
        "ai_reward_train_pairs": len(ai_train_raw),
        "ai_reward_validation_pairs": len(ai_validation_raw),
        "reward_train_pairs_total": len(rm_train),
        "ppo_prompts": len(ppo_prompts),
        "test_rows_excluded": len(test_rows),
        "seed": seed,
    }
    write_json(directory / "data_manifest.json", manifest)
    save_run_metadata(directory / "data_run", config, "build-rlhf-data", manifest)
    LOGGER.info("Built human-calibrated RLAIF data under %s", directory)
    return manifest
