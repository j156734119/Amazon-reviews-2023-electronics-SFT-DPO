from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_root
from .evaluation import bootstrap_interval
from .utils import read_jsonl, save_run_metadata, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def prepare_human_evaluation(
    config: dict[str, Any],
    samples: int,
    left_variant: str = "ppo",
    right_variant: str = "grpo",
) -> Path:
    prediction_dir = output_root(config) / "evaluation" / "predictions"
    left_rows = {
        row["id"]: row for row in read_jsonl(prediction_dir / f"{left_variant}.jsonl")
    }
    right_rows = {
        row["id"]: row for row in read_jsonl(prediction_dir / f"{right_variant}.jsonl")
    }
    common_ids = sorted(set(left_rows) & set(right_rows))
    if not common_ids:
        raise RuntimeError("No matching prediction rows were found. Run evaluate first.")
    rng = random.Random(int(config["project"]["seed"]))
    selected_ids = rng.sample(common_ids, min(samples, len(common_ids)))
    records = []
    key_records = []
    for review_id in selected_ids:
        left = left_rows[review_id]
        right = right_rows[review_id]
        swapped = bool(rng.getrandbits(1))
        if swapped:
            model_a, response_a = right_variant, right["raw_output"]
            model_b, response_b = left_variant, left["raw_output"]
        else:
            model_a, response_a = left_variant, left["raw_output"]
            model_b, response_b = right_variant, right["raw_output"]
        records.append(
            {
                "id": review_id,
                "review_text": left["text"],
                "response_a": response_a,
                "response_b": response_b,
                "choice": "",
                "notes": "",
            }
        )
        key_records.append(
            {
                "id": review_id,
                "model_a": model_a,
                "model_b": model_b,
            }
        )
    evaluation_dir = output_root(config) / "evaluation"
    path = evaluation_dir / "human_eval_responses.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    write_jsonl(evaluation_dir / "human_eval_key.jsonl", key_records)
    save_run_metadata(
        evaluation_dir / "human_eval_run",
        config,
        "human-eval-prepare",
        {
            "samples": len(records),
            "comparison": f"{left_variant}_vs_{right_variant}",
            "instructions": (
                "Choose A, B, or tie based on faithfulness, evidence, "
                "concision, and usefulness."
            ),
        },
    )
    LOGGER.info("Prepared blinded human evaluation at %s", path)
    return path


def summarize_human_evaluation(
    config: dict[str, Any],
    responses_path: str | Path,
) -> dict[str, Any]:
    frame = pd.read_csv(responses_path, keep_default_na=False)
    required = {"id", "choice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Human evaluation CSV is missing columns: {sorted(missing)}")
    normalized = frame["choice"].astype(str).str.strip().str.lower()
    valid = normalized.isin({"a", "b", "tie"})
    if not valid.all():
        invalid_rows = frame.index[~valid].tolist()[:10]
        raise ValueError(f"Choices must be A, B, or tie. Invalid rows: {invalid_rows}")
    frame["choice"] = normalized
    key_path = output_root(config) / "evaluation" / "human_eval_key.jsonl"
    if not key_path.exists():
        raise RuntimeError(f"Human evaluation key is missing: {key_path}")
    key_frame = pd.DataFrame(read_jsonl(key_path))
    frame["id"] = frame["id"].astype(str)
    key_frame["id"] = key_frame["id"].astype(str)
    frame = frame.merge(key_frame, on="id", how="left", validate="one_to_one")
    if frame[["model_a", "model_b"]].isna().any().any():
        raise ValueError("Some human evaluation rows do not match the blind key.")
    model_names = sorted(set(frame["model_a"]) | set(frame["model_b"]))
    if len(model_names) != 2:
        raise ValueError("Human evaluation summary expects exactly two model variants.")
    left, right = model_names
    right_scores = []
    winners = []
    for _, row in frame.iterrows():
        if row["choice"] == "tie":
            winner = "tie"
            score = 0.5
        else:
            winner = row["model_a"] if row["choice"] == "a" else row["model_b"]
            score = 1.0 if winner == right else 0.0
        winners.append(winner)
        right_scores.append(score)
    mean, low, high = bootstrap_interval(
        right_scores,
        int(config["evaluation"]["bootstrap_samples"]),
        int(config["project"]["seed"]),
    )
    summary = {
        "examples": len(frame),
        "left_model": left,
        "right_model": right,
        "right_model_win_rate_ties_half": mean,
        "ci_95_low": low,
        "ci_95_high": high,
        "left_wins": winners.count(left),
        "right_wins": winners.count(right),
        "ties": winners.count("tie"),
    }
    output_path = output_root(config) / "evaluation" / "human_eval_summary.json"
    write_json(output_path, summary)
    return summary
