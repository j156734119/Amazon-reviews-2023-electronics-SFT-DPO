from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import output_root
from .inference import run_inference
from .prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt
from .schemas import JudgeChoice, JudgeDecision, parse_analysis
from .utils import normalized_match, read_jsonl, save_run_metadata, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def _response_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evaluate_output(raw_output: str, review_text: str) -> dict[str, Any]:
    try:
        parsed = parse_analysis(raw_output)
    except (json.JSONDecodeError, ValueError):
        return {
            "schema_valid": False,
            "evidence_grounded": False,
            "word_limit_ok": False,
            "instruction_following": False,
            "sentiment": None,
        }
    grounded = all(normalized_match(review_text, span) for span in parsed.evidence)
    word_limit_ok = len(parsed.analysis.split()) <= 80
    return {
        "schema_valid": True,
        "evidence_grounded": grounded,
        "word_limit_ok": word_limit_ok,
        "instruction_following": grounded and word_limit_ok,
        "sentiment": parsed.sentiment.value,
    }


def evaluate_prediction_file(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    detailed = []
    for row in rows:
        result = evaluate_output(row["raw_output"], row["text"])
        detailed.append({**row, **result})
    total = max(len(detailed), 1)
    metric_names = (
        "schema_valid",
        "evidence_grounded",
        "word_limit_ok",
        "instruction_following",
    )
    metrics = {
        "variant": rows[0]["variant"] if rows else Path(path).stem,
        "examples": len(detailed),
        **{
            f"{name}_rate": sum(bool(row[name]) for row in detailed) / total
            for name in metric_names
        },
    }
    return metrics, detailed


def bootstrap_interval(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _judge_pair(
    client: Any,
    judge_model: str,
    review_text: str,
    response_a: str,
    response_b: str,
) -> JudgeDecision:
    response = client.responses.parse(
        model=judge_model,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": judge_user_prompt(review_text, response_a, response_b),
            },
        ],
        text_format=JudgeDecision,
        max_output_tokens=180,
    )
    if response.output_parsed is None:
        raise ValueError("Judge response did not contain parsed output.")
    return response.output_parsed


def run_llm_judge(
    config: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    resume_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required when LLM judging is enabled.")
    from openai import OpenAI

    client = OpenAI()
    rng = random.Random(int(config["project"]["seed"]))
    by_variant = {
        variant: {row["id"]: row for row in rows}
        for variant, rows in predictions.items()
    }
    judge_model = str(config["evaluation"]["judge_model"])
    existing_rows = (
        read_jsonl(resume_path)
        if resume_path is not None and Path(resume_path).exists()
        else []
    )
    existing = {
        (
            row.get("comparison"),
            str(row.get("id")),
            row.get("response_a_sha256"),
            row.get("response_b_sha256"),
        ): row
        for row in existing_rows
        if row.get("judge_model") == judge_model
    }
    decisions = []
    for left, right in config["evaluation"]["judge_pairs"]:
        common_ids = sorted(set(by_variant[left]) & set(by_variant[right]))
        rng.shuffle(common_ids)
        pair_limit = int(config["evaluation"].get("judge_samples_per_pair", len(common_ids)))
        selected_ids = common_ids[:pair_limit]
        for index, review_id in enumerate(selected_ids, start=1):
            comparison = f"{left}_vs_{right}"
            left_row = by_variant[left][review_id]
            right_row = by_variant[right][review_id]
            swapped = bool(rng.getrandbits(1))
            if swapped:
                shown = [(right, right_row["raw_output"]), (left, left_row["raw_output"])]
            else:
                shown = [(left, left_row["raw_output"]), (right, right_row["raw_output"])]
            response_a_hash = _response_hash(str(shown[0][1]))
            response_b_hash = _response_hash(str(shown[1][1]))
            existing_decision = existing.get(
                (
                    comparison,
                    str(review_id),
                    response_a_hash,
                    response_b_hash,
                )
            )
            if existing_decision is not None:
                decisions.append(existing_decision)
                if index % 10 == 0 or index == len(selected_ids):
                    LOGGER.info(
                        "AI judge %s: %s/%s (resumed)",
                        comparison,
                        index,
                        len(selected_ids),
                    )
                continue
            decision = _judge_pair(
                client,
                judge_model,
                left_row["text"],
                shown[0][1],
                shown[1][1],
            )
            winner = None
            if decision.choice == JudgeChoice.A:
                winner = shown[0][0]
            elif decision.choice == JudgeChoice.B:
                winner = shown[1][0]
            row = {
                "id": review_id,
                "comparison": comparison,
                "left_model": left,
                "right_model": right,
                "display_a_model": shown[0][0],
                "display_b_model": shown[1][0],
                "choice": decision.choice.value,
                "winner": winner or "tie",
                "reason": decision.reason,
                "judge_model": judge_model,
                "response_a_sha256": response_a_hash,
                "response_b_sha256": response_b_hash,
            }
            decisions.append(row)
            if resume_path is not None:
                checkpoint_path = Path(resume_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(selected_ids):
                LOGGER.info(
                    "AI judge %s: %s/%s",
                    comparison,
                    index,
                    len(selected_ids),
                )
    return decisions


def summarize_pairwise(
    decisions: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[row["comparison"]].append(row)
    summaries = []
    for comparison, rows in grouped.items():
        left = rows[0]["left_model"]
        right = rows[0]["right_model"]
        right_scores = [
            1.0 if row["winner"] == right else 0.0 if row["winner"] == left else 0.5
            for row in rows
        ]
        mean, low, high = bootstrap_interval(right_scores, bootstrap_samples, seed)
        summaries.append(
            {
                "comparison": comparison,
                "examples": len(rows),
                "left_model": left,
                "right_model": right,
                "right_model_win_rate_ties_half": mean,
                "ci_95_low": low,
                "ci_95_high": high,
                "ties": sum(row["winner"] == "tie" for row in rows),
            }
        )
    return summaries


def run_evaluation(config: dict[str, Any], force_inference: bool = False) -> dict[str, Any]:
    evaluation_dir = output_root(config) / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    variants = list(config["evaluation"].get("variants", ["base", "sft", "dpo"]))
    prediction_paths = {
        variant: run_inference(config, variant, force=force_inference)
        for variant in variants
    }
    metrics = []
    predictions: dict[str, list[dict[str, Any]]] = {}
    failures = []
    for variant, path in prediction_paths.items():
        variant_metrics, detailed = evaluate_prediction_file(path)
        metrics.append(variant_metrics)
        predictions[variant] = read_jsonl(path)
        failures.extend(
            {
                "variant": variant,
                "id": row["id"],
                "schema_valid": row["schema_valid"],
                "evidence_grounded": row["evidence_grounded"],
                "raw_output": row["raw_output"],
            }
            for row in detailed
            if not row["instruction_following"]
        )

    pd.DataFrame(metrics).to_csv(evaluation_dir / "metrics.csv", index=False)
    write_jsonl(evaluation_dir / "failure_cases.jsonl", failures)
    pd.DataFrame(failures).to_csv(evaluation_dir / "failure_cases.csv", index=False)
    result: dict[str, Any] = {"local_metrics": metrics}

    if bool(config["evaluation"].get("run_llm_judge")):
        decisions_path = evaluation_dir / "judge_decisions.jsonl"
        decisions = run_llm_judge(config, predictions, resume_path=decisions_path)
        write_jsonl(decisions_path, decisions)
        summaries = summarize_pairwise(
            decisions,
            int(config["evaluation"]["bootstrap_samples"]),
            int(config["project"]["seed"]),
        )
        pd.DataFrame(summaries).to_csv(
            evaluation_dir / "judge_pairwise_summary.csv",
            index=False,
        )
        result["judge_pairwise"] = summaries

    write_json(evaluation_dir / "evaluation_summary.json", result)
    save_run_metadata(evaluation_dir / "run", config, "evaluate", result)
    return result
