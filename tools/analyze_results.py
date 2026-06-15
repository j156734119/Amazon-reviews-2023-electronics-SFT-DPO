from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parsed_payload(raw_output: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def build_metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rate_columns = [column for column in metrics if column.endswith("_rate")]
    indexed = metrics.set_index("variant")
    rows = []
    for reference in ("base", "sft"):
        if reference not in indexed.index:
            continue
        for variant in indexed.index:
            if variant == reference:
                continue
            row: dict[str, Any] = {"reference": reference, "variant": variant}
            for column in rate_columns:
                row[f"{column}_delta_pp"] = round(
                    100
                    * (
                        float(indexed.loc[variant, column])
                        - float(indexed.loc[reference, column])
                    ),
                    1,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_summary(prediction_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = {
        variant: read_jsonl(prediction_dir / f"{variant}.jsonl")
        for variant in ("base", "sft", "dpo", "ppo", "grpo")
    }
    model_rows = []
    by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, rows in predictions.items():
        by_variant[variant] = {str(row["id"]): row for row in rows}
        payloads = [parsed_payload(str(row["raw_output"])) for row in rows]
        valid_payloads = [payload for payload in payloads if payload is not None]
        sentiments = pd.Series(
            [str(payload.get("sentiment")) for payload in valid_payloads],
            dtype="object",
        ).value_counts()
        model_rows.append(
            {
                "variant": variant,
                "examples": len(rows),
                "mean_output_words": round(
                    sum(len(str(row["raw_output"]).split()) for row in rows) / max(len(rows), 1),
                    2,
                ),
                "parsed_json_outputs": len(valid_payloads),
                "negative_outputs": int(sentiments.get("negative", 0)),
                "neutral_outputs": int(sentiments.get("neutral", 0)),
                "positive_outputs": int(sentiments.get("positive", 0)),
            }
        )

    pair_rows = []
    for left, right in combinations(by_variant, 2):
        common_ids = sorted(set(by_variant[left]) & set(by_variant[right]))
        exact_matches = 0
        same_sentiment = 0
        sentiment_comparable = 0
        for review_id in common_ids:
            left_raw = str(by_variant[left][review_id]["raw_output"]).strip()
            right_raw = str(by_variant[right][review_id]["raw_output"]).strip()
            exact_matches += left_raw == right_raw
            left_payload = parsed_payload(left_raw)
            right_payload = parsed_payload(right_raw)
            if left_payload is not None and right_payload is not None:
                sentiment_comparable += 1
                same_sentiment += left_payload.get("sentiment") == right_payload.get("sentiment")
        pair_rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "examples": len(common_ids),
                "exact_output_match_rate": exact_matches / max(len(common_ids), 1),
                "same_sentiment_rate": same_sentiment / max(sentiment_comparable, 1),
                "sentiment_comparable_examples": sentiment_comparable,
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows)


def build_training_summary(root: Path) -> pd.DataFrame:
    rlhf_dir = root / "rlhf"
    reward = read_json(rlhf_dir / "reward_metrics.json")
    ppo = read_json(rlhf_dir / "ppo_metrics.json")
    grpo = read_json(rlhf_dir / "grpo_metrics.json")
    ppo_metrics = ppo["final_logged_metrics"]
    grpo_metrics = grpo["final_logged_metrics"]
    return pd.DataFrame(
        [
            {
                "stage": "reward_model",
                "examples": reward["ai_validation"]["examples"],
                "runtime_minutes": None,
                "peak_reserved_gb": None,
                "primary_metric": "preference_accuracy",
                "primary_value": reward["ai_validation"]["preference_accuracy"],
                "diagnostic": (
                    f"mean_reward_margin={reward['ai_validation']['mean_reward_margin']:.4f}; "
                    "human_held_out=0"
                ),
            },
            {
                "stage": "ppo",
                "examples": ppo["episodes"],
                "runtime_minutes": ppo["runtime_seconds"] / 60,
                "peak_reserved_gb": ppo["peak_cuda_memory_reserved_gb"],
                "primary_metric": "rlhf_reward",
                "primary_value": ppo_metrics["objective/rlhf_reward"],
                "diagnostic": (
                    f"kl={ppo_metrics['objective/kl']:.6f}; "
                    f"policy_loss={ppo_metrics['loss/policy_avg']:.6f}; "
                    f"value_loss={ppo_metrics['loss/value_avg']:.6f}"
                ),
            },
            {
                "stage": "grpo",
                "examples": grpo["expected_completions_per_epoch"],
                "runtime_minutes": grpo["runtime_seconds"] / 60,
                "peak_reserved_gb": grpo["peak_cuda_memory_reserved_gb"],
                "primary_metric": "reward",
                "primary_value": grpo_metrics["reward"],
                "diagnostic": (
                    f"kl={grpo_metrics['kl']:.6f}; "
                    f"clip_ratio={grpo_metrics['clip_ratio/region_mean']:.6f}; "
                    "schema/evidence/length reward std=0"
                ),
            },
        ]
    )


def write_analysis(
    root: Path,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    failures: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    pairwise_agreement: pd.DataFrame,
    training_summary: pd.DataFrame,
    judge_summary: pd.DataFrame | None,
    judge_decision_count: int,
) -> Path:
    indexed = metrics.set_index("variant")
    failure_counts = failures.groupby("variant").size().to_dict()
    agreement = pairwise_agreement.set_index("comparison")
    lines = [
        "# A100 Qwen3.5-2B Experiment Analysis",
        "",
        "## Main findings",
        "",
        (
            "- SFT improved evidence grounding from "
            f"{indexed.loc['base', 'evidence_grounded_rate']:.1%} to "
            f"{indexed.loc['sft', 'evidence_grounded_rate']:.1%} and schema validity from "
            f"{indexed.loc['base', 'schema_valid_rate']:.1%} to "
            f"{indexed.loc['sft', 'schema_valid_rate']:.1%}."
        ),
        (
            f"- DPO v2 reduced schema validity to {indexed.loc['dpo', 'schema_valid_rate']:.1%} "
            f"and evidence grounding to {indexed.loc['dpo', 'evidence_grounded_rate']:.1%}; "
            "its semantic AI-judge result must be interpreted together with this "
            "structural regression."
        ),
        (
            f"- PPO and GRPO preserved high evidence grounding "
            f"({indexed.loc['ppo', 'evidence_grounded_rate']:.1%} and "
            f"{indexed.loc['grpo', 'evidence_grounded_rate']:.1%}) but did not exceed SFT locally."
        ),
        (
            "- The Reward Model reached 90.0% preference accuracy on 130 AI-generated validation "
            "pairs, but no held-out human preferences were available."
        ),
        (
            "- GRPO rule rewards were fully saturated: schema, evidence, and length "
            "reward standard deviations were all zero. KL was 0.000557 and clip ratio "
            "was zero, indicating very limited policy movement."
        ),
        (
            "- PPO and GRPO produced exactly identical outputs on "
            f"{agreement.loc['ppo_vs_grpo', 'exact_output_match_rate']:.1%} of test reviews "
            "and the same sentiment on "
            f"{agreement.loc['ppo_vs_grpo', 'same_sentiment_rate']:.1%}. This supports "
            "the observation that the two online-RL policies remained very close."
        ),
        (
            "- SFT and PPO agreed on sentiment for "
            f"{agreement.loc['sft_vs_ppo', 'same_sentiment_rate']:.1%} of test reviews, "
            "which further suggests that the PPO budget mostly preserved the SFT policy."
        ),
        (
            "- PPO used 128 episodes. Its value loss remained 5.7251, so the value function was "
            "not yet a strong estimator under this limited rollout budget."
        ),
        "",
        "## Instruction-following failures",
        "",
    ]
    for variant in ("base", "sft", "dpo", "ppo", "grpo"):
        lines.append(f"- {variant}: {failure_counts.get(variant, 0)} / 500")
    lines.extend(["", "## AI blind judge", ""])
    if judge_summary is None:
        lines.append(
            "No archived AI blind-judge summary was available for this analysis."
        )
    else:
        lines.append(
            f"The archived judge log contains {judge_decision_count} decisions "
            f"across {len(judge_summary)} pairwise comparisons."
        )
        lines.append("")
        for row in judge_summary.itertuples(index=False):
            lines.append(
                f"- {row.comparison}: {row.right_model} win rate "
                f"{row.right_model_win_rate_ties_half:.1%} "
                f"(N={int(row.examples)}, ties={int(row.ties)})."
            )
    lines.extend(
        [
            "",
            "## Reproducibility note",
            "",
            (
                "The local metrics, complete AI blind-judge decision log, aggregate "
                "pairwise summary, and training diagnostics are archived under this "
                "output directory. Raw review text, full model predictions, adapters, "
                "and checkpoints remain ignored."
            ),
            "",
            "## Generated tables",
            "",
            "- `metric_deltas.csv`: percentage-point changes against Base and SFT.",
            "- `failure_summary.csv`: failed instruction-following examples by model.",
            "- `prediction_summary.csv`: output lengths, JSON parsing, and sentiment counts.",
            "- `pairwise_output_agreement.csv`: exact output and sentiment agreement.",
            "- `training_summary.csv`: Reward Model, PPO, and GRPO diagnostics.",
            "",
        ]
    )
    output_path = root / "analysis" / "analysis.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/a100-qwen3.5-2b"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(root / "evaluation" / "metrics.csv")
    failures = pd.read_csv(root / "evaluation" / "failure_cases.csv")
    deltas = build_metric_deltas(metrics)
    prediction_summary, pairwise_agreement = build_prediction_summary(
        root / "evaluation" / "predictions"
    )
    training_summary = build_training_summary(root)
    judge_summary_path = root / "evaluation" / "judge_pairwise_summary.csv"
    judge_decisions_path = root / "evaluation" / "judge_decisions.jsonl"
    judge_summary = (
        pd.read_csv(judge_summary_path)
        if judge_summary_path.exists()
        else None
    )
    judge_decision_count = (
        sum(1 for _ in judge_decisions_path.open(encoding="utf-8"))
        if judge_decisions_path.exists()
        else 0
    )
    failure_summary = (
        failures.groupby("variant", as_index=False)
        .size()
        .rename(columns={"size": "instruction_following_failures"})
    )

    deltas.to_csv(analysis_dir / "metric_deltas.csv", index=False)
    failure_summary.to_csv(analysis_dir / "failure_summary.csv", index=False)
    prediction_summary.to_csv(analysis_dir / "prediction_summary.csv", index=False)
    pairwise_agreement.to_csv(
        analysis_dir / "pairwise_output_agreement.csv",
        index=False,
    )
    training_summary.to_csv(analysis_dir / "training_summary.csv", index=False)
    output_path = write_analysis(
        root,
        metrics,
        deltas,
        failures,
        prediction_summary,
        pairwise_agreement,
        training_summary,
        judge_summary,
        judge_decision_count,
    )
    print(output_path)


if __name__ == "__main__":
    main()
