from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import output_root
from .utils import save_run_metadata

LOGGER = logging.getLogger(__name__)


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_rate(value: Any) -> str:
    return f"{float(value):.1%}"


def _build_metrics_plot(metrics: pd.DataFrame, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; skipping the metrics plot.")
        return False
    columns = [
        "schema_valid_rate",
        "evidence_grounded_rate",
        "instruction_following_rate",
    ]
    plot_frame = metrics.set_index("variant")[columns]
    axes = plot_frame.plot(kind="bar", figsize=(8, 5), ylim=(0, 1))
    axes.set_ylabel("Rate")
    axes.set_xlabel("Model variant")
    axes.set_title("Structured review analysis quality")
    axes.legend(
        ["Schema valid", "Evidence grounded", "Instruction following"],
        loc="lower right",
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def build_report(config: dict[str, Any]) -> Path:
    root = output_root(config)
    evaluation_dir = root / "evaluation"
    metrics_path = evaluation_dir / "metrics.csv"
    if not metrics_path.exists():
        raise RuntimeError("Evaluation metrics are missing. Run evaluate first.")
    metrics = pd.read_csv(metrics_path)
    plot_path = evaluation_dir / "metrics.png"
    has_plot = _build_metrics_plot(metrics, plot_path)
    judge = (
        pd.read_csv(evaluation_dir / "judge_pairwise_summary.csv")
        if (evaluation_dir / "judge_pairwise_summary.csv").exists()
        else None
    )
    human = _read_json_if_exists(evaluation_dir / "human_eval_summary.json")
    reward = _read_json_if_exists(root / "rlhf" / "reward_metrics.json")
    ppo = _read_json_if_exists(root / "rlhf" / "ppo_metrics.json")
    grpo = _read_json_if_exists(root / "rlhf" / "grpo_metrics.json")
    human_calibration_samples = int(
        config.get("rlhf", {}).get("human_calibration_samples", 0)
    )

    lines = [
        "# Amazon Review Alignment Results",
        "",
        "## Research question",
        "",
        "How do SFT, DPO, PPO, and GRPO affect a small model's structured, "
        "evidence-grounded Amazon review analyses under a constrained compute budget?",
        "",
        "The comparison uses identical prompts, decoding settings, and held-out reviews. "
        "The previous DistilBERT classification result is historical context and is not "
        "treated as a directly comparable baseline.",
        "",
        "## Local evaluation",
        "",
        (
            "| Variant | Examples | Schema valid | Evidence grounded | "
            "Word limit | Instruction following |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['variant']} | {int(row['examples'])} | "
            f"{_format_rate(row['schema_valid_rate'])} | "
            f"{_format_rate(row['evidence_grounded_rate'])} | "
            f"{_format_rate(row['word_limit_ok_rate'])} | "
            f"{_format_rate(row['instruction_following_rate'])} |"
        )
    if has_plot:
        lines.extend(["", "![Evaluation metrics](metrics.png)"])

    baseline_labels = {
        "qwen35_2b_fewshot": "HF prompt-only baseline",
        "nlptown_template": "sentiment-classifier template baseline",
        "deepseek_v4_pro_fewshot": "API strong baseline",
    }
    present_baselines = [
        str(row["variant"])
        for _, row in metrics.iterrows()
        if str(row["variant"]) in baseline_labels
    ]
    if present_baselines:
        lines.extend(
            [
                "",
                "## Baseline comparison",
                "",
                ("These baselines are not trained by this project. They locate the "
                "post-trained Qwen policies against prompt-only, template, and "
                "external API references."),
                "",
                "| Baseline | Type |",
                "|---|---|",
            ]
        )
        for variant in present_baselines:
            lines.append(f"| {variant} | {baseline_labels[variant]} |")

    lines.extend(["", "## Pairwise evaluation", ""])
    if judge is None and human is None:
        lines.append(
            "No completed pairwise evaluation was found. This section remains intentionally "
            "open until LLM judging or blinded human responses are available."
        )
    if judge is not None:
        lines.extend(
            [
                "### LLM judge",
                "",
                "| Comparison | N | Right-model win rate | 95% CI | Ties |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in judge.iterrows():
            lines.append(
                f"| {row['comparison']} | {int(row['examples'])} | "
                f"{_format_rate(row['right_model_win_rate_ties_half'])} | "
                f"{_format_rate(row['ci_95_low'])}–{_format_rate(row['ci_95_high'])} | "
                f"{int(row['ties'])} |"
            )
    if human is not None:
        lines.extend(
            [
                "",
                "### Human blind evaluation",
                "",
                f"- Evaluated examples: {human['examples']}",
                f"- Comparison: {human['left_model']} vs {human['right_model']}",
                f"- {human['right_model']} win rate with ties counted as half: "
                f"{_format_rate(human['right_model_win_rate_ties_half'])}",
                f"- Bootstrap 95% CI: {_format_rate(human['ci_95_low'])}–"
                f"{_format_rate(human['ci_95_high'])}",
                f"- Raw outcomes: {human['left_wins']} / {human['right_wins']} / "
                f"{human['ties']} (left / right / tie)",
            ]
        )

    if reward is not None:
        lines.extend(
            [
                "",
                "## Reward Model",
                "",
                "| Split | Examples | Preference accuracy | Mean reward margin |",
                "|---|---:|---:|---:|",
            ]
        )
        for split_name, values in reward.items():
            lines.append(
                f"| {split_name} | {int(values['examples'])} | "
                f"{_format_rate(values['preference_accuracy'])} | "
                f"{float(values['mean_reward_margin']):.4f} |"
            )
    if ppo is not None:
        lines.extend(
            [
                "",
                "## PPO feasibility metrics",
                "",
                f"- Episodes: {ppo['episodes']}",
                f"- Unique prompts: {ppo['unique_prompts']}",
                f"- Runtime: {float(ppo['runtime_seconds']) / 60:.1f} minutes",
                "- Peak allocated CUDA memory: "
                f"{float(ppo['peak_cuda_memory_allocated_gb']):.2f} GiB",
                "- Peak reserved CUDA memory: "
                f"{float(ppo['peak_cuda_memory_reserved_gb']):.2f} GiB",
                "- Reference policy: merged SFT policy with the PPO adapter disabled.",
            ]
        )
        for name, value in ppo.get("final_logged_metrics", {}).items():
            lines.append(f"- `{name}`: {float(value):.6f}")
    if grpo is not None:
        lines.extend(
            [
                "",
                "## GRPO feasibility metrics",
                "",
                f"- Unique prompts: {grpo['unique_prompts']}",
                f"- Generations per prompt: {grpo['num_generations']}",
                "- Expected completions per epoch: "
                f"{grpo['expected_completions_per_epoch']}",
                f"- Runtime: {float(grpo['runtime_seconds']) / 60:.1f} minutes",
                "- Peak allocated CUDA memory: "
                f"{float(grpo['peak_cuda_memory_allocated_gb']):.2f} GiB",
                "- Peak reserved CUDA memory: "
                f"{float(grpo['peak_cuda_memory_reserved_gb']):.2f} GiB",
                "- Reference policy: merged SFT policy with the GRPO adapter disabled.",
                f"- Reward weights: `{json.dumps(grpo['reward_weights'])}`",
            ]
        )
        for name, value in grpo.get("final_logged_metrics", {}).items():
            lines.append(f"- `{name}`: {float(value):.6f}")

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Star ratings are hidden from the teacher and student and are not the main target.",
            "- Teacher-generated preferences can encode teacher bias despite validation.",
            (
                "- Exact substring matching measures evidence traceability, "
                "not full causal faithfulness."
            ),
            (
                "- A null or negative DPO result is a valid outcome and must "
                "not be reframed as success."
            ),
            (
                "- PPO and GRPO use AI-generated preferences, so they are reported "
                "as RLAIF rather than RLHF."
                if human_calibration_samples == 0
                else "- PPO and GRPO use AI-generated preferences with a small "
                "human calibration subset, so they are reported as "
                "human-calibrated RLAIF rather than pure RLHF."
            ),
            (
                "- The PPO policy receives only 256 episodes in the T4 configuration; "
                "it is a resource-constrained baseline, not a convergence claim."
            ),
            (
                "- GRPO uses 256 prompts with four sampled completions each; "
                "it is a resource-constrained RLAIF baseline, not a convergence claim."
            ),
            "- Full conclusions require the configured cloud training runs and blinded evaluation.",
            "",
        ]
    )
    report_path = evaluation_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    save_run_metadata(evaluation_dir / "report_run", config, "build-report")
    LOGGER.info("Wrote report to %s", report_path)
    return report_path
