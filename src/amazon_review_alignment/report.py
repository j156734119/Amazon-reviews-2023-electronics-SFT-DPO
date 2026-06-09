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

    lines = [
        "# Amazon Review Alignment Results",
        "",
        "## Research question",
        "",
        "Does SFT followed by DPO improve a small language model's ability to produce "
        "structured, faithful, evidence-grounded analyses of Amazon Electronics reviews?",
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
            "- Full conclusions require the configured cloud training runs and blinded evaluation.",
            "",
        ]
    )
    report_path = evaluation_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    save_run_metadata(evaluation_dir / "report_run", config, "build-report")
    LOGGER.info("Wrote report to %s", report_path)
    return report_path
