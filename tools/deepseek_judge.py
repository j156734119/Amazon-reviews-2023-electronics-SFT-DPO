from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amazon_review_alignment.config import load_config  # noqa: E402
from amazon_review_alignment.evaluation import summarize_pairwise  # noqa: E402
from amazon_review_alignment.prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt  # noqa: E402
from amazon_review_alignment.schemas import JudgeChoice, JudgeDecision  # noqa: E402
from amazon_review_alignment.utils import configure_logging, read_jsonl, write_jsonl  # noqa: E402

LOGGER = logging.getLogger(__name__)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CHAT_COMPLETIONS_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"


class DeepSeekClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = httpx.Client(timeout=90)

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def response_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_predictions(prediction_dir: Path, variants: list[str]) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        path = prediction_dir / f"{variant}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        predictions[variant] = read_jsonl(path)
    return predictions


def parse_deepseek_decision(content: str) -> JudgeDecision:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DeepSeek judge returned non-JSON content: {content!r}") from exc
    try:
        return JudgeDecision.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"DeepSeek judge JSON did not match JudgeDecision: {payload!r}") from exc


def judge_pair_deepseek(
    client: Any,
    model: str,
    review_text: str,
    response_a: str,
    response_b: str,
    max_retries: int = 3,
    retry_seconds: float = 1.0,
    max_tokens: int = 512,
) -> JudgeDecision:
    system_prompt = (
        JUDGE_SYSTEM_PROMPT
        + '\nReturn only valid JSON with exactly this shape: '
        + '{"choice":"A|B|tie","reason":"short reason"}.'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": judge_user_prompt(review_text, response_a, response_b)
            + '\n\nOutput JSON only, for example: {"choice":"A","reason":"short reason"}.',
        },
    ]
    last_error: Exception | None = None
    attempts = max(max_retries, 1)
    phases = (
        ("json_mode", {"response_format": {"type": "json_object"}}),
        ("plain_chat", {}),
    )
    for phase_name, phase_kwargs in phases:
        for attempt in range(1, attempts + 1):
            try:
                response = client.create_chat_completion(
                    {
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0,
                        "thinking": {"type": "disabled"},
                        "stream": False,
                        **phase_kwargs,
                    }
                )
                content = response["choices"][0]["message"].get("content")
                if not content:
                    finish_reason = response["choices"][0].get("finish_reason")
                    reasoning = response["choices"][0]["message"].get("reasoning_content")
                    raise ValueError(
                        "DeepSeek judge returned empty content "
                        f"(finish_reason={finish_reason!r}, "
                        f"reasoning_present={bool(reasoning)})"
                    )
                return parse_deepseek_decision(content)
            except ValueError as exc:
                last_error = exc
                LOGGER.warning(
                    "DeepSeek judge %s attempt %s/%s failed: %s",
                    phase_name,
                    attempt,
                    attempts,
                    exc,
                )
                if attempt < attempts:
                    time.sleep(retry_seconds)
        LOGGER.warning("DeepSeek judge phase %s exhausted; trying next fallback", phase_name)
    raise ValueError(
        f"DeepSeek judge failed after {attempts} json-mode and {attempts} plain-chat "
        f"attempts: {last_error}"
    ) from last_error


def run_deepseek_judge(
    config: dict[str, Any],
    predictions: dict[str, list[dict[str, Any]]],
    client: Any,
    model: str,
    samples_per_pair: int,
    resume_path: Path,
    max_retries: int = 3,
    retry_seconds: float = 1.0,
    max_tokens: int = 512,
) -> list[dict[str, Any]]:
    rng = random.Random(int(config["project"]["seed"]))
    by_variant = {
        variant: {str(row["id"]): row for row in rows}
        for variant, rows in predictions.items()
    }
    existing_rows = read_jsonl(resume_path) if resume_path.exists() else []
    existing = {
        (
            row.get("comparison"),
            str(row.get("id")),
            row.get("response_a_sha256"),
            row.get("response_b_sha256"),
        ): row
        for row in existing_rows
        if row.get("judge_provider") == "deepseek" and row.get("judge_model") == model
    }

    decisions: list[dict[str, Any]] = []
    for left, right in config["evaluation"]["judge_pairs"]:
        comparison = f"{left}_vs_{right}"
        common_ids = sorted(set(by_variant[left]) & set(by_variant[right]))
        rng.shuffle(common_ids)
        selected_ids = common_ids[:samples_per_pair]
        for index, review_id in enumerate(selected_ids, start=1):
            left_row = by_variant[left][review_id]
            right_row = by_variant[right][review_id]
            swapped = bool(rng.getrandbits(1))
            if swapped:
                shown = [(right, right_row["raw_output"]), (left, left_row["raw_output"])]
            else:
                shown = [(left, left_row["raw_output"]), (right, right_row["raw_output"])]

            response_a_hash = response_hash(str(shown[0][1]))
            response_b_hash = response_hash(str(shown[1][1]))
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
            else:
                decision = judge_pair_deepseek(
                    client,
                    model,
                    str(left_row["text"]),
                    str(shown[0][1]),
                    str(shown[1][1]),
                    max_retries=max_retries,
                    retry_seconds=retry_seconds,
                    max_tokens=max_tokens,
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
                    "judge_provider": "deepseek",
                    "judge_model": model,
                    "response_a_sha256": response_a_hash,
                    "response_b_sha256": response_b_hash,
                }
                decisions.append(row)
                resume_path.parent.mkdir(parents=True, exist_ok=True)
                with resume_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if index % 10 == 0 or index == len(selected_ids):
                LOGGER.info("DeepSeek judge %s: %s/%s", comparison, index, len(selected_ids))
    return decisions


def write_markdown_summary(summary_path: Path, summary: pd.DataFrame, model: str) -> None:
    lines = [
        "# DeepSeek LLM Judge Summary",
        "",
        f"Judge model: `{model}`",
        "",
        "| Comparison | N | Right-model win rate | 95% CI | Ties |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {int(row.examples)} | "
            f"{row.right_model_win_rate_ties_half:.1%} | "
            f"{row.ci_95_low:.1%}-{row.ci_95_high:.1%} | {int(row.ties)} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek blind pairwise judging locally.")
    parser.add_argument("--config", default="configs/rlhf_a100_online_v2.yaml")
    parser.add_argument("--root", type=Path, default=Path("outputs/a100-qwen3.5-2b"))
    parser.add_argument("--samples-per-pair", type=int, default=100)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    if args.samples_per_pair <= 0:
        raise ValueError("--samples-per-pair must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")
    if args.retry_seconds < 0:
        raise ValueError("--retry-seconds must not be negative.")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek judging.")

    config = load_config(args.config)
    root = args.root.resolve()
    evaluation_dir = root / "evaluation"
    prediction_dir = evaluation_dir / "predictions"
    variants = list(config["evaluation"].get("variants", ["base", "sft", "dpo", "ppo", "grpo"]))
    predictions = load_predictions(prediction_dir, variants)

    client = DeepSeekClient(api_key)
    decisions_path = evaluation_dir / "deepseek_judge_decisions.jsonl"
    summary_path = evaluation_dir / "deepseek_judge_pairwise_summary.csv"
    markdown_path = evaluation_dir / "deepseek_judge_summary.md"

    decisions = run_deepseek_judge(
        config,
        predictions,
        client,
        args.model,
        args.samples_per_pair,
        decisions_path,
        args.max_retries,
        args.retry_seconds,
        args.max_tokens,
    )
    write_jsonl(decisions_path, decisions)
    summaries = summarize_pairwise(
        decisions,
        int(config["evaluation"]["bootstrap_samples"]),
        int(config["project"]["seed"]),
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(summary_path, index=False)
    write_markdown_summary(markdown_path, summary, args.model)

    print(f"Decisions: {decisions_path}")
    print(f"Summary: {summary_path}")
    print(f"Markdown: {markdown_path}")
    print(summary)


if __name__ == "__main__":
    main()
