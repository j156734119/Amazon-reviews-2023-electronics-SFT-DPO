from __future__ import annotations

import json

import pandas as pd

from tools.deepseek_judge import (
    judge_pair_deepseek,
    run_deepseek_judge,
    write_markdown_summary,
)


class FakeCompletions:
    def __init__(self, contents: list[str] | None = None) -> None:
        self.calls = 0
        self.payloads = []
        self.contents = contents or [
            json.dumps(
                {
                    "choice": "A",
                    "reason": "A is better grounded.",
                }
            )
        ]

    def create_chat_completion(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        if "response_format" in payload:
            assert payload["response_format"] == {"type": "json_object"}
        assert payload["temperature"] == 0
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["max_tokens"] == 512
        content = self.contents[min(self.calls - 1, len(self.contents) - 1)]
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": content,
                    },
                }
            ]
        }


def test_deepseek_json_decision_parses() -> None:
    completions = FakeCompletions()

    decision = judge_pair_deepseek(
        completions,
        "deepseek-test",
        "review text",
        '{"sentiment":"positive"}',
        '{"sentiment":"negative"}',
    )

    assert decision.choice == "A"
    assert decision.reason == "A is better grounded."
    assert completions.calls == 1
    assert completions.payloads[0]["response_format"] == {"type": "json_object"}


def test_deepseek_json_decision_retries_empty_content() -> None:
    completions = FakeCompletions(
        [
            "",
            json.dumps(
                {
                    "choice": "tie",
                    "reason": "Both responses are comparable.",
                }
            ),
        ]
    )

    decision = judge_pair_deepseek(
        completions,
        "deepseek-test",
        "review text",
        '{"sentiment":"positive"}',
        '{"sentiment":"negative"}',
        retry_seconds=0,
    )

    assert decision.choice == "tie"
    assert completions.calls == 2


def test_deepseek_json_decision_falls_back_to_plain_chat_after_empty_json_mode() -> None:
    completions = FakeCompletions(
        [
            "",
            "",
            '```json\n{"choice":"B","reason":"B is clearer."}\n```',
        ]
    )

    decision = judge_pair_deepseek(
        completions,
        "deepseek-test",
        "review text",
        '{"sentiment":"positive"}',
        '{"sentiment":"negative"}',
        max_retries=2,
        retry_seconds=0,
    )

    assert decision.choice == "B"
    assert completions.calls == 3
    assert "response_format" in completions.payloads[0]
    assert "response_format" not in completions.payloads[2]


def test_deepseek_judge_resumes_and_does_not_overwrite_openai_files(tmp_path) -> None:
    completions = FakeCompletions()
    config = {
        "project": {"seed": 42},
        "evaluation": {
            "judge_pairs": [["base", "sft"]],
            "bootstrap_samples": 100,
        },
    }
    predictions = {
        variant: [
            {
                "id": f"id-{index}",
                "text": f"review {index}",
                "raw_output": f"{variant} response {index}",
            }
            for index in range(2)
        ]
        for variant in ("base", "sft")
    }
    resume_path = tmp_path / "deepseek_judge_decisions.jsonl"
    openai_path = tmp_path / "judge_decisions.jsonl"
    openai_path.write_text("openai stays untouched\n", encoding="utf-8")

    first = run_deepseek_judge(
        config,
        predictions,
        completions,
        "deepseek-test",
        2,
        resume_path,
    )
    second = run_deepseek_judge(
        config,
        predictions,
        completions,
        "deepseek-test",
        2,
        resume_path,
    )

    assert len(first) == 2
    assert len(second) == 2
    assert completions.calls == 2
    assert openai_path.read_text(encoding="utf-8") == "openai stays untouched\n"
    assert all(row["judge_provider"] == "deepseek" for row in second)
    assert all(row["judge_model"] == "deepseek-test" for row in second)
    assert all(row["response_a_sha256"] for row in second)
    assert all(row["response_b_sha256"] for row in second)


def test_deepseek_summary_markdown_contains_pairwise_fields(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "comparison": "base_vs_sft",
                "examples": 100,
                "left_model": "base",
                "right_model": "sft",
                "right_model_win_rate_ties_half": 0.52,
                "ci_95_low": 0.43,
                "ci_95_high": 0.61,
                "ties": 5,
            }
        ]
    )
    path = tmp_path / "deepseek_judge_summary.md"

    write_markdown_summary(path, summary, "deepseek-test")

    text = path.read_text(encoding="utf-8")
    assert "base_vs_sft" in text
    assert "52.0%" in text
    assert "43.0%-61.0%" in text
