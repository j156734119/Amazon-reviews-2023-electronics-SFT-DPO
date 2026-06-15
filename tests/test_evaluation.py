from amazon_review_alignment import evaluation
from amazon_review_alignment.evaluation import (
    bootstrap_interval,
    evaluate_output,
    run_llm_judge,
    summarize_pairwise,
)
from amazon_review_alignment.schemas import JudgeChoice, JudgeDecision


def test_evaluate_output_checks_schema_and_grounding() -> None:
    review = "Battery life is good but charging is slow."
    valid = (
        '{"sentiment":"neutral","evidence":["Battery life is good","charging is slow"],'
        '"analysis":"The review balances a benefit with a drawback."}'
    )
    fabricated = (
        '{"sentiment":"positive","evidence":["lasts for days"],'
        '"analysis":"The battery is praised."}'
    )

    assert evaluate_output(valid, review)["instruction_following"]
    result = evaluate_output(fabricated, review)
    assert result["schema_valid"]
    assert not result["evidence_grounded"]
    assert not evaluate_output("not json", review)["schema_valid"]


def test_pairwise_summary_counts_ties_as_half() -> None:
    decisions = [
        {
            "comparison": "sft_vs_dpo",
            "left_model": "sft",
            "right_model": "dpo",
            "winner": "dpo",
        },
        {
            "comparison": "sft_vs_dpo",
            "left_model": "sft",
            "right_model": "dpo",
            "winner": "sft",
        },
        {
            "comparison": "sft_vs_dpo",
            "left_model": "sft",
            "right_model": "dpo",
            "winner": "tie",
        },
    ]

    summary = summarize_pairwise(decisions, bootstrap_samples=200, seed=42)[0]
    assert summary["right_model_win_rate_ties_half"] == 0.5
    assert summary["ties"] == 1


def test_bootstrap_empty_input() -> None:
    assert bootstrap_interval([], samples=10, seed=1) == (0.0, 0.0, 0.0)


def test_llm_judge_resumes_completed_blind_decisions(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_judge(client, judge_model, review_text, response_a, response_b):
        calls.append((review_text, response_a, response_b))
        return JudgeDecision(choice=JudgeChoice.A, reason="A is better supported.")

    class FakeOpenAI:
        pass

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(evaluation, "_judge_pair", fake_judge)
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    config = {
        "project": {"seed": 42},
        "evaluation": {
            "judge_model": "judge-test",
            "judge_samples_per_pair": 2,
            "judge_pairs": [["base", "sft"]],
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
    checkpoint = tmp_path / "judge_decisions.jsonl"

    first = run_llm_judge(config, predictions, resume_path=checkpoint)
    second = run_llm_judge(config, predictions, resume_path=checkpoint)

    assert len(first) == 2
    assert len(second) == 2
    assert len(calls) == 2
    assert all(row["judge_model"] == "judge-test" for row in second)
