from amazon_review_alignment.evaluation import (
    bootstrap_interval,
    evaluate_output,
    summarize_pairwise,
)


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
