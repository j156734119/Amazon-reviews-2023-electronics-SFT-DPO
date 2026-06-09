from pathlib import Path

import pandas as pd

from amazon_review_alignment.human_eval import summarize_human_evaluation


def test_human_eval_summary(tmp_path: Path) -> None:
    responses = tmp_path / "responses.csv"
    pd.DataFrame(
        [
            {"id": "1", "choice": "A"},
            {"id": "2", "choice": "B"},
            {"id": "3", "choice": "tie"},
        ]
    ).to_csv(responses, index=False)
    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "human_eval_key.jsonl").write_text(
        "\n".join(
            [
                '{"id":"1","model_a":"dpo","model_b":"sft"}',
                '{"id":"2","model_a":"dpo","model_b":"sft"}',
                '{"id":"3","model_a":"sft","model_b":"dpo"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "project": {"seed": 42, "output_dir": str(tmp_path)},
        "evaluation": {"bootstrap_samples": 200},
    }

    summary = summarize_human_evaluation(config, responses)

    assert summary["examples"] == 3
    assert summary["right_model_win_rate_ties_half"] == 0.5
