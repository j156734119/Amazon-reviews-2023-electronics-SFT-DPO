from __future__ import annotations

import json

from amazon_review_alignment.baselines import (
    BASELINE_VARIANTS,
    deepseek_fewshot_response,
    fewshot_messages,
    select_evidence_spans,
    star_label_to_sentiment,
    template_response,
)
from amazon_review_alignment.evaluation import evaluate_output, evaluate_prediction_file


def test_baseline_variants_include_expected_names() -> None:
    assert set(BASELINE_VARIANTS) == {
        "qwen35_2b_fewshot",
        "nlptown_template",
        "deepseek_v4_pro_fewshot",
    }


def test_fewshot_messages_include_examples_without_rating() -> None:
    messages = fewshot_messages("The adapter works but feels loose.")
    joined = "\n".join(message["content"] for message in messages)

    assert len([message for message in messages if message["role"] == "assistant"]) == 3
    assert "The adapter works but feels loose" in joined
    assert "5 star" not in joined.casefold()
    assert "1 star" not in joined.casefold()


def test_nlptown_star_labels_map_to_three_way_sentiment() -> None:
    assert star_label_to_sentiment("1 star") == "negative"
    assert star_label_to_sentiment("2 stars") == "negative"
    assert star_label_to_sentiment("3 stars") == "neutral"
    assert star_label_to_sentiment("4 stars") == "positive"
    assert star_label_to_sentiment("5 stars") == "positive"


def test_template_response_is_valid_grounded_json() -> None:
    review = "Easy install. The screen is dark, but I would recommend this unit."
    raw = template_response(review, "positive")
    payload = json.loads(raw)

    assert payload["sentiment"] == "positive"
    assert 1 <= len(payload["evidence"]) <= 2
    assert evaluate_output(raw, review)["schema_valid"]
    assert evaluate_output(raw, review)["evidence_grounded"]


def test_sentence_selector_prefers_sentiment_relevant_spans() -> None:
    review = "The setup was easy. It stopped working after one day."

    assert "stopped working" in select_evidence_spans(review, "negative")[0]
    assert "setup was easy" in select_evidence_spans(review, "positive")[0]


def test_deepseek_fewshot_response_uses_env_key_and_json_mode(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"sentiment":"positive","evidence":["works well"],'
                                '"analysis":"The reviewer says it works well."}'
                            )
                        }
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "amazon_review_alignment.baselines._httpx_client",
        lambda timeout: FakeClient(timeout),
    )

    raw = deepseek_fewshot_response(
        {"model_id": "deepseek-test", "base_url": "https://api.deepseek.com"},
        "It works well.",
        {"max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0},
    )

    assert json.loads(raw)["sentiment"] == "positive"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["thinking"] == {"type": "disabled"}


def test_baseline_prediction_file_uses_existing_evaluator(tmp_path) -> None:
    review = "Works well and I would recommend it."
    path = tmp_path / "qwen35_2b_fewshot.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "r1",
                "text": review,
                "rating": 5,
                "variant": "qwen35_2b_fewshot",
                "raw_output": template_response(review, "positive"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics, detailed = evaluate_prediction_file(path)

    assert metrics["variant"] == "qwen35_2b_fewshot"
    assert metrics["schema_valid_rate"] == 1.0
    assert detailed[0]["instruction_following"]
