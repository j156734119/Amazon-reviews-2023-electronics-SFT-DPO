from amazon_review_alignment.inference import generation_prompt
from amazon_review_alignment.train_dpo import build_dpo_records
from amazon_review_alignment.train_sft import build_sft_records


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert enable_thinking is False
        rendered = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return rendered + ("|assistant:" if add_generation_prompt else "")


def _preference_rows():
    return [
        {
            "text": "Works well.",
            "chosen": '{"sentiment":"positive","evidence":["Works well"],'
            '"analysis":"The review is positive."}',
            "rejected": '{"sentiment":"negative","evidence":["Works well"],'
            '"analysis":"The review is negative."}',
        }
    ]


def test_sft_and_dpo_formatters_disable_thinking() -> None:
    tokenizer = FakeTokenizer()
    sft = build_sft_records(_preference_rows(), tokenizer)
    dpo = build_dpo_records(_preference_rows(), tokenizer)
    prompt = generation_prompt(tokenizer, "Works well.")

    assert sft[0]["text"].endswith(_preference_rows()[0]["chosen"])
    assert dpo[0]["prompt"].endswith("|assistant:")
    assert dpo[0]["chosen"] == _preference_rows()[0]["chosen"]
    assert prompt.endswith("|assistant:")
