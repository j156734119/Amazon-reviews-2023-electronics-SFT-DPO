from amazon_review_alignment.inference import generation_prompt
from amazon_review_alignment.train_dpo import build_dpo_records
from amazon_review_alignment.train_grpo import build_grpo_records
from amazon_review_alignment.train_ppo import build_ppo_records
from amazon_review_alignment.train_reward import build_reward_records
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

    def __call__(
        self,
        text,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        assert padding is False
        tokens = list(range(len(text.split())))
        if truncation:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}

    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(f"token-{token}" for token in tokens)


def _preference_rows():
    return [
        {
            "id": "review-1",
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

    assert sft[0]["prompt"].endswith("|assistant:")
    assert sft[0]["completion"] == _preference_rows()[0]["chosen"]
    assert dpo[0]["prompt"].endswith("|assistant:")
    assert dpo[0]["chosen"] == _preference_rows()[0]["chosen"]
    assert prompt.endswith("|assistant:")


def test_reward_and_ppo_formatters_use_same_chat_prompt() -> None:
    tokenizer = FakeTokenizer()
    reward = build_reward_records(_preference_rows(), tokenizer)
    ppo = build_ppo_records(_preference_rows(), tokenizer, max_prompt_length=4)

    assert reward[0]["chosen"].endswith(_preference_rows()[0]["chosen"])
    assert reward[0]["rejected"].endswith(_preference_rows()[0]["rejected"])
    assert len(ppo[0]["input_ids"]) <= 4
    grpo = build_grpo_records(_preference_rows(), tokenizer)
    assert grpo[0]["prompt"].endswith("|assistant:")
    assert grpo[0]["text"] == "Works well."
