from __future__ import annotations

import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx

from .config import output_root
from .modeling import load_base_model, load_tokenizer, render_chat
from .prompts import SYSTEM_PROMPT, analysis_user_prompt
from .schemas import Sentiment
from .utils import read_jsonl, save_run_metadata, set_seed, write_jsonl

LOGGER = logging.getLogger(__name__)

BASELINE_VARIANTS = (
    "qwen35_2b_fewshot",
    "phi4_mini_fewshot",
    "nlptown_template",
    "deepseek_v4_pro_fewshot",
)

DEFAULT_BASELINES: dict[str, dict[str, Any]] = {
    "qwen35_2b_fewshot": {
        "type": "hf_causal_lm",
        "model_id": "Qwen/Qwen3.5-2B",
        "prompt_mode": "few_shot",
        "load_in_4bit": True,
    },
    "phi4_mini_fewshot": {
        "type": "hf_causal_lm",
        "model_id": "microsoft/Phi-4-mini-instruct",
        "prompt_mode": "few_shot",
        "load_in_4bit": True,
    },
    "nlptown_template": {
        "type": "sentiment_template",
        "model_id": "nlptown/bert-base-multilingual-uncased-sentiment",
    },
    "deepseek_v4_pro_fewshot": {
        "type": "deepseek_chat",
        "model_id": "deepseek-v4-pro",
        "prompt_mode": "few_shot",
        "base_url": "https://api.deepseek.com",
        "max_retries": 3,
        "retry_seconds": 1.0,
    },
}

FEWSHOT_EXAMPLES = [
    {
        "review": (
            "Easy to install and all the buttons worked. The screen is a little dark, "
            "but I would recommend buying this unit."
        ),
        "response": {
            "sentiment": "positive",
            "evidence": ["Easy to install", "I would recommend buying this unit"],
            "analysis": (
                "The review is positive overall because installation was easy, the unit "
                "worked, and the reviewer recommends buying it despite a darker screen."
            ),
        },
    },
    {
        "review": (
            "The headphones sound great, but the battery only lasts an hour and the "
            "left side stopped working after a week."
        ),
        "response": {
            "sentiment": "negative",
            "evidence": ["battery only lasts an hour", "stopped working after a week"],
            "analysis": (
                "The review is negative because serious battery and durability problems "
                "outweigh the positive comment about sound quality."
            ),
        },
    },
    {
        "review": (
            "The cable works with my monitor. Shipping was fine, but the connector feels "
            "a bit loose and I may replace it later."
        ),
        "response": {
            "sentiment": "neutral",
            "evidence": ["works with my monitor", "connector feels a bit loose"],
            "analysis": (
                "The review is mixed because the cable works and shipping was acceptable, "
                "but the loose connector leaves the reviewer uncertain."
            ),
        },
    },
]

POSITIVE_TERMS = {
    "awesome",
    "easy",
    "excellent",
    "favorite",
    "good",
    "great",
    "love",
    "perfect",
    "recommend",
    "strong",
    "works",
}
NEGATIVE_TERMS = {
    "bad",
    "broke",
    "broken",
    "complaint",
    "cracked",
    "dark",
    "defective",
    "disappointed",
    "does not",
    "doesn't",
    "failed",
    "issue",
    "missing",
    "not",
    "poor",
    "refund",
    "returned",
    "slow",
    "stopped",
    "warning",
}


def baseline_config(config: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in DEFAULT_BASELINES:
        raise ValueError(f"Unknown baseline variant: {variant}")
    configured = config.get("baselines", {}).get(variant, {})
    merged = deepcopy(DEFAULT_BASELINES[variant])
    merged.update(configured)
    return merged


def fewshot_messages(review_text: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in FEWSHOT_EXAMPLES:
        messages.extend(
            [
                {"role": "user", "content": analysis_user_prompt(example["review"])},
                {
                    "role": "assistant",
                    "content": json.dumps(example["response"], ensure_ascii=False),
                },
            ]
        )
    messages.append({"role": "user", "content": analysis_user_prompt(review_text)})
    return messages


def fewshot_prompt(tokenizer: Any, review_text: str) -> str:
    return render_chat(tokenizer, fewshot_messages(review_text), add_generation_prompt=True)


def star_label_to_sentiment(label: str) -> str:
    match = re.search(r"[1-5]", str(label))
    if match is None:
        raise ValueError(f"Could not parse nlptown star label: {label!r}")
    stars = int(match.group(0))
    if stars <= 2:
        return Sentiment.NEGATIVE.value
    if stars == 3:
        return Sentiment.NEUTRAL.value
    return Sentiment.POSITIVE.value


def split_review_sentences(review_text: str) -> list[str]:
    text = str(review_text).strip()
    if not text:
        return []
    sentences = [item.strip() for item in re.findall(r"[^.!?\n]+[.!?]?", text)]
    return [item for item in sentences if item]


def _term_score(sentence: str, terms: set[str]) -> int:
    lowered = sentence.casefold()
    return sum(1 for term in terms if term in lowered)


def select_evidence_spans(
    review_text: str,
    sentiment: str,
    max_spans: int = 2,
) -> list[str]:
    sentences = split_review_sentences(review_text)
    if not sentences:
        stripped = str(review_text).strip()
        return [stripped[:180] if stripped else "No review text provided."]

    scored = []
    for index, sentence in enumerate(sentences):
        positive = _term_score(sentence, POSITIVE_TERMS)
        negative = _term_score(sentence, NEGATIVE_TERMS)
        if sentiment == Sentiment.POSITIVE.value:
            score = positive - negative
        elif sentiment == Sentiment.NEGATIVE.value:
            score = negative - positive
        else:
            score = min(positive, negative) + int("but" in sentence.casefold())
        scored.append((score, -index, sentence))
    ranked = sorted(scored, reverse=True)
    selected = [sentence for _, _, sentence in ranked[:max_spans]]
    if not selected:
        selected = sentences[:1]
    return [span[:180].strip() for span in selected if span.strip()]


def template_analysis(review_text: str, sentiment: str) -> str:
    evidence = select_evidence_spans(review_text, sentiment)
    joined = "; ".join(evidence)
    if sentiment == Sentiment.POSITIVE.value:
        return (
            "The review is positive overall because the selected evidence highlights "
            f"satisfaction or successful use: {joined}"
        )
    if sentiment == Sentiment.NEGATIVE.value:
        return (
            "The review is negative overall because the selected evidence highlights "
            f"problems, dissatisfaction, or return intent: {joined}"
        )
    return (
        "The review is neutral or mixed because the selected evidence includes both usable "
        f"features and reservations: {joined}"
    )


def template_response(review_text: str, sentiment: str) -> str:
    evidence = select_evidence_spans(review_text, sentiment)
    payload = {
        "sentiment": sentiment,
        "evidence": evidence[:2],
        "analysis": " ".join(template_analysis(review_text, sentiment).split()[:80]),
    }
    return json.dumps(payload, ensure_ascii=False)


def _generation_parameters(
    config: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    model_config = config["model"]
    return {
        "max_new_tokens": int(settings.get("max_new_tokens", model_config["max_new_tokens"])),
        "temperature": float(settings.get("temperature", model_config["temperature"])),
        "top_p": float(settings.get("top_p", model_config["top_p"])),
    }


def generate_hf_fewshot(
    model: Any,
    tokenizer: Any,
    review_text: str,
    generation_config: dict[str, Any],
) -> str:
    import torch

    prompt = fewshot_prompt(tokenizer, review_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = float(generation_config["temperature"]) > 0
    kwargs = {
        "max_new_tokens": int(generation_config["max_new_tokens"]),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(
            temperature=float(generation_config["temperature"]),
            top_p=float(generation_config["top_p"]),
        )
    model.eval()
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    completion = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def deepseek_fewshot_response(
    settings: dict[str, Any],
    review_text: str,
    generation_config: dict[str, Any],
) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for deepseek_v4_pro_fewshot.")
    base_url = str(settings.get("base_url", "https://api.deepseek.com")).rstrip("/")
    url = f"{base_url}/chat/completions"
    messages = fewshot_messages(review_text)
    messages[-1] = {
        "role": "user",
        "content": (
            messages[-1]["content"]
            + "\n\nReturn only valid JSON with exactly these keys: sentiment, "
            "evidence, analysis."
        ),
    }
    payload = {
        "model": settings["model_id"],
        "messages": messages,
        "max_tokens": int(generation_config["max_new_tokens"]),
        "temperature": float(generation_config["temperature"]),
        "stream": False,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    retries = int(settings.get("max_retries", 3))
    retry_seconds = float(settings.get("retry_seconds", 1.0))
    last_error: Exception | None = None
    with httpx.Client(timeout=90) as client:
        for attempt in range(1, max(retries, 1) + 1):
            try:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content")
                if not content:
                    raise ValueError("DeepSeek baseline returned empty content.")
                return str(content).strip()
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(retry_seconds)
    raise RuntimeError(f"DeepSeek baseline failed after {retries} attempts: {last_error}")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _target_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(output_root(config) / "data" / "test.jsonl")
    return rows[: int(config["evaluation"]["max_test_samples"])]


def run_baseline_inference(
    config: dict[str, Any],
    variant: str,
    force: bool = False,
) -> Path:
    if variant not in BASELINE_VARIANTS:
        raise ValueError(f"Unknown baseline variant: {variant}")
    set_seed(int(config["project"]["seed"]))
    prediction_dir = output_root(config) / "evaluation" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_path = prediction_dir / f"{variant}.jsonl"
    test_rows = _target_rows(config)
    target_ids = [str(row["id"]) for row in test_rows]

    existing_rows = [] if force or not output_path.exists() else read_jsonl(output_path)
    existing_by_id = {str(row.get("id")): row for row in existing_rows}
    if existing_by_id and all(review_id in existing_by_id for review_id in target_ids):
        LOGGER.info("Reusing existing baseline predictions: %s", output_path)
        return output_path
    if force:
        output_path.write_text("", encoding="utf-8")

    settings = baseline_config(config, variant)
    generation_config = _generation_parameters(config, settings)
    predictions = [
        existing_by_id[review_id]
        for review_id in target_ids
        if review_id in existing_by_id
    ]
    pending_rows = [row for row in test_rows if str(row["id"]) not in existing_by_id]

    if settings["type"] == "hf_causal_lm":
        model_config = deepcopy(config["model"])
        model_config["base_model"] = settings["model_id"]
        model_config["load_in_4bit"] = bool(
            settings.get("load_in_4bit", model_config["load_in_4bit"])
        )
        model_config.pop("sft_merged_dir", None)
        tokenizer = load_tokenizer(settings["model_id"], padding_side="left")
        model = load_base_model(model_config, for_training=False)
        try:
            for index, row in enumerate(pending_rows, start=1):
                raw = generate_hf_fewshot(model, tokenizer, row["text"], generation_config)
                prediction = _prediction_row(row, variant, raw)
                predictions.append(prediction)
                _append_jsonl(output_path, prediction)
                if index % 25 == 0 or index == len(pending_rows):
                    LOGGER.info("%s baseline inference: %s/%s", variant, index, len(pending_rows))
        finally:
            del model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    elif settings["type"] == "sentiment_template":
        from transformers import pipeline

        classifier = pipeline("text-classification", model=settings["model_id"])
        for index, row in enumerate(pending_rows, start=1):
            result = classifier(row["text"], truncation=True)[0]
            sentiment = star_label_to_sentiment(str(result["label"]))
            prediction = _prediction_row(row, variant, template_response(row["text"], sentiment))
            predictions.append(prediction)
            _append_jsonl(output_path, prediction)
            if index % 25 == 0 or index == len(pending_rows):
                LOGGER.info("%s baseline inference: %s/%s", variant, index, len(pending_rows))
    elif settings["type"] == "deepseek_chat":
        for index, row in enumerate(pending_rows, start=1):
            raw = deepseek_fewshot_response(settings, row["text"], generation_config)
            prediction = _prediction_row(row, variant, raw)
            predictions.append(prediction)
            _append_jsonl(output_path, prediction)
            if index % 10 == 0 or index == len(pending_rows):
                LOGGER.info("%s baseline inference: %s/%s", variant, index, len(pending_rows))
    else:
        raise ValueError(f"Unsupported baseline type for {variant}: {settings['type']}")

    ordered = {str(row["id"]): row for row in predictions}
    write_jsonl(
        output_path,
        [ordered[review_id] for review_id in target_ids if review_id in ordered],
    )
    save_run_metadata(
        prediction_dir / f"{variant}_run",
        config,
        f"inference-{variant}",
        {
            "variant": variant,
            "baseline_type": settings["type"],
            "model_id": settings["model_id"],
            "examples": len(ordered),
        },
    )
    return output_path


def _prediction_row(row: dict[str, Any], variant: str, raw_output: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "text": row["text"],
        "rating": row.get("rating"),
        "variant": variant,
        "raw_output": raw_output,
    }

