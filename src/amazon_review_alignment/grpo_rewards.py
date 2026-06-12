from __future__ import annotations

import json
from typing import Any

from .utils import normalized_match

VALID_SENTIMENTS = {"negative", "neutral", "positive"}


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        return "".join(
            str(message.get("content", ""))
            for message in completion
            if isinstance(message, dict)
        ).strip()
    return str(completion).strip()


def _reward_payload(completion: Any) -> dict[str, Any] | None:
    try:
        payload = json.loads(completion_text(completion))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        "sentiment",
        "evidence",
        "analysis",
    }:
        return None
    evidence = payload["evidence"]
    if (
        payload["sentiment"] not in VALID_SENTIMENTS
        or not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 2
        or any(not isinstance(span, str) or not span.strip() for span in evidence)
        or not isinstance(payload["analysis"], str)
        or not payload["analysis"].strip()
    ):
        return None
    return payload


def schema_reward(completions: list[Any], **_: Any) -> list[float]:
    return [1.0 if _reward_payload(completion) is not None else 0.0 for completion in completions]


def evidence_reward(
    completions: list[Any],
    text: list[str],
    **_: Any,
) -> list[float]:
    rewards = []
    for completion, review_text in zip(completions, text, strict=True):
        payload = _reward_payload(completion)
        grounded = payload is not None and all(
            normalized_match(review_text, span) for span in payload["evidence"]
        )
        rewards.append(1.0 if grounded else 0.0)
    return rewards


def length_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        payload = _reward_payload(completion)
        within_limit = (
            payload is not None and len(payload["analysis"].split()) <= 80
        )
        rewards.append(1.0 if within_limit else 0.0)
    return rewards
