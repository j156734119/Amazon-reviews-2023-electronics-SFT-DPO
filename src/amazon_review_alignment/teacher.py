from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import output_root
from .prompts import analysis_user_prompt, teacher_system_prompt
from .schemas import DefectType, TeacherPreference, rejected_has_requested_defect
from .utils import read_jsonl, save_run_metadata, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)
DEFECT_CYCLE = list(DefectType)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


def _require_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for this command.")


def _client() -> Any:
    _require_api_key()
    from openai import OpenAI

    return OpenAI()


def _usage_from_response(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )


def estimate_cost(
    usage: TokenUsage,
    teacher_config: dict[str, Any],
    discount: float = 1.0,
) -> float:
    input_cost = (
        usage.input_tokens / 1_000_000 * float(teacher_config["input_price_per_million"])
    )
    output_cost = (
        usage.output_tokens / 1_000_000 * float(teacher_config["output_price_per_million"])
    )
    return (input_cost + output_cost) * discount


def generate_preference(
    client: Any,
    review_text: str,
    defect_type: DefectType,
    teacher_config: dict[str, Any],
    sleep_seconds: float = 1.0,
) -> tuple[TeacherPreference, TokenUsage]:
    last_error: Exception | None = None
    for attempt in range(int(teacher_config["max_retries"])):
        try:
            response = client.responses.parse(
                model=teacher_config["model"],
                input=[
                    {"role": "system", "content": teacher_system_prompt(defect_type)},
                    {"role": "user", "content": analysis_user_prompt(review_text)},
                ],
                text_format=TeacherPreference,
                max_output_tokens=int(teacher_config["max_output_tokens"]),
            )
            preference = response.output_parsed
            if preference is None:
                raise ValueError("Teacher response did not contain parsed structured output.")
            if not preference.chosen_is_grounded(review_text):
                raise ValueError("Chosen evidence is not a substring of the source review.")
            if not rejected_has_requested_defect(preference, review_text):
                raise ValueError("Rejected response does not exhibit the requested defect.")
            return preference, _usage_from_response(response)
        except Exception as error:  # API exceptions vary by SDK version.
            last_error = error
            if attempt + 1 < int(teacher_config["max_retries"]):
                time.sleep(sleep_seconds * (2**attempt))
    raise RuntimeError("Teacher generation failed after all retries.") from last_error


def _preference_record(
    row: dict[str, Any],
    split: str,
    preference: TeacherPreference,
    model: str,
) -> dict[str, Any]:
    return {
        "id": row["id"],
        "split": split,
        "text": row["text"],
        "prompt": analysis_user_prompt(row["text"]),
        "chosen": preference.chosen.as_json(),
        "rejected": preference.rejected,
        "defect_type": preference.defect_type.value,
        "teacher_model": model,
    }


def _teacher_dir(config: dict[str, Any]) -> Path:
    path = output_root(config) / "teacher"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_teacher_pilot(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    teacher_config = config["teacher"]
    limit = int(limit or teacher_config["pilot_size"])
    train_rows = read_jsonl(output_root(config) / "data" / "train.jsonl")
    if not train_rows:
        raise RuntimeError("Training data is missing. Run prepare-data first.")
    rows = train_rows[: min(limit, len(train_rows))]
    client = _client()
    usage = TokenUsage()
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        defect_type = DEFECT_CYCLE[index % len(DEFECT_CYCLE)]
        try:
            preference, item_usage = generate_preference(
                client,
                row["text"],
                defect_type,
                teacher_config,
            )
            usage.add(item_usage)
            records.append(
                _preference_record(row, "train", preference, teacher_config["model"])
            )
        except RuntimeError as error:
            quarantine.append({"id": row["id"], "error": str(error)})
            LOGGER.error("Teacher generation failed for %s: %s", row["id"], error)

    teacher_dir = _teacher_dir(config)
    write_jsonl(teacher_dir / "pilot_preferences.jsonl", records)
    write_jsonl(teacher_dir / "pilot_quarantine.jsonl", quarantine)
    if not records:
        raise RuntimeError("Teacher pilot produced no validated preference records.")
    total_batch_items = sum(
        len(read_jsonl(output_root(config) / "data" / f"{split}.jsonl"))
        for split in ("train", "validation")
    )
    completed = max(len(records), 1)
    projected_usage = TokenUsage(
        input_tokens=round(usage.input_tokens / completed * total_batch_items),
        output_tokens=round(usage.output_tokens / completed * total_batch_items),
    )
    batch_discount = float(teacher_config["batch_discount"])
    summary = {
        "requested_items": len(rows),
        "completed_items": len(records),
        "quarantined_items": len(quarantine),
        "usage": usage.__dict__,
        "pilot_standard_cost_usd": round(estimate_cost(usage, teacher_config), 6),
        "projected_batch_items": total_batch_items,
        "projected_batch_usage": projected_usage.__dict__,
        "projected_batch_cost_usd": round(
            estimate_cost(projected_usage, teacher_config, batch_discount),
            6,
        ),
        "price_assumptions": {
            "input_price_per_million": teacher_config["input_price_per_million"],
            "output_price_per_million": teacher_config["output_price_per_million"],
            "batch_discount_multiplier": batch_discount,
        },
    }
    write_json(teacher_dir / "pilot_summary.json", summary)
    save_run_metadata(teacher_dir / "pilot_run", config, "teacher-pilot", summary)
    LOGGER.info(
        "Pilot completed: %s/%s items, projected batch cost $%.4f",
        len(records),
        len(rows),
        summary["projected_batch_cost_usd"],
    )
    return summary


def _structured_text_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "teacher_preference",
        "schema": TeacherPreference.model_json_schema(),
        "strict": True,
    }


def build_batch_requests(config: dict[str, Any]) -> list[dict[str, Any]]:
    teacher_config = config["teacher"]
    requests: list[dict[str, Any]] = []
    index = 0
    for split in ("train", "validation"):
        rows = read_jsonl(output_root(config) / "data" / f"{split}.jsonl")
        for row in rows:
            defect_type = DEFECT_CYCLE[index % len(DEFECT_CYCLE)]
            custom_id = f"{split}:{row['id']}:{defect_type.value}"
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": teacher_config["model"],
                        "input": [
                            {
                                "role": "system",
                                "content": teacher_system_prompt(defect_type),
                            },
                            {
                                "role": "user",
                                "content": analysis_user_prompt(row["text"]),
                            },
                        ],
                        "text": {"format": _structured_text_format()},
                        "max_output_tokens": int(teacher_config["max_output_tokens"]),
                    },
                }
            )
            index += 1
    return requests


def _extract_batch_output_text(body: dict[str, Any]) -> str:
    for output in body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    raise ValueError("Batch response body has no output_text content.")


def process_batch_results(
    config: dict[str, Any],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sources = {
        split: {
            row["id"]: row
            for row in read_jsonl(output_root(config) / "data" / f"{split}.jsonl")
        }
        for split in ("train", "validation")
    }
    records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    quarantine: list[dict[str, Any]] = []
    token_usage = TokenUsage()

    for result in result_rows:
        custom_id = result.get("custom_id", "")
        try:
            split, review_id, requested_defect = custom_id.split(":", 2)
            response = result["response"]
            if int(response["status_code"]) >= 300:
                raise ValueError(f"API status {response['status_code']}")
            body = response["body"]
            usage = body.get("usage", {})
            token_usage.input_tokens += int(usage.get("input_tokens", 0) or 0)
            token_usage.output_tokens += int(usage.get("output_tokens", 0) or 0)
            raw = _extract_batch_output_text(body)
            preference = TeacherPreference.model_validate_json(raw)
            if preference.defect_type.value != requested_defect:
                raise ValueError("Teacher returned a different defect_type than requested.")
            source = sources[split][review_id]
            if not preference.chosen_is_grounded(source["text"]):
                raise ValueError("Chosen evidence is not grounded in the review.")
            if not rejected_has_requested_defect(preference, source["text"]):
                raise ValueError("Rejected response does not exhibit the requested defect.")
            records[split].append(
                _preference_record(
                    source,
                    split,
                    preference,
                    config["teacher"]["model"],
                )
            )
        except (KeyError, ValueError, ValidationError) as error:
            quarantine.append(
                {"custom_id": custom_id, "error": str(error), "raw_result": result}
            )

    teacher_dir = _teacher_dir(config)
    for split, rows in records.items():
        write_jsonl(teacher_dir / f"preferences_{split}.jsonl", rows)
    write_jsonl(teacher_dir / "batch_quarantine.jsonl", quarantine)
    summary = {
        "completed": {split: len(rows) for split, rows in records.items()},
        "quarantined": len(quarantine),
        "usage": token_usage.__dict__,
        "batch_cost_usd": round(
            estimate_cost(
                token_usage,
                config["teacher"],
                float(config["teacher"]["batch_discount"]),
            ),
            6,
        ),
    }
    write_json(teacher_dir / "batch_summary.json", summary)
    save_run_metadata(teacher_dir / "batch_run", config, "teacher-batch", summary)
    return summary


def _batch_to_dict(batch: Any) -> dict[str, Any]:
    if hasattr(batch, "model_dump"):
        return batch.model_dump(mode="json")
    return {
        key: getattr(batch, key, None)
        for key in ("id", "status", "input_file_id", "output_file_id", "error_file_id")
    }


def run_teacher_batch(
    config: dict[str, Any],
    prepare_only: bool = False,
) -> dict[str, Any]:
    teacher_dir = _teacher_dir(config)
    pilot_path = teacher_dir / "pilot_summary.json"
    if not pilot_path.exists():
        raise RuntimeError("Run teacher-pilot before teacher-batch.")
    pilot_summary = json.loads(pilot_path.read_text(encoding="utf-8"))
    projected_cost = float(pilot_summary["projected_batch_cost_usd"])
    cap = float(config["teacher"]["max_estimated_cost_usd"])
    if projected_cost > cap:
        raise RuntimeError(
            f"Projected batch cost ${projected_cost:.4f} exceeds configured cap ${cap:.2f}."
        )

    requests = build_batch_requests(config)
    input_path = teacher_dir / "batch_input.jsonl"
    write_jsonl(input_path, requests)
    if prepare_only:
        result = {
            "status": "prepared",
            "request_count": len(requests),
            "input_path": str(input_path),
            "projected_cost_usd": projected_cost,
        }
        write_json(teacher_dir / "batch_prepare_summary.json", result)
        return result

    client = _client()
    state_path = teacher_dir / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        batch = client.batches.retrieve(state["id"])
        batch_state = _batch_to_dict(batch)
        write_json(state_path, batch_state)
        if batch_state.get("status") != "completed":
            return batch_state
        output_file_id = batch_state.get("output_file_id")
        if not output_file_id:
            raise RuntimeError("Completed batch does not have an output_file_id.")
        raw_text = client.files.content(output_file_id).text
        raw_path = teacher_dir / "batch_output.jsonl"
        raw_path.write_text(raw_text, encoding="utf-8")
        return process_batch_results(
            config,
            [json.loads(line) for line in raw_text.splitlines() if line.strip()],
        )

    with input_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"project": "amazon-review-alignment"},
    )
    batch_state = _batch_to_dict(batch)
    write_json(state_path, batch_state)
    return batch_state
