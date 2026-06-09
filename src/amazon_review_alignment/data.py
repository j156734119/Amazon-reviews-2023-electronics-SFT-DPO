from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Iterable
from typing import Any

from .config import output_root
from .utils import normalize_text, save_run_metadata, stable_id, write_json, write_jsonl

LOGGER = logging.getLogger(__name__)


def _validate_data_config(data_config: dict[str, Any]) -> None:
    sample_size = int(data_config["sample_size"])
    rating_targets = {int(key): int(value) for key, value in data_config["rating_targets"].items()}
    split_sizes = {key: int(value) for key, value in data_config["splits"].items()}
    if set(rating_targets) != {1, 2, 3, 4, 5}:
        raise ValueError("rating_targets must define ratings 1 through 5.")
    if sum(rating_targets.values()) != sample_size:
        raise ValueError("rating_targets must sum to sample_size.")
    if sum(split_sizes.values()) != sample_size:
        raise ValueError("split sizes must sum to sample_size.")


def collect_unique_reviews(
    rows: Iterable[dict[str, Any]],
    rating_targets: dict[int, int],
    max_scanned_reviews: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: dict[int, list[dict[str, Any]]] = {rating: [] for rating in rating_targets}
    seen_texts: set[str] = set()
    seen_products: set[str] = set()
    scanned = 0

    for row in rows:
        if scanned >= max_scanned_reviews:
            break
        scanned += 1
        try:
            rating = int(float(row["rating"]))
        except (KeyError, TypeError, ValueError):
            continue
        if rating not in rating_targets or len(selected[rating]) >= rating_targets[rating]:
            continue

        text = normalize_text(row.get("text", ""))
        parent_asin = str(row.get("parent_asin") or row.get("asin") or "").strip()
        if not text or not parent_asin:
            continue
        text_key = text.casefold()
        if text_key in seen_texts or parent_asin in seen_products:
            continue

        record = {
            "id": stable_id(parent_asin, text),
            "text": text,
            "rating": rating,
            "parent_asin": parent_asin,
            "asin": str(row.get("asin") or ""),
            "verified_purchase": row.get("verified_purchase"),
            "helpful_vote": row.get("helpful_vote"),
        }
        selected[rating].append(record)
        seen_texts.add(text_key)
        seen_products.add(parent_asin)
        if all(len(selected[key]) >= rating_targets[key] for key in rating_targets):
            break

    shortages = {
        rating: rating_targets[rating] - len(selected[rating])
        for rating in rating_targets
        if len(selected[rating]) < rating_targets[rating]
    }
    if shortages:
        raise RuntimeError(
            f"Unable to satisfy rating targets after scanning {scanned:,} rows: {shortages}"
        )
    return [row for rating in sorted(selected) for row in selected[rating]], scanned


def _allocation_matrix(
    rating_counts: dict[int, int],
    split_sizes: dict[str, int],
) -> dict[int, dict[str, int]]:
    total = sum(rating_counts.values())
    splits = list(split_sizes)
    allocation = {rating: {split: 0 for split in splits} for rating in rating_counts}
    remainders: list[tuple[float, int, str]] = []

    for rating, count in rating_counts.items():
        for split, split_size in split_sizes.items():
            exact = count * split_size / total
            base = int(exact)
            allocation[rating][split] = base
            remainders.append((exact - base, rating, split))

    row_remaining = {
        rating: rating_counts[rating] - sum(allocation[rating].values())
        for rating in rating_counts
    }
    column_remaining = {
        split: split_sizes[split] - sum(allocation[rating][split] for rating in rating_counts)
        for split in splits
    }
    for _, rating, split in sorted(remainders, reverse=True):
        if row_remaining[rating] and column_remaining[split]:
            allocation[rating][split] += 1
            row_remaining[rating] -= 1
            column_remaining[split] -= 1

    while any(row_remaining.values()):
        rating = next(key for key, value in row_remaining.items() if value)
        split = next(key for key, value in column_remaining.items() if value)
        allocation[rating][split] += 1
        row_remaining[rating] -= 1
        column_remaining[split] -= 1
    return allocation


def stratified_exact_split(
    records: list[dict[str, Any]],
    split_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["rating"]), []).append(record)
    allocation = _allocation_matrix(
        {rating: len(items) for rating, items in grouped.items()},
        split_sizes,
    )
    rng = random.Random(seed)
    output = {split: [] for split in split_sizes}
    for rating in sorted(grouped):
        items = list(grouped[rating])
        rng.shuffle(items)
        cursor = 0
        for split in split_sizes:
            count = allocation[rating][split]
            output[split].extend(items[cursor : cursor + count])
            cursor += count
    for split in output:
        rng.shuffle(output[split])
    return output


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    ids: set[str] = set()
    products: set[str] = set()
    texts: set[str] = set()
    for split, rows in splits.items():
        for row in rows:
            text_key = normalize_text(row["text"]).casefold()
            if row["id"] in ids:
                raise ValueError(f"Duplicate review id across splits: {row['id']}")
            if row["parent_asin"] in products:
                raise ValueError(f"Product leakage detected: {row['parent_asin']}")
            if text_key in texts:
                raise ValueError(f"Text leakage detected in {split}: {row['id']}")
            ids.add(row["id"])
            products.add(row["parent_asin"])
            texts.add(text_key)


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset

    data_config = config["data"]
    _validate_data_config(data_config)
    rating_targets = {int(key): int(value) for key, value in data_config["rating_targets"].items()}
    split_sizes = {key: int(value) for key, value in data_config["splits"].items()}
    LOGGER.info("Streaming reviews from %s", data_config["source_url"])
    stream = load_dataset(
        "json",
        data_files=data_config["source_url"],
        split="train",
        streaming=True,
    )
    records, scanned = collect_unique_reviews(
        stream,
        rating_targets,
        int(data_config["max_scanned_reviews"]),
    )
    splits = stratified_exact_split(records, split_sizes, int(config["project"]["seed"]))
    validate_splits(splits)

    data_dir = output_root(config) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_jsonl(data_dir / f"{split}.jsonl", rows)

    manifest = {
        "source_url": data_config["source_url"],
        "scanned_reviews": scanned,
        "selected_reviews": len(records),
        "selection_policy": "unique normalized text and at most one review per parent_asin",
        "rating_counts": dict(sorted(Counter(row["rating"] for row in records).items())),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "split_rating_counts": {
            split: dict(sorted(Counter(row["rating"] for row in rows).items()))
            for split, rows in splits.items()
        },
    }
    write_json(data_dir / "manifest.json", manifest)
    save_run_metadata(data_dir, config, "prepare-data", {"manifest": manifest})
    LOGGER.info("Prepared %s reviews under %s", len(records), data_dir)
    return manifest
