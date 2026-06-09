from collections import Counter

from amazon_review_alignment.data import (
    collect_unique_reviews,
    stratified_exact_split,
    validate_splits,
)


def _rows() -> list[dict]:
    rows = []
    for rating in range(1, 6):
        for index in range(6):
            rows.append(
                {
                    "rating": rating,
                    "text": f"rating {rating} review {index}",
                    "parent_asin": f"P-{rating}-{index}",
                    "asin": f"A-{rating}-{index}",
                }
            )
    rows.extend(
        [
            {
                "rating": 1,
                "text": "rating 1 review 0",
                "parent_asin": "duplicate-text-product",
            },
            {
                "rating": 2,
                "text": "different text for same product",
                "parent_asin": "P-2-0",
            },
        ]
    )
    return rows


def test_collect_and_split_are_exact_and_leak_free() -> None:
    targets = {rating: 4 for rating in range(1, 6)}
    selected, scanned = collect_unique_reviews(_rows(), targets, max_scanned_reviews=100)

    assert scanned <= 100
    assert Counter(row["rating"] for row in selected) == targets

    splits = stratified_exact_split(
        selected,
        {"train": 14, "validation": 2, "test": 4},
        seed=42,
    )
    validate_splits(splits)

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 14,
        "validation": 2,
        "test": 4,
    }


def test_split_is_deterministic() -> None:
    selected, _ = collect_unique_reviews(
        _rows(),
        {rating: 4 for rating in range(1, 6)},
        max_scanned_reviews=100,
    )
    sizes = {"train": 14, "validation": 2, "test": 4}
    first = stratified_exact_split(selected, sizes, seed=7)
    second = stratified_exact_split(selected, sizes, seed=7)

    assert {
        split: [row["id"] for row in rows] for split, rows in first.items()
    } == {
        split: [row["id"] for row in rows] for split, rows in second.items()
    }
