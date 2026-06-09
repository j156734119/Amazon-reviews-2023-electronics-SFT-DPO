import json

import pytest
from pydantic import ValidationError

from amazon_review_alignment.schemas import (
    DefectType,
    ReviewAnalysis,
    TeacherPreference,
    parse_analysis,
    rejected_has_requested_defect,
)


def test_analysis_is_valid_and_grounded() -> None:
    review = "Sound is excellent, but the cable is too short."
    analysis = ReviewAnalysis(
        sentiment="neutral",
        evidence=["Sound is excellent", "the cable is too short"],
        analysis="The review contains both praise and a concrete limitation.",
    )

    assert analysis.evidence_is_grounded(review)
    assert parse_analysis(analysis.as_json()) == analysis


def test_analysis_rejects_extra_fields_and_long_text() -> None:
    payload = {
        "sentiment": "positive",
        "evidence": ["good"],
        "analysis": "word " * 81,
        "confidence": 0.9,
    }

    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate(payload)


def test_parse_analysis_requires_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_analysis("positive")


def test_rejected_defect_is_machine_checked() -> None:
    chosen = ReviewAnalysis(
        sentiment="positive",
        evidence=["Works well"],
        analysis="The reviewer reports a good experience.",
    )
    valid = TeacherPreference(
        chosen=chosen,
        rejected='{"sentiment":"negative","evidence":["Works well"],'
        '"analysis":"The reviewer reports a bad experience."}',
        defect_type=DefectType.WRONG_SENTIMENT,
    )
    invalid = TeacherPreference(
        chosen=chosen,
        rejected=chosen.as_json(),
        defect_type=DefectType.WRONG_SENTIMENT,
    )

    assert rejected_has_requested_defect(valid, "Works well.")
    assert not rejected_has_requested_defect(invalid, "Works well.")
