from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .utils import normalized_match


class Sentiment(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class DefectType(str, Enum):
    WRONG_SENTIMENT = "wrong_sentiment"
    FABRICATED_EVIDENCE = "fabricated_evidence"
    OVER_INFERENCE = "over_inference"
    FORMAT_ERROR = "format_error"
    IRRELEVANT_VERBOSE = "irrelevant_verbose"


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: Sentiment
    evidence: list[str] = Field(min_length=1, max_length=2)
    analysis: str

    @field_validator("evidence")
    @classmethod
    def non_empty_evidence(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Evidence spans must not be empty.")
        return value

    @field_validator("analysis")
    @classmethod
    def concise_analysis(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Analysis must not be empty.")
        if len(value.split()) > 80:
            raise ValueError("Analysis must contain at most 80 whitespace-delimited words.")
        return value

    def evidence_is_grounded(self, review_text: str) -> bool:
        return all(normalized_match(review_text, span) for span in self.evidence)

    def as_json(self) -> str:
        return self.model_dump_json()


class TeacherPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chosen: ReviewAnalysis
    rejected: str
    defect_type: DefectType

    @field_validator("rejected")
    @classmethod
    def non_empty_rejected(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Rejected response must not be empty.")
        return value.strip()

    def chosen_is_grounded(self, review_text: str) -> bool:
        return self.chosen.evidence_is_grounded(review_text)


class JudgeChoice(str, Enum):
    A = "A"
    B = "B"
    TIE = "tie"


class JudgeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: JudgeChoice
    reason: str = Field(min_length=1, max_length=500)


def parse_analysis(raw: str) -> ReviewAnalysis:
    return ReviewAnalysis.model_validate(json.loads(raw))


def schema_valid_and_grounded(raw: str, review_text: str) -> tuple[bool, bool]:
    try:
        analysis = parse_analysis(raw)
    except (json.JSONDecodeError, ValueError):
        return False, False
    return True, analysis.evidence_is_grounded(review_text)


def rejected_has_requested_defect(
    preference: TeacherPreference,
    review_text: str,
) -> bool:
    defect = preference.defect_type
    if defect == DefectType.FORMAT_ERROR:
        try:
            parse_analysis(preference.rejected)
        except (json.JSONDecodeError, ValueError):
            return True
        return False

    try:
        payload = json.loads(preference.rejected)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False

    if defect == DefectType.IRRELEVANT_VERBOSE:
        analysis = str(payload.get("analysis", ""))
        return len(analysis.split()) > 80

    try:
        rejected = ReviewAnalysis.model_validate(payload)
    except ValueError:
        return False
    if defect == DefectType.WRONG_SENTIMENT:
        return rejected.sentiment != preference.chosen.sentiment
    if defect == DefectType.FABRICATED_EVIDENCE:
        return not rejected.evidence_is_grounded(review_text)
    if defect == DefectType.OVER_INFERENCE:
        # Unsupported inference is semantic and is checked later by blind judges.
        # Here we still require a plausible schema-valid contrast response.
        return rejected.as_json() != preference.chosen.as_json()
    return False
