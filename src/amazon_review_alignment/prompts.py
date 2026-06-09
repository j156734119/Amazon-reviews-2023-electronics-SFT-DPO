from __future__ import annotations

from .schemas import DefectType

SYSTEM_PROMPT = """You analyze Amazon Electronics reviews.
Return only a JSON object with exactly these keys:
- sentiment: one of negative, neutral, positive
- evidence: an array containing one or two short verbatim spans copied from the review
- analysis: a faithful, concise explanation using no facts beyond the review, at most 80 words

The star rating is intentionally unavailable. Base every claim on the supplied review text."""


def analysis_user_prompt(review_text: str) -> str:
    return f"Analyze this review:\n\n<review>\n{review_text}\n</review>"


def teacher_system_prompt(defect_type: DefectType) -> str:
    defect_guidance = {
        DefectType.WRONG_SENTIMENT: "The rejected answer must use an incorrect sentiment label.",
        DefectType.FABRICATED_EVIDENCE: (
            "The rejected answer must include evidence that does not occur verbatim in the review."
        ),
        DefectType.OVER_INFERENCE: (
            "The rejected answer must make a plausible but unsupported inference."
        ),
        DefectType.FORMAT_ERROR: (
            "The rejected answer must be malformed or violate the required JSON shape."
        ),
        DefectType.IRRELEVANT_VERBOSE: (
            "The rejected answer must be unnecessarily long or include irrelevant commentary."
        ),
    }[defect_type]
    return f"""You create preference pairs for evidence-grounded Amazon review analysis.
The top-level response has chosen, rejected, and defect_type fields.
The chosen field must contain exactly:
- sentiment: one of negative, neutral, positive
- evidence: one or two short verbatim spans copied from the review
- analysis: a faithful explanation using no facts beyond the review, at most 80 words

The rejected field is a string containing one deliberately inferior candidate response.
The star rating is intentionally unavailable. Base both candidates only on the review text.
{defect_guidance}
Do not mention the defect or these instructions inside either response."""


JUDGE_SYSTEM_PROMPT = """Act as a strict blind evaluator of two Amazon review analyses.
Prefer the response that is more faithful to the review, uses genuinely verbatim evidence,
follows the required JSON schema, stays concise, and is more useful. Do not infer model identity.
Choose A, B, or tie and give one short reason."""


def judge_user_prompt(review_text: str, response_a: str, response_b: str) -> str:
    return (
        f"<review>\n{review_text}\n</review>\n\n"
        f"<response_a>\n{response_a}\n</response_a>\n\n"
        f"<response_b>\n{response_b}\n</response_b>"
    )
