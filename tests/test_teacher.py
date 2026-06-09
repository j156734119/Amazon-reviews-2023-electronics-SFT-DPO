from types import SimpleNamespace

from amazon_review_alignment.schemas import DefectType, ReviewAnalysis, TeacherPreference
from amazon_review_alignment.teacher import generate_preference


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            parsed = TeacherPreference(
                chosen=ReviewAnalysis(
                    sentiment="positive",
                    evidence=["not in source"],
                    analysis="Unsupported on the first attempt.",
                ),
                rejected="{}",
                defect_type=DefectType.FORMAT_ERROR,
            )
        else:
            parsed = TeacherPreference(
                chosen=ReviewAnalysis(
                    sentiment="positive",
                    evidence=["Excellent sound"],
                    analysis="The reviewer explicitly praises the sound.",
                ),
                rejected="{sentiment: positive}",
                defect_type=DefectType.FORMAT_ERROR,
            )
        return SimpleNamespace(
            output_parsed=parsed,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


def test_teacher_retries_ungrounded_chosen_without_real_api() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    config = {
        "model": "fake-model",
        "max_retries": 2,
        "max_output_tokens": 200,
    }

    preference, usage = generate_preference(
        client,
        "Excellent sound for the price.",
        DefectType.FORMAT_ERROR,
        config,
        sleep_seconds=0,
    )

    assert responses.calls == 2
    assert preference.chosen.evidence == ["Excellent sound"]
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
