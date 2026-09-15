"""Tests for the LLM judge, using a fake provider (no API calls)."""

from litmus import Case, JudgeRubric, LLMJudge


class FakeProvider:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list = []

    async def complete(self, messages, *, json_mode=False):
        self.calls.append(messages)
        return self.reply


def _rubric():
    return JudgeRubric(
        name="quality", criteria="Be good.", min_score=1, max_score=5, pass_threshold=4
    )


def _case():
    return Case(id="c1", input="the input", expected="the reference")


async def test_judge_top_score_passes():
    judge = LLMJudge(_rubric(), FakeProvider('{"score": 5, "explanation": "excellent"}'))
    score = await judge.score(_case(), "the output")
    assert score.passed is True
    assert score.value == 1.0
    assert score.explanation == "excellent"


async def test_judge_mid_score_fails_and_normalizes():
    judge = LLMJudge(_rubric(), FakeProvider('{"score": 3, "explanation": "meh"}'))
    score = await judge.score(_case(), "the output")
    assert score.passed is False
    assert score.value == 0.5  # (3-1)/(5-1)


async def test_judge_score_is_clamped():
    judge = LLMJudge(_rubric(), FakeProvider('{"score": 99, "explanation": "wild"}'))
    score = await judge.score(_case(), "the output")
    assert score.value == 1.0
    assert score.details["raw_score"] == 5


async def test_judge_unparseable_output_fails_safe():
    judge = LLMJudge(_rubric(), FakeProvider("I refuse to output JSON"))
    score = await judge.score(_case(), "the output")
    assert score.passed is False
    assert score.value == 0.0
    assert "unparseable" in (score.explanation or "")


async def test_judge_sends_rubric_and_case_in_prompt():
    provider = FakeProvider('{"score": 4, "explanation": "ok"}')
    judge = LLMJudge(_rubric(), provider)
    await judge.score(_case(), "the output")
    prompt = provider.calls[0][0]["content"]
    assert "Be good." in prompt
    assert "the input" in prompt and "the reference" in prompt and "the output" in prompt
