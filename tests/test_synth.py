"""Tests for synthetic traffic generation and load testing."""

import json

import pytest

from litmus import tracing
from litmus.scorers.base import Score
from litmus.scorers.exact import ContainsScorer
from litmus.synth import (
    SYNTHETIC_ATTR,
    Generator,
    Intent,
    Persona,
    load_personas,
    run_synth,
)


class StubProvider:
    """Deterministic stand-in for a ModelProvider."""

    def __init__(self, text="stub request text", judge_score=5):
        self.text = text
        self.judge_score = judge_score
        self.calls = 0

    async def complete(self, messages, *, json_mode=False):
        self.calls += 1
        if json_mode:
            return json.dumps({"score": self.judge_score, "explanation": "stub"})
        return self.text


def make_personas():
    return [
        Persona(
            name="a",
            description="persona a",
            intents=[
                Intent(name="i1", description="intent one", weight=3.0),
                Intent(name="i2", description="intent two", weight=1.0),
            ],
        ),
        Persona(
            name="b",
            description="persona b",
            intents=[Intent(name="i3", description="intent three", weight=1.0)],
        ),
    ]


async def test_generate_is_deterministic_under_seed():
    g1 = await Generator(make_personas(), StubProvider()).generate(20, seed=7)
    g2 = await Generator(make_personas(), StubProvider()).generate(20, seed=7)
    assert [(r.persona, r.intent, r.text) for r in g1] == [
        (r.persona, r.intent, r.text) for r in g2
    ]
    assert [r.id for r in g1] == [f"synth-{i + 1:04d}" for i in range(20)]


async def test_different_seeds_differ():
    g1 = await Generator(make_personas(), StubProvider()).generate(20, seed=1)
    g2 = await Generator(make_personas(), StubProvider()).generate(20, seed=2)
    assert [r.intent for r in g1] != [r.intent for r in g2]


async def test_intent_weights_hold_at_scale():
    reqs = await Generator(make_personas(), StubProvider()).generate(4000, seed=0)
    a_reqs = [r for r in reqs if r.persona == "a"]
    i1 = sum(1 for r in a_reqs if r.intent == "i1") / len(a_reqs)
    assert 0.70 < i1 < 0.80  # weight 3:1 -> 0.75


async def test_generate_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one persona"):
        Generator([], StubProvider())
    with pytest.raises(ValueError, match="n must be positive"):
        await Generator(make_personas(), StubProvider()).generate(0)


def test_load_personas_validation(tmp_path):
    good = tmp_path / "p.json"
    good.write_text(
        json.dumps(
            {
                "personas": [
                    {
                        "name": "x",
                        "description": "d",
                        "intents": [{"name": "i", "description": "d"}],
                    }
                ]
            }
        )
    )
    assert load_personas(good)[0].intents[0].weight == 1.0

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"personas": []}))
    with pytest.raises(ValueError, match="no personas"):
        load_personas(bad)


async def test_run_synth_end_to_end():
    async def system(text):
        return f"echo: {text}"

    report = await run_synth(
        system,
        make_personas(),
        StubProvider(),
        n=6,
        seed=3,
        scorers=[ContainsScorer()],
        concurrency=3,
        threshold=0.0,  # containment vs stub text scores 0; gate on errors only
    )
    assert report.n_requests == 6
    assert report.error_rate == 0.0
    assert report.latency_p50 is not None
    assert report.latency_p95 >= report.latency_p50
    assert set(report.per_persona) <= {"a", "b"}
    assert report.passed is True
    assert "SYNTHETIC" in report.to_markdown()


async def test_run_synth_default_judge_scores_quality():
    async def system(text):
        return "a helpful answer"

    report = await run_synth(
        system, make_personas(), StubProvider(judge_score=5), n=4, seed=0, threshold=0.9
    )
    assert report.error_rate == 0.0
    assert report.eval_report.summaries["judge_synth_quality"].mean == 1.0
    assert report.passed is True


async def test_run_synth_captures_errors_and_fails_gate():
    async def system(text):
        # fail exactly one request, deterministically
        if text.endswith("2"):
            raise RuntimeError("kaput")
        return text

    texts = [f"request {i}" for i in range(4)]

    class SeqProvider(StubProvider):
        def __init__(self):
            self.i = 0

        async def complete(self, messages, *, json_mode=False):
            if json_mode:
                return json.dumps({"score": 5, "explanation": "stub"})
            t = texts[self.i % len(texts)]
            self.i += 1
            return t

    report = await run_synth(system, make_personas(), SeqProvider(), n=4, seed=0, threshold=0.0)
    assert report.error_rate == 0.25
    assert report.passed is False
    assert report.per_persona  # still aggregated


async def test_synthetic_spans_are_labeled():
    tracing.shutdown_tracing()
    try:
        tracing.init_tracing(exporter="memory")

        async def system(text):
            return text

        await run_synth(
            system,
            make_personas(),
            StubProvider(),
            n=3,
            seed=0,
            scorers=[],
            trace=True,
        )
        spans = tracing.get_finished_spans()
        req_spans = [s for s in spans if s.name == "litmus.case"]
        assert len(req_spans) == 3
        assert all(s.attributes[SYNTHETIC_ATTR] is True for s in req_spans)
        assert {s.attributes["litmus.persona"] for s in req_spans} <= {"a", "b"}
        batch = [s for s in spans if s.name == "litmus.synthetic.batch"]
        assert len(batch) == 1 and batch[0].attributes[SYNTHETIC_ATTR] is True
    finally:
        tracing.shutdown_tracing()


async def test_trace_flag_requires_init():
    async def system(text):
        return text

    with pytest.raises(tracing.TracingNotInitialized):
        await run_synth(system, make_personas(), StubProvider(), n=2, trace=True)


async def test_judge_retries_malformed_then_succeeds():
    from litmus.datasets import Case
    from litmus.scorers.llm_judge import JudgeRubric, LLMJudge

    calls = 0

    class FlakyProvider:
        async def complete(self, messages, *, json_mode=False):
            nonlocal calls
            calls += 1
            if calls == 1:
                return "not json at all"
            return json.dumps({"score": 4, "explanation": "fine"})

    judge = LLMJudge(JudgeRubric(name="r", criteria="c"), FlakyProvider())
    score = await judge.score(Case(id="c", input="hi", expected="hello"), "hello there")
    assert isinstance(score, Score)
    assert score.value == pytest.approx(0.75)  # (4-1)/(5-1)
    assert calls == 2


async def test_judge_prompt_omits_reference_when_none():
    from litmus.datasets import Case
    from litmus.scorers.llm_judge import JudgeRubric, LLMJudge

    judge = LLMJudge(JudgeRubric(name="r", criteria="c"), StubProvider())
    prompt = judge._prompt(Case(id="c", input="hi", expected=None), "hello")
    assert "on its own merits" in prompt
    assert "## Reference (expected) answer" not in prompt
