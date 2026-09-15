"""Tests for the eval runner: concurrency, timeouts, retries, error capture."""

import asyncio

from litmus import Case, Dataset, ExactMatchScorer, Runner


def _ds(n=6):
    return Dataset(
        name="t", version="1.0", cases=[Case(id=f"c{i}", input=i, expected=i) for i in range(n)]
    )


async def test_concurrency_is_bounded():
    peak = 0
    current = 0

    async def system(case):
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return case.expected

    runner = Runner(system, [ExactMatchScorer()], max_concurrency=3)
    result = await runner.run(_ds(9))
    assert peak <= 3
    assert all(r.error is None for r in result.results)
    assert all(r.scores["exact_match"].passed for r in result.results)


async def test_timeout_records_error_and_retries():
    calls = 0

    async def slow(case):
        nonlocal calls
        calls += 1
        await asyncio.sleep(5)

    runner = Runner(slow, [], timeout_s=0.1, retries=2)
    result = await runner.run(_ds(1))
    r = result.results[0]
    assert r.error is not None and "Timeout" in r.error
    assert r.attempts == 3
    assert calls == 3


async def test_retry_then_success():
    calls = 0

    async def flaky(case):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        return case.expected

    runner = Runner(flaky, [ExactMatchScorer()], retries=1)
    result = await runner.run(_ds(1))
    r = result.results[0]
    assert r.error is None
    assert r.attempts == 2
    assert r.scores["exact_match"].passed is True


async def test_case_error_skips_scorers_but_keeps_case():
    async def broken(case):
        raise ValueError("always fails")

    runner = Runner(broken, [ExactMatchScorer()], retries=0)
    result = await runner.run(_ds(2))
    assert len(result.results) == 2
    assert all(r.error and "ValueError" in r.error for r in result.results)
    assert all(r.scores == {} for r in result.results)


async def test_scorer_crash_becomes_failed_score():
    class BadScorer:
        name = "bad"

        async def score(self, case, output):
            raise RuntimeError("scorer bug")

    async def ok(case):
        return case.expected

    runner = Runner(ok, [BadScorer()], retries=0)
    result = await runner.run(_ds(1))
    score = result.results[0].scores["bad"]
    assert score.passed is False
    assert "Scorer crashed" in (score.explanation or "")


async def test_trace_id_is_attached():
    from litmus import tracing

    tracing.init_tracing(exporter="memory")
    try:

        async def ok(case):
            return case.expected

        runner = Runner(ok, [], retries=0)
        result = await runner.run(_ds(1))
        assert result.results[0].trace_id is not None
        assert result.results[0].trace_id != "0" * 32
    finally:
        tracing.shutdown_tracing()
