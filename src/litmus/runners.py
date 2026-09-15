"""Eval runner: executes a dataset against a system under test.

The system is any async callable ``async def system(case: Case) -> Any``.
The runner handles concurrency, per-case timeouts, and retries with
backoff — the boring reliability work that makes evals trustworthy.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from litmus import tracing
from litmus.datasets import Case, Dataset
from litmus.scorers.base import Score, Scorer

SystemFn = Callable[[Case], Awaitable[Any]]
SpanAttributesFn = Callable[[Case], dict[str, Any]]


@dataclass
class CaseResult:
    case_id: str
    output: Any
    scores: dict[str, Score] = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1
    error: str | None = None
    trace_id: str | None = None

    def passed(self, scorer_name: str) -> bool | None:
        s = self.scores.get(scorer_name)
        return s.passed if s else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "output": self.output,
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
            "latency_s": self.latency_s,
            "attempts": self.attempts,
            "error": self.error,
            "trace_id": self.trace_id,
        }


@dataclass
class EvalRunResult:
    dataset_name: str
    dataset_version: str
    results: list[CaseResult] = field(default_factory=list)

    def errors(self) -> list[CaseResult]:
        return [r for r in self.results if r.error]


class Runner:
    def __init__(
        self,
        system: SystemFn,
        scorers: list[Scorer] | None = None,
        max_concurrency: int = 8,
        timeout_s: float = 120.0,
        retries: int = 1,
        tracer: Any | None = None,
        extra_span_attributes: SpanAttributesFn | None = None,
    ):
        self.system = system
        self.scorers = scorers or []
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.retries = retries
        # A caller-supplied OTel tracer; when None, litmus.tracing.case_span
        # is used instead (a no-op unless init_tracing() was called).
        self.tracer = tracer
        # Per-case span attributes, e.g. lambda case: {"litmus.synthetic": True}.
        self.extra_span_attributes = extra_span_attributes

    @contextmanager
    def _span_for(self, case: Case) -> Iterator[tuple[Any, str | None]]:
        extra = self.extra_span_attributes(case) if self.extra_span_attributes else {}
        if self.tracer is not None:
            with self.tracer.start_as_current_span("litmus.case") as span:
                span.set_attribute("litmus.case_id", case.id)
                for k, v in extra.items():
                    span.set_attribute(k, v)
                ctx = span.get_span_context()
                trace_id = format(ctx.trace_id, "032x") if ctx else None
                yield span, trace_id
        else:
            with tracing.case_span(case.id, extra) as (span, trace_id):
                yield span, trace_id

    async def _run_case(self, case: Case, sem: asyncio.Semaphore) -> CaseResult:
        async with sem:
            with self._span_for(case) as (span, trace_id):
                start = time.perf_counter()
                output: Any = None
                error: str | None = None
                attempts = 0
                for attempt in range(self.retries + 1):
                    attempts = attempt + 1
                    try:
                        output = await asyncio.wait_for(self.system(case), timeout=self.timeout_s)
                        error = None
                        break
                    except Exception as exc:  # noqa: BLE001 — evals must not die on one bad case
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < self.retries:
                            await asyncio.sleep(0.5 * (attempt + 1))
                latency = time.perf_counter() - start
                scores: dict[str, Score] = {}
                if error is None:
                    for scorer in self.scorers:
                        try:
                            scores[scorer.name] = await scorer.score(case, output)
                        except Exception as exc:  # noqa: BLE001
                            scores[scorer.name] = Score(
                                name=scorer.name,
                                value=0.0,
                                passed=False,
                                explanation=f"Scorer crashed: {type(exc).__name__}: {exc}",
                            )
                span.set_attribute("litmus.error", bool(error))
                return CaseResult(
                    case_id=case.id,
                    output=output,
                    scores=scores,
                    latency_s=round(latency, 3),
                    attempts=attempts,
                    error=error,
                    trace_id=trace_id,
                )

    async def run(self, dataset: Dataset) -> EvalRunResult:
        sem = asyncio.Semaphore(self.max_concurrency)
        results = await asyncio.gather(*[self._run_case(c, sem) for c in dataset.cases])
        return EvalRunResult(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            results=list(results),
        )
