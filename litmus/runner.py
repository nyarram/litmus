"""Eval runner: execute a target over a golden dataset and aggregate scores.

The target is any ``Callable[[str], str]`` — typically a thin wrapper around
the pipeline or agent under test. The runner is deliberately synchronous and
single-process in piece 1: determinism and debuggability beat throughput
until the harness proves itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

from litmus.dataset import GoldenCase
from litmus.judges import Judge, Score
from litmus import tracing

@dataclass(frozen=True)
class CaseResult:
    case_id: str
    prediction: str
    scores: list[Score]


@dataclass
class EvalReport:
    suite: str
    threshold: float
    results: list[CaseResult] = field(default_factory=list)
    judge_means: dict[str, float] = field(default_factory=dict)
    overall: float = 0.0
    passed: bool = False
    ran_at: str = ""

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "threshold": self.threshold,
            "overall": round(self.overall, 4),
            "passed": self.passed,
            "judge_means": {k: round(v, 4) for k, v in self.judge_means.items()},
            "ran_at": self.ran_at,
            "results": [
                {
                    "case_id": r.case_id,
                    "prediction": r.prediction,
                    "scores": [
                        {"judge": s.judge, "score": s.score, "passed": s.passed}
                        for s in r.scores
                    ],
                }
                for r in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        lines = [
            f"# litmus eval report — {self.suite}",
            "",
            f"overall: **{self.overall:.3f}** (threshold {self.threshold:.3f}) → "
            f"{'PASS' if self.passed else 'FAIL'}",
            "",
            "| judge | mean score |",
            "|---|---|",
        ]
        for name, mean in sorted(self.judge_means.items()):
            lines.append(f"| {name} | {mean:.3f} |")
        lines += ["", "| case | prediction | scores |", "|---|---|---|"]
        for r in self.results:
            scores = ", ".join(f"{s.judge}={s.score:.1f}" for s in r.scores)
            pred = r.prediction.replace("|", "\\|")
            if len(pred) > 60:
                pred = pred[:57] + "..."
            lines.append(f"| {r.case_id} | {pred} | {scores} |")
        return "\n".join(lines) + "\n"


def run_eval(
    cases: Sequence[GoldenCase],
    target: Callable[[str], str],
    judges: Sequence[Judge],
    *,
    threshold: float = 1.0,
    suite: str = "",
    trace: bool = False,
) -> EvalReport:
    """Run every case through ``target``, score with ``judges``, aggregate.

    With ``trace=True`` each case runs inside a ``litmus.eval.case`` span with
    ``litmus.eval.target`` and ``litmus.eval.judge`` children (requires
    ``litmus.tracing.init_tracing()`` first). Spans are no-ops when tracing
    is not initialized, so results never depend on it.
    """
    if not cases:
        raise ValueError("no cases to evaluate")
    if not judges:
        raise ValueError("at least one judge is required")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if trace:
        tracing.ensure_initialized()

    report = EvalReport(
        suite=suite,
        threshold=threshold,
        ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    totals: dict[str, float] = {j.name: 0.0 for j in judges}

    for case in cases:
        with tracing.span(
            "litmus.eval.case",
            attributes={"litmus.case.id": case.id, "litmus.suite": suite},
        ):
            with tracing.span("litmus.eval.target"):
                prediction = target(case.input)
            scores = []
            for judge in judges:
                with tracing.span(
                    "litmus.eval.judge",
                    attributes={"litmus.judge.name": judge.name},
                ) as judge_span:
                    s = judge.score(prediction, case.reference)
                    judge_span.set_attribute("litmus.score", s.score)
                scores.append(s)
            for s in scores:
                totals[s.judge] += s.score
            report.results.append(
                CaseResult(case_id=case.id, prediction=prediction, scores=scores)
            )

    n = len(report.results)
    report.judge_means = {name: total / n for name, total in totals.items()}
    report.overall = sum(report.judge_means.values()) / len(report.judge_means)
    report.passed = report.overall >= threshold
    return report
