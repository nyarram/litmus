"""Eval reports: aggregate stats, markdown/JSON rendering, and baseline diffs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from litmus.runners import CaseResult, EvalRunResult
from litmus.scorers.base import Score


@dataclass
class ScorerSummary:
    name: str
    n: int
    mean: float
    pass_rate: float | None
    n_passed: int
    n_failed: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "mean": round(self.mean, 4),
            "pass_rate": round(self.pass_rate, 4) if self.pass_rate is not None else None,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScorerSummary:
        return cls(
            name=d["name"],
            n=d["n"],
            mean=d["mean"],
            pass_rate=d.get("pass_rate"),
            n_passed=d["n_passed"],
            n_failed=d["n_failed"],
        )


@dataclass
class EvalReport:
    run_name: str
    dataset_name: str
    dataset_version: str
    created_at: str
    summaries: dict[str, ScorerSummary] = field(default_factory=dict)
    results: list[CaseResult] = field(default_factory=list)
    n_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at,
            "n_cases": len(self.results),
            "n_errors": self.n_errors,
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalReport:
        return cls(
            run_name=d["run_name"],
            dataset_name=d["dataset_name"],
            dataset_version=d["dataset_version"],
            created_at=d["created_at"],
            summaries={k: ScorerSummary.from_dict(v) for k, v in d["summaries"].items()},
            results=[_case_result_from_dict(r) for r in d["results"]],
            n_errors=d.get("n_errors", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            f"# Eval report: {self.run_name}",
            "",
            f"Dataset `{self.dataset_name}` v{self.dataset_version} — "
            f"{len(self.results)} cases, {self.n_errors} errors — {self.created_at}",
            "",
            "| scorer | n | mean | pass rate | passed | failed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for s in self.summaries.values():
            pr = f"{s.pass_rate:.1%}" if s.pass_rate is not None else "n/a"
            lines.append(
                f"| {s.name} | {s.n} | {s.mean:.3f} | {pr} | {s.n_passed} | {s.n_failed} |"
            )
        failing = [
            r.case_id
            for r in self.results
            if r.error or any(sc.passed is False for sc in r.scores.values())
        ]
        if failing:
            failing_list = ", ".join(f"`{c}`" for c in failing)
            lines += ["", f"**{len(failing)} failing/errored cases:** {failing_list}"]
        return "\n".join(lines) + "\n"


def _case_result_from_dict(d: dict) -> CaseResult:
    return CaseResult(
        case_id=d["case_id"],
        output=d.get("output"),
        scores={k: Score.from_dict(v) for k, v in d.get("scores", {}).items()},
        latency_s=d.get("latency_s", 0.0),
        attempts=d.get("attempts", 1),
        error=d.get("error"),
        trace_id=d.get("trace_id"),
    )


def build_report(run: EvalRunResult, run_name: str) -> EvalReport:
    summaries: dict[str, ScorerSummary] = {}
    by_scorer: dict[str, list] = {}
    for r in run.results:
        for name, score in r.scores.items():
            by_scorer.setdefault(name, []).append(score)
    for name, scores in by_scorer.items():
        n = len(scores)
        passed_flags = [s.passed for s in scores if s.passed is not None]
        summaries[name] = ScorerSummary(
            name=name,
            n=n,
            mean=sum(s.value for s in scores) / n if n else 0.0,
            pass_rate=(sum(passed_flags) / len(passed_flags)) if passed_flags else None,
            n_passed=sum(1 for p in passed_flags if p),
            n_failed=sum(1 for p in passed_flags if not p),
        )
    return EvalReport(
        run_name=run_name,
        dataset_name=run.dataset_name,
        dataset_version=run.dataset_version,
        created_at=datetime.now(UTC).isoformat(),
        summaries=summaries,
        results=run.results,
        n_errors=len(run.errors()),
    )


@dataclass
class ScorerDelta:
    name: str
    baseline_mean: float
    candidate_mean: float
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None

    @property
    def mean_delta(self) -> float:
        return self.candidate_mean - self.baseline_mean


@dataclass
class ReportDiff:
    baseline_run: str
    candidate_run: str
    deltas: list[ScorerDelta] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)  # passed in baseline, fail now
    fixed: list[str] = field(default_factory=list)  # failed in baseline, pass now

    def has_regression(self, threshold: float = 0.0) -> bool:
        return bool(self.new_failures) or any(d.mean_delta < -threshold for d in self.deltas)

    def to_markdown(self) -> str:
        lines = [f"# Diff: `{self.baseline_run}` → `{self.candidate_run}`", ""]
        for d in self.deltas:
            arrow = "🔻" if d.mean_delta < 0 else "🔺" if d.mean_delta > 0 else "➖"
            lines.append(
                f"{arrow} **{d.name}**: mean {d.baseline_mean:.3f} → {d.candidate_mean:.3f} "
                f"({d.mean_delta:+.3f})"
            )
        if self.new_failures:
            new = ", ".join(f"`{c}`" for c in self.new_failures)
            lines.append(f"\n**New failures ({len(self.new_failures)}):** {new}")
        if self.fixed:
            fixed = ", ".join(f"`{c}`" for c in self.fixed)
            lines.append(f"\n**Fixed ({len(self.fixed)}):** {fixed}")
        if not self.deltas and not self.new_failures and not self.fixed:
            lines.append("No comparable scorers or cases.")
        return "\n".join(lines) + "\n"


def _case_passed(result: CaseResult) -> bool:
    return result.error is None and all(s.passed is not False for s in result.scores.values())


def compare_reports(baseline: EvalReport, candidate: EvalReport) -> ReportDiff:
    deltas = []
    for name, b in baseline.summaries.items():
        c = candidate.summaries.get(name)
        if c is None:
            continue
        deltas.append(
            ScorerDelta(
                name=name,
                baseline_mean=b.mean,
                candidate_mean=c.mean,
                baseline_pass_rate=b.pass_rate,
                candidate_pass_rate=c.pass_rate,
            )
        )
    b_map = {r.case_id: r for r in baseline.results}
    c_map = {r.case_id: r for r in candidate.results}
    new_failures, fixed = [], []
    for cid in b_map.keys() & c_map.keys():
        was, now = _case_passed(b_map[cid]), _case_passed(c_map[cid])
        if was and not now:
            new_failures.append(cid)
        elif now and not was:
            fixed.append(cid)
    return ReportDiff(
        baseline_run=baseline.run_name,
        candidate_run=candidate.run_name,
        deltas=deltas,
        new_failures=sorted(new_failures),
        fixed=sorted(fixed),
    )
