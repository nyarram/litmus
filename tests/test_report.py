"""Tests for report building and baseline diffing."""

import json

import pytest

from litmus import EvalRunResult, build_report, compare_reports
from litmus.runners import CaseResult
from litmus.scorers.base import Score


def _result(case_id, passed: bool):
    return CaseResult(
        case_id=case_id,
        output="o",
        scores={"acc": Score(name="acc", value=1.0 if passed else 0.0, passed=passed)},
    )


def _report(name, verdicts: dict):
    run = EvalRunResult(
        dataset_name="d",
        dataset_version="1.0",
        results=[_result(cid, p) for cid, p in verdicts.items()],
    )
    return build_report(run, run_name=name)


def test_build_report_aggregates():
    report = _report("r1", {"a": True, "b": True, "c": False, "d": True})
    s = report.summaries["acc"]
    assert s.n == 4 and s.n_passed == 3 and s.n_failed == 1
    assert s.mean == 0.75 and s.pass_rate == 0.75
    assert "acc" in report.to_markdown()


def test_compare_detects_regression():
    baseline = _report("base", {"a": True, "b": True, "c": False})
    candidate = _report("cand", {"a": True, "b": False, "c": True})
    diff = compare_reports(baseline, candidate)
    assert diff.new_failures == ["b"]
    assert diff.fixed == ["c"]
    assert diff.has_regression() is True
    delta = diff.deltas[0]
    assert delta.baseline_mean == pytest.approx(2 / 3)
    assert delta.candidate_mean == pytest.approx(2 / 3)


def test_compare_no_regression():
    baseline = _report("base", {"a": True})
    candidate = _report("cand", {"a": True})
    diff = compare_reports(baseline, candidate)
    assert diff.has_regression() is False
    assert diff.new_failures == []


def test_report_json_round_trips():
    report = _report("r1", {"a": True})
    parsed = json.loads(report.to_json())
    assert parsed["run_name"] == "r1"
    assert parsed["summaries"]["acc"]["mean"] == 1.0
