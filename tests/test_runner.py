"""Tests for the eval runner and aggregation."""
from pathlib import Path

import pytest

from litmus.dataset import GoldenCase, load_dataset
from litmus.judges import ContainmentJudge, ExactMatchJudge
from litmus.runner import run_eval

EXAMPLES = Path(__file__).resolve().parents[1] / "datasets" / "examples" / "basic.jsonl"


def echo(text: str) -> str:
    return text


def test_run_eval_aggregates_against_echo_target():
    cases = load_dataset(EXAMPLES)
    report = run_eval(cases, echo, [ExactMatchJudge(), ContainmentJudge()], threshold=0.0)
    assert len(report.results) == 4
    # ex-001 and ex-004 echo exactly; ex-002 echoes a superset; ex-003 echoes wrong content
    exact = report.judge_means["exact_match"]
    containment = report.judge_means["containment"]
    assert exact == pytest.approx(0.5)
    assert containment == pytest.approx(0.75)
    assert report.overall == pytest.approx((0.5 + 0.75) / 2)


def test_threshold_gate():
    cases = [GoldenCase(id="a", input="x", reference="x")]
    passing = run_eval(cases, echo, [ExactMatchJudge()], threshold=1.0)
    failing = run_eval(cases, lambda _: "wrong", [ExactMatchJudge()], threshold=0.5)
    assert passing.passed
    assert not failing.passed


def test_report_serializes():
    cases = [GoldenCase(id="a", input="x", reference="x")]
    report = run_eval(cases, echo, [ExactMatchJudge()], threshold=1.0, suite="demo")
    d = report.to_dict()
    assert d["suite"] == "demo" and d["passed"] is True
    md = report.to_markdown()
    assert "demo" in md and "PASS" in md


def test_runner_rejects_empty_inputs():
    with pytest.raises(ValueError, match="no cases"):
        run_eval([], echo, [ExactMatchJudge()])
    with pytest.raises(ValueError, match="at least one judge"):
        run_eval([GoldenCase(id="a", input="x", reference="x")], echo, [])
    with pytest.raises(ValueError, match="threshold"):
        run_eval([GoldenCase(id="a", input="x", reference="x")], echo,
                 [ExactMatchJudge()], threshold=1.5)
