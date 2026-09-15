"""Tests for report serialization and the `litmus diff` CLI command."""

import json

from typer.testing import CliRunner

from litmus.cli import app
from litmus.datasets import Case
from litmus.report import EvalReport, build_report, compare_reports
from litmus.runners import CaseResult, EvalRunResult
from litmus.scorers import ExactMatchScorer
from litmus.scorers.base import Score

runner = CliRunner()


def _report(name="r"):
    results = []
    for i, expected in enumerate(["a", "b"]):
        results.append(
            CaseResult(
                case_id=f"c{i}",
                output=expected,
                scores={
                    "exact_match": Score(
                        name="exact_match", value=1.0, passed=True, explanation="ok"
                    )
                },
                latency_s=0.1,
            )
        )
    run = EvalRunResult(dataset_name="d", dataset_version="1.0", results=results)
    return build_report(run, run_name=name)


def test_report_round_trip():
    original = _report()
    restored = EvalReport.from_dict(json.loads(original.to_json()))
    assert restored.run_name == original.run_name
    assert restored.dataset_name == original.dataset_name
    assert restored.summaries["exact_match"].mean == 1.0
    assert restored.results[0].scores["exact_match"].passed is True
    assert restored.results[1].case_id == "c1"
    # diff of a report against itself finds nothing
    d = compare_reports(original, restored)
    assert not d.has_regression()
    assert not d.new_failures and not d.fixed


def test_diff_cli_no_regression(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(_report("base").to_json())
    result = runner.invoke(app, ["diff", "--baseline", str(base), "--candidate", str(base)])
    assert result.exit_code == 0, result.output
    assert "No regression" in result.output


def test_diff_cli_detects_regression(tmp_path):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(_report("base").to_json())
    bad = _report("cand")
    bad.results[0].scores["exact_match"] = Score(
        name="exact_match", value=0.0, passed=False, explanation="regressed"
    )
    # rebuild summaries to stay consistent
    bad = build_report(
        EvalRunResult(
            dataset_name=bad.dataset_name,
            dataset_version=bad.dataset_version,
            results=bad.results,
        ),
        run_name="cand",
    )
    cand.write_text(bad.to_json())
    result = runner.invoke(app, ["diff", "--baseline", str(base), "--candidate", str(cand)])
    assert result.exit_code == 1, result.output
    assert "REGRESSION" in result.output
    assert "c0" in result.output


def test_diff_cli_missing_file(tmp_path):
    result = runner.invoke(
        app,
        ["diff", "--baseline", str(tmp_path / "nope.json"), "--candidate", "x.json"],
    )
    assert result.exit_code == 2


def test_exact_match_scorer_still_async():
    # guard the async scorer contract the runner relies on
    import asyncio

    scorer = ExactMatchScorer()
    score = asyncio.run(scorer.score(Case(id="c", input="x", expected="X"), "x"))
    assert score.passed is True
