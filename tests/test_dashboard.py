"""Tests for the dashboard store, app routes, and tracing flag plumbing."""

import pytest
from fastapi.testclient import TestClient

from litmus.dashboard import create_app, store
from litmus.report import build_report
from litmus.runners import CaseResult, EvalRunResult
from litmus.scorers.base import Score


def _report(name, created_at, means=(1.0, 0.5)):
    results = []
    for i, value in enumerate(means):
        results.append(
            CaseResult(
                case_id=f"c{i}",
                output="x",
                scores={
                    "quality": Score(
                        name="quality",
                        value=value,
                        passed=value >= 0.7,
                        explanation="ok",
                    )
                },
                latency_s=0.1,
                trace_id="ab" * 16,
            )
        )
    run = EvalRunResult(dataset_name="d", dataset_version="1.0", results=results)
    report = build_report(run, run_name=name)
    report.created_at = created_at
    return report


def _write_reports(tmp_path):
    for name, ts in (("run-a", "2026-09-14T10:00:00"), ("run-b", "2026-09-15T10:00:00")):
        (tmp_path / f"{name}.json").write_text(_report(name, ts).to_json())
    (tmp_path / "not-a-report.json").write_text('{"hello": "world"}')
    (tmp_path / "garbage.json").write_text("not json at all {{{")
    return tmp_path


def test_load_reports_sorts_and_skips_bad_files(tmp_path):
    reports = store.load_reports(_write_reports(tmp_path))
    assert [r.run_name for r in reports] == ["run-a", "run-b"]


def test_load_reports_missing_dir(tmp_path):
    assert store.load_reports(tmp_path / "nope") == []


def test_trend_points(tmp_path):
    reports = store.load_reports(_write_reports(tmp_path))
    pts = store.trend_points(reports, "quality")
    assert [p["run_name"] for p in pts] == ["run-a", "run-b"]
    assert pts[0]["mean"] == pytest.approx(0.75)
    assert pts[0]["pass_rate"] == pytest.approx(0.5)
    assert store.trend_points(reports, "nope") == []


def test_index_lists_runs(tmp_path):
    client = TestClient(create_app(reports_dir=_write_reports(tmp_path)))
    resp = client.get("/")
    assert resp.status_code == 200
    assert "run-a" in resp.text and "run-b" in resp.text


def test_run_detail_and_404(tmp_path):
    client = TestClient(create_app(reports_dir=_write_reports(tmp_path)))
    resp = client.get("/runs/run-a")
    assert resp.status_code == 200
    assert "c0" in resp.text
    # no jaeger url configured: trace ids render as text, not links
    assert "/trace/" not in resp.text
    assert client.get("/runs/unknown").status_code == 404


def test_run_detail_jaeger_links(tmp_path):
    client = TestClient(
        create_app(reports_dir=_write_reports(tmp_path), jaeger_url="http://localhost:16686/")
    )
    resp = client.get("/runs/run-a")
    assert "http://localhost:16686/trace/" in resp.text


def test_trends_and_api(tmp_path):
    client = TestClient(create_app(reports_dir=_write_reports(tmp_path)))
    resp = client.get("/trends")
    assert resp.status_code == 200
    assert "quality" in resp.text
    api = client.get("/api/runs").json()
    assert [r["run_name"] for r in api] == ["run-a", "run-b"]
    assert api[0]["summaries"]["quality"]["mean"] == pytest.approx(0.75)


def test_init_tracing_prefers_otlp(monkeypatch):
    import litmus.cli as cli
    import litmus.tracing as tracing

    calls = []
    monkeypatch.setattr(tracing, "init_tracing", lambda **kw: calls.append(kw) or None)
    monkeypatch.setattr(
        tracing, "init_console_tracing", lambda **kw: calls.append({"console": True})
    )

    cli._init_tracing(trace=True, otlp_endpoint="http://localhost:4318")
    assert calls[-1]["exporter"] == "otlp"
    assert calls[-1]["endpoint"] == "http://localhost:4318"

    calls.clear()
    monkeypatch.setenv("LITMUS_OTLP_ENDPOINT", "http://collector:4318")
    cli._init_tracing(trace=False, otlp_endpoint=None)
    assert calls[-1]["exporter"] == "otlp"

    calls.clear()
    monkeypatch.delenv("LITMUS_OTLP_ENDPOINT")
    cli._init_tracing(trace=True, otlp_endpoint=None)
    assert calls[-1]["exporter"] == "console"

    calls.clear()
    cli._init_tracing(trace=False, otlp_endpoint=None)
    assert calls == []
