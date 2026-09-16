"""Tests for tracing exporter setup."""

import pytest

otlp_module = pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")

from litmus import tracing  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_tracing():
    yield
    tracing.shutdown_tracing()


def test_otlp_endpoint_gets_traces_path(monkeypatch):
    seen = {}

    class FakeExporter:
        def __init__(self, endpoint=None):
            seen["endpoint"] = endpoint

    monkeypatch.setattr(otlp_module, "OTLPSpanExporter", FakeExporter)

    tracing.init_tracing(exporter="otlp", endpoint="http://collector:4318")
    assert seen["endpoint"] == "http://collector:4318/v1/traces"

    tracing.init_tracing(exporter="otlp", endpoint="http://collector:4318/")
    assert seen["endpoint"] == "http://collector:4318/v1/traces"

    tracing.init_tracing(exporter="otlp", endpoint="http://collector:4318/v1/traces")
    assert seen["endpoint"] == "http://collector:4318/v1/traces"
