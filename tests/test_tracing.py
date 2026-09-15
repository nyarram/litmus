"""Tests for OTel tracing: spans, semconv attributes, and safe defaults."""
import sys

import pytest

from litmus import tracing
from litmus.tracing import (
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    TracingNotInitialized,
    get_finished_spans,
    init_tracing,
    shutdown_tracing,
    span,
    trace_llm_call,
    trace_tool_call,
    traced,
)


@pytest.fixture(autouse=True)
def fresh_tracing():
    shutdown_tracing()
    yield
    shutdown_tracing()


def names(spans):
    return [s.name for s in spans]


def test_spans_record_names_attributes_and_nesting():
    init_tracing(exporter="memory")
    with span("parent", attributes={"k": "v"}):
        with span("child"):
            pass
    spans = get_finished_spans()
    assert names(spans) == ["child", "parent"]
    parent = next(s for s in spans if s.name == "parent")
    child = next(s for s in spans if s.name == "child")
    assert parent.attributes["k"] == "v"
    assert child.parent.span_id == parent.context.span_id


def test_span_is_noop_when_tracing_disabled():
    with span("whatever") as s:  # must not raise, must not record
        s.set_attribute("k", "v")
        s.add_event("e")
    with pytest.raises(RuntimeError, match="memory exporter"):
        get_finished_spans()


def test_ensure_initialized_guards_trace_flag():
    with pytest.raises(TracingNotInitialized):
        tracing.ensure_initialized()
    init_tracing(exporter="memory")
    tracing.ensure_initialized()  # no raise once initialized


def test_traced_decorator():
    init_tracing(exporter="memory")

    @traced("my-op")
    def work(x):
        return x * 2

    assert work(21) == 42
    assert names(get_finished_spans()) == ["my-op"]


def test_trace_llm_call_records_semconv():
    init_tracing(exporter="memory")
    text = trace_llm_call(
        "acme", "acme-large", "hello", lambda p: "world",
        input_tokens=5, output_tokens=1,
    )
    assert text == "world"
    (s,) = get_finished_spans()
    assert s.name == "gen_ai.chat"
    assert s.attributes[GEN_AI_SYSTEM] == "acme"
    assert s.attributes[GEN_AI_REQUEST_MODEL] == "acme-large"
    assert s.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert s.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 1
    assert s.kind.name == "CLIENT"


def test_content_capture_off_by_default():
    init_tracing(exporter="memory")
    trace_llm_call("acme", "m", "secret prompt", lambda p: "secret completion")
    (s,) = get_finished_spans()
    assert s.events == ()


def test_content_capture_opt_in():
    init_tracing(exporter="memory", capture_content=True)
    trace_llm_call("acme", "m", "hello", lambda p: "world")
    (s,) = get_finished_spans()
    event_names = [e.name for e in s.events]
    assert "gen_ai.content.prompt" in event_names
    assert "gen_ai.content.completion" in event_names


def test_trace_tool_call():
    init_tracing(exporter="memory")
    result = trace_tool_call("search", lambda: {"hits": 3}, arguments={"q": "x"})
    assert result == {"hits": 3}
    (s,) = get_finished_spans()
    assert s.name == "gen_ai.tool.search"
    assert s.attributes[GEN_AI_TOOL_NAME] == "search"


def test_unknown_exporter_rejected():
    with pytest.raises(ValueError, match="unknown exporter"):
        init_tracing(exporter="nope")


def test_missing_sdk_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", None)
    # force re-import attempt to fail
    import importlib

    with pytest.raises(RuntimeError, match="pip install litmus\\[tracing\\]"):
        importlib.reload(tracing)
        tracing.init_tracing(exporter="memory")
    importlib.reload(tracing)  # restore module state
