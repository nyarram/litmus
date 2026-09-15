"""OpenTelemetry tracing for litmus, following GenAI semantic conventions.

Tracing is strictly additive: every litmus API works identically with
tracing disabled, and enabling it never changes eval results. The OTel SDK
is an optional dependency — importing this module is always safe; only
``init_tracing()`` requires the SDK to be installed.

Content capture (prompts, completions, tool payloads) is opt-in because
traces get exported to backends you may not control. Metadata — model,
tokens, latency, scores — is always recorded.
"""
from __future__ import annotations

import functools
import json
from contextlib import contextmanager
from typing import Any, Callable, Iterator

# ---- GenAI semantic convention attribute keys (stable subset) ----
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

_provider: Any = None
_tracer: Any = None
_memory_exporter: Any = None
_capture_content = False
_initialized = False


class TracingNotInitialized(RuntimeError):
    """Raised when tracing is used without init_tracing()."""


def _require_otel() -> Any:
    """Import the OTel API, or raise a helpful error if the extra is missing."""
    try:
        import opentelemetry.trace as trace_api  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "litmus tracing needs the 'tracing' extra: pip install litmus[tracing]"
        ) from e
    return trace_api


class _NoopSpan:
    """Stand-in span used when tracing is disabled; every method is a no-op."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@contextmanager
def _noop_span() -> Iterator[_NoopSpan]:
    yield _NoopSpan()


def init_tracing(
    *,
    service_name: str = "litmus",
    exporter: str = "console",
    endpoint: str | None = None,
    capture_content: bool = False,
):
    """Initialize the global tracer provider.

    exporter: "console" (default, prints spans to stdout — no infra needed),
    "otlp" (needs the opentelemetry-exporter-otlp package), or "memory"
    (in-memory, for tests; read spans with get_finished_spans()).
    """
    global _provider, _tracer, _memory_exporter, _capture_content, _initialized
    trace_api = _require_otel()
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    shutdown_tracing()  # safe no-op when nothing is running

    if exporter == "console":
        span_exporter: Any = ConsoleSpanExporter()
        processor = SimpleSpanProcessor(span_exporter)  # print spans immediately
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise RuntimeError(
                "the 'otlp' exporter needs opentelemetry-exporter-otlp installed"
            ) from e
        span_exporter = (
            OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        )
        processor = BatchSpanProcessor(span_exporter)
    elif exporter == "memory":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        span_exporter = InMemorySpanExporter()
        _memory_exporter = span_exporter
        processor = SimpleSpanProcessor(span_exporter)
    else:
        raise ValueError(
            f"unknown exporter {exporter!r}; expected 'console', 'otlp', or 'memory'"
        )

    _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    _provider.add_span_processor(processor)
    _tracer = _provider.get_tracer("litmus")
    _capture_content = capture_content
    _initialized = True
    try:
        trace_api.set_tracer_provider(_provider)
    except Exception:
        pass  # a provider is already set globally; ours is what span() uses
    return _tracer


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider; safe to call when idle."""
    global _provider, _tracer, _memory_exporter, _capture_content, _initialized
    if _provider is not None:
        try:
            _provider.force_flush()
            _provider.shutdown()
        except Exception:
            pass
    _provider = _tracer = _memory_exporter = None
    _capture_content = False
    _initialized = False


def ensure_initialized() -> None:
    if not _initialized:
        raise TracingNotInitialized(
            "tracing is not initialized; call litmus.tracing.init_tracing() first "
            "(or pass --trace to the litmus CLI)"
        )


def get_finished_spans() -> list:
    """Return spans recorded by the 'memory' exporter (tests)."""
    if _memory_exporter is None:
        raise RuntimeError("no memory exporter active; init_tracing(exporter='memory') first")
    return _memory_exporter.get_finished_spans()


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None, kind: Any = None):
    """Create a span; a no-op when tracing is not initialized."""
    if _tracer is None:
        with _noop_span() as s:
            yield s
        return
    trace_api = _require_otel()
    kind = kind if kind is not None else trace_api.SpanKind.INTERNAL
    with _tracer.start_as_current_span(name, kind=kind, attributes=attributes) as s:
        yield s


def traced(name: str | None = None, *, attributes: dict[str, Any] | None = None):
    """Decorator that runs the function inside a span named ``name``."""

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, attributes=attributes):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def trace_llm_call(
    system: str,
    model: str,
    prompt: str,
    client: Callable[[str], str],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    capture_content: bool | None = None,
) -> str:
    """Run ``client(prompt)`` inside a GenAI-semconv span; return the completion.

    Records gen_ai.system, gen_ai.request.model, and token usage as span
    attributes. Prompt/completion payloads are recorded as span events only
    when content capture is enabled (argument or init_tracing flag).
    """
    trace_api = _require_otel()
    capture = _capture_content if capture_content is None else capture_content
    attributes: dict[str, Any] = {
        GEN_AI_SYSTEM: system,
        GEN_AI_REQUEST_MODEL: model,
    }
    if input_tokens is not None:
        attributes[GEN_AI_USAGE_INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        attributes[GEN_AI_USAGE_OUTPUT_TOKENS] = output_tokens
    with span("gen_ai.chat", attributes=attributes, kind=trace_api.SpanKind.CLIENT) as s:
        if capture:
            s.add_event("gen_ai.content.prompt", {"content": prompt})
        completion = client(prompt)
        if capture:
            s.add_event("gen_ai.content.completion", {"content": completion})
        return completion


def trace_tool_call(
    tool_name: str,
    func: Callable[[], Any],
    *,
    arguments: Any = None,
    capture_content: bool | None = None,
) -> Any:
    """Run ``func()`` inside a tool span; return its result."""
    trace_api = _require_otel()
    capture = _capture_content if capture_content is None else capture_content
    with span(
        f"gen_ai.tool.{tool_name}",
        attributes={GEN_AI_TOOL_NAME: tool_name},
        kind=trace_api.SpanKind.CLIENT,
    ) as s:
        if capture and arguments is not None:
            s.add_event("gen_ai.tool.arguments", {"arguments": json.dumps(arguments)[:2000]})
        result = func()
        if capture:
            try:
                payload = json.dumps(result)[:2000]
            except (TypeError, ValueError):
                payload = str(result)[:2000]
            s.add_event("gen_ai.tool.result", {"result": payload})
        return result
