"""OpenTelemetry tracing with GenAI semantic conventions.

Tracing is strictly additive:

- The OTel SDK is an **optional** dependency. Importing this module — and
  importing ``litmus`` itself — never requires it. Only ``init_tracing()``
  needs the SDK, and it raises a helpful error telling you how to install it.
- Every helper is a **no-op when tracing is not initialized**, so
  instrumented code runs identically with tracing on or off.

Content capture (prompts, completions) is **opt-in** because traces get
exported to backends you may not control. Metadata — model, tokens,
latency, scores — is always recorded; payloads are recorded as span
*events* (never attributes, which backends index and retain) only when
capture is enabled.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# ---- GenAI semantic convention attribute keys (stable subset) ----
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

_provider: Any = None
_memory_exporter: Any = None
_capture_content = False
_initialized = False


class TracingNotInitialized(RuntimeError):
    """Raised when trace output is requested without init_tracing()."""


def _require_otel() -> tuple[Any, Any, Any, Any, Any]:
    """Import the OTel SDK, or raise a helpful error if the extra is missing."""
    try:
        import opentelemetry.trace as trace_api
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError as e:
        raise RuntimeError(
            "litmus tracing needs the 'tracing' extra: pip install 'litmus[tracing]'"
        ) from e
    return trace_api, TracerProvider, BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor


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

    def get_span_context(self) -> None:
        return None

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


@contextmanager
def _noop_span() -> Iterator[_NoopSpan]:
    yield _NoopSpan()


def init_tracing(
    *,
    service_name: str = "litmus",
    exporter: str | Any = "console",
    endpoint: str | None = None,
    capture_content: bool = False,
) -> Any:
    """Initialize the global tracer provider and return a tracer.

    exporter: ``"console"`` (default, prints spans to stdout — no infra
    needed), ``"otlp"`` (needs the ``opentelemetry-exporter-otlp`` package),
    ``"memory"`` (in-memory, for tests — read spans with
    ``get_finished_spans()``), a ready-made OTel ``SpanExporter``, or
    ``None`` (provider with no exporters; spans are created and dropped).
    """
    global _provider, _memory_exporter, _capture_content, _initialized
    trace_api, TracerProvider, BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor = (
        _require_otel()
    )
    shutdown_tracing()  # safe no-op when nothing is running

    processor: Any = None
    if exporter is None:
        pass  # provider with no exporters: spans are created and dropped
    elif exporter == "console":
        processor = SimpleSpanProcessor(ConsoleSpanExporter())  # print immediately
    elif exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise RuntimeError(
                "the 'otlp' exporter needs the opentelemetry-exporter-otlp package"
            ) from e
        # The OTLP HTTP exporter posts to the endpoint verbatim, so a bare
        # base URL (e.g. http://collector:4318, as documented for
        # --otlp-endpoint) would 404. Normalize to the traces path.
        if endpoint and not endpoint.rstrip("/").endswith("/v1/traces"):
            endpoint = endpoint.rstrip("/") + "/v1/traces"
        otlp = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
        processor = BatchSpanProcessor(otlp)
    elif exporter == "memory":
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        _memory_exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(_memory_exporter)
    elif hasattr(exporter, "export"):
        processor = BatchSpanProcessor(exporter)
    else:
        raise ValueError(
            f"unknown exporter {exporter!r}; expected 'console', 'otlp', 'memory', "
            "a SpanExporter, or None"
        )

    from opentelemetry.sdk.resources import Resource

    _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if processor is not None:
        _provider.add_span_processor(processor)
    _capture_content = capture_content
    _initialized = True
    try:
        trace_api.set_tracer_provider(_provider)
    except Exception:
        pass  # a provider is already set globally; ours is what litmus uses
    return _provider.get_tracer("litmus")


def init_console_tracing(service_name: str = "litmus") -> Any:
    """Tracing that prints spans to stdout — handy for debugging eval runs."""
    return init_tracing(service_name=service_name, exporter="console")


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider; safe to call when idle."""
    global _provider, _memory_exporter, _capture_content, _initialized
    if _provider is not None:
        try:
            _provider.force_flush()
            _provider.shutdown()
        except Exception:
            pass
    _provider = _memory_exporter = None
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


def _current_tracer() -> Any | None:
    return _provider.get_tracer("litmus") if _provider is not None else None


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Create a span; a no-op when tracing is not initialized."""
    tracer = _current_tracer()
    if tracer is None:
        with _noop_span() as s:
            yield s
        return
    trace_api, _, _, _, _ = _require_otel()
    with tracer.start_as_current_span(
        name, kind=trace_api.SpanKind.INTERNAL, attributes=attributes
    ) as s:
        yield s


@contextmanager
def case_span(
    case_id: str, attributes: dict[str, Any] | None = None
) -> Iterator[tuple[Any, str | None]]:
    """Span for one eval case; yields ``(span, trace_id)``. No-op safe.

    A score in a report carries the trace id, so it can be traced back to
    the exact execution that produced it.
    """
    attrs = {"litmus.case_id": case_id, **(attributes or {})}
    with span("litmus.case", attrs) as s:
        ctx = s.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx is not None else None
        yield s, trace_id


@contextmanager
def start_llm_span(
    system: str,
    model: str,
    operation: str = "chat",
    *,
    capture_content: bool | None = None,
) -> Iterator[Any]:
    """Span for one LLM call, using GenAI semantic conventions.

    Usage:
        with start_llm_span("groq", "openai/gpt-oss-120b") as llm:
            text = await provider.complete(messages)
            set_span_io(llm, input_messages=messages, output_text=text,
                        input_tokens=..., output_tokens=...)
    """
    attrs = {
        GEN_AI_SYSTEM: system,
        GEN_AI_REQUEST_MODEL: model,
        GEN_AI_OPERATION_NAME: operation,
    }
    with span(f"gen_ai.{operation}", attrs) as s:
        yield s
    # content capture is consumed by set_span_io, not here


def set_span_io(
    span_obj: Any,
    input_messages: list[dict[str, str]] | None = None,
    output_text: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
    *,
    capture_content: bool | None = None,
) -> None:
    """Attach IO metadata to an LLM span.

    Token counts and ``extra`` are always recorded as attributes. Message
    content is recorded as span *events* only when content capture is
    enabled (argument, else the ``init_tracing`` default) — never by
    default, and never as attributes.
    """
    capture = _capture_content if capture_content is None else capture_content
    if input_tokens is not None:
        span_obj.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    if output_tokens is not None:
        span_obj.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    for key, value in (extra or {}).items():
        span_obj.set_attribute(key, value)
    if capture:
        if input_messages:
            span_obj.add_event("gen_ai.content.prompt", {"content": str(input_messages)[:4000]})
        if output_text:
            span_obj.add_event("gen_ai.content.completion", {"content": output_text[:4000]})
