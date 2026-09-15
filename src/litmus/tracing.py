"""OpenTelemetry tracing with GenAI semantic conventions.

Every eval case runs inside a ``litmus.case`` span carrying a trace id, so a
score in a report can be traced back to the exact model call that produced
it. Instrument your system under test with ``start_llm_span`` to get nested
spans with standard ``gen_ai.*`` attributes that any OTel backend understands.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

_initialized = False


def init_tracing(
    service_name: str = "litmus", exporter: SpanExporter | None = None
) -> trace.Tracer:
    """Install the global tracer provider. Safe to call multiple times; first call wins."""
    global _initialized
    if not _initialized:
        provider = TracerProvider()
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
    return trace.get_tracer(service_name)


def init_console_tracing(service_name: str = "litmus") -> trace.Tracer:
    """Tracing that prints spans to stdout — handy for debugging eval runs."""
    return init_tracing(service_name, exporter=ConsoleSpanExporter())


@contextmanager
def start_llm_span(
    tracer: trace.Tracer,
    system: str,
    model: str,
    operation: str = "chat",
) -> Iterator[trace.Span]:
    """Span for one LLM call, using GenAI semantic conventions.

    Usage:
        with start_llm_span(tracer, "groq", "openai/gpt-oss-120b") as span:
            text = await provider.complete(messages)
            set_span_io(span, messages, text, input_tokens, output_tokens)
    """
    with tracer.start_as_current_span(f"gen_ai.{operation}") as span:
        span.set_attribute("gen_ai.system", system)
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.operation.name", operation)
        yield span


def set_span_io(
    span: trace.Span,
    input_messages: list[dict[str, str]] | None = None,
    output_text: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    if input_messages:
        span.set_attribute("gen_ai.input.messages", str(input_messages)[:4000])
    if output_text:
        span.set_attribute("gen_ai.output.text", output_text[:4000])
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    for key, value in (extra or {}).items():
        span.set_attribute(key, value)
