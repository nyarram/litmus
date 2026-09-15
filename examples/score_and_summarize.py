"""Example system under test, modeled on AstroDigest's news-scoring worker.

AstroDigest scores astronomy news items for newsworthiness before summarizing
them. This module replays that shape: input is a news item, output is a
structured verdict ``{"newsworthy": bool, "score": float, "reason": str}``.

Two systems:
- ``score_stub``: deterministic keyword heuristic. Runs with no API key —
  deliberately imperfect, so the eval report has something to say.
- ``score_with_groq``: calls Groq's free tier like the real pipeline.
  Needs the ``GROQ_API_KEY`` environment variable.

Scorer sets:
- ``scorers_basic``: shape validation + newsworthiness accuracy (no API key).
- ``scorers_full()``: adds an LLM judge of the written reason (needs Groq key).
"""

from __future__ import annotations

import json
import os
from typing import Any

from opentelemetry import trace

from litmus import (
    Case,
    ExactMatchScorer,
    GroqProvider,
    JSONSchemaScorer,
    JudgeRubric,
    LLMJudge,
)
from litmus.tracing import set_span_io, start_llm_span

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["newsworthy", "score", "reason"],
    "properties": {
        "newsworthy": {"type": "boolean"},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1},
    },
}

_KEYWORDS = {
    "nasa", "webb", "exoplanet", "black hole", "spacex", "artemis",
    "astronomers", "telescope", "jwst", "iss", "esa",
}


def _text(case: Case) -> str:
    data = case.input if isinstance(case.input, dict) else {"text": str(case.input)}
    return f"{data.get('title', '')} {data.get('snippet', '')}".lower()


async def score_stub(case: Case) -> dict[str, Any]:
    """Deterministic heuristic: newsworthy if it mentions known space keywords."""
    text = _text(case)
    hit = any(k in text for k in _KEYWORDS)
    return {
        "newsworthy": hit,
        "score": 0.85 if hit else 0.15,
        "reason": (
            "Mentions recognized space program keywords."
            if hit
            else "No space-news indicators found."
        ),
    }


_SCORE_PROMPT = """You score astronomy news for a space-news digest. Reply with ONLY valid JSON.

Item:
Title: {title}
Snippet: {snippet}

Decide: is this genuine astronomy/space news worth including in a digest?
Respond: {{"newsworthy": <true|false>, "score": <0..1>, "reason": "<one sentence>"}}"""


async def score_with_groq(case: Case) -> dict[str, Any]:
    """The real shape: one LLM call with structured output, traced with OTel."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    data = case.input if isinstance(case.input, dict) else {"title": str(case.input), "snippet": ""}
    provider = GroqProvider(api_key=api_key)
    messages = [
        {
            "role": "user",
            "content": _SCORE_PROMPT.format(title=data.get("title"), snippet=data.get("snippet")),
        }
    ]
    tracer = trace.get_tracer("astrodigest.worker")
    with start_llm_span(tracer, "groq", provider.model) as span:
        raw = await provider.complete(messages, json_mode=True)
        set_span_io(span, messages, raw)
    parsed = json.loads(raw)
    return {
        "newsworthy": bool(parsed["newsworthy"]),
        "score": max(0.0, min(1.0, float(parsed["score"]))),
        "reason": str(parsed["reason"]),
    }


scorers_basic = [
    JSONSchemaScorer("valid_shape", OUTPUT_SCHEMA),
    ExactMatchScorer("newsworthy_match", field="newsworthy"),
]

REASON_RUBRIC = JudgeRubric(
    name="reason_quality",
    criteria=(
        "The 'reason' must justify the newsworthy verdict with specific reference to the "
        "item's content. 5 = cites concrete facts from the item and connects them to "
        "newsworthiness; 3 = generic but plausible; 1 = contradicts the verdict or the item."
    ),
)


def scorers_full() -> list:
    """Basic scorers plus an LLM judge on reason quality (needs GROQ_API_KEY)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set — cannot build the judge scorers")
    return [
        *scorers_basic,
        LLMJudge(REASON_RUBRIC, GroqProvider(api_key=api_key)),
    ]
