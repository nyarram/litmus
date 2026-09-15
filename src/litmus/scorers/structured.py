"""Structured-output scorer: validates the system's output against a JSON Schema."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from litmus.datasets import Case
from litmus.scorers.base import Score


class JSONSchemaScorer:
    """Passes when the output (dict, or JSON string) conforms to the given JSON Schema.

    Catches the most common LLM failure mode in pipelines: malformed or
    wrongly-shaped structured output that breaks downstream workers.
    """

    def __init__(self, name: str = "valid_shape", schema: dict[str, Any] | None = None):
        self.name = name
        self.schema = schema or {"type": "object"}

    async def score(self, case: Case, output: Any) -> Score:
        parsed = output
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as exc:
                return Score(
                    name=self.name,
                    value=0.0,
                    passed=False,
                    explanation=f"Output is not valid JSON: {exc}",
                )
        try:
            jsonschema.validate(instance=parsed, schema=self.schema)
        except jsonschema.ValidationError as exc:
            return Score(
                name=self.name,
                value=0.0,
                passed=False,
                explanation=f"Schema violation: {exc.message}",
                details={"path": list(exc.absolute_path)},
            )
        return Score(name=self.name, value=1.0, passed=True)
