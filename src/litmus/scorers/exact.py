"""Deterministic string scorers: exact match, substring, and regex."""

from __future__ import annotations

import re
from typing import Any

from litmus.datasets import Case
from litmus.scorers.base import Score


def _normalize(value: Any, case_insensitive: bool, strip: bool) -> str:
    text = "" if value is None else str(value)
    if strip:
        text = text.strip()
    if case_insensitive:
        text = text.lower()
    return text


def _pluck(obj: Any, field: str | None) -> Any:
    if field is None:
        return obj
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


class ExactMatchScorer:
    """Passes when the (optionally field-plucked) output equals the expected value."""

    def __init__(
        self, name: str = "exact_match", field: str | None = None, case_insensitive: bool = True
    ):
        self.name = name
        self.field = field
        self.case_insensitive = case_insensitive

    async def score(self, case: Case, output: Any) -> Score:
        expected = _normalize(_pluck(case.expected, self.field), self.case_insensitive, True)
        actual = _normalize(_pluck(output, self.field), self.case_insensitive, True)
        matched = expected == actual
        return Score(
            name=self.name,
            value=1.0 if matched else 0.0,
            passed=matched,
            details={"expected": expected, "actual": actual},
        )


class ContainsScorer:
    """Passes when the expected string appears inside the output string."""

    def __init__(
        self, name: str = "contains", field: str | None = None, case_insensitive: bool = True
    ):
        self.name = name
        self.field = field
        self.case_insensitive = case_insensitive

    async def score(self, case: Case, output: Any) -> Score:
        needle = _normalize(_pluck(case.expected, self.field), self.case_insensitive, True)
        haystack = _normalize(_pluck(output, self.field), self.case_insensitive, True)
        found = bool(needle) and needle in haystack
        return Score(
            name=self.name,
            value=1.0 if found else 0.0,
            passed=found,
            details={"needle": needle},
        )


class RegexScorer:
    """Passes when the output matches a regex. Useful for format checks (ids, citations, units)."""

    def __init__(self, name: str = "regex", pattern: str = r".*", field: str | None = None):
        self.name = name
        self.pattern = re.compile(pattern)
        self.field = field

    async def score(self, case: Case, output: Any) -> Score:
        actual = str(_pluck(output, self.field) or "")
        matched = self.pattern.search(actual) is not None
        return Score(
            name=self.name,
            value=1.0 if matched else 0.0,
            passed=matched,
            details={"pattern": self.pattern.pattern, "actual": actual[:200]},
        )
