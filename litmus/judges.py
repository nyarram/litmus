"""Judges: how a prediction is scored against a reference.

A Judge is any object with a ``name`` and a ``score(prediction, reference)``
method returning a ``Score`` in [0, 1]. Piece 1 ships deterministic judges
that need no model calls. Model-graded judging lives in ``litmus.judges_llm``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Score:
    judge: str
    score: float  # 0.0 .. 1.0
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class Judge(Protocol):
    name: str

    def score(self, prediction: str, reference: str) -> Score: ...


class ExactMatchJudge:
    """1.0 when the prediction equals the reference, else 0.0."""

    name = "exact_match"

    def __init__(self, *, case_sensitive: bool = True, strip: bool = True) -> None:
        self.case_sensitive = case_sensitive
        self.strip = strip

    def _norm(self, s: str) -> str:
        if self.strip:
            s = s.strip()
        return s if self.case_sensitive else s.casefold()

    def score(self, prediction: str, reference: str) -> Score:
        match = self._norm(prediction) == self._norm(reference)
        return Score(
            judge=self.name,
            score=1.0 if match else 0.0,
            passed=match,
            details={"case_sensitive": self.case_sensitive},
        )


class ContainmentJudge:
    """1.0 when the reference appears verbatim inside the prediction."""

    name = "containment"

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def _norm(self, s: str) -> str:
        return s if self.case_sensitive else s.casefold()

    def score(self, prediction: str, reference: str) -> Score:
        hit = self._norm(reference) in self._norm(prediction)
        return Score(
            judge=self.name,
            score=1.0 if hit else 0.0,
            passed=hit,
            details={"case_sensitive": self.case_sensitive},
        )
