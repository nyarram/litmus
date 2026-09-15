"""Scorer primitives shared by every scorer implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from litmus.datasets import Case


@dataclass
class Score:
    """The verdict of one scorer on one case output."""

    name: str
    value: float  # usually normalized to 0..1
    passed: bool | None = None
    explanation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "explanation": self.explanation,
            "details": self.details,
        }


class Scorer(Protocol):
    """Anything that grades a system's output for a case. Implementations must be async."""

    name: str

    async def score(self, case: Case, output: Any) -> Score: ...
