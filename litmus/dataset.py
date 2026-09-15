"""Golden datasets: the frozen inputs + references every eval runs against.

A dataset is a JSONL file, one object per line:

    {"id": "ex-001", "input": "hello", "reference": "hello",
     "tags": ["smoke"], "metadata": {"source": "hand-written"}}

``id`` and ``input`` and ``reference`` are required. Everything else is
optional context that judges and reports can use but never score on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """Raised when a dataset file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class GoldenCase:
    id: str
    input: str
    reference: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def _validate(obj: dict[str, Any], lineno: int) -> GoldenCase:
    for key in ("id", "input", "reference"):
        if key not in obj:
            raise DatasetError(f"line {lineno}: missing required field {key!r}")
        if not isinstance(obj[key], str):
            raise DatasetError(f"line {lineno}: field {key!r} must be a string")
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"line {lineno}: field 'metadata' must be an object")
    tags = obj.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise DatasetError(f"line {lineno}: field 'tags' must be a list of strings")
    return GoldenCase(
        id=obj["id"],
        input=obj["input"],
        reference=obj["reference"],
        metadata=metadata,
        tags=tags,
    )


def load_dataset(path: str | Path) -> list[GoldenCase]:
    """Load and validate a golden dataset from a JSONL file."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"line {lineno}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise DatasetError(f"line {lineno}: each line must be a JSON object")
            case = _validate(obj, lineno)
            if case.id in seen:
                raise DatasetError(f"line {lineno}: duplicate case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise DatasetError(f"dataset is empty: {path}")
    return cases
