"""Datasets: the unit of an eval is a Case; a named, versioned list of them is a Dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """One eval case: an input to the system under test, plus the expected answer."""

    id: str
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    """A named, versioned collection of cases. Version it — regressions are measured per version."""

    name: str
    version: str = "1.0"
    cases: list[Case] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        path = Path(path)
        raw = json.loads(path.read_text()) if path.suffix == ".json" else None
        if raw is not None:
            return cls(**raw)
        header: dict[str, Any] = {}
        cases: list[Case] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if set(obj) == {"name", "version"} or (set(obj) <= {"name", "version"} and not cases):
                header = obj
                continue
            cases.append(Case(**obj))
        return cls(
            name=header.get("name", path.stem),
            version=header.get("version", "1.0"),
            cases=cases,
        )

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        lines = [json.dumps({"name": self.name, "version": self.version})]
        lines += [c.model_dump_json() for c in self.cases]
        path.write_text("\n".join(lines) + "\n")
