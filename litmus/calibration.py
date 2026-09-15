"""Calibration: measure how well a judge agrees with human labels.

A judge is only as trustworthy as its agreement with human graders. Feed
``calibrate`` a labeled dataset (prediction/reference pairs with human
scores) and it reports mean absolute error, bias, Pearson correlation, and
a calibration curve of binned predicted-vs-human means. Run it once to vet
a judge before trusting it inside the CI gate.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from litmus.judges import Judge

CURVE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.01),  # 1.01 so a perfect 1.0 lands in the top bin
)


class LabeledDatasetError(ValueError):
    """Raised when a labeled dataset file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class LabeledCase:
    id: str
    prediction: str
    reference: str
    human_score: float  # 0.0 .. 1.0


def load_labeled(path: str | Path) -> list[LabeledCase]:
    """Load human-labeled (prediction, reference, score) rows from JSONL."""
    path = Path(path)
    if not path.is_file():
        raise LabeledDatasetError(f"labeled dataset not found: {path}")
    cases: list[LabeledCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise LabeledDatasetError(f"line {lineno}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise LabeledDatasetError(f"line {lineno}: each line must be a JSON object")
            for key in ("id", "prediction", "reference", "human_score"):
                if key not in obj:
                    raise LabeledDatasetError(f"line {lineno}: missing field {key!r}")
            try:
                human = float(obj["human_score"])
            except (TypeError, ValueError) as e:
                raise LabeledDatasetError(
                    f"line {lineno}: 'human_score' must be numeric"
                ) from e
            if not 0.0 <= human <= 1.0:
                raise LabeledDatasetError(
                    f"line {lineno}: 'human_score' must be in [0, 1]"
                )
            if obj["id"] in seen:
                raise LabeledDatasetError(f"line {lineno}: duplicate id {obj['id']!r}")
            seen.add(obj["id"])
            cases.append(
                LabeledCase(
                    id=str(obj["id"]),
                    prediction=str(obj["prediction"]),
                    reference=str(obj["reference"]),
                    human_score=human,
                )
            )
    if not cases:
        raise LabeledDatasetError(f"labeled dataset is empty: {path}")
    return cases


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or None when it is undefined (< 2 points or zero variance)."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if den == 0.0:
        return None
    return num / den


@dataclass
class CalibrationReport:
    judge: str
    n: int
    mae: float = 0.0
    bias: float = 0.0  # mean(predicted - human); >0 means the judge is generous
    pearson: float | None = None
    curve: list[dict] = field(default_factory=list)
    ran_at: str = ""

    def to_dict(self) -> dict:
        return {
            "judge": self.judge,
            "n": self.n,
            "mae": round(self.mae, 4),
            "bias": round(self.bias, 4),
            "pearson": None if self.pearson is None else round(self.pearson, 4),
            "curve": self.curve,
            "ran_at": self.ran_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        p = "n/a" if self.pearson is None else f"{self.pearson:.3f}"
        lines = [
            f"# litmus calibration report — {self.judge}",
            "",
            f"n={self.n} · MAE **{self.mae:.3f}** · bias **{self.bias:+.3f}** · Pearson {p}",
            "",
            "| predicted bin | n | mean predicted | mean human |",
            "|---|---|---|---|",
        ]
        for b in self.curve:
            mp = "—" if b["mean_predicted"] is None else f"{b['mean_predicted']:.3f}"
            mh = "—" if b["mean_human"] is None else f"{b['mean_human']:.3f}"
            lines.append(f"| {b['bin'][0]:.1f}–{b['bin'][1]:.1f} | {b['n']} | {mp} | {mh} |")
        return "\n".join(lines) + "\n"


def calibrate(cases: Sequence[LabeledCase], judge: Judge) -> CalibrationReport:
    """Score every labeled case with ``judge`` and compare against human scores."""
    if not cases:
        raise ValueError("no labeled cases to calibrate on")
    predicted = [judge.score(c.prediction, c.reference).score for c in cases]
    human = [c.human_score for c in cases]
    n = len(cases)

    report = CalibrationReport(
        judge=judge.name,
        n=n,
        ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    report.mae = sum(abs(p - h) for p, h in zip(predicted, human)) / n
    report.bias = sum(p - h for p, h in zip(predicted, human)) / n
    report.pearson = _pearson(predicted, human)

    for lo, hi in CURVE_BINS:
        in_bin = [(p, h) for p, h in zip(predicted, human) if lo <= p < hi]
        if in_bin:
            mp = sum(p for p, _ in in_bin) / len(in_bin)
            mh = sum(h for _, h in in_bin) / len(in_bin)
        else:
            mp, mh = None, None
        report.curve.append(
            {
                "bin": [lo, min(hi, 1.0)],
                "n": len(in_bin),
                "mean_predicted": None if mp is None else round(mp, 4),
                "mean_human": None if mh is None else round(mh, 4),
            }
        )
    return report
