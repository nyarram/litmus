"""Load eval report JSON files from a directory. The filesystem is the database."""

from __future__ import annotations

import json
from pathlib import Path

from litmus.report import EvalReport


def load_reports(reports_dir: Path) -> list[EvalReport]:
    """Load every ``*.json`` litmus report in ``reports_dir``, oldest first.

    Files that are not valid litmus reports are skipped, not fatal: the
    directory is a drop-box, and a stray file should not take the dashboard
    down.
    """
    reports: list[EvalReport] = []
    if not reports_dir.is_dir():
        return reports
    for path in sorted(reports_dir.glob("*.json")):
        try:
            reports.append(EvalReport.from_dict(json.loads(path.read_text())))
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            continue
    reports.sort(key=lambda r: r.created_at)
    return reports


def scorer_names(reports: list[EvalReport]) -> list[str]:
    """Every scorer seen across all reports, in first-seen order."""
    names: list[str] = []
    for report in reports:
        for name in report.summaries:
            if name not in names:
                names.append(name)
    return names


def trend_points(reports: list[EvalReport], scorer: str) -> list[dict]:
    """One point per report that contains ``scorer``: time, mean, pass rate."""
    points = []
    for report in reports:
        summary = report.summaries.get(scorer)
        if summary is None:
            continue
        points.append(
            {
                "run_name": report.run_name,
                "created_at": report.created_at,
                "mean": summary.mean,
                "pass_rate": summary.pass_rate,
                "n": summary.n,
            }
        )
    return points
