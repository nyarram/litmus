"""CLI: ``litmus run`` executes a golden suite and enforces a quality gate.

The ``--target`` spec is either ``module:attr`` (importable from the current
working directory) or ``path/to/file.py:attr`` for a standalone script.
The process exit code is the CI gate: 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from litmus.dataset import load_dataset
from litmus.judges import ContainmentJudge, ExactMatchJudge, Judge
from litmus.runner import run_eval

JUDGES: dict[str, Callable[[], Judge]] = {
    "exact_match": ExactMatchJudge,
    "containment": ContainmentJudge,
}


def load_target(spec: str) -> Callable[[str], str]:
    module_part, sep, attr = spec.rpartition(":")
    if not sep or not attr:
        raise ValueError(
            f"bad --target {spec!r}; expected 'module:attr' or 'path/to/file.py:attr'"
        )
    if module_part.endswith(".py") and os.path.isfile(module_part):
        name = "_litmus_target_" + Path(module_part).stem
        mod_spec = importlib.util.spec_from_file_location(name, module_part)
        if mod_spec is None or mod_spec.loader is None:
            raise ValueError(f"cannot load target file {module_part!r}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    target = getattr(module, attr, None)
    if not callable(target):
        raise ValueError(f"target {spec!r} is not callable")
    return target


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_dataset(args.suite)
    judges = [JUDGES[name]() for name in args.judges]
    target = load_target(args.target)

    report = run_eval(
        cases,
        target,
        judges,
        threshold=args.threshold,
        suite=Path(args.suite).stem,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = report_dir / f"eval_report_{report.suite}_{stamp}"
    base.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    base.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")

    print(report.to_markdown())
    print(f"reports written to {base}.json / {base}.md")
    return 0 if report.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="litmus", description="offline evals for LLM pipelines")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a golden suite against a target")
    run.add_argument("--suite", required=True, help="path to a golden dataset (.jsonl)")
    run.add_argument("--target", required=True,
                     help="'module:attr' or 'path/to/file.py:attr' returning Callable[[str], str]")
    run.add_argument("--judges", default="exact_match",
                     help=f"comma-separated; available: {', '.join(sorted(JUDGES))}")
    run.add_argument("--threshold", type=float, default=1.0,
                     help="minimum overall mean score to pass (default 1.0)")
    run.add_argument("--report-dir", default="reports",
                     help="where to write eval reports (default ./reports)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        names = [n.strip() for n in args.judges.split(",") if n.strip()]
        unknown = [n for n in names if n not in JUDGES]
        if unknown:
            print(f"unknown judges: {', '.join(unknown)}", file=sys.stderr)
            return 2
        args.judges = names
        try:
            return cmd_run(args)
        except (ValueError, OSError) as e:
            print(f"litmus: {e}", file=sys.stderr)
            return 2
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
