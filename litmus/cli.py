"""CLI: ``litmus run`` executes a golden suite and enforces a quality gate;
``litmus calibrate`` vets a judge against human labels.

The ``--target`` and ``--client`` specs are either ``module:attr``
(importable from the current working directory) or ``path/to/file.py:attr``
for a standalone script. For ``run``, the process exit code is the CI gate:
0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from litmus.calibration import calibrate, load_labeled
from litmus.dataset import load_dataset
from litmus.judges import ContainmentJudge, ExactMatchJudge, Judge
from litmus.judges_llm import DEFAULT_RUBRIC, LLMJudge
from litmus.runner import run_eval

JUDGE_NAMES = ("exact_match", "containment", "llm_judge")


def load_callable(spec: str, flag: str) -> Callable:
    module_part, sep, attr = spec.rpartition(":")
    if not sep or not attr:
        raise ValueError(
            f"bad {flag} {spec!r}; expected 'module:attr' or 'path/to/file.py:attr'"
        )
    if module_part.endswith(".py") and os.path.isfile(module_part):
        name = "_litmus_dynamic_" + Path(module_part).stem
        mod_spec = importlib.util.spec_from_file_location(name, module_part)
        if mod_spec is None or mod_spec.loader is None:
            raise ValueError(f"cannot load file {module_part!r}")
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    target = getattr(module, attr, None)
    if not callable(target):
        raise ValueError(f"{flag} {spec!r} is not callable")
    return target


def load_judge_config(path: str | None) -> dict:
    config: dict = {}
    if path:
        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot read judge config {path!r}: {e}") from e
        if not isinstance(config, dict):
            raise ValueError(f"judge config {path!r} must be a JSON object")
    return config


def build_judges(names: list[str], client: Callable | None, config: dict) -> list[Judge]:
    judges: list[Judge] = []
    for name in names:
        if name == "exact_match":
            judges.append(ExactMatchJudge())
        elif name == "containment":
            judges.append(ContainmentJudge())
        elif name == "llm_judge":
            if client is None:
                raise ValueError("--client is required when using the llm_judge")
            judges.append(
                LLMJudge(
                    client,
                    prompt_version=config.get("prompt_version", "v1"),
                    rubric=config.get("rubric", DEFAULT_RUBRIC),
                    pass_threshold=config.get("pass_threshold", 0.5),
                    max_retries=config.get("max_retries", 1),
                )
            )
        else:
            raise ValueError(f"unknown judge {name!r}; available: {', '.join(JUDGE_NAMES)}")
    return judges


def _write_reports(report_dir: str, stem: str, report) -> Path:
    report_dir_p = Path(report_dir)
    report_dir_p.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = report_dir_p / f"{stem}_{stamp}"
    base.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    base.with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
    return base


def cmd_run(args: argparse.Namespace) -> int:
    cases = load_dataset(args.suite)
    config = load_judge_config(args.judge_config)
    client = load_callable(args.client, "--client") if args.client else None
    judges = build_judges(args.judges, client, config)
    target = load_callable(args.target, "--target")

    report = run_eval(
        cases, target, judges,
        threshold=args.threshold, suite=Path(args.suite).stem,
    )
    base = _write_reports(args.report_dir, f"eval_report_{report.suite}", report)
    print(report.to_markdown())
    print(f"reports written to {base}.json / {base}.md")
    return 0 if report.passed else 1


def cmd_calibrate(args: argparse.Namespace) -> int:
    cases = load_labeled(args.labeled)
    config = load_judge_config(args.judge_config)
    client = load_callable(args.client, "--client")
    judge = LLMJudge(
        client,
        prompt_version=config.get("prompt_version", "v1"),
        rubric=config.get("rubric", DEFAULT_RUBRIC),
        pass_threshold=config.get("pass_threshold", 0.5),
        max_retries=config.get("max_retries", 1),
    )
    report = calibrate(cases, judge)
    base = _write_reports(args.report_dir, "calibration_llm_judge", report)
    print(report.to_markdown())
    print(f"reports written to {base}.json / {base}.md")
    return 0


def _judge_list(value: str) -> list[str]:
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in JUDGE_NAMES]
    if unknown:
        raise ValueError(f"unknown judges: {', '.join(unknown)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="litmus", description="offline evals for LLM pipelines")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run a golden suite against a target")
    run.add_argument("--suite", required=True, help="path to a golden dataset (.jsonl)")
    run.add_argument("--target", required=True,
                     help="'module:attr' or 'path/to/file.py:attr' returning Callable[[str], str]")
    run.add_argument("--judges", default="exact_match",
                     help=f"comma-separated; available: {', '.join(JUDGE_NAMES)}")
    run.add_argument("--client",
                     help="LLM client spec ('module:attr' or 'path/to/file.py:attr'); "
                          "required when llm_judge is in --judges")
    run.add_argument("--judge-config", default=None,
                     help="JSON file with llm_judge options "
                          "(prompt_version, rubric, pass_threshold, max_retries)")
    run.add_argument("--threshold", type=float, default=1.0,
                     help="minimum overall mean score to pass (default 1.0)")
    run.add_argument("--report-dir", default="reports",
                     help="where to write eval reports (default ./reports)")

    cal = sub.add_parser("calibrate", help="vet a judge against human labels")
    cal.add_argument("--labeled", required=True, help="path to a labeled dataset (.jsonl)")
    cal.add_argument("--client", required=True,
                     help="LLM client spec ('module:attr' or 'path/to/file.py:attr')")
    cal.add_argument("--judge-config", default=None,
                     help="JSON file with llm_judge options "
                          "(prompt_version, rubric, pass_threshold, max_retries)")
    cal.add_argument("--report-dir", default="reports",
                     help="where to write calibration reports (default ./reports)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            args.judges = _judge_list(args.judges)
            return cmd_run(args)
        if args.cmd == "calibrate":
            return cmd_calibrate(args)
    except (ValueError, OSError) as e:
        print(f"litmus: {e}", file=sys.stderr)
        return 2
    return 2  # unreachable


if __name__ == "__main__":
    sys.exit(main())
