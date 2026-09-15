"""Litmus CLI: `litmus run` executes an eval and optionally gates on quality."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import typer

from litmus.datasets import Dataset
from litmus.report import build_report
from litmus.runners import Runner
from litmus.scorers.base import Scorer
from litmus.tracing import init_console_tracing

app = typer.Typer(help="Litmus: eval harness for LLM pipelines and agents.")


def _import_attr(path: str) -> Any:
    """Import a ``module:attr`` reference (attr may be dotted)."""
    if ":" not in path:
        raise typer.BadParameter(f"Expected 'module:attr', got {path!r}")
    module_name, attr = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_scorers(paths: list[str]) -> list[Scorer]:
    scorers: list[Scorer] = []
    for p in paths:
        obj = _import_attr(p)
        if callable(obj):
            try:
                obj = obj()
            except TypeError:
                pass
        if isinstance(obj, (list, tuple)):
            scorers.extend(obj)
        else:
            scorers.append(obj)
    for s in scorers:
        if not (hasattr(s, "name") and hasattr(s, "score")):
            raise typer.BadParameter(f"{s!r} is not a Scorer (needs .name and async .score)")
    return scorers


@app.command()
def run(
    dataset: Path = typer.Option(..., "--dataset", help="Dataset .jsonl file"),
    system: str = typer.Option(
        ..., "--system", help="System under test as 'module:attr' (async callable)"
    ),
    scorer: list[str] = typer.Option([], "--scorer", help="Scorer as 'module:attr' (repeatable)"),
    name: str = typer.Option("eval", "--name", help="Run name for the report"),
    concurrency: int = typer.Option(8, "--concurrency"),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-case timeout in seconds"),
    retries: int = typer.Option(1, "--retries"),
    report_md: Path | None = typer.Option(None, "--report-md", help="Write markdown report here"),
    report_json: Path | None = typer.Option(None, "--report-json", help="Write JSON report here"),
    fail_below: float | None = typer.Option(
        None, "--fail-below", help="Exit 1 if GATE_SCORER mean drops below this"
    ),
    gate_scorer: str | None = typer.Option(
        None, "--gate-scorer", help="Scorer the quality gate watches"
    ),
    trace: bool = typer.Option(False, "--trace", help="Print OTel spans to stdout"),
):
    """Run a dataset against a system and print a score report."""
    if trace:
        init_console_tracing()
    # Without --trace, spans are no-ops and tracing stays uninitialized.

    ds = Dataset.from_jsonl(dataset)
    system_fn = _import_attr(system)
    if not callable(system_fn):
        raise typer.BadParameter(f"--system {system!r} is not callable")
    scorers = _resolve_scorers(scorer)

    typer.echo(f"Running {len(ds)} cases from {ds.name} v{ds.version} against {system} ...")
    runner = Runner(
        system_fn, scorers, max_concurrency=concurrency, timeout_s=timeout, retries=retries
    )
    run_result = asyncio.run(runner.run(ds))
    report = build_report(run_result, run_name=name)

    typer.echo("")
    typer.echo(report.to_markdown())
    if report_json:
        report_json.write_text(report.to_json())
        typer.echo(f"Wrote JSON report → {report_json}")
    if report_md:
        report_md.write_text(report.to_markdown())
        typer.echo(f"Wrote markdown report → {report_md}")

    if fail_below is not None:
        if not gate_scorer or gate_scorer not in report.summaries:
            typer.echo(f"Quality gate misconfigured: unknown scorer {gate_scorer!r}", err=True)
            raise typer.Exit(code=2)
        mean = report.summaries[gate_scorer].mean
        if mean < fail_below:
            typer.echo(
                f"QUALITY GATE FAILED: {gate_scorer} mean {mean:.3f} < {fail_below}", err=True
            )
            raise typer.Exit(code=1)
        typer.echo(f"Quality gate passed: {gate_scorer} mean {mean:.3f} >= {fail_below}")


@app.command()
def diff(
    baseline: Path = typer.Option(..., "--baseline", help="Baseline report JSON file"),
    candidate: Path = typer.Option(..., "--candidate", help="Candidate report JSON file"),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Fail if any scorer mean drops more than this vs the baseline",
    ),
):
    """Diff two eval reports; exit 1 on regression, 2 on bad inputs."""
    import json

    from litmus.report import EvalReport, compare_reports

    for label, path in (("baseline", baseline), ("candidate", candidate)):
        if not path.exists():
            typer.echo(f"error: {label} report not found: {path}", err=True)
            raise typer.Exit(code=2)
    base = EvalReport.from_dict(json.loads(baseline.read_text()))
    cand = EvalReport.from_dict(json.loads(candidate.read_text()))
    report_diff = compare_reports(base, cand)
    typer.echo(report_diff.to_markdown())
    if report_diff.has_regression(threshold):
        typer.echo("REGRESSION DETECTED vs baseline", err=True)
        raise typer.Exit(code=1)
    typer.echo("No regression vs baseline.")


@app.command()
def version():
    """Print the litmus version."""
    from litmus import __version__

    typer.echo(f"litmus {__version__}")


@app.command()
def synth(
    personas: Path = typer.Option(..., "--personas", help="Personas JSON file"),
    provider: str = typer.Option(
        ..., "--provider", help="ModelProvider as 'module:attr' (async); used to write requests"
    ),
    system: str = typer.Option(
        ..., "--system", help="System under test as 'module:attr' (async callable)"
    ),
    scorer: list[str] = typer.Option(
        [], "--scorer", help="Scorer as 'module:attr' (repeatable); default: quality rubric judge"
    ),
    n: int = typer.Option(20, "--n", help="Number of synthetic requests"),
    seed: int = typer.Option(0, "--seed", help="Seed for reproducible generation"),
    concurrency: int = typer.Option(8, "--concurrency"),
    timeout: float = typer.Option(120.0, "--timeout", help="Per-request timeout in seconds"),
    retries: int = typer.Option(1, "--retries"),
    report_md: Path | None = typer.Option(None, "--report-md", help="Write markdown report here"),
    report_json: Path | None = typer.Option(None, "--report-json", help="Write JSON report here"),
    fail_below: float | None = typer.Option(
        None, "--fail-below", help="Exit 1 if the lowest scorer mean drops below this"
    ),
    trace: bool = typer.Option(False, "--trace", help="Print OTel spans to stdout"),
):
    """Generate synthetic persona traffic, load-test a system, and score it.

    Every span is labeled litmus.synthetic=true so synthetic traffic is
    trivially filterable — and never confused with organic traffic.
    """
    import asyncio

    from litmus import tracing
    from litmus.synth import default_quality_judge, load_personas, run_synth

    if trace:
        tracing.init_console_tracing()

    persona_list = load_personas(personas)
    provider_obj = _import_attr(provider)
    if isinstance(provider_obj, type):
        provider_obj = provider_obj()
    system_fn = _import_attr(system)
    if not callable(system_fn):
        raise typer.BadParameter(f"--system {system!r} is not callable")
    scorers = _resolve_scorers(scorer) or [default_quality_judge(provider_obj)]

    typer.echo(f"Generating {n} synthetic requests (seed {seed}) ...")
    report = asyncio.run(
        run_synth(
            system_fn,
            persona_list,
            provider_obj,
            n=n,
            seed=seed,
            scorers=scorers,
            concurrency=concurrency,
            timeout_s=timeout,
            retries=retries,
            trace=trace,
        )
    )
    typer.echo("")
    typer.echo(report.to_markdown())
    if report_json:
        report_json.write_text(report.to_json())
        typer.echo(f"Wrote JSON report → {report_json}")
    if report_md:
        report_md.write_text(report.to_markdown())
        typer.echo(f"Wrote markdown report → {report_md}")

    if fail_below is not None:
        means = [s.mean for s in report.eval_report.summaries.values()]
        worst = min(means) if means else 1.0
        if report.error_rate > 0 or worst < fail_below:
            typer.echo(
                f"SYNTH GATE FAILED: error_rate={report.error_rate:.1%}, "
                f"worst scorer mean {worst:.3f} < {fail_below}",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"Synth gate passed: worst scorer mean {worst:.3f} >= {fail_below}")


if __name__ == "__main__":
    app()
