"""Synthetic persona-based traffic generation and load testing.

Generates realistic-looking requests from weighted personas, fires them at a
system under test through the standard async ``Runner`` (concurrency,
timeouts, retries), and scores the responses. Every span is labeled
``litmus.synthetic: true`` so synthetic traffic is trivially filterable —
and never confused with organic traffic — in any trace backend.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litmus import tracing
from litmus.datasets import Case, Dataset
from litmus.report import EvalReport, build_report
from litmus.runners import EvalRunResult, Runner
from litmus.scorers.base import Scorer
from litmus.scorers.llm_judge import JudgeRubric, LLMJudge, ModelProvider

SYNTHETIC_ATTR = "litmus.synthetic"

SystemFn = Callable[[str], Awaitable[Any]]


@dataclass
class Intent:
    name: str
    description: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> Intent:
        return cls(
            name=d["name"],
            description=d["description"],
            weight=float(d.get("weight", 1.0)),
        )


@dataclass
class Persona:
    name: str
    description: str
    intents: list[Intent]

    @classmethod
    def from_dict(cls, d: dict) -> Persona:
        return cls(
            name=d["name"],
            description=d["description"],
            intents=[Intent.from_dict(i) for i in d["intents"]],
        )


@dataclass
class SyntheticRequest:
    id: str
    persona: str
    intent: str
    text: str


def load_personas(path: str | Path) -> list[Persona]:
    """Load personas from JSON: ``{"personas": [{name, description, intents}]}``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    personas = [Persona.from_dict(p) for p in data["personas"]]
    if not personas:
        raise ValueError(f"no personas defined in {path}")
    for p in personas:
        if not p.intents:
            raise ValueError(f"persona {p.name!r} has no intents")
        if any(i.weight < 0 for i in p.intents):
            raise ValueError(f"persona {p.name!r} has a negative intent weight")
    return personas


class Generator:
    """Turns personas into concrete request strings via a model provider.

    Deterministic under a fixed seed when the provider is deterministic:
    persona and intent sampling both draw from ``random.Random(seed)``.
    """

    def __init__(self, personas: list[Persona], provider: ModelProvider):
        if not personas:
            raise ValueError("at least one persona is required")
        self.personas = personas
        self.provider = provider

    @staticmethod
    def _prompt(persona: Persona, intent: Intent) -> str:
        return (
            "You are simulating a user for load-testing an AI assistant. "
            "Write ONE realistic user request, and nothing else.\n\n"
            f"Persona: {persona.name} — {persona.description}\n"
            f"Request type: {intent.name} — {intent.description}\n\n"
            "Reply with only the request text, no quotes or commentary."
        )

    async def generate(self, n: int, seed: int = 0) -> list[SyntheticRequest]:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = random.Random(seed)
        requests = []
        with tracing.span(
            "litmus.synthetic.batch",
            {
                SYNTHETIC_ATTR: True,
                "litmus.request.count": n,
                "litmus.seed": seed,
                "litmus.personas": ",".join(p.name for p in self.personas),
            },
        ):
            for i in range(n):
                persona = rng.choice(self.personas)
                intent = _weighted_choice(rng, persona.intents)
                text = await self.provider.complete(
                    [{"role": "user", "content": self._prompt(persona, intent)}]
                )
                requests.append(
                    SyntheticRequest(
                        id=f"synth-{i + 1:04d}",
                        persona=persona.name,
                        intent=intent.name,
                        text=text.strip(),
                    )
                )
        return requests


def _weighted_choice(rng: random.Random, intents: list[Intent]) -> Intent:
    total = sum(i.weight for i in intents)
    if total <= 0:
        raise ValueError("intent weights sum to zero")
    r = rng.uniform(0, total)
    for intent in intents:
        r -= intent.weight
        if r <= 0:
            return intent
    return intents[-1]


def default_quality_judge(provider: ModelProvider) -> LLMJudge:
    """Reference-free quality judge used when ``run_synth`` gets no scorers."""
    return LLMJudge(
        JudgeRubric(
            name="synth_quality",
            criteria=(
                "Rate the assistant's response on overall quality: helpfulness, "
                "correctness, clarity, and safety. There is no reference answer; "
                "judge the response on its own merits against the user's request."
            ),
            min_score=1,
            max_score=5,
            pass_threshold=4,
        ),
        provider,
    )


@dataclass
class SynthReport:
    eval_report: EvalReport
    n_requests: int
    seed: int
    error_rate: float
    latency_p50: float | None
    latency_p95: float | None
    per_persona: dict[str, dict[str, Any]]
    passed: bool
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        return {
            "n_requests": self.n_requests,
            "seed": self.seed,
            "synthetic": True,
            "error_rate": round(self.error_rate, 4),
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "per_persona": self.per_persona,
            "passed": self.passed,
            "created_at": self.created_at,
            "eval": self.eval_report.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        lines = [
            "# litmus synthetic traffic report",
            "",
            f"- requests: {self.n_requests} (seed {self.seed}) — "
            "**SYNTHETIC traffic, not organic**",
            f"- error rate: {self.error_rate:.1%}",
        ]
        if self.latency_p50 is not None:
            lines.append(f"- latency p50/p95: {self.latency_p50:.2f}s / {self.latency_p95:.2f}s")
        lines += ["", "## per persona"]
        for persona, stats in sorted(self.per_persona.items()):
            lines.append(
                f"- {persona}: n={stats['n']}, errors={stats['errors']}, p50={stats['p50']:.2f}s"
            )
        lines += ["", self.eval_report.to_markdown()]
        lines.append(f"**{'PASS' if self.passed else 'FAIL'}**")
        return "\n".join(lines) + "\n"


async def run_synth(
    system: SystemFn,
    personas: list[Persona],
    provider: ModelProvider,
    *,
    n: int = 20,
    seed: int = 0,
    scorers: list[Scorer] | None = None,
    concurrency: int = 8,
    timeout_s: float = 120.0,
    retries: int = 1,
    threshold: float = 0.7,
    trace: bool = False,
) -> SynthReport:
    """Generate synthetic traffic and load-test ``system`` with it.

    Generation uses ``provider`` to write natural request text; scoring uses
    ``scorers`` (default: a reference-free quality rubric judge on the same
    provider). With ``trace=True`` every request gets a span labeled
    ``litmus.synthetic: true`` (requires ``init_tracing()`` first).
    """
    if trace:
        tracing.ensure_initialized()

    requests = await Generator(personas, provider).generate(n, seed=seed)
    by_id = {r.id: r for r in requests}
    dataset = Dataset(
        name="synthetic",
        version=f"seed-{seed}",
        cases=[
            Case(
                id=r.id,
                input=r.text,
                expected=None,
                metadata={"synthetic": True, "persona": r.persona, "intent": r.intent},
            )
            for r in requests
        ],
    )

    async def system_fn(case: Case) -> Any:
        return await system(case.input)

    runner = Runner(
        system_fn,
        scorers if scorers is not None else [default_quality_judge(provider)],
        max_concurrency=concurrency,
        timeout_s=timeout_s,
        retries=retries,
        extra_span_attributes=lambda case: {
            SYNTHETIC_ATTR: True,
            "litmus.persona": case.metadata.get("persona", ""),
            "litmus.intent": case.metadata.get("intent", ""),
        },
    )
    run_result: EvalRunResult = await runner.run(dataset)
    return summarize(run_result, by_id, seed=seed, threshold=threshold)


def summarize(
    run_result: EvalRunResult,
    by_id: dict[str, SyntheticRequest],
    *,
    seed: int,
    threshold: float = 0.7,
) -> SynthReport:
    """Aggregate a synthetic run. Pass = zero errors and every scorer mean >= threshold."""
    ok = [r for r in run_result.results if r.error is None]
    latencies = sorted(r.latency_s for r in ok)

    def pct(p: float) -> float | None:
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, int(p * len(latencies)))]

    per_persona: dict[str, dict[str, Any]] = {}
    for r in run_result.results:
        req = by_id.get(r.case_id)
        persona = req.persona if req else "unknown"
        stats = per_persona.setdefault(persona, {"n": 0, "errors": 0, "latencies": []})
        stats["n"] += 1
        if r.error:
            stats["errors"] += 1
        else:
            stats["latencies"].append(r.latency_s)
    for stats in per_persona.values():
        lats = sorted(stats.pop("latencies"))
        stats["p50"] = statistics.median(lats) if lats else 0.0

    eval_report = build_report(run_result, run_name=f"synth-seed-{seed}")
    means = [s.mean for s in eval_report.summaries.values()]
    error_rate = (len(run_result.results) - len(ok)) / len(run_result.results)
    passed = error_rate == 0 and all(m >= threshold for m in means)
    return SynthReport(
        eval_report=eval_report,
        n_requests=len(run_result.results),
        seed=seed,
        error_rate=error_rate,
        latency_p50=pct(0.50),
        latency_p95=pct(0.95),
        per_persona=per_persona,
        passed=passed,
    )
