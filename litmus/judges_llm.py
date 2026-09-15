"""LLM-as-judge: model-graded scoring behind the Judge protocol.

The judge never imports a provider SDK. It takes a ``client`` — any
``Callable[[str], str]`` mapping a prompt to raw model text — so provider
choice, model name, temperature, retries, and auth all live in the caller's
client factory (see ``examples/stub_client.py`` for the pattern). The judge
owns only what is provider-independent: the prompt template, verdict parsing,
and score normalization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from litmus.judges import Judge, Score

DEFAULT_RUBRIC = (
    "Correctness relative to the reference answer. Ignore style, formatting, "
    "and verbosity; penalize hallucinations and contradictions."
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class VerdictParseError(ValueError):
    """Raised when model output cannot be parsed into a verdict."""


def load_prompt_template(version: str) -> str:
    path = PROMPTS_DIR / f"judge_{version}.txt"
    if not path.is_file():
        known = sorted(p.stem.replace("judge_", "") for p in PROMPTS_DIR.glob("judge_*.txt"))
        raise ValueError(f"unknown prompt version {version!r}; known: {known}")
    return path.read_text(encoding="utf-8")


def extract_verdict(text: str) -> dict:
    """Pull the verdict JSON out of raw model output.

    Tolerates markdown code fences and surrounding chatter; requires one
    top-level JSON object with a numeric ``score``.
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        lines = lines[1:]  # drop opening fence (``` or ```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise VerdictParseError("no JSON object found in model output")
    try:
        verdict = json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise VerdictParseError(f"malformed verdict JSON: {e}") from e
    if not isinstance(verdict, dict):
        raise VerdictParseError("verdict must be a JSON object")
    if "score" not in verdict:
        raise VerdictParseError("verdict missing required field 'score'")
    try:
        float(verdict["score"])
    except (TypeError, ValueError) as e:
        raise VerdictParseError(f"verdict 'score' is not numeric: {e}") from e
    return verdict


class LLMJudge:
    """Model-graded judge implementing the Judge protocol."""

    name = "llm_judge"

    def __init__(
        self,
        client: Callable[[str], str],
        *,
        prompt_version: str = "v1",
        rubric: str = DEFAULT_RUBRIC,
        pass_threshold: float = 0.5,
        max_retries: int = 1,
    ) -> None:
        if not callable(client):
            raise ValueError("client must be a callable taking a prompt string")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        self.client = client
        self.prompt_version = prompt_version
        self.template = load_prompt_template(prompt_version)
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.max_retries = max_retries

    def render_prompt(self, prediction: str, reference: str) -> str:
        return self.template.format(
            prediction=prediction, reference=reference, rubric=self.rubric
        )

    def score(self, prediction: str, reference: str) -> Score:
        prompt = self.render_prompt(prediction, reference)
        raw: str | None = None
        verdict: dict | None = None
        attempts = 0
        for attempts in range(self.max_retries + 1):
            raw = self.client(prompt)  # client errors propagate: fail loudly
            try:
                verdict = extract_verdict(raw)
                break
            except VerdictParseError:
                verdict = None
        if verdict is None:
            return Score(
                judge=self.name,
                score=0.0,
                passed=False,
                details={
                    "parse_error": True,
                    "prompt_version": self.prompt_version,
                    "raw_excerpt": (raw or "")[:200],
                },
            )
        score = min(1.0, max(0.0, float(verdict["score"])))
        return Score(
            judge=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            details={
                "rationale": str(verdict.get("rationale", "")),
                "flags": list(verdict.get("flags", [])),
                "prompt_version": self.prompt_version,
                "retries": attempts,
            },
        )
