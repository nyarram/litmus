"""LLM-as-judge scorer with pluggable, free inference providers.

The judge is the least trustworthy scorer by default, which is why litmus
pairs it with calibration (see litmus.calibration): never ship a judge you
haven't measured against human labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from litmus.datasets import Case
from litmus.scorers.base import Score


@dataclass
class JudgeRubric:
    """What the judge grades and where the pass/fail line is."""

    name: str
    criteria: str
    min_score: int = 1
    max_score: int = 5
    pass_threshold: int = 4


class ModelProvider(Protocol):
    """Any chat-completion backend the judge can call."""

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str: ...


class GroqProvider:
    """Groq's OpenAI-compatible API (free tier). Model default is an open-weights model."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 60.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


class OllamaProvider:
    """Fully local inference via Ollama — $0 and no data leaves the machine."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json()["message"]["content"]


_JUDGE_PROMPT = """You are an impartial judge evaluating an AI system's output.

## Rubric: {rubric_name}
{criteria}

Rate the output on an integer scale from {min_score} to {max_score} (inclusive).

## Input given to the system
{case_input}

## Reference (expected) answer
{expected}

## System's actual output
{actual_output}

Respond with ONLY valid JSON: {{"score": <int>, "explanation": "<1-2 sentences>"}}"""


class LLMJudge:
    """Grades output against a rubric using an LLM. Score is normalized to 0..1."""

    def __init__(self, rubric: JudgeRubric, provider: ModelProvider, name: str | None = None):
        self.rubric = rubric
        self.provider = provider
        self.name = name or f"judge_{rubric.name}"

    def _prompt(self, case: Case, output: Any) -> str:
        def fmt(v: Any) -> str:
            return v if isinstance(v, str) else json.dumps(v, indent=2, default=str)

        return _JUDGE_PROMPT.format(
            rubric_name=self.rubric.name,
            criteria=self.rubric.criteria,
            min_score=self.rubric.min_score,
            max_score=self.rubric.max_score,
            case_input=fmt(case.input),
            expected=fmt(case.expected),
            actual_output=fmt(output),
        )

    async def score(self, case: Case, output: Any) -> Score:
        raw = await self.provider.complete(
            [{"role": "user", "content": self._prompt(case, output)}], json_mode=True
        )
        try:
            parsed = json.loads(raw)
            value = int(parsed["score"])
            explanation = str(parsed.get("explanation", ""))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return Score(
                name=self.name,
                value=0.0,
                passed=False,
                explanation="Judge returned unparseable output",
                details={"raw": raw[:500]},
            )
        value = max(self.rubric.min_score, min(self.rubric.max_score, value))
        span = self.rubric.max_score - self.rubric.min_score
        normalized = (value - self.rubric.min_score) / span if span else 1.0
        passed = value >= self.rubric.pass_threshold
        return Score(
            name=self.name,
            value=normalized,
            passed=passed,
            explanation=explanation,
            details={"raw_score": value, "threshold": self.rubric.pass_threshold},
        )
