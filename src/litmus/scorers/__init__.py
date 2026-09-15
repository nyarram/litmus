"""Scorer package: deterministic, structured, and LLM-judge scorers."""

from litmus.scorers.base import Score, Scorer
from litmus.scorers.exact import ContainsScorer, ExactMatchScorer, RegexScorer
from litmus.scorers.llm_judge import (
    GroqProvider,
    JudgeRubric,
    LLMJudge,
    ModelProvider,
    OllamaProvider,
)
from litmus.scorers.structured import JSONSchemaScorer

__all__ = [
    "ContainsScorer",
    "ExactMatchScorer",
    "GroqProvider",
    "JSONSchemaScorer",
    "JudgeRubric",
    "LLMJudge",
    "ModelProvider",
    "OllamaProvider",
    "RegexScorer",
    "Score",
    "Scorer",
]
