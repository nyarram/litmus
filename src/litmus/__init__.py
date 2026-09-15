"""Litmus: a free, open-source evaluation and observability harness for LLM pipelines and agents."""

__version__ = "0.1.0"

from litmus.calibration import CalibrationReport, calibrate
from litmus.datasets import Case, Dataset
from litmus.report import EvalReport, ReportDiff, build_report, compare_reports
from litmus.runners import CaseResult, EvalRunResult, Runner
from litmus.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    GroqProvider,
    JSONSchemaScorer,
    JudgeRubric,
    LLMJudge,
    ModelProvider,
    OllamaProvider,
    RegexScorer,
    Score,
    Scorer,
)
from litmus.tracing import init_tracing

__all__ = [
    "CalibrationReport",
    "Case",
    "CaseResult",
    "ContainsScorer",
    "Dataset",
    "EvalReport",
    "EvalRunResult",
    "ExactMatchScorer",
    "GroqProvider",
    "JSONSchemaScorer",
    "JudgeRubric",
    "LLMJudge",
    "ModelProvider",
    "OllamaProvider",
    "RegexScorer",
    "ReportDiff",
    "Runner",
    "Score",
    "Scorer",
    "build_report",
    "calibrate",
    "compare_reports",
    "init_tracing",
]
