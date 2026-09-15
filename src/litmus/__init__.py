"""Litmus: a free, open-source evaluation and observability harness for LLM pipelines and agents."""

__version__ = "0.2.0"

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
from litmus.synth import (
    Generator,
    Intent,
    Persona,
    SyntheticRequest,
    SynthReport,
    default_quality_judge,
    load_personas,
    run_synth,
)
from litmus.tracing import (
    TracingNotInitialized,
    case_span,
    ensure_initialized,
    get_finished_spans,
    init_console_tracing,
    init_tracing,
    set_span_io,
    shutdown_tracing,
    span,
    start_llm_span,
)

__all__ = [
    "CalibrationReport",
    "Case",
    "CaseResult",
    "ContainsScorer",
    "Dataset",
    "EvalReport",
    "EvalRunResult",
    "ExactMatchScorer",
    "Generator",
    "GroqProvider",
    "Intent",
    "JSONSchemaScorer",
    "JudgeRubric",
    "LLMJudge",
    "ModelProvider",
    "OllamaProvider",
    "Persona",
    "RegexScorer",
    "ReportDiff",
    "Runner",
    "Score",
    "Scorer",
    "SynthReport",
    "SyntheticRequest",
    "TracingNotInitialized",
    "build_report",
    "calibrate",
    "case_span",
    "compare_reports",
    "default_quality_judge",
    "ensure_initialized",
    "get_finished_spans",
    "init_console_tracing",
    "init_tracing",
    "load_personas",
    "run_synth",
    "set_span_io",
    "shutdown_tracing",
    "span",
    "start_llm_span",
]
