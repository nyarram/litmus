"""litmus: offline regression evals for agents and LLM pipelines."""

from litmus.calibration import (
    CalibrationReport,
    LabeledCase,
    LabeledDatasetError,
    calibrate,
    load_labeled,
)
from litmus.dataset import DatasetError, GoldenCase, load_dataset
from litmus.judges import ContainmentJudge, ExactMatchJudge, Judge, Score
from litmus.judges_llm import DEFAULT_RUBRIC, LLMJudge, VerdictParseError
from litmus.runner import CaseResult, EvalReport, run_eval

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_RUBRIC",
    "CalibrationReport",
    "CaseResult",
    "ContainmentJudge",
    "DatasetError",
    "EvalReport",
    "ExactMatchJudge",
    "GoldenCase",
    "Judge",
    "LLMJudge",
    "LabeledCase",
    "LabeledDatasetError",
    "Score",
    "VerdictParseError",
    "__version__",
    "calibrate",
    "load_dataset",
    "load_labeled",
    "run_eval",
]
