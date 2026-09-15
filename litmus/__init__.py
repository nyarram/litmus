"""litmus: offline regression evals for agents and LLM pipelines."""

from litmus.dataset import DatasetError, GoldenCase, load_dataset
from litmus.judges import ContainmentJudge, ExactMatchJudge, Judge, LLMJudge, Score
from litmus.runner import CaseResult, EvalReport, run_eval

__version__ = "0.1.0"

__all__ = [
    "CaseResult",
    "ContainmentJudge",
    "DatasetError",
    "EvalReport",
    "ExactMatchJudge",
    "GoldenCase",
    "Judge",
    "LLMJudge",
    "Score",
    "__version__",
    "load_dataset",
    "run_eval",
]
