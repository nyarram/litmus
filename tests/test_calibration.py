"""Tests for calibration: agreement metrics against human labels."""
import json
from pathlib import Path

import pytest

from litmus.calibration import (
    LabeledCase,
    LabeledDatasetError,
    _pearson,
    calibrate,
    load_labeled,
)
from litmus.judges import ExactMatchJudge

EXAMPLES = Path(__file__).resolve().parents[1] / "datasets" / "examples" / "labeled.jsonl"


def perfect_judge_factory():
    class J:
        name = "perfect"

        def score(self, prediction, reference):
            from litmus.judges import Score

            return Score(judge="perfect", score=1.0, passed=True)

    return J()


def test_load_labeled_example():
    cases = load_labeled(EXAMPLES)
    assert len(cases) == 8
    assert cases[0].human_score == 1.0
    assert cases[3].prediction == "London"


def test_labeled_validation(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "a", "prediction": "p", "reference": "r"}\n', encoding="utf-8")
    with pytest.raises(LabeledDatasetError, match="human_score"):
        load_labeled(bad)

    out_of_range = tmp_path / "oor.jsonl"
    out_of_range.write_text(
        '{"id": "a", "prediction": "p", "reference": "r", "human_score": 1.5}\n',
        encoding="utf-8",
    )
    with pytest.raises(LabeledDatasetError, match="in \\[0, 1\\]"):
        load_labeled(out_of_range)


def test_pearson_perfect_correlation():
    assert _pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == pytest.approx(1.0)


def test_pearson_undefined_on_zero_variance():
    assert _pearson([0.5, 0.5, 0.5], [0.1, 0.9, 0.3]) is None
    assert _pearson([0.5], [0.5]) is None


def test_calibrate_perfect_judge_has_zero_mae():
    cases = [
        LabeledCase(id="a", prediction="x", reference="x", human_score=1.0),
        LabeledCase(id="b", prediction="y", reference="y", human_score=1.0),
    ]
    report = calibrate(cases, perfect_judge_factory())
    assert report.mae == 0.0
    assert report.bias == 0.0
    assert report.n == 2


def test_calibrate_reports_bias_direction():
    cases = [
        LabeledCase(id="a", prediction="x", reference="x", human_score=0.2),
        LabeledCase(id="b", prediction="y", reference="y", human_score=0.4),
    ]
    report = calibrate(cases, perfect_judge_factory())  # always predicts 1.0
    assert report.mae == pytest.approx(0.7)
    assert report.bias == pytest.approx(0.7)  # generous judge


def test_calibration_curve_bins():
    cases = [
        LabeledCase(id="a", prediction="x", reference="x", human_score=0.9),
        LabeledCase(id="b", prediction="y", reference="y", human_score=0.1),
    ]
    report = calibrate(cases, perfect_judge_factory())
    top = [b for b in report.curve if b["bin"] == [0.8, 1.0]][0]
    assert top["n"] == 2
    assert top["mean_predicted"] == 1.0
    assert top["mean_human"] == pytest.approx(0.5)
    empty = [b for b in report.curve if b["bin"] == [0.0, 0.2]][0]
    assert empty["n"] == 0 and empty["mean_human"] is None


def test_calibrate_serializes():
    cases = load_labeled(EXAMPLES)
    report = calibrate(cases, ExactMatchJudge())
    d = json.loads(report.to_json())
    assert d["n"] == 8 and d["judge"] == "exact_match"
    assert "MAE" in report.to_markdown()


def test_calibrate_rejects_empty():
    with pytest.raises(ValueError, match="no labeled cases"):
        calibrate([], ExactMatchJudge())
