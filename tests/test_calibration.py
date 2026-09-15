"""Tests for judge calibration against human labels."""

import pytest

from litmus import calibrate


def test_perfect_agreement():
    report = calibrate(
        {"a": (True, "good"), "b": (False, "bad")},
        {"a": True, "b": False},
    )
    assert report.n == 2
    assert report.agreement_rate == 1.0
    assert report.disagreements == []


def test_partial_agreement_reports_disagreements():
    report = calibrate(
        {"a": (True, "x"), "b": (True, "y"), "c": (False, "z"), "d": (False, "w")},
        {"a": True, "b": False, "c": False, "d": True},
    )
    assert report.n == 4
    assert report.agreement_rate == 0.5
    assert {d.case_id for d in report.disagreements} == {"b", "d"}
    assert report.judge_pass_rate == 0.5
    assert report.human_pass_rate == 0.5


def test_no_overlap_raises():
    with pytest.raises(ValueError):
        calibrate({"a": (True, "x")}, {"b": True})


def test_only_shared_cases_count():
    report = calibrate({"a": (True, "x"), "b": (False, "y")}, {"a": True})
    assert report.n == 1
    assert report.agreement_rate == 1.0
