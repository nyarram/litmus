"""Tests for the deterministic judges."""

from litmus.judges import ContainmentJudge, ExactMatchJudge


def test_exact_match_pass():
    s = ExactMatchJudge().score("hello", "hello")
    assert s.score == 1.0 and s.passed


def test_exact_match_fail():
    s = ExactMatchJudge().score("hello", "goodbye")
    assert s.score == 0.0 and not s.passed


def test_exact_match_case_insensitive_option():
    s = ExactMatchJudge(case_sensitive=False).score("Hello", "hello")
    assert s.passed


def test_containment_pass():
    s = ContainmentJudge().score("the cat sat on the mat", "cat sat")
    assert s.passed


def test_containment_fail():
    s = ContainmentJudge().score("nothing relevant", "bonjour")
    assert not s.passed
