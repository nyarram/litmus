"""Tests for the LLM-as-judge: parsing, retries, and provider independence."""
import json

import pytest

from litmus.judges_llm import (
    DEFAULT_RUBRIC,
    LLMJudge,
    VerdictParseError,
    extract_verdict,
    load_prompt_template,
)


def ok_client(prompt: str) -> str:
    return json.dumps({"score": 0.8, "rationale": "good", "flags": []})


def test_clean_verdict_parses():
    v = extract_verdict('{"score": 0.75, "rationale": "ok"}')
    assert v["score"] == 0.75


def test_fenced_verdict_parses():
    raw = '```json\n{"score": 0.5, "rationale": "meh"}\n```'
    assert extract_verdict(raw)["score"] == 0.5


def test_chatter_around_verdict_parses():
    raw = 'Here is my verdict:\n{"score": 0.2, "rationale": "bad"}\nHope this helps.'
    assert extract_verdict(raw)["score"] == 0.2


def test_garbage_raises_parse_error():
    with pytest.raises(VerdictParseError):
        extract_verdict("I refuse to answer in JSON.")


def test_missing_score_raises_parse_error():
    with pytest.raises(VerdictParseError, match="'score'"):
        extract_verdict('{"rationale": "no score here"}')


def test_judge_scores_through_client():
    s = LLMJudge(ok_client).score("Paris", "Paris")
    assert s.judge == "llm_judge"
    assert s.score == pytest.approx(0.8)
    assert s.passed  # 0.8 >= default pass_threshold 0.5
    assert s.details["rationale"] == "good"


def test_score_clamped_to_unit_interval():
    def generous(prompt: str) -> str:
        return json.dumps({"score": 99, "rationale": "x"})

    assert LLMJudge(generous).score("a", "b").score == 1.0


def test_pass_threshold_respected():
    s = LLMJudge(ok_client, pass_threshold=0.9).score("a", "b")
    assert not s.passed


def test_retry_then_success():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({"score": 0.6, "rationale": "recovered"})

    s = LLMJudge(flaky, max_retries=1).score("a", "b")
    assert s.score == pytest.approx(0.6)
    assert calls["n"] == 2
    assert s.details["retries"] == 1


def test_persistent_parse_failure_falls_back_to_zero():
    def broken(prompt: str) -> str:
        return "definitely not json"

    s = LLMJudge(broken, max_retries=1).score("a", "b")
    assert s.score == 0.0
    assert not s.passed
    assert s.details["parse_error"] is True


def test_client_errors_propagate_loudly():
    def down(prompt: str) -> str:
        raise RuntimeError("provider outage")

    with pytest.raises(RuntimeError, match="provider outage"):
        LLMJudge(down).score("a", "b")


def test_unknown_prompt_version_rejected():
    with pytest.raises(ValueError, match="unknown prompt version"):
        LLMJudge(ok_client, prompt_version="nope")


def test_prompt_template_renders_slots():
    judge = LLMJudge(ok_client)
    prompt = judge.render_prompt("pred", "ref")
    assert "pred" in prompt and "ref" in prompt
    assert DEFAULT_RUBRIC.split()[0] in prompt
    assert "{prediction}" not in prompt  # no unrendered slots


def test_load_prompt_template_v1():
    assert "Respond with ONLY a JSON object" in load_prompt_template("v1")
