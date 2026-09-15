"""Tests for the MCP demo agent (litmus's second consumer)."""

import pytest

pytest.importorskip("mcp")

from examples.mcp_agent.agent import (
    STUB_SCRIPTS,
    StubProvider,
    parse_tool_call,
    run_agent,
    run_agent_with_server,
)
from examples.mcp_agent.scorers import TOOL_CALL_REGISTRY, ToolSequenceScorer
from examples.mcp_agent.server import calculator, summarize, weather_lookup
from litmus.datasets import Case


def test_parse_tool_call():
    call = parse_tool_call(
        '```tool\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```'
    )
    assert call == {"name": "calculator", "arguments": {"expression": "2+2"}}
    assert parse_tool_call("just an answer") is None
    assert parse_tool_call("```tool\nnot json\n```") is None


def test_server_tools_are_deterministic():
    assert calculator("17 * 23") == "391"
    assert calculator("10 / 4") == "2.5"
    assert calculator("__import__('os')").startswith("error:")
    assert weather_lookup("Minneapolis") == "sunny, 72F in Minneapolis"
    assert weather_lookup("Nowhere") == "no weather data for Nowhere"
    assert summarize("First. Second. Third.", max_sentences=1) == "First."


class FakeSession:
    """Stand-in MCP session: no subprocess, canned tool results."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "calculator":
            return _text(calculator(arguments["expression"]))
        raise RuntimeError(f"unknown tool {name}")


class _Text:
    def __init__(self, text):
        self.text = text


def _text(s):
    return type("R", (), {"content": [_Text(s)]})()


async def test_react_loop_with_fake_session():
    provider = StubProvider(STUB_SCRIPTS)
    result = await run_agent("What is 17 * 23?", provider, FakeSession())
    assert result.answer == "391"
    assert [c["name"] for c in result.tool_calls] == ["calculator"]
    assert result.steps == 2


async def test_tool_error_reaches_agent():
    class Boom:
        async def complete(self, messages, *, json_mode=False):
            return '```tool\n{"name": "nope", "arguments": {}}\n```'

    session = FakeSession()
    result = await run_agent("What is 17 * 23?", Boom(), session, max_steps=1)
    assert result.tool_calls[0]["result"].startswith("tool error:")
    assert "max steps" in result.answer


async def test_agent_over_real_stdio_server():
    provider = StubProvider(STUB_SCRIPTS)
    result = await run_agent_with_server("What is the weather in Minneapolis?", provider)
    assert result.answer == "It is sunny, 72F in Minneapolis."
    assert [c["name"] for c in result.tool_calls] == ["weather_lookup"]


async def test_tool_sequence_scorer():
    scorer = ToolSequenceScorer()
    case = Case(id="c1", input="x", expected="y", metadata={"tools": ["calculator"]})
    TOOL_CALL_REGISTRY["c1"] = [{"name": "calculator", "arguments": {}, "result": "1"}]
    score = await scorer.score(case, "y")
    assert score.passed is True and score.value == 1.0

    TOOL_CALL_REGISTRY["c1"] = []
    score = await scorer.score(case, "y")
    assert score.passed is False and "expected tools" in score.explanation


async def test_agent_system_registers_tool_calls():
    from examples.mcp_agent.agent import agent_system

    case = Case(id="smoke-1", input="What is 17 * 23?", expected="391")
    answer = await agent_system(case)
    assert answer == "391"
    assert [c["name"] for c in TOOL_CALL_REGISTRY["smoke-1"]] == ["calculator"]
