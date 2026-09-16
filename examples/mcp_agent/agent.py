"""Minimal ReAct-style MCP agent: litmus's second consumer.

The agent is a plain ``async`` callable — ``agent_system`` — so it plugs
straight into ``litmus run`` like any other system under test::

    litmus run --dataset examples/mcp_agent/tasks.jsonl \\
        --system examples.mcp_agent.agent:agent_system \\
        --scorer examples.mcp_agent.scorers:tool_scorer \\
        --scorer litmus.scorers.exact:ExactMatchScorer

Provider selection via ``LITMUS_AGENT_PROVIDER``: ``stub`` (default —
scripted responses for the demo tasks, no API keys), ``groq`` (needs
``GROQ_API_KEY``), or ``ollama`` (needs a local Ollama server).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from examples.mcp_agent.scorers import TOOL_CALL_REGISTRY
from litmus import tracing

SYSTEM_PROMPT = """You are a helpful assistant with access to tools. When you need
a tool, output exactly one fenced block and nothing else:

```tool
{"name": "<tool_name>", "arguments": {...}}
```

Available tools:
- calculator(expression: string) — evaluate arithmetic,
  e.g. {"expression": "17 * 23"}
- weather_lookup(city: string) — demo weather data,
  e.g. {"city": "Minneapolis"}
- summarize(text: string, max_sentences: integer) — extractive summary

When you have the final answer, output it as plain text with no tool block."""

TOOL_RE = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a fenced tool block; None when the model gave a final answer."""
    m = TOOL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    return {"name": call["name"], "arguments": call.get("arguments", {})}


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0


async def call_tool(session: Any, name: str, arguments: dict) -> str:
    """Call an MCP tool; tool errors come back as text the agent can see."""
    with tracing.span("mcp.tool", attributes={"tool.name": name}):
        try:
            result = await session.call_tool(name, arguments)
        except Exception as e:  # unknown tool, bad args, server died, ...
            return f"tool error: {e}"
    parts = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) or "(empty tool result)"


async def run_agent(task: str, provider: Any, session: Any, max_steps: int = 6) -> AgentResult:
    """ReAct loop: prompt -> tool block? -> execute -> repeat -> final answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    result = AgentResult(answer="")
    for _ in range(max_steps):
        response = (await provider.complete(messages)).strip()
        result.steps += 1
        call = parse_tool_call(response)
        if call is None:
            result.answer = response
            return result
        observation = await call_tool(session, call["name"], call["arguments"])
        result.tool_calls.append(
            {"name": call["name"], "arguments": call["arguments"], "result": observation}
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Tool result:\n{observation}"})
    result.answer = "(agent hit max steps without a final answer)"
    return result


async def run_agent_with_server(task: str, provider: Any, max_steps: int = 6) -> AgentResult:
    """Spawn the demo MCP server over stdio and run the agent against it."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "examples.mcp_agent.server"],
        # The SDK only inherits a small allowlist of env vars by default
        # (no PYTHONPATH), so pass the full environment through.
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await run_agent(task, provider, session, max_steps=max_steps)


class StubProvider:
    """Scripted provider for the demo tasks: no API keys, fully deterministic.

    ``scripts`` maps task text -> the assistant messages to return in order.
    """

    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = scripts
        self._counts: dict[str, int] = {}

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        task = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        # the user message may be a tool-result follow-up; find the original task
        if task.startswith("Tool result:"):
            task = next(m["content"] for m in messages if m["role"] == "user")
        script = self.scripts.get(task)
        if not script:
            return f"(stub has no script for: {task[:60]})"
        i = self._counts.get(task, 0)
        self._counts[task] = i + 1
        return script[min(i, len(script) - 1)]


def _tool_block(name: str, arguments: dict) -> str:
    return "```tool\n" + json.dumps({"name": name, "arguments": arguments}) + "\n```"


_SUMMARIZE_TASK = (
    "Summarize in one sentence: The James Webb Space Telescope launched in "
    "December 2021. It observes the universe in infrared light. "
    "Its primary mirror is 6.5 meters across."
)
_SUMMARIZE_TEXT = (
    "The James Webb Space Telescope launched in December 2021. "
    "It observes the universe in infrared light. "
    "Its primary mirror is 6.5 meters across."
)


STUB_SCRIPTS: dict[str, list[str]] = {
    "What is 17 * 23?": [
        _tool_block("calculator", {"expression": "17 * 23"}),
        "391",
    ],
    "What is the weather in Minneapolis?": [
        _tool_block("weather_lookup", {"city": "Minneapolis"}),
        "It is sunny, 72F in Minneapolis.",
    ],
    "What is 12 * 12? Also, what is the weather in Duluth?": [
        _tool_block("calculator", {"expression": "12 * 12"}),
        _tool_block("weather_lookup", {"city": "Duluth"}),
        "144, and it is cloudy, 58F in Duluth.",
    ],
    _SUMMARIZE_TASK: [
        _tool_block(
            "summarize",
            {"text": _SUMMARIZE_TEXT, "max_sentences": 1},
        ),
        "The James Webb Space Telescope launched in December 2021.",
    ],
    "What is the capital of France?": ["Paris"],
}


def resolve_provider() -> Any:
    """Pick the agent's LLM from ``LITMUS_AGENT_PROVIDER``."""
    from litmus.scorers.llm_judge import GroqProvider, OllamaProvider

    which = os.environ.get("LITMUS_AGENT_PROVIDER", "stub").lower()
    if which == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("LITMUS_AGENT_PROVIDER=groq needs GROQ_API_KEY")
        return GroqProvider(api_key=api_key)
    if which == "ollama":
        return OllamaProvider()
    if which == "stub":
        return StubProvider(STUB_SCRIPTS)
    raise RuntimeError(f"unknown LITMUS_AGENT_PROVIDER={which!r} (stub|groq|ollama)")


async def agent_system(case: Any) -> str:
    """CLI entry point: run the MCP agent on a case, return its final answer.

    Tool calls are recorded in ``TOOL_CALL_REGISTRY[case.id]`` for the
    ``ToolSequenceScorer`` to check.
    """
    provider = resolve_provider()
    result = await run_agent_with_server(case.input, provider)
    TOOL_CALL_REGISTRY[case.id] = result.tool_calls
    return result.answer


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "What is 17 * 23?"
    res = asyncio.run(run_agent_with_server(task, StubProvider(STUB_SCRIPTS)))
    print("answer:", res.answer)
    print("tools:", [c["name"] for c in res.tool_calls])
