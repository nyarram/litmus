"""Demo MCP server with deterministic tools.

Run it directly (stdio transport)::

    python -m examples.mcp_agent.server

The agent in ``agent.py`` spawns this automatically; you never need to run
it by hand.
"""

from __future__ import annotations

import ast
import asyncio
import operator as op
import re

from mcp.server.mcpserver import MCPServer

server = MCPServer("litmus-demo-tools")

# ---- calculator: safe arithmetic only (no eval) ----

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


@server.tool()
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '17 * 23'."""
    try:
        value = _eval(ast.parse(expression.strip(), mode="eval"))
    except Exception as e:
        return f"error: {e}"
    return str(int(value)) if float(value).is_integer() else str(value)


# ---- weather_lookup: canned answers (deterministic by design) ----

_CANNED = {
    "minneapolis": "sunny, 72F",
    "duluth": "cloudy, 58F",
    "san francisco": "foggy, 64F",
}


@server.tool()
def weather_lookup(city: str) -> str:
    """Look up the current weather for a city (demo data)."""
    key = city.strip().lower()
    if key in _CANNED:
        return f"{_CANNED[key]} in {city.strip()}"
    return f"no weather data for {city.strip()}"


# ---- summarize: extractive, no LLM involved ----


@server.tool()
def summarize(text: str, max_sentences: int = 2) -> str:
    """Summarize text by returning its first N sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[: max(1, max_sentences)])


def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
