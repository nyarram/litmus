"""Scorers for the MCP demo agent.

``TOOL_CALL_REGISTRY`` is the side channel between the agent system and the
``ToolSequenceScorer``: ``agent_system`` records each case's tool calls under
``case.id``; the scorer reads them back. (Litmus scorers only see
``(case, output)``, so a registry is the honest way to grade behavior that
isn't in the final answer string.)
"""

from __future__ import annotations

from typing import Any

from litmus.scorers.base import Score

TOOL_CALL_REGISTRY: dict[str, list[dict[str, Any]]] = {}


class ToolSequenceScorer:
    """Did the agent call exactly the tools the task expected, in order?"""

    name = "tools_used"

    async def score(self, case: Any, output: Any) -> Score:
        expected: list[str] = list((case.metadata or {}).get("tools", []))
        actual: list[str] = [c["name"] for c in TOOL_CALL_REGISTRY.get(case.id, [])]
        passed = actual == expected
        return Score(
            name=self.name,
            value=1.0 if passed else 0.0,
            passed=passed,
            explanation=f"expected tools {expected}, agent used {actual}",
        )


tool_scorer = ToolSequenceScorer()
