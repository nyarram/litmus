"""Example eval target: echoes the input back unchanged.

Intentionally trivial — it exists so the harness can be exercised end to end
without any model calls. A real target wraps the pipeline or agent under test.
"""


def echo(case_input: str) -> str:
    return case_input
