"""Tests for the deterministic string scorers."""

from litmus import Case, ContainsScorer, ExactMatchScorer, RegexScorer


def _case(expected):
    return Case(id="c1", input="x", expected=expected)


async def test_exact_match_basic():
    s = ExactMatchScorer()
    assert (await s.score(_case("hello"), "hello")).passed is True
    assert (await s.score(_case("hello"), "goodbye")).passed is False


async def test_exact_match_case_insensitive_and_strip():
    s = ExactMatchScorer()
    assert (await s.score(_case("Hello"), "  hello ")).passed is True


async def test_exact_match_field_pluck():
    s = ExactMatchScorer(field="newsworthy")
    assert (await s.score(_case({"newsworthy": True}), {"newsworthy": True})).passed is True
    assert (await s.score(_case({"newsworthy": True}), {"newsworthy": False})).passed is False


async def test_contains():
    s = ContainsScorer()
    assert (await s.score(_case("world"), "hello world")).passed is True
    assert (await s.score(_case("world"), "hello")).passed is False


async def test_regex():
    s = RegexScorer(pattern=r"^\d{4}-\d{2}-\d{2}$")
    assert (await s.score(_case(None), "2026-09-15")).passed is True
    assert (await s.score(_case(None), "Sept 15")).passed is False
