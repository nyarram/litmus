"""Tests for the JSON Schema scorer."""

from litmus import Case, JSONSchemaScorer

SCHEMA = {
    "type": "object",
    "required": ["newsworthy", "score"],
    "properties": {"newsworthy": {"type": "boolean"}, "score": {"type": "number"}},
}


def _case():
    return Case(id="c1", input="x")


async def test_valid_dict_passes():
    s = JSONSchemaScorer(schema=SCHEMA)
    assert (await s.score(_case(), {"newsworthy": True, "score": 0.9})).passed is True


async def test_json_string_is_parsed():
    s = JSONSchemaScorer(schema=SCHEMA)
    assert (await s.score(_case(), '{"newsworthy": false, "score": 0.1}')).passed is True


async def test_wrong_type_fails():
    s = JSONSchemaScorer(schema=SCHEMA)
    score = await s.score(_case(), {"newsworthy": "yes", "score": 0.9})
    assert score.passed is False
    assert "Schema violation" in (score.explanation or "")


async def test_missing_field_fails():
    s = JSONSchemaScorer(schema=SCHEMA)
    assert (await s.score(_case(), {"newsworthy": True})).passed is False


async def test_invalid_json_string_fails():
    s = JSONSchemaScorer(schema=SCHEMA)
    score = await s.score(_case(), "{not json")
    assert score.passed is False
    assert "not valid JSON" in (score.explanation or "")
