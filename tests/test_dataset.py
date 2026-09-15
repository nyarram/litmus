"""Tests for the golden-dataset loader."""
from pathlib import Path

import pytest

from litmus.dataset import DatasetError, load_dataset

EXAMPLES = Path(__file__).resolve().parents[1] / "datasets" / "examples" / "basic.jsonl"


def test_load_example_dataset():
    cases = load_dataset(EXAMPLES)
    assert len(cases) == 4
    assert cases[0].id == "ex-001"
    assert cases[0].tags == ["smoke"]
    assert cases[1].reference == "the cat sat on the mat"


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")


def test_missing_required_field(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "x", "input": "y"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="missing required field 'reference'"):
        load_dataset(p)


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"id": "a", "input": "i", "reference": "r"}\n'
        '{"id": "a", "input": "i", "reference": "r"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_dataset(p)


def test_empty_dataset_rejected(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(p)
