"""Loader tests (Phase 7, Task 2).

These guard the properties the eval harness depends on: loaders are
deterministic, they return the documented shapes, and — most importantly — a
missing or unreadable dataset raises instead of silently yielding an empty list.
A silent empty dataset produces a RESULTS.md full of 0.000 that reads like a
finished run, which is the failure mode Phase 5 called out explicitly.
"""

import pytest

from eval.datasets import (
    load_fabrication_pairs,
    load_gold,
    load_public_gold,
    load_real_gold,
)
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume


def test_load_gold_returns_path_and_gold_json():
    items = load_gold()
    assert items, "synthetic gold set is empty — regenerate with eval.datasets.synthetic"
    path, gold = items[0]
    assert path.exists()
    assert {"skills", "work", "education"} <= set(gold)


def test_load_gold_is_deterministic_in_order():
    assert [p.name for p, _ in load_gold()] == [p.name for p, _ in load_gold()]


def test_load_gold_carries_provenance_labels():
    """C1 needs gold spans; without them provenance accuracy is unmeasurable."""
    _, gold = load_gold()[0]
    assert gold["gold_prov_values"], "gold record has no provenance labels"


def test_load_gold_covers_both_document_formats():
    """Both ingest paths (text adapter and Docling) must be exercised."""
    suffixes = {p.suffix for p, _ in load_gold()}
    assert {".txt", ".docx"} <= suffixes


def test_load_gold_limit_truncates():
    assert len(load_gold(limit=5)) == 5


def test_load_gold_raises_when_dataset_missing(tmp_path):
    """An absent dataset is an error, never an empty run."""
    with pytest.raises(FileNotFoundError):
        load_gold(data_dir=tmp_path / "does-not-exist")


def test_load_real_gold_returns_labelled_items_only():
    items = load_real_gold()
    assert len(items) == 30
    for path, gold in items:
        assert path.exists()
        assert "skills" in gold and "work_titles" in gold


def test_load_real_gold_skips_readme_key():
    """`_README` documents the schema; it is not a résumé."""
    assert all(not p.name.startswith("_") for p, _ in load_real_gold())


def test_load_fabrication_pairs_shape():
    pairs = load_fabrication_pairs()
    assert pairs
    first = pairs[0]
    assert isinstance(first["resume"], StructuredResume)
    assert isinstance(first["prov"], ProvenanceMap)
    assert "gaps" in first


def test_load_public_gold_returns_text_and_labels():
    """The public set is the independently-annotated one (Table 1c)."""
    items = load_public_gold()
    assert len(items) == 150
    text, gold = items[0]
    assert isinstance(text, str) and len(text) > 500
    assert gold["skills"], "public gold record has no skill labels"


def test_load_public_gold_dedups_repeat_mentions():
    """Skills are annotated at every mention; the metric wants the claim set."""
    for _, gold in load_public_gold(limit=25):
        lowered = [s.lower() for s in gold["skills"]]
        assert len(lowered) == len(set(lowered))
