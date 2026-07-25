from pathlib import Path

import pytest

from rho.ingestion import ingest, structural_check
from rho.models.provenance import ProvenanceMap, SourceSpan

FIX = Path(__file__).parent.parent / "fixtures"


def test_text_ingest_offsets_map_back():
    data = (FIX / "clean.txt").read_bytes()
    md, pm = ingest(data, "clean.txt")
    assert "Alice Johnson" in md
    assert len(pm.spans) >= 4
    # every span's raw_text equals the markdown slice at its offsets
    for span in pm.spans.values():
        assert md[span.char_start:span.char_end] == span.raw_text


def test_docling_ingest_produces_markdown_and_spans():
    docx = FIX / "simple.docx"
    if not docx.exists():
        pytest.skip("fixture simple.docx missing")
    md, pm = ingest(docx.read_bytes(), "simple.docx")
    assert isinstance(md, str) and len(md) > 0
    assert len(pm.spans) >= 1
    for span in pm.spans.values():
        # offsets are valid indices into md
        assert 0 <= span.char_start <= span.char_end <= len(md)
        # and they resolve back to the exact source text
        assert md[span.char_start:span.char_end] == span.raw_text


def test_docling_repeated_lines_get_distinct_offsets():
    """A line appearing twice must map to two different locations, not collapse
    onto the first occurrence — otherwise the provenance chain points at the
    wrong source (breaks C1)."""
    docx = FIX / "simple.docx"
    if not docx.exists():
        pytest.skip("fixture simple.docx missing")
    md, pm = ingest(docx.read_bytes(), "simple.docx")
    dupe = "Built REST APIs serving 2M requests per day"
    starts = [s.char_start for s in pm.spans.values() if s.raw_text == dupe]
    assert len(starts) == 2
    assert starts[0] != starts[1]


def test_structural_check_true_when_no_geometry():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=3, raw_text="abc"))
    assert structural_check("abc", pm) is True


def test_structural_check_flags_backward_jumps():
    pm = ProvenanceMap(doc_id="d")
    # y decreasing then jumping = out-of-order columns
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=1, page=1, bbox=(0, 700, 1, 710), raw_text="a"))
    pm.add(SourceSpan(doc_id="d", char_start=1, char_end=2, page=1, bbox=(0, 100, 1, 110), raw_text="b"))
    pm.add(SourceSpan(doc_id="d", char_start=2, char_end=3, page=1, bbox=(300, 690, 301, 700), raw_text="c"))
    assert structural_check("abc", pm) is False
