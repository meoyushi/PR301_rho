"""Corpus-backed fabrication benchmark (Phase 5, option 2).

`Resume.csv` stores each résumé as a single line with multi-space runs where the
original had line breaks, so the plain ingest path yields one span covering the
whole document. That is technically provenance but useless as a gate: a single
5000-char span "supports" nearly anything by substring match, and C1 claims an
*exact* source location. These tests pin the segmentation that restores real
per-line spans.
"""

from eval.fabrication_corpus import corpus_prov, segment_corpus_text


def test_segmentation_splits_on_multi_space_runs():
    text = "Alice Johnson    Skills: Python, SQL    Acme Corp - Engineer 2019-2022"
    assert segment_corpus_text(text).splitlines() == [
        "Alice Johnson",
        "Skills: Python, SQL",
        "Acme Corp - Engineer 2019-2022",
    ]


def test_segmentation_preserves_single_spaces():
    """Only run-of-space breaks are structural; normal word spacing must survive."""
    assert segment_corpus_text("Senior Data Engineer    Acme Corp") == (
        "Senior Data Engineer\nAcme Corp"
    )


def test_corpus_prov_yields_one_span_per_segment():
    text = "Alice Johnson    Skills: Python, SQL    Acme Corp - Engineer"
    prov = corpus_prov(text, "r1")
    raws = [s.raw_text for s in prov.spans.values()]
    assert raws == ["Alice Johnson", "Skills: Python, SQL", "Acme Corp - Engineer"]


def test_corpus_prov_offsets_point_into_segmented_text():
    """A prov_id must locate its value, not just assert it exists somewhere."""
    text = "Alice Johnson    Skills: Python, SQL"
    segmented = segment_corpus_text(text)
    prov = corpus_prov(text, "r1")
    for span in prov.spans.values():
        assert segmented[span.char_start : span.char_end] == span.raw_text


def test_corpus_prov_beats_single_span_specificity():
    """Segmented provenance localises a value; the unsegmented blob does not."""
    text = "Alice Johnson    Skills: Python, SQL    Acme Corp - Engineer"
    prov = corpus_prov(text, "r1")
    from rho.extraction.provenance_attach import find_prov

    hits = find_prov("Python", prov)
    assert len(hits) == 1
    assert prov.get(hits[0]).raw_text == "Skills: Python, SQL"
