"""Unit tests for the evaluation metrics (Phase 7, Task 1).

Metrics are scored on tiny hand-computable inputs so a regression shows up as a
wrong number, not a wrong-looking number: every expected value here is derivable
by hand from the definition, which is what makes the paper's tables auditable.
"""

from eval.metrics import field_f1, long_text_f1, provenance_accuracy
from rho.models.provenance import ProvenanceMap, SourceSpan
from rho.models.resume import Education, StructuredResume, WorkExperience


def test_field_f1_on_skills():
    m = field_f1({"skills": ["python", "aws", "sql"]}, {"skills": ["python", "aws", "gcp"]}, "skills")
    assert round(m["precision"], 2) == 0.67 and round(m["recall"], 2) == 0.67
    assert round(m["f1"], 2) == 0.67


def test_field_f1_is_case_insensitive():
    """Extraction casing varies with the source document; the claim does not."""
    m = field_f1({"skills": ["Python", "AWS"]}, {"skills": ["python", "aws"]}, "skills")
    assert m["f1"] == 1.0


def test_field_f1_empty_prediction_scores_zero_not_crash():
    """A dropped section must score 0, never divide-by-zero into a fake pass."""
    m = field_f1({"skills": []}, {"skills": ["python"]}, "skills")
    assert m == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_field_f1_both_empty_is_perfect():
    """Nothing to find and nothing claimed is a correct extraction, not a miss."""
    m = field_f1({"skills": []}, {"skills": []}, "skills")
    assert m["f1"] == 1.0


def test_field_f1_aligns_dict_entities_by_key_subset():
    """Work/education are lists of dicts; alignment compares the labelled keys only."""
    pred = {"work": [{"company": "Acme Corp", "title": "Data Engineer", "start_date": "2020"}]}
    gold = {"work": [{"company": "acme corp", "title": "data engineer"}]}
    m = field_f1(pred, gold, "work", keys=("company", "title"))
    assert m["f1"] == 1.0


def test_long_text_f1_token_overlap():
    f = long_text_f1("built scalable python api", "built python api service")
    assert 0.0 < f < 1.0


def test_long_text_f1_identical_is_one():
    assert long_text_f1("built python api", "built python api") == 1.0


def test_long_text_f1_missing_field_scores_zero():
    """`summary` is often None; a None-vs-text comparison is a miss, not a crash."""
    assert long_text_f1("", "built python api") == 0.0


def test_long_text_f1_both_missing_is_perfect():
    assert long_text_f1("", "") == 1.0


def _prov_with(*texts: str) -> tuple[ProvenanceMap, list[str]]:
    """A ProvenanceMap holding one span per text, plus the ids in order."""
    prov = ProvenanceMap(doc_id="d")
    ids = [
        prov.add(SourceSpan(doc_id="d", char_start=i * 100, char_end=i * 100 + len(t), raw_text=t))
        for i, t in enumerate(texts)
    ]
    return prov, ids


def test_provenance_accuracy_all_correct():
    """Every attached prov_id points at the span the gold map names."""
    prov, ids = _prov_with("Python", "Acme Corp")
    resume = StructuredResume(name="X", skills=["Python"], skills_prov=[[ids[0]]])
    gold = {"skills[0]": ids[0]}
    assert provenance_accuracy(resume, gold, prov) == 1.0


def test_provenance_accuracy_counts_wrong_span_as_miss():
    """Pointing at *a* span is not enough — it must be the right one."""
    prov, ids = _prov_with("Python", "Acme Corp")
    resume = StructuredResume(name="X", skills=["Python"], skills_prov=[[ids[1]]])
    gold = {"skills[0]": ids[0]}
    assert provenance_accuracy(resume, gold, prov) == 0.0


def test_provenance_accuracy_counts_unattached_field_as_miss():
    """A field with no prov at all is a broken chain (C1), so it scores 0."""
    prov, ids = _prov_with("Python")
    resume = StructuredResume(name="X", skills=["Python"], skills_prov=[[]])
    gold = {"skills[0]": ids[0]}
    assert provenance_accuracy(resume, gold, prov) == 0.0


def test_provenance_accuracy_scores_work_and_education_paths():
    """The field paths match `hard_content_tokens`, so C1 is measured everywhere."""
    prov, ids = _prov_with("Acme Corp", "Data Engineer", "State University")
    resume = StructuredResume(
        name="X",
        work=[WorkExperience(company="Acme Corp", company_prov=[ids[0]], title="Data Engineer", title_prov=[ids[1]])],
        education=[Education(institution="State University", institution_prov=[ids[2]])],
    )
    gold = {
        "work[0].company": ids[0],
        "work[0].title": ids[1],
        "education[0].institution": ids[2],
    }
    assert provenance_accuracy(resume, gold, prov) == 1.0


def test_provenance_accuracy_partial():
    """Two of three correct scores 2/3 — the fraction, not a pass/fail."""
    prov, ids = _prov_with("Python", "AWS", "Acme Corp")
    resume = StructuredResume(
        name="X",
        skills=["Python", "AWS"],
        skills_prov=[[ids[0]], [ids[2]]],
        work=[WorkExperience(company="Acme Corp", company_prov=[ids[2]], title="T")],
    )
    gold = {"skills[0]": ids[0], "skills[1]": ids[1], "work[0].company": ids[2]}
    assert round(provenance_accuracy(resume, gold, prov), 2) == 0.67


def test_provenance_accuracy_empty_gold_is_undefined_not_one():
    """No gold spans means the metric has nothing to say; 0.0 with no crash."""
    prov, _ = _prov_with("Python")
    assert provenance_accuracy(StructuredResume(name="X"), {}, prov) == 0.0


def test_provenance_accuracy_accepts_span_offsets_as_gold():
    """Gold may name a char range instead of a prov_id; the span it resolves to counts."""
    prov, ids = _prov_with("Python")
    resume = StructuredResume(name="X", skills=["Python"], skills_prov=[[ids[0]]])
    # (char_start, char_end) of the span the value should trace to.
    assert provenance_accuracy(resume, {"skills[0]": (0, 6)}, prov) == 1.0
