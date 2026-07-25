from rho.graph.review import check_provenance_invariant, compute_final_score
from rho.models.provenance import ProvenanceMap, SourceSpan
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport
from rho.models.scoring import ComponentVector, MatchResult


def _pm():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    return pm


def test_invariant_passes_when_all_sourced():
    ok, viol = check_provenance_invariant(
        StructuredResume(name="A", skills=["Python"]), _pm()
    )
    assert ok and viol == []


def test_invariant_fails_on_unsourced_token():
    ok, viol = check_provenance_invariant(
        StructuredResume(name="A", skills=["Rust"]), _pm()
    )
    assert not ok and "Rust" in viol


def test_final_score_penalized_by_fabrication():
    cv = ComponentVector(
        keyword_coverage=1,
        semantic_similarity=1,
        fuzzy_coverage=1,
        must_have_coverage=1,
        nice_have_coverage=1,
    )
    mr = MatchResult(component_vector=cv, predicted_score=80.0)
    clean = FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)
    assert compute_final_score(mr, clean) == 80.0
