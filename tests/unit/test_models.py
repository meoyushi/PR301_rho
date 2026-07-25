from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.models.resume import StructuredResume, WorkExperience
from rho.models.jd import RequirementSet, Requirement
from rho.models.scoring import MatchResult, ComponentVector, Gap
from rho.models.rewrite import TailoredResume, FabricationReport
from rho.models.api import PipelineResponse


def test_provmap_add_and_get():
    pm = ProvenanceMap(doc_id="d1")
    pid = pm.add(SourceSpan(doc_id="d1", char_start=0, char_end=5, raw_text="Alice"))
    assert pid == "p:d1:0"
    assert pm.get(pid).raw_text == "Alice"
    pid2 = pm.add(SourceSpan(doc_id="d1", char_start=6, char_end=9, raw_text="Bob"))
    assert pid2 == "p:d1:1"


def test_models_construct_with_provenance_fields():
    w = WorkExperience(company="Acme", title="Eng", company_prov=["p:d1:0"])
    r = StructuredResume(name="Alice", work=[w], skills=["python"], skills_prov=[["p:d1:1"]])
    assert r.work[0].company_prov == ["p:d1:0"]
    fr = FabricationReport(total_edits=2, verified_edits=2, fabrication_rate=0.0)
    assert fr.fabrication_rate == 0.0
    cv = ComponentVector(keyword_coverage=0.5, semantic_similarity=0.5,
                         fuzzy_coverage=0.5, must_have_coverage=0.5, nice_have_coverage=0.5)
    mr = MatchResult(component_vector=cv, predicted_score=0.0)
    assert mr.predicted_score == 0.0
