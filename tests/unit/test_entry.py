# tests/unit/test_entry.py
from rho.api.entry import build_prov_from_resume, run_from_structured
from rho.models.resume import StructuredResume, WorkExperience
from rho.models.jd import RequirementSet, Requirement
from rho.models.scoring import MatchResult, ComponentVector


def _resume():
    return StructuredResume(
        name="Jane Doe",
        skills=["Python", "AWS"],
        work=[WorkExperience(company="Acme", title="Engineer", bullets=["Built pipelines in Python"])],
    )


def test_build_prov_covers_every_value():
    prov = build_prov_from_resume(_resume())
    joined = " ".join(s.raw_text for s in prov.spans.values()).lower()
    assert "python" in joined and "acme" in joined and "jane doe" in joined


def test_run_from_structured_enters_at_match_and_returns_full_response(monkeypatch):
    # Stub the two LLM legs so no network is touched.
    reqs = RequirementSet(requirements=[Requirement(text="Python", kind="skill", priority="must")])
    monkeypatch.setattr("rho.api.entry.analyze_jd", lambda jd, _schema_fn=None: reqs)

    from rho.models.rewrite import TailoredResume, FabricationReport
    def fake_rewrite(resume, gaps, prov, _rewrite_fn=None):
        return TailoredResume(resume=resume, fabrication_report=FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0))
    monkeypatch.setattr("rho.api.entry.rewrite", fake_rewrite)

    resp = run_from_structured(_resume(), "We need a Python engineer.")
    assert resp.structured_resume.name == "Jane Doe"
    assert isinstance(resp.match_result, MatchResult)
    assert resp.tailored_resume.fabrication_report.total_edits == 0
    assert 0.0 <= resp.final_score <= 100.0
