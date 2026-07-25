from rho.extraction import extract
from rho.extraction.provenance_attach import attach_provenance, find_prov
from rho.extraction.schema import ExtractionSchema, to_structured
from rho.models.provenance import ProvenanceMap, SourceSpan
from rho.models.resume import StructuredResume, WorkExperience


def test_schema_maps_to_structured_resume():
    es = ExtractionSchema(
        reasoning="found name+skill",
        name="Alice",
        skills=["python"],
        work=[
            {
                "company": "Acme",
                "title": "Eng",
                "start_date": "2019",
                "end_date": "2022",
                "bullets": ["Built API"],
            }
        ],
    )
    sr = to_structured(es)
    assert sr.name == "Alice"
    assert sr.skills == ["python"]
    assert sr.work[0].company == "Acme"
    # prov fields exist but empty until attach step
    assert sr.name_prov == []


def _pm():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=13, raw_text="Alice Johnson"))
    pm.add(
        SourceSpan(
            doc_id="d", char_start=14, char_end=45, raw_text="Skills: Python, FastAPI, AWS"
        )
    )
    pm.add(
        SourceSpan(
            doc_id="d", char_start=46, char_end=70, raw_text="Acme Corp Backend Engineer"
        )
    )
    return pm


def test_find_prov_exact_and_fuzzy():
    pm = _pm()
    assert find_prov("Alice Johnson", pm) == ["p:d:0"]
    assert find_prov("Python", pm) == ["p:d:1"]  # substring of skills line


def test_attach_sets_prov_fields():
    pm = _pm()
    r = StructuredResume(
        name="Alice Johnson",
        skills=["Python", "AWS"],
        work=[WorkExperience(company="Acme Corp", title="Backend Engineer")],
    )
    r2 = attach_provenance(r, pm)
    assert r2.name_prov == ["p:d:0"]
    assert r2.skills_prov[0] == ["p:d:1"]
    assert "p:d:2" in r2.work[0].company_prov


def test_attach_leaves_empty_when_no_support():
    pm = _pm()
    r = StructuredResume(name="Nonexistent Person")
    r2 = attach_provenance(r, pm)
    assert r2.name_prov == []  # never invent provenance


def test_extract_end_to_end_with_fake_llm():
    pm = _pm()
    md = "Alice Johnson\nSkills: Python, FastAPI, AWS\nAcme Corp Backend Engineer"
    fake = lambda m: ExtractionSchema(
        reasoning="x",
        name="Alice Johnson",
        skills=["Python", "AWS"],
        work=[{"company": "Acme Corp", "title": "Backend Engineer"}],
    )
    r = extract(md, pm, _schema_fn=fake)
    assert r.name == "Alice Johnson"
    assert r.name_prov == ["p:d:0"]
    assert r.work[0].company_prov == ["p:d:2"]
