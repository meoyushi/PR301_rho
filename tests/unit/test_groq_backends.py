"""Groq-backed rewrite and JD analysis, with the network stubbed out."""

import json

from rho.jd.groq import analyze_jd_schema_groq
from rho.llm.groq import GroqClient
from rho.models.resume import StructuredResume, WorkExperience
from rho.rewrite.groq import rewrite_schema_groq


def _client(payload: dict) -> GroqClient:
    def transport(url, headers, body, timeout):
        return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]})

    return GroqClient(api_keys=["k"], transport=transport)


def test_rewrite_groq_returns_structured_resume():
    client = _client(
        {
            "reasoning": "kept everything truthful",
            "name": "Dana Reed",
            "skills": ["Python", "SQL"],
            "work": [
                {
                    "company": "Acme",
                    "title": "Engineer",
                    "start_date": "2019",
                    "end_date": "2022",
                    "bullets": ["Built reporting tools"],
                }
            ],
            "education": [],
            "certifications": [],
        }
    )
    source = StructuredResume(
        name="Dana Reed",
        skills=["Python", "SQL"],
        work=[WorkExperience(company="Acme", title="Engineer")],
    )
    out = rewrite_schema_groq(source, gaps=[], client=client)
    assert out.name == "Dana Reed"
    assert out.skills == ["Python", "SQL"]
    assert out.work[0].bullets == ["Built reporting tools"]


def test_rewrite_groq_drops_malformed_work_entry():
    """No silent fills: an item missing required fields is dropped, not defaulted."""
    client = _client(
        {
            "reasoning": "",
            "name": "A",
            "skills": [],
            "work": [{"title": "Engineer"}],  # no company
            "education": [],
            "certifications": [],
        }
    )
    out = rewrite_schema_groq(StructuredResume(name="A"), gaps=[], client=client)
    assert out.work == []


def test_analyze_jd_groq_parses_requirements():
    client = _client(
        {
            "reasoning": "",
            "title": "Backend Engineer",
            "requirements": [
                {"text": "Python", "kind": "skill", "priority": "must", "years": None},
                {"text": "Kubernetes", "kind": "tool", "priority": "nice", "years": None},
            ],
        }
    )
    reqs = analyze_jd_schema_groq("some jd text", client=client)
    assert reqs.title == "Backend Engineer"
    assert [r.text for r in reqs.requirements] == ["Python", "Kubernetes"]


def test_analyze_jd_groq_drops_sentence_length_requirements():
    """Coverage matches literally, so a restated sentence can never match."""
    client = _client(
        {
            "reasoning": "",
            "title": None,
            "requirements": [
                {"text": "Python", "kind": "skill", "priority": "must", "years": None},
                {
                    "text": "must be authorized to work in the united states",
                    "kind": "experience",
                    "priority": "must",
                    "years": None,
                },
            ],
        }
    )
    reqs = analyze_jd_schema_groq("jd", client=client)
    assert [r.text for r in reqs.requirements] == ["Python"]
