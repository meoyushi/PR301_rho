"""Gemini-backed grounded rewriting (`gemini-3.1-flash-lite`).

Same grounded prompt as `rho.rewrite.llm`/`rho.rewrite.groq`, same temperature
(0.6), different backend — via Gemini's `responseSchema` constrained decoding.

The prompt is still not the safety mechanism. Truthfulness is enforced downstream
by `rho.rewrite.verifier`, which is deterministic and LLM-free.
"""

from rho.extraction.schema import (
    EduItem,
    ExtractionSchema,
    ProjectItem,
    WorkItem,
    to_structured,
)
from rho.llm.gemini import GeminiClient
from rho.models.resume import StructuredResume
from rho.models.scoring import Gap

_PROMPT = """You tailor a résumé toward a job's requirements. Rules:
- The MASTER RÉSUMÉ below is the ONLY source of truth.
- You MAY reorder and rephrase existing content, and emphasize the parts most
  relevant to the target requirements.
- Return EVERY work entry and EVERY project from the master résumé, and for each
  one KEEP ALL of its bullets — rewritten and re-emphasized, but never dropped.
  A résumé that comes back with fewer bullets than it went in with is WRONG.
  Do not delete a bullet because it seems less relevant; rephrase it instead.
- Keep every skill from the master résumé. You may reorder skills so the ones the
  job asks for come first, but do not remove any.
- You MAY add a skill to the skills list ONLY when the master résumé already
  demonstrates it in a work bullet, a project bullet, or a project's tech list —
  a tool used in the work but missing from the skills list is under-reported, not
  new. Copy the term as the résumé spells it. If it is not already written
  somewhere in the master résumé, it is an invention and must NOT be added.
- You MUST NOT invent skills, tools, employers, titles, metrics, dates, or
  certifications. If the résumé does not claim it, it does not go in.
- If a target requirement cannot be satisfied truthfully, leave it unsatisfied.
  An honest gap is the correct output; a fabricated match is a failure.
- Keep the person's name, employers, titles, and dates exactly as given.
- Fill the `reasoning` field first, briefly, then the data fields.
{gaps_block}
MASTER RÉSUMÉ (JSON):
---
{resume_json}
---"""

_GAPS_HEADER = """
TARGET REQUIREMENTS (emphasize existing evidence for these; never invent):
{gaps}
"""

_REWRITE_TEMPERATURE = 0.6

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {"type": "STRING"},
        "name": {"type": "STRING"},
        "headline": {"type": "STRING", "nullable": True},
        "summary": {"type": "STRING", "nullable": True},
        "emails": {"type": "ARRAY", "items": {"type": "STRING"}},
        "phones": {"type": "ARRAY", "items": {"type": "STRING"}},
        "urls": {"type": "ARRAY", "items": {"type": "STRING"}},
        "work": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "company": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "start_date": {"type": "STRING", "nullable": True},
                    "end_date": {"type": "STRING", "nullable": True},
                    "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                # bullets required so the decoder must emit them; optional, the
                # model drops the work history's bullets to "select" relevance.
                "required": ["company", "title", "bullets"],
            },
        },
        "education": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "institution": {"type": "STRING"},
                    "degree": {"type": "STRING", "nullable": True},
                    "field": {"type": "STRING", "nullable": True},
                    "end_year": {"type": "STRING", "nullable": True},
                },
                "required": ["institution"],
            },
        },
        "projects": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "url": {"type": "STRING", "nullable": True},
                    "tech": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "bullets": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["name"],
            },
        },
        "skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "certifications": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["reasoning", "name", "work", "education", "projects", "skills"],
}

_client: GeminiClient | None = None


def _default_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def _source_json(resume: StructuredResume) -> str:
    """The résumé as the model sees it — values only, no prov chain noise."""
    return resume.model_dump_json(
        indent=2,
        exclude={
            "name_prov": True,
            "contact_prov": True,
            "skills_prov": True,
            "work": {
                "__all__": {
                    "company_prov": True,
                    "title_prov": True,
                    "date_prov": True,
                    "bullet_prov": True,
                }
            },
            "education": {"__all__": {"institution_prov": True, "edu_prov": True}},
            "projects": {
                "__all__": {
                    "name_prov": True,
                    "url_prov": True,
                    "tech_prov": True,
                    "bullet_prov": True,
                }
            },
        },
    )


def _gaps_block(gaps: list[Gap]) -> str:
    targets = [g.requirement.text for g in gaps if g.status != "present"]
    if not targets:
        return ""
    return _GAPS_HEADER.format(gaps="\n".join(f"- {t}" for t in targets))


def _coerce(data: dict) -> ExtractionSchema:
    """Validate item-by-item so one malformed entry cannot lose the whole résumé."""
    work = []
    for item in data.get("work") or []:
        try:
            work.append(WorkItem(**item))
        except Exception:
            continue  # No silent fills.
    education = []
    for item in data.get("education") or []:
        try:
            education.append(EduItem(**item))
        except Exception:
            continue
    projects = []
    for item in data.get("projects") or []:
        try:
            projects.append(ProjectItem(**item))
        except Exception:
            continue
    as_str = lambda xs: [str(x) for x in (xs or []) if str(x).strip()]  # noqa: E731
    return ExtractionSchema(
        reasoning=data.get("reasoning") or "",
        name=str(data.get("name") or ""),
        headline=data.get("headline"),
        summary=data.get("summary"),
        emails=as_str(data.get("emails")),
        phones=as_str(data.get("phones")),
        urls=as_str(data.get("urls")),
        work=work,
        education=education,
        projects=projects,
        skills=as_str(data.get("skills")),
        certifications=as_str(data.get("certifications")),
    )


def rewrite_schema_gemini(
    resume: StructuredResume, gaps: list[Gap], client: GeminiClient | None = None
) -> StructuredResume:
    """Tailor `resume` toward `gaps` via Gemini. Output is UNVERIFIED — gate it."""
    client = client or _default_client()
    prompt = _PROMPT.format(
        gaps_block=_gaps_block(gaps), resume_json=_source_json(resume)
    )
    data = client.complete_json(
        prompt, response_schema=_SCHEMA, temperature=_REWRITE_TEMPERATURE
    )
    tailored = to_structured(_coerce(data))
    # Achievements (awards/honors) are facts, not prose to tailor. The rewriter
    # never sees them, so carry them through verbatim rather than let the
    # reconstruction drop them to [].
    tailored.achievements = list(resume.achievements)
    tailored.achievements_prov = [list(p) for p in resume.achievements_prov]
    return tailored
