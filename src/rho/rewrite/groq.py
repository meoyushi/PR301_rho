"""Groq-backed grounded rewriting (`qwen/qwen3.6-27b`).

Same grounded prompt as `rho.rewrite.llm`, same temperature (0.6), different
backend. Qwen3.6 on Groq does not support schema-constrained decoding, so the
output is validated by Pydantic afterwards rather than constrained during
generation — malformed entries are dropped, never defaulted.

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
from rho.llm.groq import GroqClient, shared_budget
from rho.models.resume import StructuredResume
from rho.models.scoring import Gap

_PROMPT = """You tailor a résumé toward a job's requirements. Rules:
- The MASTER RÉSUMÉ below is the ONLY source of truth.
- You MAY reorder and rephrase existing content, and emphasize the parts most
  relevant to the target requirements.
- Return EVERY work entry and project, and KEEP ALL of each one's bullets —
  rephrased and re-emphasized, but never dropped. Fewer bullets out than in is
  WRONG; rephrase a less-relevant bullet, do not delete it.
- Keep every skill; you may reorder so the job's terms come first.
- You MUST NOT invent skills, tools, employers, titles, metrics, dates, or
  certifications. If the résumé does not claim it, it does not go in.
- If a target requirement cannot be satisfied truthfully, leave it unsatisfied.
  An honest gap is the correct output; a fabricated match is a failure.
- Keep the person's name, employers, titles, and dates exactly as given.
{gaps_block}
Return ONLY a JSON object of this shape, with no commentary:
{{"reasoning": "<brief>", "name": "...", "headline": null, "summary": null,
  "emails": [], "phones": [], "urls": [],
  "work": [{{"company": "...", "title": "...", "start_date": null,
             "end_date": null, "bullets": ["..."]}}],
  "education": [{{"institution": "...", "degree": null, "field": null,
                  "end_year": null}}],
  "projects": [{{"name": "...", "url": null, "tech": ["..."], "bullets": ["..."]}}],
  "skills": ["..."], "certifications": ["..."]}}

MASTER RÉSUMÉ (JSON):
---
{resume_json}
---"""

_GAPS_HEADER = """
TARGET REQUIREMENTS (emphasize existing evidence for these; never invent):
{gaps}
"""

_REWRITE_TEMPERATURE = 0.6
_client: GroqClient | None = None


def _default_client() -> GroqClient:
    global _client
    if _client is None:
        # Share the account-wide token budget with every other Groq caller.
        _client = GroqClient(budget=shared_budget())
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


def rewrite_schema_groq(
    resume: StructuredResume, gaps: list[Gap], client: GroqClient | None = None
) -> StructuredResume:
    """Tailor `resume` toward `gaps` via Groq. Output is UNVERIFIED — gate it."""
    client = client or _default_client()
    prompt = _PROMPT.format(
        gaps_block=_gaps_block(gaps), resume_json=_source_json(resume)
    )
    data = client.complete_json(prompt, temperature=_REWRITE_TEMPERATURE)
    tailored = to_structured(_coerce(data))
    # Achievements are facts, not prose to tailor — carry them through verbatim.
    tailored.achievements = list(resume.achievements)
    tailored.achievements_prov = [list(p) for p in resume.achievements_prov]
    return tailored
