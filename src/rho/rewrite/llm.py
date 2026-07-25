"""Grounded résumé rewriting with server-side constrained decoding.

Deviation from the plan (same one Phase 4 took, same reason): the plan specifies
vLLM + Outlines, but the calibration host has no CUDA, so `outlines.models.vllm`
cannot load. Ollama's `format` parameter enforces the JSON schema during
decoding, so this is still constrained generation rather than parse-and-hope,
and the C3 ablation can actually be run on this machine.

The prompt is grounded — master résumé as sole source of truth — but the prompt
is *not* the safety mechanism. Truthfulness is enforced downstream by
`rho.rewrite.verifier`, which is deterministic and LLM-free. Temperature is 0.6
per the plan: the rewriter is meant to be creative, and the gate is what makes
that safe.
"""

import json
import urllib.error
import urllib.request

from rho.config import settings
from rho.extraction.schema import ExtractionSchema, to_structured
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

# Mirrors ExtractionSchema; Ollama enforces this during decoding.
_FORMAT = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "name": {"type": "string"},
        "headline": {"type": ["string", "null"]},
        "summary": {"type": ["string", "null"]},
        "emails": {"type": "array", "items": {"type": "string"}},
        "phones": {"type": "array", "items": {"type": "string"}},
        "urls": {"type": "array", "items": {"type": "string"}},
        "work": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "start_date": {"type": ["string", "null"]},
                    "end_date": {"type": ["string", "null"]},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                # bullets required: optional, the model drops the work history's
                # bullets when told to "select" for relevance.
                "required": ["company", "title", "bullets"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": ["string", "null"]},
                    "field": {"type": ["string", "null"]},
                    "end_year": {"type": ["string", "null"]},
                },
                "required": ["institution"],
            },
        },
        "projects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "url": {"type": ["string", "null"]},
                    "tech": {"type": "array", "items": {"type": "string"}},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name"],
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "certifications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "name"],
}

_TIMEOUT_SECONDS = 600
_REWRITE_TEMPERATURE = 0.6


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


def _build_payload(resume: StructuredResume, gaps: list[Gap], model: str | None = None) -> dict:
    prompt = _PROMPT.format(
        gaps_block=_gaps_block(gaps), resume_json=_source_json(resume)
    )
    return {
        "model": model or settings.rewrite_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": _FORMAT,
        "options": {"temperature": _REWRITE_TEMPERATURE},
    }


def _parse_response(raw: dict) -> StructuredResume:
    content = raw.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned non-JSON content: {content[:200]}") from exc
    return to_structured(ExtractionSchema(**data))


def rewrite_schema(resume: StructuredResume, gaps: list[Gap]) -> StructuredResume:
    """Tailor `resume` toward `gaps`. Output is UNVERIFIED — gate it before use."""
    payload = json.dumps(_build_payload(resume, gaps)).encode()
    request = urllib.request.Request(
        f"{settings.ollama_base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"ollama request failed: {exc}") from exc
    tailored = _parse_response(raw)
    # Achievements are facts, not prose to tailor — carry them through verbatim.
    tailored.achievements = list(resume.achievements)
    tailored.achievements_prov = [list(p) for p in resume.achievements_prov]
    return tailored
