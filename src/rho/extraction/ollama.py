"""Ollama-backed résumé extraction with server-side constrained decoding.

Same contract as `rho.extraction.llm` (vLLM + Outlines), but reachable without a
CUDA GPU — the same deviation Phases 4 and 5 took for JD analysis and rewriting,
for the same reason. Ollama's `format` parameter enforces the JSON schema during
decoding, so this is constrained generation rather than parse-and-hope.

Temperature is pinned to 0: extraction is a faithfulness task, and the whole
provenance chain starts from what this returns. Anything invented here is
invented for the rest of the pipeline, so `attach_provenance` will fail to find
a span for it and the reviewer will report it — but the correct fix is to not
invent it in the first place.
"""

import json
import urllib.error
import urllib.request

from rho.config import settings
from rho.extraction.schema import ExtractionSchema

_PROMPT = """You extract structured data from a resume. Rules:
- Extract ONLY information present in the text. Never invent, infer, or fill.
- If a field is absent, leave it empty ("" or []).
- Copy values verbatim from the resume. Do not paraphrase names, employers,
  titles, or skills — a value that does not appear in the text is a failure.
- Each skill MUST be a short token as it appears in the resume ("Python",
  "PostgreSQL"), not a sentence about the skill.
- `achievements` are standalone accomplishments, awards, honors, or recognition
  NOT tied to one specific job's bullet list (e.g. "Winner, ACM ICPC Regionals
  2021"). Copy each verbatim. Leave [] if none.
- Dates in ISO-8601 (2019, 2019-06). Use "" for present.
- Fill the `reasoning` field first, briefly, then the data fields.
Resume:
---
{markdown}
---"""

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
                "required": ["company", "title"],
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
        "achievements": {"type": "array", "items": {"type": "string"}},
    },
    # work/education/projects/skills/achievements are required so the decoder must
    # emit the arrays. Left optional, qwen2.5:14b closes the object early and
    # drops the entire section — a silent fill by omission, which reads
    # identically to a résumé that genuinely has no such entries.
    "required": ["reasoning", "name", "work", "education", "projects", "skills", "achievements"],
}

# Résumés are long and CPU inference is slow; generous on purpose.
_TIMEOUT_SECONDS = 600
_MAX_RESUME_CHARS = 12000
_EXTRACTION_TEMPERATURE = 0


def _build_payload(markdown: str, model: str | None = None) -> dict:
    return {
        "model": model or settings.extraction_model_ollama,
        "messages": [
            {
                "role": "user",
                "content": _PROMPT.format(markdown=markdown[:_MAX_RESUME_CHARS]),
            }
        ],
        "stream": False,
        "format": _FORMAT,
        "options": {"temperature": _EXTRACTION_TEMPERATURE},
    }


def _parse_response(raw: dict) -> ExtractionSchema:
    content = raw.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned non-JSON content: {content[:200]}") from exc
    return ExtractionSchema(**data)


def extract_schema(markdown: str) -> ExtractionSchema:
    """markdown -> ExtractionSchema. Provenance is attached by the caller."""
    payload = json.dumps(_build_payload(markdown)).encode()
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
    return _parse_response(raw)
