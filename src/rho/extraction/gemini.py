"""Gemini-backed résumé extraction (`gemini-3.1-flash-lite`).

Same contract as `rho.extraction.ollama`, but via Gemini's `responseSchema`
constrained decoding rather than Ollama's `format` parameter.
"""

from rho.extraction.schema import ExtractionSchema
from rho.llm.gemini import GeminiClient

_PROMPT = """You extract structured data from a resume. Rules:
- Extract ONLY information present in the text. Never invent, infer, or fill.
- If a field is absent, leave it empty ("" or []).
- Copy values verbatim from the resume. Do not paraphrase names, employers,
  titles, or skills — a value that does not appear in the text is a failure.
- Each skill MUST be a short token as it appears in the resume ("Python",
  "PostgreSQL"), not a sentence about the skill.
- `achievements` are standalone accomplishments, awards, honors, or recognition
  NOT tied to one specific job's bullet list (e.g. "Winner, ACM ICPC Regionals
  2021", "Patent US1234567"). Copy each verbatim. Leave [] if none.
- Dates in ISO-8601 (2019, 2019-06). Use "" for present.
- Fill the `reasoning` field first, briefly, then the data fields.
Resume:
---
{markdown}
---"""

_MAX_RESUME_CHARS = 12000

# work/education/skills are required so the decoder must emit the arrays; see
# rho.extraction.ollama for why leaving them optional silently drops sections.
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
                "required": ["company", "title"],
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
        "achievements": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    # projects/achievements required so the decoder emits the array (empty if
    # none) rather than closing the object early and dropping the section — same
    # reason work/education are required.
    "required": ["reasoning", "name", "work", "education", "projects", "skills", "achievements"],
}

_client: GeminiClient | None = None


def _default_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def extract_schema_gemini(
    markdown: str, client: GeminiClient | None = None
) -> ExtractionSchema:
    """markdown -> ExtractionSchema, via Gemini. Provenance attached by caller."""
    client = client or _default_client()
    data = client.complete_json(
        _PROMPT.format(markdown=markdown[:_MAX_RESUME_CHARS]),
        response_schema=_SCHEMA,
        temperature=0.0,
    )
    return ExtractionSchema(**data)
