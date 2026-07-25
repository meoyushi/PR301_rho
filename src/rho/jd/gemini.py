"""Gemini-backed JD analysis (`gemini-3.1-flash-lite`).

Same contract and prompt discipline as `rho.jd.ollama`, but through Gemini's
`responseSchema` so JD analysis and rewriting can share this backend the way
the Ollama and Groq paths do.
"""

from rho.jd.schema import JDSchema, ReqItem, to_requirement_set
from rho.llm.gemini import GeminiClient
from rho.models.jd import RequirementSet

_PROMPT = """You extract structured requirements from a job description. Rules:
- Extract ONLY requirements stated in the text. Never invent or infer.
- Each `text` MUST be a short skill/tool token of 1-4 words, as it would appear
  in a résumé's skills list. Name the skill; do not restate the sentence.
    GOOD: "Python", "AWS", "customer service", "Bootstrap", "SQL"
    BAD:  "must be authorized to work in the usa", "communicate with customers
          via phone email and chat", "5+ years of relevant experience"
- Skip boilerplate that is not a skill: work authorization, location, age,
  company names, "must be excited to learn".
- Classify each requirement's `kind`: skill, tool, title, cert, or experience.
- Classify `priority`: "must" for required/essential items, "nice" for
  preferred/bonus/plus items. If the text does not mark it as optional, use "must".
- Set `years` only when the text states a number of years; otherwise null.
- Fill the `reasoning` field first, briefly, then the data fields.
Job description:
---
{jd_text}
---"""

# Mirrors rho.jd.ollama: coverage matches requirement text literally against
# résumé strings, so a sentence-length requirement can never match and would
# silently pin keyword_coverage and fuzzy_coverage identically at 0.
_MAX_REQUIREMENT_WORDS = 5
_MAX_JD_CHARS = 6000

_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning": {"type": "STRING"},
        "title": {"type": "STRING", "nullable": True},
        "requirements": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "kind": {
                        "type": "STRING",
                        "enum": ["skill", "tool", "title", "cert", "experience"],
                    },
                    "priority": {"type": "STRING", "enum": ["must", "nice"]},
                    "years": {"type": "NUMBER", "nullable": True},
                },
                "required": ["text", "kind", "priority"],
            },
        },
    },
    "required": ["reasoning", "requirements"],
}

_client: GeminiClient | None = None


def _default_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def _parse(data: dict) -> JDSchema:
    requirements = []
    for item in data.get("requirements") or []:
        try:
            req = ReqItem(**item)
        except Exception:
            continue  # No silent fills: a malformed item is dropped.
        if len(req.text.split()) > _MAX_REQUIREMENT_WORDS:
            continue  # The model restated a sentence instead of naming the skill.
        requirements.append(req)
    return JDSchema(
        reasoning=data.get("reasoning") or "",
        title=data.get("title"),
        requirements=requirements,
    )


def analyze_jd_schema_gemini(
    jd_text: str, client: GeminiClient | None = None
) -> RequirementSet:
    """jd text -> RequirementSet, via Gemini."""
    client = client or _default_client()
    data = client.complete_json(
        _PROMPT.format(jd_text=jd_text[:_MAX_JD_CHARS]),
        response_schema=_SCHEMA,
        temperature=0.0,
    )
    return to_requirement_set(_parse(data))
