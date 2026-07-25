"""Groq-backed JD analysis.

Same contract and prompt discipline as `rho.jd.ollama`, but through the Groq
endpoint so JD analysis and rewriting share one backend. Groq's Qwen3.6 does not
support `json_schema` decoding (see `rho.llm.groq`), so the schema is enforced by
Pydantic after generation rather than during it: malformed items are dropped, never
defaulted.
"""

from rho.jd.schema import JDSchema, ReqItem, to_requirement_set
from rho.llm.groq import GroqClient, shared_budget
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

Return ONLY a JSON object of this shape, with no commentary:
{{"reasoning": "<brief>", "title": "<job title or null>",
  "requirements": [{{"text": "...", "kind": "skill", "priority": "must", "years": null}}]}}

Job description:
---
{jd_text}
---"""

# Mirrors rho.jd.ollama: coverage matches requirement text literally against
# résumé strings, so a sentence-length requirement can never match and would
# silently pin keyword_coverage and fuzzy_coverage at 0.
_MAX_REQUIREMENT_WORDS = 5
_MAX_JD_CHARS = 6000

_client: GroqClient | None = None


def _default_client() -> GroqClient:
    global _client
    if _client is None:
        # Share the account-wide token budget with every other Groq caller.
        _client = GroqClient(budget=shared_budget())
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


def analyze_jd_schema_groq(jd_text: str, client: GroqClient | None = None) -> RequirementSet:
    """jd text -> RequirementSet, via Groq."""
    client = client or _default_client()
    data = client.complete_json(
        _PROMPT.format(jd_text=jd_text[:_MAX_JD_CHARS]), temperature=0.0
    )
    return to_requirement_set(_parse(data))
