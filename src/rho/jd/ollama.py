"""Ollama-backed JD analysis with server-side constrained decoding.

Same contract as `rho.jd.llm` (vLLM + Outlines), but reachable without a CUDA
GPU. Ollama's `format` parameter enforces the JSON schema during decoding, so
this is constrained generation rather than parse-and-hope.
"""

import json
import urllib.error
import urllib.request

from rho.config import settings
from rho.jd.schema import JDSchema, ReqItem

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

# Coverage matches requirement text literally against résumé strings, so a
# sentence-length requirement can never match and would silently pin
# keyword_coverage and fuzzy_coverage at 0.
_MAX_REQUIREMENT_WORDS = 5

# Mirrors JDSchema; Ollama enforces this during decoding.
_FORMAT = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "title": {"type": ["string", "null"]},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"enum": ["skill", "tool", "title", "cert", "experience"]},
                    "priority": {"enum": ["must", "nice"]},
                    "years": {"type": ["number", "null"]},
                },
                "required": ["text", "kind", "priority"],
            },
        },
    },
    "required": ["reasoning", "requirements"],
}

# JDs are long and CPU inference is slow; this is generous on purpose.
_TIMEOUT_SECONDS = 600
_MAX_JD_CHARS = 6000


def _build_payload(jd_text: str, model: str | None = None) -> dict:
    return {
        "model": model or settings.jd_model,
        "messages": [{"role": "user", "content": _PROMPT.format(jd_text=jd_text[:_MAX_JD_CHARS])}],
        "stream": False,
        "format": _FORMAT,
        "options": {"temperature": 0},
    }


def _parse_response(raw: dict) -> JDSchema:
    content = raw.get("message", {}).get("content", "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned non-JSON content: {content[:200]}") from exc

    requirements = []
    for item in data.get("requirements", []):
        try:
            req = ReqItem(**item)
        except Exception:
            # No silent fills: a malformed item is dropped, never defaulted.
            continue
        if len(req.text.split()) > _MAX_REQUIREMENT_WORDS:
            # The model restated a sentence instead of naming the skill.
            continue
        requirements.append(req)

    return JDSchema(
        reasoning=data.get("reasoning", ""),
        title=data.get("title"),
        requirements=requirements,
    )


def analyze_jd_schema(jd_text: str) -> JDSchema:
    payload = json.dumps(_build_payload(jd_text)).encode()
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
