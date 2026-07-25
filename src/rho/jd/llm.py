import outlines

from rho.config import settings
from rho.jd.schema import JDSchema

_PROMPT = """You extract structured requirements from a job description. Rules:
- Extract ONLY requirements stated in the text. Never invent or infer.
- Classify each requirement's `kind`: skill, tool, title, cert, or experience.
- Classify `priority`: "must" for required/essential items, "nice" for
  preferred/bonus/plus items. If the text does not mark it as optional, use "must".
- Set `years` only when the text states a number of years; otherwise null.
- Fill the `reasoning` field first, briefly, then the data fields.
Job description:
---
{jd_text}
---"""

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = outlines.models.vllm(settings.extraction_model)
    return _model


def analyze_jd_schema(jd_text: str) -> JDSchema:
    model = _get_model()
    generator = outlines.generate.json(model, JDSchema)
    return generator(_PROMPT.format(jd_text=jd_text), temperature=settings.temperature)
