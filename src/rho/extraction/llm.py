import outlines

from rho.config import settings
from rho.extraction.schema import ExtractionSchema

_PROMPT = """You extract structured data from a resume. Rules:
- Extract ONLY information present in the text. Never invent, infer, or fill.
- If a field is absent, leave it empty ("" or []).
- Dates in ISO-8601 (2019, 2019-06). Use "" for present.
- Fill the `reasoning` field first, briefly, then the data fields.
Resume:
---
{markdown}
---"""

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = outlines.models.vllm(settings.extraction_model)
    return _model


def extract_schema(markdown: str) -> ExtractionSchema:
    model = _get_model()
    generator = outlines.generate.json(model, ExtractionSchema)
    return generator(
        _PROMPT.format(markdown=markdown), temperature=settings.temperature
    )
