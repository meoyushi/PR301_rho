from rho.config import settings
from rho.extraction.provenance_attach import attach_provenance
from rho.extraction.schema import to_structured
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume


def _resolve_schema_fn(backend: str):
    """Pick the extraction backend by name.

    Imports are lazy and per-branch: `rho.extraction.llm` imports `outlines`
    at module level, which is not installed on a CUDA-less host, so resolving
    "ollama" must not drag it in.
    """
    if backend == "ollama":
        from rho.extraction.ollama import extract_schema

        return extract_schema
    if backend == "vllm":
        from rho.extraction.llm import extract_schema

        return extract_schema
    if backend == "gemini":
        from rho.extraction.gemini import extract_schema_gemini

        return extract_schema_gemini
    raise ValueError(
        f"unknown extraction backend: {backend!r} (want 'ollama'/'vllm'/'gemini')"
    )


def extract(markdown: str, prov: ProvenanceMap, _schema_fn=None) -> StructuredResume:
    """markdown+prov -> StructuredResume with *_prov filled"""
    if _schema_fn is None:
        _schema_fn = _resolve_schema_fn(settings.extraction_backend)
    es = _schema_fn(markdown)  # ExtractionSchema (validated by Pydantic already)
    resume = to_structured(es)
    return attach_provenance(resume, prov)
