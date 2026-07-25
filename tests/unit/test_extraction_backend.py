"""Default extraction backend selection.

`extract()` previously hardcoded the vLLM+Outlines path, which needs CUDA. On a
CUDA-less host that made the whole pipeline unrunnable end-to-end, so the
backend is now chosen by config, defaulting to the Ollama path that works here.
"""

import pytest

from rho.extraction import _resolve_schema_fn


def test_default_backend_is_ollama():
    """Ollama is the path that runs on the calibration host, so it is default."""
    from rho.config import settings

    assert settings.extraction_backend == "ollama"


def test_resolve_returns_ollama_schema_fn():
    from rho.extraction.ollama import extract_schema

    assert _resolve_schema_fn("ollama") is extract_schema


def test_resolve_vllm_backend_is_still_reachable():
    """The plan's vLLM path stays selectable on a CUDA host."""
    pytest.importorskip("outlines")
    from rho.extraction.llm import extract_schema

    assert _resolve_schema_fn("vllm") is extract_schema


def test_resolve_returns_gemini_schema_fn():
    from rho.extraction.gemini import extract_schema_gemini

    assert _resolve_schema_fn("gemini") is extract_schema_gemini


def test_unknown_backend_fails_loudly():
    """A typo'd backend must not silently fall through to some default."""
    with pytest.raises(ValueError, match="unknown extraction backend"):
        _resolve_schema_fn("nope")


def test_extract_uses_injected_fn_over_backend(monkeypatch):
    """`_schema_fn=` injection still wins, so tests never hit a real model."""
    from rho.extraction import extract
    from rho.extraction.schema import ExtractionSchema
    from rho.models.provenance import ProvenanceMap, SourceSpan

    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    stub = lambda md: ExtractionSchema(reasoning="", name="A", skills=["Python"])
    resume = extract("Python", pm, _schema_fn=stub)
    assert resume.skills == ["Python"]
    # provenance still attached by the real attach_provenance
    assert resume.skills_prov[0] != []


def test_extract_carries_achievements_with_provenance(monkeypatch):
    """Achievements flow schema -> to_structured -> attach_provenance."""
    from rho.extraction import extract
    from rho.extraction.schema import ExtractionSchema
    from rho.models.provenance import ProvenanceMap, SourceSpan

    award = "Winner, ACM ICPC Regionals 2021"
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=len(award), raw_text=award))
    stub = lambda md: ExtractionSchema(reasoning="", name="A", achievements=[award])
    resume = extract(award, pm, _schema_fn=stub)
    assert resume.achievements == [award]
    assert resume.achievements_prov[0] != []
