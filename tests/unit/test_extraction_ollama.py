"""Ollama-backed résumé extraction — structured extraction without a CUDA GPU.

Mirrors `tests/unit/test_jd_ollama.py`: the vLLM+Outlines path in
`rho.extraction.llm` needs CUDA the calibration host does not have, so the
default extraction path must work through Ollama's constrained decoding.
"""

import json

import pytest

from rho.extraction.ollama import _build_payload, _parse_response


def test_prompt_forbids_invention():
    """Extraction must never fill a field the résumé does not claim."""
    payload = _build_payload("Alice\nPython", model="gemma3:4b")
    prompt = payload["messages"][0]["content"]
    assert "Never invent" in prompt
    assert "Alice" in prompt


def test_payload_pins_temperature_and_schema():
    payload = _build_payload("Alice\nPython", model="gemma3:4b")
    assert payload["model"] == "gemma3:4b"
    assert payload["stream"] is False
    # Extraction is a faithfulness task, not a creative one: pinned low.
    assert payload["options"]["temperature"] == 0
    # Constrained decoding: the résumé schema is enforced by the server.
    assert payload["format"]["properties"]["work"]["type"] == "array"
    assert payload["format"]["properties"]["skills"]["type"] == "array"


def test_parse_response_maps_to_extraction_schema():
    raw = {
        "message": {
            "content": json.dumps(
                {
                    "reasoning": "r",
                    "name": "Alice Chen",
                    "emails": ["alice@example.com"],
                    "skills": ["Python", "SQL"],
                    "work": [
                        {
                            "company": "Acme Corp",
                            "title": "Data Engineer",
                            "start_date": "2020",
                            "end_date": "2024",
                            "bullets": ["Built ETL pipelines"],
                        }
                    ],
                    "education": [
                        {
                            "institution": "State University",
                            "degree": "BS",
                            "field": "CS",
                            "end_year": "2019",
                        }
                    ],
                }
            )
        }
    }
    es = _parse_response(raw)
    assert es.name == "Alice Chen"
    assert es.skills == ["Python", "SQL"]
    assert es.work[0].company == "Acme Corp"
    assert es.work[0].bullets == ["Built ETL pipelines"]
    assert es.education[0].institution == "State University"


def test_parse_response_rejects_malformed_json():
    with pytest.raises(ValueError):
        _parse_response({"message": {"content": "not json"}})


def test_format_requires_section_arrays():
    """The model must emit work/education/skills, even when empty.

    Left optional, qwen2.5:14b closes the object after `certifications` and
    silently drops a résumé's entire work history — indistinguishable from a
    résumé that genuinely has none. Requiring the keys forces the decoder to
    produce the arrays.
    """
    payload = _build_payload("Alice\nPython", model="gemma3:4b")
    required = payload["format"]["required"]
    for key in ("name", "work", "education", "skills"):
        assert key in required


def test_parse_response_leaves_absent_fields_empty():
    """No silent fills: a résumé with no work history yields [], not a stub entry."""
    raw = {"message": {"content": json.dumps({"reasoning": "r", "name": "Alice"})}}
    es = _parse_response(raw)
    assert es.work == []
    assert es.education == []
    assert es.skills == []
    assert es.headline is None
