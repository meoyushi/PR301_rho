import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RHO_LLM_ENABLED") != "1", reason="no LLM backend"
)


def test_extract_schema_on_simple_markdown():
    from rho.extraction.llm import extract_schema

    md = (
        "Alice Johnson\nSenior Python Engineer\n"
        "Skills: Python, FastAPI, AWS\nAcme Corp Backend Engineer 2019-2022"
    )
    es = extract_schema(md)
    assert es.name.startswith("Alice")
    assert any("python" in s.lower() for s in es.skills)
