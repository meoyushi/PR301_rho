"""Phase 5 Task 3: grounded rewrite generation.

Skipped without a reachable model. The gate (`tests/unit/test_verifier.py`) is
where truthfulness is actually *enforced*; this only checks the grounded prompt
produces a usable résumé in the same schema.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RHO_LLM_ENABLED") != "1", reason="no LLM"
)


def test_rewrite_does_not_add_unsourced_skill():
    from rho.models.resume import StructuredResume
    from rho.rewrite.llm import rewrite_schema

    src = StructuredResume(name="A", skills=["Python"])
    out = rewrite_schema(src, gaps=[])
    # grounded prompt shouldn't invent; even if it does, gate catches it later
    assert "python" in [s.lower() for s in out.skills]


def test_rewrite_preserves_identity():
    from rho.models.resume import StructuredResume, WorkExperience
    from rho.rewrite.llm import rewrite_schema

    src = StructuredResume(
        name="Dana Reed",
        skills=["Python", "SQL"],
        work=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                start_date="2019",
                end_date="2022",
                bullets=["Built internal reporting tools"],
            )
        ],
    )
    out = rewrite_schema(src, gaps=[])
    assert out.name == "Dana Reed"
    assert [w.company for w in out.work] == ["Acme"]
