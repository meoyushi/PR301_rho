"""Shared stubs for graph-level integration tests.

`/optimize` and `run_pipeline` both drive the real LangGraph pipeline, whose
`extract`, `analyze_jd` and `rewrite` nodes are LLM-backed. Integration tests
here assert wiring and response shape, so those three are replaced with
deterministic stubs; model behaviour is covered by the LLM tests.
"""

import pytest

from rho.models.jd import Requirement, RequirementSet
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume


@pytest.fixture
def stub_nodes(monkeypatch):
    import rho.graph.nodes as N

    monkeypatch.setattr(
        N,
        "extract",
        lambda md, prov: StructuredResume(
            name="A", skills=["Python"], skills_prov=[["x"]]
        ),
    )
    monkeypatch.setattr(
        N,
        "analyze_jd",
        lambda jd: RequirementSet(
            requirements=[Requirement(text="Python", kind="skill", priority="must")]
        ),
    )
    monkeypatch.setattr(
        N,
        "rewrite",
        lambda resume, gaps, prov: TailoredResume(
            resume=resume,
            fabrication_report=FabricationReport(
                total_edits=0, verified_edits=0, fabrication_rate=0.0
            ),
        ),
    )
    return N
