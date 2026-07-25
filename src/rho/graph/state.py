"""Typed state carried through the LangGraph pipeline.

`total=False` because the graph fills the dict incrementally: each node returns a
partial update and LangGraph merges it. Only the three inputs are present at
START; everything else appears as its producing node runs.
"""

from typing import TypedDict

from rho.models.jd import RequirementSet
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import TailoredResume
from rho.models.scoring import MatchResult


class PipelineState(TypedDict, total=False):
    # inputs
    file_bytes: bytes
    filename: str
    jd_text: str
    # ingest branch
    markdown: str
    prov: ProvenanceMap
    resume: StructuredResume
    # jd branch
    reqs: RequirementSet
    # fan-in and downstream
    match_result: MatchResult
    tailored: TailoredResume
    # reviewer
    final_score: float
    invariant_ok: bool
    invariant_violations: list[str]
