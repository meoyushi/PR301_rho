"""Phase 6 — LangGraph orchestration.

Topology (two branches off START, fanning in at `match`):

    START -> ingest -> extract ---+
    START -> jd ------------------+--> match -> score -> rewrite -> review -> END

LangGraph only schedules a node once every inbound edge has fired, so `match`
is a real barrier: it cannot see a half-built state with `resume` but no `reqs`.
Résumé parsing and JD analysis are independent, so they overlap.
"""

from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rho.graph import nodes as N
from rho.graph.state import PipelineState
from rho.models.api import PipelineResponse

__all__ = ["build_graph", "run_pipeline"]


def build_graph():
    """Compile the pipeline graph.

    `MemorySaver` keeps checkpoints in-process; swap `PostgresSaver` in prod to
    survive restarts.
    """
    g = StateGraph(PipelineState)
    g.add_node("ingest", N.ingest_node)
    g.add_node("extract", N.extract_node)
    g.add_node("jd", N.jd_node)
    # defer=True makes `match` a real fan-in barrier. Without it LangGraph's
    # default any-of triggering fires `match` as soon as the short `jd` branch
    # lands (superstep 2) and *again* after `extract` (superstep 3) — the first
    # call blowing up on a state that has no `resume` yet.
    g.add_node("match", N.match_node, defer=True)
    g.add_node("score", N.score_node)
    g.add_node("rewrite", N.rewrite_node)
    g.add_node("review", N.review_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge(START, "jd")
    g.add_edge("extract", "match")  # fan-in barrier: both edges must fire
    g.add_edge("jd", "match")
    g.add_edge("match", "score")
    g.add_edge("score", "rewrite")
    g.add_edge("rewrite", "review")
    g.add_edge("review", END)
    return g.compile(checkpointer=MemorySaver())


_graph = None


def run_pipeline(file_bytes: bytes, filename: str, jd_text: str) -> PipelineResponse:
    """Run one résumé + JD through the whole pipeline (frozen signature)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    final = _graph.invoke(
        {"file_bytes": file_bytes, "filename": filename, "jd_text": jd_text},
        # Fresh thread_id per call: checkpoints exist for retry/resume within a
        # run, not to accumulate state across unrelated résumés.
        config={"configurable": {"thread_id": f"run-{uuid4()}"}},
    )
    return PipelineResponse(
        structured_resume=final["resume"],
        provenance_map=final["prov"],
        match_result=final["match_result"],
        tailored_resume=final["tailored"],
        final_score=final["final_score"],
    )
