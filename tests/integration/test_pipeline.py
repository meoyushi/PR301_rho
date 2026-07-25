"""End-to-end graph runs with the LLM-backed components monkeypatched out.

The point is the wiring — parallel branches, fan-in, reviewer — not model
quality, so `extract`, `analyze_jd` and `rewrite` are stubbed (see conftest).
"""

from rho.models.resume import StructuredResume


def test_pipeline_end_to_end(stub_nodes):
    from rho.graph import run_pipeline

    resp = run_pipeline(b"Alice\nPython", "r.txt", "need python")
    assert resp.structured_resume.name == "A"
    assert resp.match_result.gaps[0].requirement.text == "Python"
    assert isinstance(resp.final_score, float)


def test_pipeline_fans_in_after_both_branches(stub_nodes, monkeypatch):
    """`match` must not run until both `extract` and `jd` have landed."""
    N = stub_nodes
    calls = []
    real_match = N.match

    def spy_match(resume, reqs):
        calls.append((resume, reqs))
        return real_match(resume, reqs)

    monkeypatch.setattr(N, "match", spy_match)
    from rho.graph import run_pipeline

    run_pipeline(b"Alice\nPython", "r.txt", "need python")
    # Exactly once: default any-of triggering would fire `match` early on the
    # short jd branch and then again after extract.
    assert len(calls) == 1
    resume, reqs = calls[0]
    assert resume.skills == ["Python"]
    assert reqs.requirements[0].text == "Python"


def test_pipeline_reports_invariant(stub_nodes, monkeypatch):
    """A skill with no supporting span is reported, not silently shipped."""
    N = stub_nodes
    monkeypatch.setattr(
        N,
        "extract",
        lambda md, prov: StructuredResume(name="A", skills=["Kubernetes"]),
    )
    ok_calls = []
    real_review = N.review_node

    def spy_review(state):
        out = real_review(state)
        ok_calls.append(out)
        return out

    monkeypatch.setattr(N, "review_node", spy_review)
    from rho.graph import build_graph

    graph = build_graph()
    graph.invoke(
        {"file_bytes": b"Alice\nPython", "filename": "r.txt", "jd_text": "need python"},
        config={"configurable": {"thread_id": "t-invariant"}},
    )
    assert ok_calls[0]["invariant_ok"] is False
    assert "Kubernetes" in ok_calls[0]["invariant_violations"]
