import time

from rho.api.jobs import JobStore
from rho.models.api import OptimizeJobRequest, OptimizeResult
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport


def _fake_response():
    return OptimizeResult(
        match_result=MatchResult(component_vector=ComponentVector(
            keyword_coverage=0, semantic_similarity=0, fuzzy_coverage=0,
            must_have_coverage=0, nice_have_coverage=0), predicted_score=70.0),
        tailored_resume=TailoredResume(resume=StructuredResume(name="X"),
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)),
        final_score=70.0,
        display_score=100.0,
    )


def _await(store, jid, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        js = store.get(jid)
        if js.state in ("done", "error"):
            return js
        time.sleep(0.01)
    raise AssertionError(f"job {jid} did not finish; last={store.get(jid)}")


def test_job_runs_to_done_with_result():
    store = JobStore()
    jid = store.create(OptimizeJobRequest(resume=StructuredResume(name="X"), jd_text="jd"),
                       runner=lambda req, on_stage: _fake_response())
    js = _await(store, jid)
    assert js.state == "done" and js.result.final_score == 70.0


def test_job_failure_is_reported_not_swallowed():
    store = JobStore()
    def boom(req, on_stage):
        raise RuntimeError("model down")
    jid = store.create(OptimizeJobRequest(resume=StructuredResume(name="X"), jd_text="jd"), runner=boom)
    js = _await(store, jid)
    assert js.state == "error" and "model down" in js.error


def test_unknown_job_is_none():
    assert JobStore().get("nope") is None
