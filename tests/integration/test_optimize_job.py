from fastapi.testclient import TestClient

from rho.api.app import app
from rho.models.api import OptimizeResult, ScoreComponent
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport


def _result():
    return OptimizeResult(
        match_result=MatchResult(component_vector=ComponentVector(
            keyword_coverage=0.9, semantic_similarity=0.9, fuzzy_coverage=0,
            must_have_coverage=0.9, nice_have_coverage=1.0), predicted_score=80.0),
        tailored_resume=TailoredResume(resume=StructuredResume(name="X"),
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)),
        final_score=80.0,
        display_score=95.0,
        baseline_score=60.0,
        baseline_display_score=70.0,
        components=[ScoreComponent(label="Keyword match", before=0.5, after=0.9)],
    )


def test_optimize_job_lifecycle(monkeypatch):
    # Patch the store's default runner path (run_optimize) with an LLM-free stub.
    monkeypatch.setattr("rho.api.jobs.run_optimize",
                        lambda resume, jd_text, on_stage=None: _result())
    client = TestClient(app)
    start = client.post("/optimize", json={"resume": {"name": "X"}, "jd_text": "jd"})
    assert start.status_code == 200
    jid = start.json()["id"]

    import time
    for _ in range(200):
        poll = client.get(f"/optimize/{jid}")
        if poll.json()["state"] in ("done", "error"):
            break
        time.sleep(0.01)
    body = poll.json()
    assert body["state"] == "done"
    assert body["result"]["final_score"] == 80.0
    assert body["result"]["display_score"] == 95.0
    assert body["result"]["baseline_display_score"] == 70.0
    assert body["result"]["components"][0]["label"] == "Keyword match"


def test_optimize_unknown_job_404():
    assert TestClient(app).get("/optimize/nope").status_code == 404
