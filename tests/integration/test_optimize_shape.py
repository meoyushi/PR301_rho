from fastapi.testclient import TestClient

from rho.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_optimize_returns_job_shape(stub_nodes):
    # /optimize is now an async job endpoint (résumé is already structured by
    # the time it's submitted) — this test asserts the response shape, not
    # model behaviour. stub_nodes is kept even though this route no longer
    # drives the LangGraph nodes directly, to preserve the fixture wiring.
    r = client.post(
        "/optimize",
        json={"resume": {"name": "Alice", "skills": ["python"]}, "jd_text": "need python"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["state"] in ("queued", "running", "done")
