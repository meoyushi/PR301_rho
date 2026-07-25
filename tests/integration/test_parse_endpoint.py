import io

from fastapi.testclient import TestClient

import rho.api.app as appmod
from rho.api.app import app
from rho.models.resume import StructuredResume


def test_parse_returns_structured_resume(monkeypatch):
    # Stub extraction so /parse needs no LLM; ingest runs for real on .txt.
    monkeypatch.setattr(appmod, "extract",
                        lambda md, prov: StructuredResume(name="Parsed Person", skills=["python"]))
    client = TestClient(app)
    resp = client.post("/parse", files={"file": ("r.txt", io.BytesIO(b"Parsed Person\nSkills: python"), "text/plain")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["structured_resume"]["name"] == "Parsed Person"
    assert "provenance_map" in body
