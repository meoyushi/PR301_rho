"""One doc's extraction failing must not lose the whole table (Phase 7).

Regression: `eval_synthetic`/`eval_real`/`eval_public` had no per-item error
handling, so a single bad résumé (a permanently denied API key, a truncated
response) killed the whole 300-résumé Phase-7 run at document 14 of 30,
losing every doc after it — including Table 1c entirely.
"""

from pathlib import Path
from unittest.mock import patch

import eval.run_all as run_all


def test_eval_synthetic_survives_one_failing_doc(monkeypatch):
    calls = []

    def flaky_extract(path, use_cache=True):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("all 7 live Gemini keys failed")
        from rho.models.provenance import ProvenanceMap

        return (
            {"name": "A", "skills": [], "work": [], "education": []},
            1.0,
            ProvenanceMap(doc_id="d"),
        )

    fake_gold = [
        (Path(f"path{i}.txt"), {"id": f"g{i}", "name": "A", "gold_prov_values": {}})
        for i in range(4)
    ]
    with patch.object(run_all, "_extract_doc", flaky_extract), patch.object(
        run_all, "load_gold", lambda limit=None: fake_gold
    ):
        result = run_all.eval_synthetic(limit=None, use_cache=True)

    assert len(result["rows"]) == 3  # 3 succeeded
    assert len(result["failures"]) == 1  # 1 recorded, not silently dropped
    assert "g1" in result["failures"][0]


def test_eval_real_survives_one_failing_doc(monkeypatch):
    calls = []

    def flaky_extract(path, use_cache=True):
        calls.append(path)
        if len(calls) == 1:
            raise ValueError("truncated JSON")
        return ({"skills": [], "work": [], "education": []}, 1.0, None)

    fake_gold = [(f"path{i}", {"id": f"r{i}"}) for i in range(3)]
    with patch.object(run_all, "_extract_doc", flaky_extract), patch.object(
        run_all, "load_real_gold", lambda limit=None: fake_gold
    ):
        result = run_all.eval_real(limit=None, use_cache=True)

    assert len(result["rows"]) == 2
    assert len(result["failures"]) == 1


def test_eval_public_survives_one_failing_doc(monkeypatch):
    calls = []

    def flaky_extract_text(doc, filename, use_cache=True):
        calls.append(filename)
        if len(calls) == 1:
            raise RuntimeError("403 denied")
        from rho.models.provenance import ProvenanceMap

        return (
            {"name": "A", "skills": [], "work": [], "education": []},
            1.0,
            ProvenanceMap(doc_id="d"),
        )

    fake_gold = [("some resume text", {"id": f"p{i}", "gold_prov_values": {}}) for i in range(3)]
    with patch.object(run_all, "_extract_text", flaky_extract_text), patch.object(
        run_all, "load_public_gold", lambda limit=None: fake_gold
    ):
        result = run_all.eval_public(limit=None, use_cache=True)

    assert len(result["rows"]) == 2
    assert len(result["failures"]) == 1
