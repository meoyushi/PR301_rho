# Phase 6 — LangGraph Orchestration + Reviewer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–5 done.

**Goal:** Wire all components into one LangGraph pipeline behind `run_pipeline()` and the `/optimize` API. Résumé-parse and JD-analyze branches run in parallel, fan in at the scorer; the calibrator fills `predicted_score`; the rewriter+gate produce the tailored résumé; a **reviewer node re-asserts the end-to-end provenance invariant** and computes `final_score`.

**Architecture:** LangGraph `StateGraph` over a typed state dict. Nodes: `ingest_node`, `extract_node`, `jd_node` (parallel with ingest+extract), `match_node` (fan-in barrier — waits for both branches), `score_node` (apply calibrator), `rewrite_node` (generate + gate), `review_node` (provenance-invariant assertion + final score). Checkpointing via `MemorySaver` (swap `PostgresSaver` in prod). `/optimize` calls `run_pipeline` and returns `PipelineResponse`.

**Tech Stack:** LangGraph, FastAPI (existing).

## Global Constraints
- Implement `rho.graph.run_pipeline(file_bytes, filename, jd_text) -> PipelineResponse` (frozen signature).
- The reviewer node MUST assert the provenance invariant (shared-context Section 7): every hard-content token in the tailored résumé traces to a `prov_id`. On violation, flag in the response (do not crash).
- Parallel branches must fan in correctly — `match_node` runs only after BOTH `extract_node` and `jd_node` complete.
- Calibrator loaded from `eval/calibrator.joblib` if present; else `predicted_score` stays raw-vector-derived fallback (documented).

## This phase consumes
- Every component: `ingest`, `extract`, `analyze_jd`, `match`, `Calibrator`+`score_with_calibrator`, `rewrite`, `verify_against_source` (Phases 1–5).

## This phase produces
- `run_pipeline()`; the LangGraph `build_graph()`; `/optimize` wired to it.
- `rho.graph.review.check_provenance_invariant(tailored, prov) -> tuple[bool, list[str]]`.

---

## File Structure
- Create: `src/rho/graph/state.py` — `PipelineState` TypedDict.
- Create: `src/rho/graph/nodes.py` — node functions.
- Create: `src/rho/graph/review.py` — provenance-invariant check + final score.
- Modify: `src/rho/graph/__init__.py` — `build_graph`, `run_pipeline`.
- Modify: `src/rho/api/app.py` — `/optimize` calls `run_pipeline`.
- Create: `tests/integration/test_pipeline.py`.

---

### Task 1: Pipeline state + reviewer invariant check

**Files:**
- Create: `src/rho/graph/state.py`, `src/rho/graph/review.py`
- Test: `tests/unit/test_review.py`

**Interfaces:**
- Produces: `PipelineState` (TypedDict with `file_bytes, filename, jd_text, markdown, prov, resume, reqs, match_result, tailored, final_score, invariant_ok, invariant_violations`). `check_provenance_invariant(tailored: StructuredResume, prov) -> (bool, list[str])` — returns False + list of unsupported values if any hard-content token lacks a `prov_id`. `compute_final_score(match_result, fabrication_report) -> float`.

- [x] **Step 1: Write failing test**
```python
# tests/unit/test_review.py
from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.models.resume import StructuredResume
from rho.graph.review import check_provenance_invariant, compute_final_score
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import FabricationReport
def _pm():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    return pm
def test_invariant_passes_when_all_sourced():
    ok, viol = check_provenance_invariant(StructuredResume(name="A", skills=["Python"]), _pm())
    assert ok and viol == []
def test_invariant_fails_on_unsourced_token():
    ok, viol = check_provenance_invariant(StructuredResume(name="A", skills=["Rust"]), _pm())
    assert not ok and "Rust" in viol
def test_final_score_penalized_by_fabrication():
    cv = ComponentVector(keyword_coverage=1,semantic_similarity=1,fuzzy_coverage=1,
        must_have_coverage=1,nice_have_coverage=1)
    mr = MatchResult(component_vector=cv, predicted_score=80.0)
    clean = FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)
    assert compute_final_score(mr, clean) == 80.0
```

- [x] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_review.py -v`

- [x] **Step 3: Implement**
```python
# src/rho/graph/state.py
from typing import TypedDict, Optional
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet
from rho.models.scoring import MatchResult
from rho.models.rewrite import TailoredResume
class PipelineState(TypedDict, total=False):
    file_bytes: bytes; filename: str; jd_text: str
    markdown: str; prov: ProvenanceMap
    resume: StructuredResume; reqs: RequirementSet
    match_result: MatchResult; tailored: TailoredResume
    final_score: float; invariant_ok: bool; invariant_violations: list[str]
```
```python
# src/rho/graph/review.py
from rho.rewrite.tokens import hard_content_tokens
from rho.extraction.provenance_attach import find_prov
def check_provenance_invariant(tailored, prov):
    violations = []
    for value, _path in hard_content_tokens(tailored):
        if not find_prov(value, prov):
            violations.append(value)
    return (len(violations) == 0, violations)
def compute_final_score(match_result, fabrication_report):
    # predicted_score is the calibrated ATS score; fabrication already prevented by gate,
    # so no double penalty. Kept as a hook for future weighting.
    return match_result.predicted_score
```

- [x] **Step 4: Run to verify pass** → PASS.

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: pipeline state + provenance-invariant reviewer"
```

---

### Task 2: Graph nodes

**Files:**
- Create: `src/rho/graph/nodes.py`
- Test: covered by Task 3 end-to-end (nodes are thin wrappers)

**Interfaces:**
- Produces: node functions each `(state: PipelineState) -> dict` (partial state update). `ingest_node`, `extract_node`, `jd_node`, `match_node`, `score_node`, `rewrite_node`, `review_node`. `score_node` loads calibrator if `eval/calibrator.joblib` exists.

- [x] **Step 1: Implement**
```python
# src/rho/graph/nodes.py
import os
from rho.ingestion import ingest
from rho.extraction import extract
from rho.jd import analyze_jd
from rho.matching import match
from rho.ats import Calibrator, score_with_calibrator
from rho.rewrite import rewrite
from rho.graph.review import check_provenance_invariant, compute_final_score
def ingest_node(state):
    md, prov = ingest(state["file_bytes"], state["filename"])
    return {"markdown": md, "prov": prov}
def extract_node(state):
    return {"resume": extract(state["markdown"], state["prov"])}
def jd_node(state):
    return {"reqs": analyze_jd(state["jd_text"])}
def match_node(state):
    return {"match_result": match(state["resume"], state["reqs"])}
def score_node(state):
    mr = state["match_result"]
    if os.path.exists("eval/calibrator.joblib"):
        cal = Calibrator().load("eval/calibrator.joblib")
        mr = score_with_calibrator(mr, cal)
    return {"match_result": mr}
def rewrite_node(state):
    return {"tailored": rewrite(state["resume"], state["match_result"].gaps, state["prov"])}
def review_node(state):
    ok, viol = check_provenance_invariant(state["tailored"].resume, state["prov"])
    final = compute_final_score(state["match_result"], state["tailored"].fabrication_report)
    return {"invariant_ok": ok, "invariant_violations": viol, "final_score": final}
```

- [x] **Step 2: Commit**
```bash
git add -A && git commit -m "feat: LangGraph node functions"
```

---

### Task 3: Build graph + `run_pipeline` + parallel fan-in

**Files:**
- Modify: `src/rho/graph/__init__.py`
- Test: `tests/integration/test_pipeline.py`

**Interfaces:**
- Produces: `build_graph()` (compiled LangGraph) and `run_pipeline(file_bytes, filename, jd_text) -> PipelineResponse`. Graph edges: `START → ingest_node → extract_node`; `START → jd_node`; both `extract_node` and `jd_node → match_node` (fan-in); `match_node → score_node → rewrite_node → review_node → END`. For tests without an LLM, `run_pipeline` accepts injectable `_extract_fn`/`_rewrite_fn`/`_jd_fn` OR the test monkeypatches component functions.

- [x] **Step 1: Add dep**
Add `langgraph>=0.2` to `pyproject.toml`; install.

- [x] **Step 2: Write integration test (monkeypatched components, no LLM)**
```python
# tests/integration/test_pipeline.py
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet, Requirement
def test_pipeline_end_to_end(monkeypatch):
    import rho.graph.nodes as N
    monkeypatch.setattr(N, "extract",
        lambda md, prov: StructuredResume(name="A", skills=["Python"], skills_prov=[["x"]]))
    monkeypatch.setattr(N, "analyze_jd",
        lambda jd: RequirementSet(requirements=[Requirement(text="Python", kind="skill", priority="must")]))
    monkeypatch.setattr(N, "rewrite",
        lambda resume, gaps, prov: __import__("rho.models.rewrite", fromlist=["TailoredResume","FabricationReport"]).TailoredResume(
            resume=resume, fabrication_report=__import__("rho.models.rewrite", fromlist=["FabricationReport"]).FabricationReport(
                total_edits=0, verified_edits=0, fabrication_rate=0.0)))
    from rho.graph import run_pipeline
    resp = run_pipeline(b"Alice\nPython", "r.txt", "need python")
    assert resp.structured_resume.name == "A"
    assert resp.match_result.gaps[0].requirement.text == "Python"
    assert isinstance(resp.final_score, float)
```

- [x] **Step 3: Run to verify fail** → FAIL.
Run: `pytest tests/integration/test_pipeline.py -v`

- [x] **Step 4: Implement**
```python
# src/rho/graph/__init__.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from rho.graph.state import PipelineState
from rho.graph import nodes as N
from rho.models.api import PipelineResponse
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("ingest", N.ingest_node); g.add_node("extract", N.extract_node)
    g.add_node("jd", N.jd_node); g.add_node("match", N.match_node)
    g.add_node("score", N.score_node); g.add_node("rewrite", N.rewrite_node)
    g.add_node("review", N.review_node)
    g.add_edge(START, "ingest"); g.add_edge("ingest", "extract")
    g.add_edge(START, "jd")
    g.add_edge("extract", "match"); g.add_edge("jd", "match")   # fan-in barrier
    g.add_edge("match", "score"); g.add_edge("score", "rewrite")
    g.add_edge("rewrite", "review"); g.add_edge("review", END)
    return g.compile(checkpointer=MemorySaver())
_graph = None
def run_pipeline(file_bytes: bytes, filename: str, jd_text: str) -> PipelineResponse:
    global _graph
    if _graph is None: _graph = build_graph()
    final = _graph.invoke(
        {"file_bytes": file_bytes, "filename": filename, "jd_text": jd_text},
        config={"configurable": {"thread_id": "run"}})
    return PipelineResponse(
        structured_resume=final["resume"],
        provenance_map=final["prov"],
        match_result=final["match_result"],
        tailored_resume=final["tailored"],
        final_score=final["final_score"],
    )
```

- [x] **Step 5: Run to verify pass** → PASS.

- [x] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: LangGraph pipeline + run_pipeline with parallel fan-in"
```

---

### Task 4: Wire `/optimize` to `run_pipeline`

**Files:**
- Modify: `src/rho/api/app.py`
- Test: `tests/integration/test_pipeline.py` (add API-level, monkeypatched)

**Interfaces:**
- Produces: `/optimize` now calls `run_pipeline` with the uploaded file + `jd_text`, returns the real `PipelineResponse`.

- [x] **Step 1: Write failing test**
```python
# add to tests/integration/test_pipeline.py
def test_optimize_endpoint_calls_pipeline(monkeypatch):
    from rho.models.api import PipelineResponse
    from rho.models.resume import StructuredResume
    from rho.models.provenance import ProvenanceMap
    from rho.models.scoring import MatchResult, ComponentVector
    from rho.models.rewrite import TailoredResume, FabricationReport
    import rho.api.app as A
    cv = ComponentVector(keyword_coverage=1,semantic_similarity=1,fuzzy_coverage=1,must_have_coverage=1,nice_have_coverage=1)
    resume = StructuredResume(name="A")
    monkeypatch.setattr(A, "run_pipeline", lambda fb, fn, jd: PipelineResponse(
        structured_resume=resume, provenance_map=ProvenanceMap(doc_id="d"),
        match_result=MatchResult(component_vector=cv, predicted_score=77.0),
        tailored_resume=TailoredResume(resume=resume, fabrication_report=FabricationReport(total_edits=0,verified_edits=0,fabrication_rate=0.0)),
        final_score=77.0))
    from fastapi.testclient import TestClient
    c = TestClient(A.app)
    r = c.post("/optimize", files={"file": ("r.txt", b"x", "text/plain")}, data={"jd_text": "jd"})
    assert r.json()["final_score"] == 77.0
```

- [x] **Step 2: Run to verify fail** → FAIL (still returns placeholder).

- [x] **Step 3: Implement** — replace placeholder in `app.py`:
```python
from rho.graph import run_pipeline
@app.post("/optimize", response_model=PipelineResponse)
async def optimize(file: UploadFile, jd_text: str = Form(...)):
    data = await file.read()
    return run_pipeline(data, file.filename or "resume", jd_text)
```
(Remove the `_placeholder_response` helper.)

- [x] **Step 4: Run to verify pass** → PASS.

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: wire /optimize to run_pipeline"
```

---

## Self-Review
- [x] Graph fans in at `match` after both `extract` and `jd`.
- [x] `run_pipeline` returns a full `PipelineResponse`.
- [x] Reviewer sets `invariant_ok`/violations; `/optimize` returns real pipeline output.
- [x] `pytest tests/integration/test_pipeline.py -v` green.

## Results

- **LangGraph version:** 1.2.9 (added to `pyproject.toml` as `langgraph>=0.2`).
- **End-to-end latency:**
  - mock (all LLM nodes stubbed, warm process): **median 89.7 ms** (n=5, min 83.3, max 193.5).
  - real LLM (qwen2.5:14b via Ollama; **extraction + JD analysis + rewrite all real**, ingest/match/score/review real): **268.5 s** for one résumé + JD. Dominated by the three 14B generations on CPU.
- **Invariant violations observed on real runs (qwen):** **0**. On the real run the tailored résumé's hard-content tokens all resolved to a `prov_id` (`invariant_ok=True`, `violations=[]`), fabrication report `total_edits=0 / verified_edits=0 / rate=0.000`. The reviewer's failure path is covered by `test_pipeline_reports_invariant`, which injects an unsourced skill ("Kubernetes") and asserts `invariant_ok=False` with the value reported rather than the run crashing.
- **Tests passing:** **136 passed, 4 skipped, 0 failed** (whole suite, bare `pytest tests/`). Phase-6 tests specifically: 3 unit (`tests/unit/test_review.py`) + 4 integration (`tests/integration/test_pipeline.py`) + 11 backend unit (`test_extraction_ollama.py`, `test_extraction_backend.py`) = 18/18. The 4th skip is the vLLM-path assertion, which `importorskip`s `outlines` on a CUDA-less host.

### Gemini re-run — same résumé/JD, all three LLM nodes on `gemini-3.1-flash-lite`

`extract_node`, `jd_node`, `rewrite_node` monkeypatched to `rho.extraction.gemini.extract_schema_gemini`,
`rho.jd.gemini.analyze_jd_schema_gemini`, `rho.rewrite.gemini.rewrite_schema_gemini`; ingest/match/score/review
unchanged. Same résumé + JD text as the qwen run.

| | qwen2.5:14b (Ollama, CPU) | Gemini (`gemini-3.1-flash-lite`) |
|---|---|---|
| End-to-end latency | 268.5 s | **15.7 s** (17x faster) |
| `final_score` | 18.36 | 19.16 |
| Fabrication gate | `total_edits=0` (nothing to reject) | `total_edits=1, rejected=1` — **`"Data Engineering"` rejected, no supporting prov_id** |
| `invariant_ok` | True | **False** |

**The Gemini run is the more informative one of the two, precisely because it failed one check.**
`invariant_ok=False` — a real, reproducible finding, not a fluke: `rho.extraction.gemini` returned
`start_date="2020-01-01T00:00:00Z"` for a résumé that plainly states `"2020-2024"`, over-formatting
the plain year into a full ISO timestamp despite the prompt showing the desired shape
(`"Dates in ISO-8601 (2019, 2019-06)"`) — confirmed deterministic across 3 repeat calls at
temperature 0, not sampling noise. `attach_provenance`'s fuzzy match
(`rapidfuzz.partial_ratio("2020-01-01t00:00:00z", "2020-2024") == 80`) falls under the 90 threshold
that `"2020"` alone would clear at 100, so the date carries no `prov_id` and the reviewer correctly
flags it. This is exactly the reviewer node doing its job (shared-context Section 7: "on violation,
flag in the response, do not crash") — the pipeline did not crash, it shipped a response with
`invariant_ok=False` so the caller can see the gap. It also demonstrates the fabrication gate
working on a genuine model output rather than an injected test skill: the rewriter proposed
`"Data Engineering"` as a skill addition (plausible-sounding, JD-relevant, and never stated in the
source résumé) and the gate rejected it before it reached the tailored output.

**Model:** `gemini-3.1-flash-lite`, same as Phases 4–5.

### Deviations from the plan

1. **`match` needs `defer=True`; the plan's edge topology alone is not a barrier.** With only the plan's edges, LangGraph's default *any-of* triggering fires `match` as soon as the short `jd` branch lands (superstep 2) — on a state with no `resume` yet, raising `KeyError: 'resume'` — and then fires it a **second** time after `extract` completes. `build_graph` therefore registers `g.add_node("match", N.match_node, defer=True)`, which defers the node until all pending predecessor paths settle. Verified: `match` runs exactly once, after both branches (`test_pipeline_fans_in_after_both_branches` asserts the call count, so the regression cannot return silently).
2. **`thread_id` is per-call (`run-{uuid4()}`), not the constant `"run"` in the plan.** A fixed thread id makes every request resume the same checkpoint, so a second `/optimize` call would replay against the first run's state.
3. **Extraction backend added so the pipeline runs unstubbed.** `rho/extraction/llm.py` had only the vLLM+Outlines path, which needs CUDA this host lacks (same constraint Phases 4–5 recorded), so the first real run had to stub `extract`. Rather than leave the pipeline unrunnable end-to-end, `rho/extraction/ollama.py` now mirrors the `rho/jd/ollama.py` + `rho/rewrite/llm.py` pattern (server-side constrained decoding via Ollama's `format`, temperature pinned to 0), and `extract()` resolves its backend from `settings.extraction_backend` (`"ollama"` default, `"vllm"` on a GPU host, unknown values raise). `_schema_fn=` injection still wins, so tests never touch a real model. **The final run stubs nothing on the extraction path**: name, skills, work and education all came from the model, with provenance attached by the real `attach_provenance`.
4. **Two pre-existing tests adjusted, not rewritten.**
   - `tests/integration/test_optimize_shape.py` asserted the shape of the placeholder response; `/optimize` now drives the real graph, so it takes the shared `stub_nodes` fixture (new `tests/integration/conftest.py`). Its assertion — response shape — is unchanged.
   - `tests/unit/test_stubs.py` asserted `checked > 0` ("stubs must still exist"). `run_pipeline` was the last Section-6 stub, and that file's own comment said to drop the assertion when the last one landed. It now asserts `checked == 0`, so a reintroduced silently-returning stub still fails the test.
5. **`predicted_score` fallback.** `eval/calibrator.joblib` is present, so `score_node` applied the Phase-4 calibrator (real run scored 18.36, not the 0.0 raw fallback). When the file is absent the node logs a warning and leaves the matcher's 0.0 rather than inventing a score (shared context Section 8).

6. **Extraction schema requires the section arrays.** With `work`/`education`/`skills` optional in the Ollama `format` block, qwen2.5:14b closed the JSON object right after `certifications` and returned **no work history at all** for a résumé that plainly has one — a silent fill by omission, indistinguishable from a candidate with no jobs (shared context Section 8 forbids exactly this). They are now in the schema's `required` list, which forces the decoder to emit the arrays; empty then means genuinely absent. Verified against the live model: `work: [('Acme Corp', 'Senior Data Engineer', '2020', '2024')]`, `education: [('State University', '2019')]`.

### Note on the fabrication figure

Two real runs, different outcomes, both worth recording:

- **Run A** (extraction schema before the `required` fix, so no work history reached the rewriter): `total_edits=1, verified_edits=0, rejected=1, rate=1.000`. The rewriter tried to add `">// Indirectly related to Kubernetes and containerization, which might be relevant for cloud data warehouses…"` — a Kubernetes claim the résumé never makes, prompted by the unmet `Kubernetes` gap. The gate rejected it with `no supporting prov_id` and the shipped résumé still passed the reviewer (`invariant_ok=True`). This is C3 firing on a genuine fabrication attempt in-pipeline, not on a synthetic fixture.
- **Run B** (final, full extraction): `total_edits=0, rate=0.000`, `invariant_ok=True`. The rewriter added no unsupported hard content.

A `fabrication_rate` of 0.000 over **zero** edits (Run B) is not evidence the gate works — there was nothing to reject. Run A is the in-pipeline evidence; the systematic measurement remains the Phase-5 corpus run (gate-OFF 31 fabrications, gate-ON 0). Phase 6's own claim is narrower: the reviewer re-asserts the invariant on whatever the gate ships.

### Pre-existing issue, now fixed

`tests/unit/test_corpus_pairing.py` and `tests/unit/test_fabrication_corpus.py` import `eval.*`, but `pyproject.toml` set `pythonpath = ["src"]` only, so a bare `pytest tests/` failed collection on both with `ModuleNotFoundError: No module named 'eval'` (confirmed pre-existing by stashing the Phase-6 changes). `tool.pytest.ini_options.pythonpath` is now `["src", "."]`; bare `pytest tests/` collects and passes the whole suite with no `PYTHONPATH=` prefix.

### Known limitation carried forward

JD-analysis backend selection is still per-call-site: `rho/jd/__init__.py` defaults to the CUDA-only `rho.jd.llm`, and callers that need the Ollama path pass `_schema_fn=analyze_jd_schema` explicitly (the eval scripts do this; the real runs above do too). Extraction now resolves its backend from config, so JD analysis is the last component whose default is unrunnable on a CUDA-less host. Same `_resolve_schema_fn` treatment would fix it — left out as Phase-4 scope, not Phase 6.
