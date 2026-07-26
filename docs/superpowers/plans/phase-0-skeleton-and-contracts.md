# Phase 0 — Skeleton & Frozen Contracts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
> **Read `00-SHARED-CONTEXT.md` first.** It holds the thesis, full contract definitions, and conventions this phase makes real.

**Goal:** Boot a FastAPI backend with every Pydantic contract defined and every pipeline component stubbed, wired end-to-end so `/optimize` returns a typed (stub) response.

**Architecture:** Define the frozen data contracts (Section 5 of shared context) as real Pydantic v2 models. Stub each component function (Section 6) with the exact signature, raising `NotImplementedError`. Wire a FastAPI route that calls the stubs and returns a `PipelineResponse` assembled from placeholder objects. One integration test asserts the response *shape*.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Pydantic v2, pytest.

## Global Constraints

- Package name is `rho` (Resume Holistic Optimizer). All imports `from rho...`.
- Contract field names/types are FROZEN after this phase — copy them verbatim from shared context Section 5. Do not rename or retype later.
- Pydantic v2 syntax (`model_config`, `BaseModel`, `| None` unions).
- TDD: failing test → fail → impl → pass → commit.

---

## This phase consumes
Nothing (first phase).

## This phase produces (later phases depend on)
- All Pydantic models importable from `rho.models.*` exactly as in shared-context Section 5.
- Stub functions importable at the paths in shared-context Section 6.
- A running FastAPI app with `GET /health` and `POST /optimize`.

---

## File Structure

- Create: `pyproject.toml` — package + deps.
- Create: `src/rho/__init__.py`
- Create: `src/rho/config.py` — settings.
- Create: `src/rho/models/{__init__,provenance,resume,jd,scoring,rewrite,api}.py`
- Create: `src/rho/{ingestion,extraction,jd,matching,ats,rewrite,graph}/__init__.py` — stub functions.
- Create: `src/rho/api/app.py` — FastAPI app + routes.
- Create: `tests/unit/test_models.py`, `tests/integration/test_optimize_shape.py`
- Create: `tests/fixtures/.gitkeep`

---

### Task 1: Project scaffold + package boots

**Files:**
- Create: `pyproject.toml`, `src/rho/__init__.py`, `src/rho/config.py`
- Test: `tests/unit/test_boot.py`

**Interfaces:**
- Produces: importable package `rho`; `rho.config.settings` object.

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/test_boot.py
def test_package_imports():
    import rho
    from rho.config import settings
    assert settings.app_name == "rho"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_boot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rho'`

- [ ] **Step 3: Create pyproject + package**
```toml
# pyproject.toml
[project]
name = "rho"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["fastapi>=0.110", "uvicorn>=0.29", "pydantic>=2.6", "pydantic-settings>=2.2", "python-multipart>=0.0.9"]
[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27", "hypothesis>=6"]
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
[tool.pytest.ini_options]
pythonpath = ["src"]
```
```python
# src/rho/__init__.py
__version__ = "0.1.0"
```
```python
# src/rho/config.py
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    app_name: str = "rho"
    extraction_model: str = "Qwen/Qwen3-0.6B"      # override in P2
    temperature: float = 0.2
settings = Settings()
```

- [ ] **Step 4: Install + run**
Run: `pip install -e ".[dev]"` then `pytest tests/unit/test_boot.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "chore: scaffold rho package + config"
```

---

### Task 2: Provenance models

**Files:**
- Create: `src/rho/models/__init__.py`, `src/rho/models/provenance.py`
- Test: `tests/unit/test_models.py`

**Interfaces:**
- Produces: `SourceSpan`, `ProvenanceMap` with `add()` returning `p:<doc>:<seq>` and `get()`.

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/test_models.py
from rho.models.provenance import SourceSpan, ProvenanceMap
def test_provmap_add_and_get():
    pm = ProvenanceMap(doc_id="d1")
    pid = pm.add(SourceSpan(doc_id="d1", char_start=0, char_end=5, raw_text="Alice"))
    assert pid == "p:d1:0"
    assert pm.get(pid).raw_text == "Alice"
    pid2 = pm.add(SourceSpan(doc_id="d1", char_start=6, char_end=9, raw_text="Bob"))
    assert pid2 == "p:d1:1"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_models.py::test_provmap_add_and_get -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```python
# src/rho/models/provenance.py
from pydantic import BaseModel
class SourceSpan(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    raw_text: str
class ProvenanceMap(BaseModel):
    doc_id: str
    spans: dict[str, SourceSpan] = {}
    def add(self, span: SourceSpan) -> str:
        pid = f"p:{self.doc_id}:{len(self.spans)}"
        self.spans[pid] = span
        return pid
    def get(self, prov_id: str) -> SourceSpan:
        return self.spans[prov_id]
```
```python
# src/rho/models/__init__.py
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_models.py::test_provmap_add_and_get -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: provenance SourceSpan + ProvenanceMap"
```

---

### Task 3: Resume, JD, scoring, rewrite, api models

**Files:**
- Create: `src/rho/models/{resume,jd,scoring,rewrite,api}.py`
- Test: append to `tests/unit/test_models.py`

**Interfaces:**
- Produces: every model in shared-context Section 5. Copy field names/types EXACTLY.

- [ ] **Step 1: Write the failing test**
```python
# append to tests/unit/test_models.py
from rho.models.resume import StructuredResume, WorkExperience
from rho.models.jd import RequirementSet, Requirement
from rho.models.scoring import MatchResult, ComponentVector, Gap
from rho.models.rewrite import TailoredResume, FabricationReport
from rho.models.api import PipelineResponse
def test_models_construct_with_provenance_fields():
    w = WorkExperience(company="Acme", title="Eng", company_prov=["p:d1:0"])
    r = StructuredResume(name="Alice", work=[w], skills=["python"], skills_prov=[["p:d1:1"]])
    assert r.work[0].company_prov == ["p:d1:0"]
    fr = FabricationReport(total_edits=2, verified_edits=2, fabrication_rate=0.0)
    assert fr.fabrication_rate == 0.0
    cv = ComponentVector(keyword_coverage=0.5, semantic_similarity=0.5,
                         fuzzy_coverage=0.5, must_have_coverage=0.5, nice_have_coverage=0.5)
    mr = MatchResult(component_vector=cv, predicted_score=0.0)
    assert mr.predicted_score == 0.0
```

- [ ] **Step 2: Run to verify it fails**
Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement all five model files**
Copy verbatim from shared-context Section 5 (`resume.py`, `jd.py`, `scoring.py`, `rewrite.py`, `api.py`). Add needed imports: `from typing import Literal`; `from pydantic import BaseModel`; cross-file imports (`from rho.models.provenance import ProvenanceMap`, `from rho.models.resume import StructuredResume`, etc.). For `api.py`'s `PipelineResponse` import `ProvenanceMap`, `MatchResult`, `TailoredResume`, `StructuredResume`.

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_models.py -v`
Expected: PASS (all model tests)

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: resume/jd/scoring/rewrite/api Pydantic contracts"
```

---

### Task 4: Component stubs

**Files:**
- Create: `src/rho/{ingestion,extraction,jd,matching,ats,rewrite,graph}/__init__.py`
- Test: `tests/unit/test_stubs.py`

**Interfaces:**
- Produces: functions at shared-context Section 6 paths, each raising `NotImplementedError`. These signatures are frozen; later phases replace the body only.

- [ ] **Step 1: Write the failing test**
```python
# tests/unit/test_stubs.py
import pytest
def test_stubs_raise_not_implemented():
    from rho.ingestion import ingest
    from rho.extraction import extract
    from rho.jd import analyze_jd
    from rho.matching import match
    from rho.ats import harvest_ats, Calibrator
    from rho.rewrite import rewrite, verify
    from rho.graph import run_pipeline
    with pytest.raises(NotImplementedError):
        ingest(b"", "x.pdf")
```

- [ ] **Step 2: Run to verify it fails**
Run: `pytest tests/unit/test_stubs.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement stubs**
Each `__init__.py` defines the exact-signature function(s) from Section 6, body `raise NotImplementedError`. Example:
```python
# src/rho/ingestion/__init__.py
from rho.models.provenance import ProvenanceMap
def ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]:
    raise NotImplementedError
```
Do the same for `extraction.extract`, `jd.analyze_jd`, `matching.match`, `ats.harvest_ats`, `ats.Calibrator` (class with `fit`/`predict` raising `NotImplementedError`), `rewrite.rewrite`, `rewrite.verify`, `graph.run_pipeline`. Import the referenced models in each file.

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_stubs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: stub pipeline component signatures"
```

---

### Task 5: FastAPI app + `/health` + `/optimize` shape

**Files:**
- Create: `src/rho/api/__init__.py`, `src/rho/api/app.py`
- Test: `tests/integration/test_optimize_shape.py`

**Interfaces:**
- Consumes: all models + stubs above.
- Produces: `app` (FastAPI); `GET /health` → `{"status":"ok"}`; `POST /optimize` (multipart: file + `jd_text`) → `PipelineResponse` JSON. In P0 it returns a hand-built placeholder `PipelineResponse` (does NOT call stubs yet — they raise). Wiring to `run_pipeline` happens in P6.

- [ ] **Step 1: Write the failing test**
```python
# tests/integration/test_optimize_shape.py
from fastapi.testclient import TestClient
from rho.api.app import app
client = TestClient(app)
def test_health():
    assert client.get("/health").json() == {"status": "ok"}
def test_optimize_returns_pipeline_shape():
    r = client.post("/optimize",
        files={"file": ("r.txt", b"Alice\npython", "text/plain")},
        data={"jd_text": "need python"})
    assert r.status_code == 200
    body = r.json()
    for key in ["structured_resume","provenance_map","match_result","tailored_resume","final_score"]:
        assert key in body
```

- [ ] **Step 2: Run to verify it fails**
Run: `pytest tests/integration/test_optimize_shape.py -v`
Expected: FAIL — `rho.api.app` not found.

- [ ] **Step 3: Implement app with placeholder response**
```python
# src/rho/api/app.py
from fastapi import FastAPI, UploadFile, Form
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport
from rho.models.api import PipelineResponse
app = FastAPI(title="rho")

@app.get("/health")
def health():
    return {"status": "ok"}

def _placeholder_response() -> PipelineResponse:
    resume = StructuredResume(name="")
    cv = ComponentVector(keyword_coverage=0, semantic_similarity=0,
        fuzzy_coverage=0, must_have_coverage=0, nice_have_coverage=0)
    return PipelineResponse(
        structured_resume=resume,
        provenance_map=ProvenanceMap(doc_id="d0"),
        match_result=MatchResult(component_vector=cv, predicted_score=0.0),
        tailored_resume=TailoredResume(resume=resume,
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0, fabrication_rate=0.0)),
        final_score=0.0,
    )

@app.post("/optimize", response_model=PipelineResponse)
async def optimize(file: UploadFile, jd_text: str = Form(...)):
    _ = await file.read()          # consumed; real wiring in P6
    return _placeholder_response()
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/integration/test_optimize_shape.py -v`
Expected: PASS both tests.

- [ ] **Step 5: Boot check + commit**
Run: `uvicorn rho.api.app:app --port 8000` (Ctrl-C after confirming it starts), then:
```bash
git add -A && git commit -m "feat: FastAPI app with /health and /optimize shape"
```

---

## Self-Review checklist (run before marking phase done)
- [ ] All models import from `rho.models.*` with exact Section-5 field names/types.
- [ ] All stubs import from Section-6 paths and raise `NotImplementedError`.
- [ ] `pytest` full run green: `pytest -v`.
- [ ] `uvicorn rho.api.app:app` boots.

## Results (fill in when done — read by final verification session)
- Tests passing: 6 / 6 (`pytest -v` green)
- Contracts frozen: yes — all Section-5 models live in `rho.models.*` with verbatim field names/types.
- Notes / deviations:
  - Env: `uv venv` with **CPython 3.13.14** (system python is 3.14, too new for the dep set). Deps installed via `uv pip install -e ".[dev]"`. Run tests with `.venv/bin/python -m pytest`.
  - Added `.gitignore` (`.venv/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`) — not in plan, needed to keep the venv out of git.
  - Uvicorn boot verified on port 8123: `GET /health` → `{"status":"ok"}`.
  - Known warning (harmless): `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
  - `POST /optimize` returns a hand-built placeholder `PipelineResponse`; stubs are not called (they raise). Real wiring lands in P6.
