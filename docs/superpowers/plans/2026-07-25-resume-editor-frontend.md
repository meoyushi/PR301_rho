# Résumé Editor Frontend + API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app to upload a résumé, display it parsed, edit every part (text, bullets, skills, section order, visual styling), and optimise its ATS-match score against a pasted job description with one button — running the provenance-verified pipeline so bullets are tailored to the JD and nothing is fabricated.

**Architecture:** Two new FastAPI endpoints (`/parse` synchronous, `/optimize` async job + polling) over the existing `rho` pipeline, plus a Next.js frontend in `web/` with a two-pane editor/preview. Optimize enters the pipeline at `match` from an edited résumé (skipping ingest/extract), rebuilding provenance from the edited résumé's own values.

**Tech Stack:** FastAPI, Pydantic v2, Python threading (backend); Next.js App Router, TypeScript, Tailwind, Zustand, Vitest (frontend). Gemini backend for LLM calls.

**Spec:** `../specs/2026-07-25-resume-editor-frontend-design.md`

## Global Constraints

- Backend LLM calls use the **Gemini** backend (the one Phase 7 ran end-to-end). `analyze_jd`, `extract`, and `rewrite` default to CUDA-only backends, so the Gemini schema functions MUST be injected explicitly (`analyze_jd_schema_gemini`, `extract_schema_gemini`, `rewrite_schema_gemini`).
- **No silent fills** (shared context §8): a failed job sets `state="error"` with the message; never a fake result or a swallowed exception.
- **No new Python dependencies.** FastAPI, Pydantic, threading are already present.
- Visual styling (fonts, margins, spacing, order) is **frontend-only CSS** — never sent to or stored by the backend.
- Frontend dev server runs on `localhost:3000`; backend allows it via CORS.
- Package name is `rho` everywhere in Python.
- No `Co-Authored-By: Claude` trailer on commits (project CLAUDE.md).

---

## File Structure

**Backend (create):**
- `src/rho/api/entry.py` — `run_from_structured(resume, jd_text)` composing the pipeline from `match` onward with Gemini backends; `build_prov_from_resume(resume)`.
- `src/rho/api/jobs.py` — in-process `JobStore` + background worker.
- `tests/integration/test_parse_endpoint.py`, `tests/integration/test_optimize_job.py`, `tests/unit/test_entry.py`, `tests/unit/test_jobs.py`.

**Backend (modify):**
- `src/rho/models/api.py` — add `ParseResponse`, `OptimizeJobRequest`, `OptimizeResult`, `JobStatus`.
- `src/rho/api/app.py` — add `/parse`, `POST /optimize`, `GET /optimize/{id}`, CORS.

**Frontend (create, all under `web/`):**
- `package.json`, `next.config.mjs`, `tsconfig.json`, `tailwind.config.ts`, `postcss.config.mjs`, `vitest.config.ts`, `app/globals.css`, `app/layout.tsx`, `app/page.tsx`.
- `lib/types.ts`, `lib/api.ts`, `lib/resumeStore.ts`.
- `components/Editor/*`, `components/Preview/ResumePreview.tsx`, `components/UploadDropzone.tsx`.
- Test files colocated: `lib/resumeStore.test.ts`, `lib/api.test.ts`, `components/Preview/ResumePreview.test.tsx`.

---

### Task 1: Pipeline entry from a structured résumé

**Files:**
- Create: `src/rho/api/entry.py`
- Test: `tests/unit/test_entry.py`

**Interfaces:**
- Consumes: `rho.graph.nodes` node functions; `rho.models.resume.StructuredResume`; `rho.models.api.PipelineResponse`.
- Produces: `run_from_structured(resume: StructuredResume, jd_text: str, *, jd_fn=None, rewrite_fn=None) -> PipelineResponse`; `build_prov_from_resume(resume: StructuredResume) -> ProvenanceMap`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_entry.py
from rho.api.entry import build_prov_from_resume, run_from_structured
from rho.models.resume import StructuredResume, WorkExperience
from rho.models.jd import RequirementSet, Requirement
from rho.models.scoring import MatchResult, ComponentVector


def _resume():
    return StructuredResume(
        name="Jane Doe",
        skills=["Python", "AWS"],
        work=[WorkExperience(company="Acme", title="Engineer", bullets=["Built pipelines in Python"])],
    )


def test_build_prov_covers_every_value():
    prov = build_prov_from_resume(_resume())
    joined = " ".join(s.raw_text for s in prov.spans.values()).lower()
    assert "python" in joined and "acme" in joined and "jane doe" in joined


def test_run_from_structured_enters_at_match_and_returns_full_response(monkeypatch):
    # Stub the two LLM legs so no network is touched.
    reqs = RequirementSet(requirements=[Requirement(text="Python", kind="skill", priority="must")])
    monkeypatch.setattr("rho.api.entry.analyze_jd", lambda jd, _schema_fn=None: reqs)

    from rho.models.rewrite import TailoredResume, FabricationReport
    def fake_rewrite(resume, gaps, prov, _rewrite_fn=None):
        return TailoredResume(resume=resume, fabrication_report=FabricationReport(total_edits=0, verified_edits=0))
    monkeypatch.setattr("rho.api.entry.rewrite", fake_rewrite)

    resp = run_from_structured(_resume(), "We need a Python engineer.")
    assert resp.structured_resume.name == "Jane Doe"
    assert isinstance(resp.match_result, MatchResult)
    assert resp.tailored_resume.fabrication_report.total_edits == 0
    assert 0.0 <= resp.final_score <= 100.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_entry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rho.api.entry'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rho/api/entry.py
"""Run the pipeline from an already-structured (edited) résumé.

The graph in `rho.graph` starts from file bytes: ingest -> extract -> ... The
editor sends back a résumé the user has already parsed and edited, so this entry
skips ingest/extract and composes the remaining stages (jd, match, score,
rewrite, review) directly from the same node functions the graph uses.

Provenance is rebuilt from the edited résumé's own values via the real ingest
path (the technique `eval/fabrication_corpus.py` uses), so the C3 gate still has
spans to verify against. Consequence, surfaced in the UI: "sourced" now means
"traces to the current résumé", not "traces to the original upload".

LLM legs use the Gemini backend explicitly: `analyze_jd`/`rewrite` default to
CUDA-only backends otherwise.
"""

from pathlib import Path

from rho.graph import nodes as N
from rho.ingestion import ingest
from rho.jd import analyze_jd
from rho.models.api import PipelineResponse
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.rewrite import rewrite


def build_prov_from_resume(resume: StructuredResume) -> ProvenanceMap:
    """ProvenanceMap over the résumé's own values, one span per line."""
    lines = [resume.name, resume.headline or "", resume.summary or "", *resume.skills, *resume.certifications]
    for w in resume.work:
        lines += [w.company, w.title, w.start_date or "", w.end_date or "", *w.bullets]
    for e in resume.education:
        lines += [e.institution, e.degree or "", e.field or "", e.end_year or ""]
    doc = "\n".join(ln.strip() for ln in lines if ln and ln.strip())
    _, prov = ingest(doc.encode(), "edited.txt")
    return prov


def _gemini_jd_fn():
    from rho.jd.gemini import analyze_jd_schema_gemini
    return analyze_jd_schema_gemini


def _gemini_rewrite_fn():
    from rho.rewrite.gemini import rewrite_schema_gemini
    return rewrite_schema_gemini


def run_from_structured(
    resume: StructuredResume,
    jd_text: str,
    *,
    jd_fn=None,
    rewrite_fn=None,
    on_stage=None,
) -> PipelineResponse:
    """Score and tailor an edited résumé against `jd_text`.

    `on_stage(name)` is called before each stage so a caller (the job worker)
    can report progress. `jd_fn`/`rewrite_fn` override the Gemini defaults in
    tests.
    """
    def stage(name):
        if on_stage:
            on_stage(name)

    prov = build_prov_from_resume(resume)

    stage("analyzing_jd")
    reqs = analyze_jd(jd_text, _schema_fn=jd_fn or _gemini_jd_fn())

    stage("matching")
    state = {"resume": resume, "reqs": reqs, "prov": prov}
    state.update(N.match_node(state))

    stage("scoring")
    state.update(N.score_node(state))

    stage("rewriting")
    tailored = rewrite(resume, state["match_result"].gaps, prov, _rewrite_fn=rewrite_fn or _gemini_rewrite_fn())
    state["tailored"] = tailored

    stage("reviewing")
    state.update(N.review_node(state))

    return PipelineResponse(
        structured_resume=resume,
        provenance_map=prov,
        match_result=state["match_result"],
        tailored_resume=tailored,
        final_score=state["final_score"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_entry.py -v`
Expected: PASS (2 tests). If `analyze_jd` is called despite the monkeypatch, confirm the patch targets `rho.api.entry.analyze_jd` and that `rewrite` is patched as `rho.api.entry.rewrite`.

- [ ] **Step 5: Commit**

```bash
git add src/rho/api/entry.py tests/unit/test_entry.py
git commit -m "feat: run_from_structured pipeline entry for edited resumes"
```

---

### Task 2: API response/request models

**Files:**
- Modify: `src/rho/models/api.py`
- Test: `tests/unit/test_api_models.py`

**Interfaces:**
- Consumes: existing `StructuredResume`, `ProvenanceMap`, `MatchResult`, `TailoredResume`.
- Produces: `ParseResponse`, `OptimizeJobRequest`, `OptimizeResult`, `JobStatus`, `JobState` (Literal).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_api_models.py
from rho.models.api import JobStatus, OptimizeJobRequest, OptimizeResult, ParseResponse
from rho.models.resume import StructuredResume


def test_parse_response_holds_resume_and_prov():
    r = ParseResponse(structured_resume=StructuredResume(name="X"), provenance_map={"doc_id": "d", "spans": {}})
    assert r.structured_resume.name == "X"


def test_optimize_request_requires_resume_and_jd():
    req = OptimizeJobRequest(resume=StructuredResume(name="X"), jd_text="jd")
    assert req.jd_text == "jd"


def test_job_status_defaults_to_queued_with_no_result():
    js = JobStatus(id="abc")
    assert js.state == "queued" and js.result is None and js.error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_api_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'JobStatus'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/rho/models/api.py`:

```python
from typing import Literal

JobState = Literal["queued", "running", "done", "error"]


class ParseResponse(BaseModel):
    structured_resume: StructuredResume
    provenance_map: ProvenanceMap


class OptimizeJobRequest(BaseModel):
    resume: StructuredResume
    jd_text: str


class OptimizeResult(BaseModel):
    match_result: MatchResult
    tailored_resume: TailoredResume
    final_score: float
    previous_score: float | None = None


class JobStatus(BaseModel):
    id: str
    state: JobState = "queued"
    stage: str | None = None
    result: OptimizeResult | None = None
    error: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_api_models.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rho/models/api.py tests/unit/test_api_models.py
git commit -m "feat: api models for parse + optimize job"
```

---

### Task 3: In-process job store + background worker

**Files:**
- Create: `src/rho/api/jobs.py`
- Test: `tests/unit/test_jobs.py`

**Interfaces:**
- Consumes: `run_from_structured` (Task 1), `JobStatus`/`OptimizeResult`/`OptimizeJobRequest` (Task 2).
- Produces: `JobStore` with `create(req: OptimizeJobRequest, runner=...) -> str`, `get(job_id) -> JobStatus | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_jobs.py
import time

from rho.api.jobs import JobStore
from rho.models.api import OptimizeJobRequest, PipelineResponse
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport
from rho.models.provenance import ProvenanceMap


def _fake_response():
    return PipelineResponse(
        structured_resume=StructuredResume(name="X"),
        provenance_map=ProvenanceMap(doc_id="d"),
        match_result=MatchResult(component_vector=ComponentVector(
            keyword_coverage=0, semantic_similarity=0, fuzzy_coverage=0,
            must_have_coverage=0, nice_have_coverage=0), predicted_score=70.0),
        tailored_resume=TailoredResume(resume=StructuredResume(name="X"),
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0)),
        final_score=70.0,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rho.api.jobs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rho/api/jobs.py
"""In-process async job store for the optimise pipeline.

Single-user local tool: jobs live in a dict and run on daemon threads. A failed
job is recorded as state="error" with the exception message — never swallowed,
per shared context §8. `runner` is injectable so tests drive the store without
an LLM.
"""

import threading
import uuid

from rho.api.entry import run_from_structured
from rho.models.api import JobStatus, OptimizeJobRequest, OptimizeResult


def _default_runner(req: OptimizeJobRequest, on_stage):
    return run_from_structured(req.resume, req.jd_text, on_stage=on_stage)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create(self, req: OptimizeJobRequest, runner=_default_runner) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = JobStatus(id=job_id, state="queued")
        threading.Thread(target=self._run, args=(job_id, req, runner), daemon=True).start()
        return job_id

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _set(self, job_id: str, **fields) -> None:
        with self._lock:
            current = self._jobs[job_id]
            self._jobs[job_id] = current.model_copy(update=fields)

    def _run(self, job_id: str, req: OptimizeJobRequest, runner) -> None:
        self._set(job_id, state="running")
        try:
            resp = runner(req, lambda name: self._set(job_id, stage=name))
            result = OptimizeResult(
                match_result=resp.match_result,
                tailored_resume=resp.tailored_resume,
                final_score=resp.final_score,
            )
            self._set(job_id, state="done", stage="done", result=result)
        except Exception as exc:  # a dead model must not look like a clean run
            self._set(job_id, state="error", error=f"{type(exc).__name__}: {exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_jobs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rho/api/jobs.py tests/unit/test_jobs.py
git commit -m "feat: in-process job store for async optimize"
```

---

### Task 4: Wire endpoints into the FastAPI app

**Files:**
- Modify: `src/rho/api/app.py`
- Test: `tests/integration/test_parse_endpoint.py`, `tests/integration/test_optimize_job.py`

**Interfaces:**
- Consumes: `JobStore` (Task 3), `run_from_structured` (Task 1), `ingest`, `extract`, models (Task 2).
- Produces: routes `POST /parse`, `POST /optimize`, `GET /optimize/{job_id}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_parse_endpoint.py
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
```

```python
# tests/integration/test_optimize_job.py
from fastapi.testclient import TestClient

import rho.api.app as appmod
from rho.api.app import app
from rho.models.api import PipelineResponse
from rho.models.resume import StructuredResume
from rho.models.provenance import ProvenanceMap
from rho.models.scoring import MatchResult, ComponentVector
from rho.models.rewrite import TailoredResume, FabricationReport


def _resp():
    return PipelineResponse(
        structured_resume=StructuredResume(name="X"),
        provenance_map=ProvenanceMap(doc_id="d"),
        match_result=MatchResult(component_vector=ComponentVector(
            keyword_coverage=0, semantic_similarity=0, fuzzy_coverage=0,
            must_have_coverage=0, nice_have_coverage=0), predicted_score=80.0),
        tailored_resume=TailoredResume(resume=StructuredResume(name="X"),
            fabrication_report=FabricationReport(total_edits=0, verified_edits=0)),
        final_score=80.0,
    )


def test_optimize_job_lifecycle(monkeypatch):
    # Replace the store's default runner path by patching run_from_structured.
    monkeypatch.setattr("rho.api.jobs.run_from_structured",
                        lambda resume, jd_text, on_stage=None: _resp())
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


def test_optimize_unknown_job_404():
    assert TestClient(app).get("/optimize/nope").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/integration/test_parse_endpoint.py tests/integration/test_optimize_job.py -v`
Expected: FAIL (`/parse` 404 / no route).

- [ ] **Step 3: Write minimal implementation**

Replace `src/rho/api/app.py` with:

```python
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rho.api.jobs import JobStore
from rho.extraction import extract
from rho.ingestion import ingest
from rho.models.api import JobStatus, OptimizeJobRequest, ParseResponse

app = FastAPI(title="rho")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs = JobStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
async def parse(file: UploadFile):
    data = await file.read()
    try:
        md, prov = ingest(data, file.filename or "resume.txt")
        resume = extract(md, prov)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"parse failed: {exc}")
    return ParseResponse(structured_resume=resume, provenance_map=prov)


@app.post("/optimize", response_model=JobStatus)
def optimize(req: OptimizeJobRequest):
    job_id = _jobs.create(req)
    return _jobs.get(job_id)


@app.get("/optimize/{job_id}", response_model=JobStatus)
def optimize_status(job_id: str):
    js = _jobs.get(job_id)
    if js is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return js
```

Note: the old `/optimize` (file upload → `run_pipeline`) is intentionally replaced; the frontend uses `/parse` then `/optimize` on the edited résumé. If `tests/integration/test_optimize_shape.py` asserts the old shape, update it to the new job shape (it takes the `stub_nodes` fixture; assert `state`/`id` instead of the old `PipelineResponse` fields).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/integration/test_parse_endpoint.py tests/integration/test_optimize_job.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole backend suite; fix the one pre-existing optimize-shape test if it breaks**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS. If `test_optimize_shape.py` fails on the removed route, rewrite its assertion to the job shape (`state == "queued"` and an `id` present), keeping its `stub_nodes` fixture.

- [ ] **Step 6: Commit**

```bash
git add src/rho/api/app.py tests/integration/test_parse_endpoint.py tests/integration/test_optimize_job.py tests/integration/test_optimize_shape.py
git commit -m "feat: /parse and async /optimize endpoints with CORS"
```

---

### Task 5: Frontend scaffold + types + API client

**Files:**
- Create: `web/package.json`, `web/next.config.mjs`, `web/tsconfig.json`, `web/tailwind.config.ts`, `web/postcss.config.mjs`, `web/vitest.config.ts`, `web/app/globals.css`, `web/app/layout.tsx`, `web/lib/types.ts`, `web/lib/api.ts`
- Test: `web/lib/api.test.ts`

**Interfaces:**
- Produces: `parseResume(file: File): Promise<ParseResponse>`, `startOptimize(resume, jdText): Promise<{id: string}>`, `pollOptimize(jobId, {intervalMs, timeoutMs}): Promise<JobStatus>`; TS types mirroring the backend models. Base URL from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

- [ ] **Step 1: Scaffold config files**

`web/package.json`:
```json
{
  "name": "rho-web",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "vitest run"
  },
  "dependencies": {
    "next": "^14.2.5",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^4.5.4"
  },
  "devDependencies": {
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.4.8",
    "@types/node": "^20",
    "@types/react": "^18",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.19",
    "jsdom": "^24.1.1",
    "postcss": "^8.4.40",
    "tailwindcss": "^3.4.7",
    "typescript": "^5.5.4",
    "vitest": "^2.0.5"
  }
}
```

`web/next.config.mjs`:
```js
/** @type {import('next').NextConfig} */
export default { reactStrictMode: true };
```

`web/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020", "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false, "strict": true, "noEmit": true, "esModuleInterop": true,
    "module": "esnext", "moduleResolution": "bundler", "resolveJsonModule": true,
    "isolatedModules": true, "jsx": "preserve", "incremental": true,
    "paths": { "@/*": ["./*"] },
    "plugins": [{ "name": "next" }]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`web/tailwind.config.ts`:
```ts
import type { Config } from "tailwindcss";
export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
} satisfies Config;
```

`web/postcss.config.mjs`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`web/vitest.config.ts`:
```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true },
});
```

`web/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

`web/app/layout.tsx`:
```tsx
import "./globals.css";
export const metadata = { title: "rho — résumé editor" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
```

- [ ] **Step 2: Write `web/lib/types.ts`**

```ts
// Mirrors the rho backend Pydantic models (partial: only fields the UI uses).
export interface WorkExperience {
  company: string; title: string;
  start_date?: string | null; end_date?: string | null;
  bullets: string[]; bullet_prov?: string[][];
  company_prov?: string[]; title_prov?: string[];
}
export interface Education {
  institution: string; degree?: string | null; field?: string | null; end_year?: string | null;
}
export interface StructuredResume {
  name: string; headline?: string | null; summary?: string | null;
  emails: string[]; phones: string[]; urls: string[];
  work: WorkExperience[]; education: Education[];
  skills: string[]; certifications: string[];
  skills_prov?: string[][];
}
export interface ParseResponse { structured_resume: StructuredResume; provenance_map: unknown; }
export interface Gap { requirement: { text: string; priority: string }; status: string; }
export interface MatchResult { predicted_score: number; gaps: Gap[]; }
export interface FabricationReport { total_edits: number; verified_edits: number; fabrication_rate?: number; rejected_edits: { added_text: string; reason: string }[]; }
export interface TailoredResume { resume: StructuredResume; fabrication_report: FabricationReport; }
export interface OptimizeResult { match_result: MatchResult; tailored_resume: TailoredResume; final_score: number; previous_score?: number | null; }
export type JobState = "queued" | "running" | "done" | "error";
export interface JobStatus { id: string; state: JobState; stage?: string | null; result?: OptimizeResult | null; error?: string | null; }
```

- [ ] **Step 3: Write the failing api-client test**

```ts
// web/lib/api.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { pollOptimize, startOptimize } from "./api";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("startOptimize posts resume + jd and returns the job id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "j1", state: "queued" }) });
    vi.stubGlobal("fetch", fetchMock);
    const res = await startOptimize({ name: "X" } as any, "jd");
    expect(res.id).toBe("j1");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("pollOptimize resolves when the job reaches done", async () => {
    const states = [
      { id: "j1", state: "running", stage: "matching" },
      { id: "j1", state: "done", result: { final_score: 77 } },
    ];
    const fetchMock = vi.fn().mockImplementation(async () => ({ ok: true, json: async () => states.shift() }));
    vi.stubGlobal("fetch", fetchMock);
    const js = await pollOptimize("j1", { intervalMs: 1, timeoutMs: 1000 });
    expect(js.state).toBe("done");
    expect(js.result!.final_score).toBe(77);
  });

  it("pollOptimize rejects on the error state", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ id: "j1", state: "error", error: "model down" }) });
    vi.stubGlobal("fetch", fetchMock);
    await expect(pollOptimize("j1", { intervalMs: 1, timeoutMs: 1000 })).rejects.toThrow("model down");
  });
});
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd web && npm install && npm test`
Expected: FAIL (`./api` has no `startOptimize`/`pollOptimize`).

- [ ] **Step 5: Write `web/lib/api.ts`**

```ts
import type { JobStatus, ParseResponse, StructuredResume } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class BackendUnreachable extends Error {}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.json() as Promise<T>;
}

export async function parseResume(file: File): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  let res: Response;
  try { res = await fetch(`${BASE}/parse`, { method: "POST", body: form }); }
  catch { throw new BackendUnreachable("backend unreachable"); }
  return json<ParseResponse>(res);
}

export async function startOptimize(resume: StructuredResume, jdText: string): Promise<JobStatus> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/optimize`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume, jd_text: jdText }),
    });
  } catch { throw new BackendUnreachable("backend unreachable"); }
  return json<JobStatus>(res);
}

export async function pollOptimize(
  jobId: string,
  { intervalMs = 1000, timeoutMs = 120000 }: { intervalMs?: number; timeoutMs?: number } = {},
): Promise<JobStatus> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const js = await json<JobStatus>(await fetch(`${BASE}/optimize/${jobId}`));
    if (js.state === "done") return js;
    if (js.state === "error") throw new Error(js.error ?? "optimize failed");
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("optimize timed out");
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd web && npm test`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add web/package.json web/next.config.mjs web/tsconfig.json web/tailwind.config.ts web/postcss.config.mjs web/vitest.config.ts web/app/globals.css web/app/layout.tsx web/lib/types.ts web/lib/api.ts web/lib/api.test.ts web/package-lock.json
git commit -m "feat: web scaffold, backend types, api client with poll loop"
```

---

### Task 6: Editor store (Zustand)

**Files:**
- Create: `web/lib/resumeStore.ts`
- Test: `web/lib/resumeStore.test.ts`

**Interfaces:**
- Consumes: `StructuredResume`, `OptimizeResult` (types).
- Produces: `useResumeStore` with state `{resume, style, optimize}` and actions `setResume`, `setField`, `addBullet`, `editBullet`, `removeBullet`, `addSkill`, `removeSkill`, `setStyle`, `setOptimize`, `applyTailored`.

- [ ] **Step 1: Write the failing test**

```ts
// web/lib/resumeStore.test.ts
import { beforeEach, describe, expect, it } from "vitest";
import { useResumeStore } from "./resumeStore";

const base = () => ({
  name: "Jane", headline: null, summary: null, emails: [], phones: [], urls: [],
  work: [{ company: "Acme", title: "Eng", bullets: ["Built X"] }],
  education: [], skills: ["python"], certifications: [],
});

beforeEach(() => useResumeStore.getState().setResume(base() as any));

describe("resume store", () => {
  it("edits a top-level field", () => {
    useResumeStore.getState().setField("summary", "Senior engineer");
    expect(useResumeStore.getState().resume!.summary).toBe("Senior engineer");
  });

  it("adds and removes a bullet on a work entry", () => {
    useResumeStore.getState().addBullet(0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toHaveLength(2);
    useResumeStore.getState().editBullet(0, 1, "Led Y");
    expect(useResumeStore.getState().resume!.work[0].bullets[1]).toBe("Led Y");
    useResumeStore.getState().removeBullet(0, 0);
    expect(useResumeStore.getState().resume!.work[0].bullets).toEqual(["Led Y"]);
  });

  it("adds and removes skills without duplicates", () => {
    useResumeStore.getState().addSkill("python"); // dup ignored
    useResumeStore.getState().addSkill("aws");
    expect(useResumeStore.getState().resume!.skills).toEqual(["python", "aws"]);
    useResumeStore.getState().removeSkill("python");
    expect(useResumeStore.getState().resume!.skills).toEqual(["aws"]);
  });

  it("updates style settings", () => {
    useResumeStore.getState().setStyle({ fontSize: 12 });
    expect(useResumeStore.getState().style.fontSize).toBe(12);
  });

  it("applyTailored swaps in tailored resume and records previous score", () => {
    useResumeStore.getState().applyTailored(
      { name: "Jane", work: [{ company: "Acme", title: "Eng", bullets: ["Built X in Python"] }], skills: ["python"], emails: [], phones: [], urls: [], education: [], certifications: [] } as any,
      88, 60,
    );
    const s = useResumeStore.getState();
    expect(s.resume!.work[0].bullets[0]).toBe("Built X in Python");
    expect(s.optimize?.score).toBe(88);
    expect(s.optimize?.previousScore).toBe(60);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm test resumeStore`
Expected: FAIL (`./resumeStore` has no `useResumeStore`).

- [ ] **Step 3: Write `web/lib/resumeStore.ts`**

```ts
import { create } from "zustand";
import type { StructuredResume } from "./types";

export interface StyleSettings {
  fontSize: number; margin: number; lineSpacing: number; accent: string;
  sectionOrder: string[];
}
export interface OptimizeView {
  score: number; previousScore: number | null;
  gaps: { text: string; priority: string; status: string }[];
  fabricationsBlocked: number;
  originalResume: StructuredResume; // pre-optimize, for before/after
}

interface State {
  resume: StructuredResume | null;
  provenance: unknown;
  style: StyleSettings;
  optimize: OptimizeView | null;
  setResume: (r: StructuredResume) => void;
  setProvenance: (p: unknown) => void;
  setField: <K extends keyof StructuredResume>(k: K, v: StructuredResume[K]) => void;
  addBullet: (workIdx: number) => void;
  editBullet: (workIdx: number, bulletIdx: number, text: string) => void;
  removeBullet: (workIdx: number, bulletIdx: number) => void;
  addSkill: (s: string) => void;
  removeSkill: (s: string) => void;
  setStyle: (patch: Partial<StyleSettings>) => void;
  applyTailored: (tailored: StructuredResume, score: number, previousScore: number | null) => void;
  setGaps: (gaps: OptimizeView["gaps"], fabricationsBlocked: number) => void;
}

const DEFAULT_STYLE: StyleSettings = {
  fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#2563eb",
  sectionOrder: ["summary", "skills", "work", "education"],
};

function mutate(r: StructuredResume, fn: (draft: StructuredResume) => void): StructuredResume {
  const copy: StructuredResume = JSON.parse(JSON.stringify(r));
  fn(copy);
  return copy;
}

export const useResumeStore = create<State>((set, get) => ({
  resume: null, provenance: null, style: DEFAULT_STYLE, optimize: null,
  setResume: (r) => set({ resume: r, optimize: null }),
  setProvenance: (p) => set({ provenance: p }),
  setField: (k, v) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { (d as any)[k] = v; }) })),
  addBullet: (wi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.push(""); }) })),
  editBullet: (wi, bi, text) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets[bi] = text; }) })),
  removeBullet: (wi, bi) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.work[wi].bullets.splice(bi, 1); }) })),
  addSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { if (!d.skills.map((x) => x.toLowerCase()).includes(skill.toLowerCase())) d.skills.push(skill); }) })),
  removeSkill: (skill) => set((s) => ({ resume: s.resume && mutate(s.resume, (d) => { d.skills = d.skills.filter((x) => x.toLowerCase() !== skill.toLowerCase()); }) })),
  setStyle: (patch) => set((s) => ({ style: { ...s.style, ...patch } })),
  applyTailored: (tailored, score, previousScore) => set((s) => ({
    optimize: {
      score, previousScore,
      gaps: s.optimize?.gaps ?? [], fabricationsBlocked: s.optimize?.fabricationsBlocked ?? 0,
      originalResume: s.resume!,
    },
    resume: tailored,
  })),
  setGaps: (gaps, fabricationsBlocked) => set((s) => ({
    optimize: s.optimize ? { ...s.optimize, gaps, fabricationsBlocked } : {
      score: 0, previousScore: null, gaps, fabricationsBlocked, originalResume: s.resume!,
    },
  })),
}));
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm test resumeStore`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add web/lib/resumeStore.ts web/lib/resumeStore.test.ts
git commit -m "feat: zustand editor store with content + style state"
```

---

### Task 7: Preview component with before/after bullets

**Files:**
- Create: `web/components/Preview/ResumePreview.tsx`
- Test: `web/components/Preview/ResumePreview.test.tsx`

**Interfaces:**
- Consumes: `StructuredResume`, `StyleSettings`, `OptimizeView` (store).
- Produces: `<ResumePreview resume style optimize />` — a styled résumé sheet. When `optimize` is present, bullets changed by tailoring render original (struck) + tailored.

- [ ] **Step 1: Write the failing test**

```tsx
// web/components/Preview/ResumePreview.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResumePreview } from "./ResumePreview";

const style = { fontSize: 14, margin: 48, lineSpacing: 1.4, accent: "#000", sectionOrder: ["summary", "skills", "work", "education"] };
const resume = {
  name: "Jane Doe", headline: "Engineer", summary: "Builds things",
  emails: [], phones: [], urls: [], education: [], certifications: [],
  skills: ["python", "aws"],
  work: [{ company: "Acme", title: "Engineer", bullets: ["Built X in Python"] }],
};

describe("ResumePreview", () => {
  it("renders name, skills and bullets", () => {
    render(<ResumePreview resume={resume as any} style={style as any} optimize={null} />);
    expect(screen.getByText("Jane Doe")).toBeDefined();
    expect(screen.getByText("python")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });

  it("shows before/after when a bullet was tailored", () => {
    const original = { ...resume, work: [{ company: "Acme", title: "Engineer", bullets: ["Built X"] }] };
    render(<ResumePreview resume={resume as any} style={style as any}
      optimize={{ score: 80, previousScore: 60, gaps: [], fabricationsBlocked: 0, originalResume: original } as any} />);
    // original struck-through text present alongside the tailored version
    expect(screen.getByText("Built X")).toBeDefined();
    expect(screen.getByText(/Built X in Python/)).toBeDefined();
  });
});
```

Note the test needs `@testing-library/jest-dom` matchers; add `import "@testing-library/jest-dom";` at the top of the test (or a `web/vitest.setup.ts` referenced from `vitest.config.ts`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm test ResumePreview`
Expected: FAIL (no `ResumePreview`).

- [ ] **Step 3: Write `web/components/Preview/ResumePreview.tsx`**

```tsx
"use client";
import type { StructuredResume } from "@/lib/types";
import type { OptimizeView, StyleSettings } from "@/lib/resumeStore";

function bulletBefore(original: StructuredResume | undefined, wi: number, bi: number): string | null {
  const b = original?.work?.[wi]?.bullets?.[bi];
  return b !== undefined ? b : null;
}

export function ResumePreview({ resume, style, optimize }: {
  resume: StructuredResume; style: StyleSettings; optimize: OptimizeView | null;
}) {
  const sheet: React.CSSProperties = {
    fontSize: style.fontSize, padding: style.margin, lineHeight: style.lineSpacing,
    ["--accent" as any]: style.accent,
  };
  return (
    <article style={sheet} className="mx-auto max-w-3xl bg-white text-neutral-900 shadow">
      <header>
        <h1 className="text-2xl font-bold">{resume.name}</h1>
        {resume.headline && <p className="text-[color:var(--accent)]">{resume.headline}</p>}
      </header>
      {style.sectionOrder.map((section) => {
        if (section === "summary" && resume.summary)
          return <section key="summary"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Summary</h2><p>{resume.summary}</p></section>;
        if (section === "skills" && resume.skills.length)
          return <section key="skills"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Skills</h2>
            <ul className="flex flex-wrap gap-2">{resume.skills.map((s) => <li key={s} className="rounded bg-neutral-100 px-2">{s}</li>)}</ul></section>;
        if (section === "work" && resume.work.length)
          return <section key="work"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Experience</h2>
            {resume.work.map((w, wi) => (
              <div key={wi} className="mt-2">
                <div className="flex justify-between"><strong>{w.title}</strong><span>{w.start_date}{w.end_date ? `–${w.end_date}` : ""}</span></div>
                <div className="italic">{w.company}</div>
                <ul className="list-disc pl-5">
                  {w.bullets.map((b, bi) => {
                    const before = optimize ? bulletBefore(optimize.originalResume, wi, bi) : null;
                    const changed = before !== null && before !== b;
                    return (
                      <li key={bi}>
                        {changed && <span className="mr-1 text-neutral-400 line-through">{before}</span>}
                        <span>{b}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}</section>;
        if (section === "education" && resume.education.length)
          return <section key="education"><h2 className="mt-4 font-semibold uppercase text-sm text-[color:var(--accent)]">Education</h2>
            {resume.education.map((e, ei) => <div key={ei}>{e.institution}{e.degree ? ` — ${e.degree}` : ""}{e.field ? `, ${e.field}` : ""}{e.end_year ? ` (${e.end_year})` : ""}</div>)}</section>;
        return null;
      })}
    </article>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm test ResumePreview`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add web/components/Preview/ResumePreview.tsx web/components/Preview/ResumePreview.test.tsx web/vitest.setup.ts web/vitest.config.ts
git commit -m "feat: resume preview with before/after tailored bullets"
```

---

### Task 8: Editor panels, upload, and page wiring

**Files:**
- Create: `web/components/UploadDropzone.tsx`, `web/components/Editor/FieldEditors.tsx`, `web/components/Editor/WorkEditor.tsx`, `web/components/Editor/SkillsEditor.tsx`, `web/components/Editor/StyleControls.tsx`, `web/components/Editor/JdBox.tsx`, `web/app/page.tsx`
- Test: manual (UI wiring) — logic is already covered by store/api/preview tests. Use `frontend-design` skill for the visual layer of this task.

**Interfaces:**
- Consumes: `useResumeStore` actions, `parseResume`, `startOptimize`, `pollOptimize`.
- Produces: the two-pane page.

- [ ] **Step 1: Invoke the frontend-design skill**

Before writing the components, invoke `frontend-design:frontend-design` for aesthetic direction (typography, the paper-sheet preview, the two-pane layout, accent usage). Apply its guidance to the components below.

- [ ] **Step 2: Write `web/components/UploadDropzone.tsx`**

```tsx
"use client";
import { useState } from "react";
import { BackendUnreachable, parseResume } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function UploadDropzone() {
  const setResume = useResumeStore((s) => s.setResume);
  const setProvenance = useResumeStore((s) => s.setProvenance);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onFile(file: File) {
    setBusy(true); setError(null);
    try {
      const res = await parseResume(file);
      setResume(res.structured_resume);
      setProvenance(res.provenance_map);
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable. Start it: uvicorn rho.api.app:app --reload"
        : `Parse failed: ${(e as Error).message}`);
    } finally { setBusy(false); }
  }

  return (
    <div>
      <label className="block cursor-pointer rounded border-2 border-dashed p-6 text-center">
        <input type="file" accept=".pdf,.docx,.txt" className="hidden"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        {busy ? "Parsing…" : "Upload résumé (PDF, DOCX, TXT)"}
      </label>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 3: Write the editor panels**

`web/components/Editor/FieldEditors.tsx`:
```tsx
"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function FieldEditors() {
  const resume = useResumeStore((s) => s.resume);
  const setField = useResumeStore((s) => s.setField);
  if (!resume) return null;
  return (
    <div className="space-y-2">
      <input className="w-full rounded border p-2" value={resume.name} onChange={(e) => setField("name", e.target.value)} placeholder="Name" />
      <input className="w-full rounded border p-2" value={resume.headline ?? ""} onChange={(e) => setField("headline", e.target.value)} placeholder="Headline" />
      <textarea className="w-full rounded border p-2" value={resume.summary ?? ""} onChange={(e) => setField("summary", e.target.value)} placeholder="Summary" />
    </div>
  );
}
```

`web/components/Editor/WorkEditor.tsx`:
```tsx
"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function WorkEditor() {
  const resume = useResumeStore((s) => s.resume);
  const { addBullet, editBullet, removeBullet } = useResumeStore.getState();
  if (!resume) return null;
  return (
    <div className="space-y-4">
      {resume.work.map((w, wi) => (
        <div key={wi} className="rounded border p-2">
          <div className="font-medium">{w.title} — {w.company}</div>
          <ul className="mt-2 space-y-1">
            {w.bullets.map((b, bi) => (
              <li key={bi} className="flex gap-1">
                <textarea className="w-full rounded border p-1 text-sm" value={b} onChange={(e) => editBullet(wi, bi, e.target.value)} />
                <button className="px-2 text-red-500" onClick={() => removeBullet(wi, bi)}>×</button>
              </li>
            ))}
          </ul>
          <button className="mt-1 text-sm text-blue-600" onClick={() => addBullet(wi)}>+ bullet</button>
        </div>
      ))}
    </div>
  );
}
```

`web/components/Editor/SkillsEditor.tsx`:
```tsx
"use client";
import { useState } from "react";
import { useResumeStore } from "@/lib/resumeStore";

export function SkillsEditor() {
  const resume = useResumeStore((s) => s.resume);
  const { addSkill, removeSkill } = useResumeStore.getState();
  const [draft, setDraft] = useState("");
  if (!resume) return null;
  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {resume.skills.map((s) => (
          <span key={s} className="rounded bg-neutral-200 px-2 py-0.5 text-sm">
            {s} <button onClick={() => removeSkill(s)}>×</button>
          </span>
        ))}
      </div>
      <form className="mt-2 flex gap-1" onSubmit={(e) => { e.preventDefault(); if (draft.trim()) { addSkill(draft.trim()); setDraft(""); } }}>
        <input className="rounded border p-1 text-sm" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Add skill" />
        <button className="text-sm text-blue-600" type="submit">Add</button>
      </form>
    </div>
  );
}
```

`web/components/Editor/StyleControls.tsx`:
```tsx
"use client";
import { useResumeStore } from "@/lib/resumeStore";

export function StyleControls() {
  const style = useResumeStore((s) => s.style);
  const setStyle = useResumeStore((s) => s.setStyle);
  return (
    <div className="space-y-2 text-sm">
      <label className="block">Font size {style.fontSize}px
        <input type="range" min={10} max={20} value={style.fontSize} onChange={(e) => setStyle({ fontSize: +e.target.value })} className="w-full" /></label>
      <label className="block">Margin {style.margin}px
        <input type="range" min={16} max={80} value={style.margin} onChange={(e) => setStyle({ margin: +e.target.value })} className="w-full" /></label>
      <label className="block">Line spacing {style.lineSpacing}
        <input type="range" min={1} max={2} step={0.1} value={style.lineSpacing} onChange={(e) => setStyle({ lineSpacing: +e.target.value })} className="w-full" /></label>
      <label className="block">Accent
        <input type="color" value={style.accent} onChange={(e) => setStyle({ accent: e.target.value })} /></label>
    </div>
  );
}
```

`web/components/Editor/JdBox.tsx`:
```tsx
"use client";
import { useState } from "react";
import { startOptimize, pollOptimize, BackendUnreachable } from "@/lib/api";
import { useResumeStore } from "@/lib/resumeStore";

export function JdBox() {
  const resume = useResumeStore((s) => s.resume);
  const optimize = useResumeStore((s) => s.optimize);
  const { applyTailored, setGaps } = useResumeStore.getState();
  const [jd, setJd] = useState("");
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (!resume || !jd.trim()) return;
    setBusy(true); setError(null); setStage("starting");
    const prevScore = optimize?.score ?? null;
    try {
      const started = await startOptimize(resume, jd);
      const done = await pollOptimize(started.id, {
        intervalMs: 1500, timeoutMs: 180000,
      });
      const r = done.result!;
      applyTailored(r.tailored_resume.resume, r.final_score, prevScore);
      setGaps(
        r.match_result.gaps.map((g) => ({ text: g.requirement.text, priority: g.requirement.priority, status: g.status })),
        r.tailored_resume.fabrication_report.rejected_edits.length,
      );
    } catch (e) {
      setError(e instanceof BackendUnreachable
        ? "Backend unreachable. Start it: uvicorn rho.api.app:app --reload"
        : (e as Error).message);
    } finally { setBusy(false); setStage(null); }
  }

  return (
    <div className="space-y-2">
      <textarea className="h-32 w-full rounded border p-2 text-sm" value={jd} onChange={(e) => setJd(e.target.value)} placeholder="Paste the target job description…" />
      <button disabled={!resume || !jd.trim() || busy}
        className="w-full rounded bg-blue-600 py-2 text-white disabled:opacity-40"
        onClick={run}>
        {busy ? `Optimising… ${stage ?? ""}` : "Optimise score →"}
      </button>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {optimize && !busy && (
        <div className="rounded border p-2 text-sm">
          <div>Score: <strong>{optimize.score.toFixed(0)}</strong>/100
            {optimize.previousScore !== null && <span className="ml-1 text-green-600">▲ from {optimize.previousScore.toFixed(0)}</span>}</div>
          <div>Unsourced edits blocked: {optimize.fabricationsBlocked} (fabrication gate)</div>
          {optimize.gaps.length > 0 && <div>Gaps: {optimize.gaps.filter((g) => g.status !== "present").map((g) => g.text).join(", ")}</div>}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Write `web/app/page.tsx`**

```tsx
"use client";
import { UploadDropzone } from "@/components/UploadDropzone";
import { FieldEditors } from "@/components/Editor/FieldEditors";
import { WorkEditor } from "@/components/Editor/WorkEditor";
import { SkillsEditor } from "@/components/Editor/SkillsEditor";
import { StyleControls } from "@/components/Editor/StyleControls";
import { JdBox } from "@/components/Editor/JdBox";
import { ResumePreview } from "@/components/Preview/ResumePreview";
import { useResumeStore } from "@/lib/resumeStore";

export default function Page() {
  const resume = useResumeStore((s) => s.resume);
  const style = useResumeStore((s) => s.style);
  const optimize = useResumeStore((s) => s.optimize);
  return (
    <main className="grid min-h-screen grid-cols-1 gap-4 bg-neutral-100 p-4 lg:grid-cols-2">
      <section className="space-y-4 overflow-y-auto">
        <h1 className="text-xl font-bold">rho — résumé editor</h1>
        <UploadDropzone />
        {resume && (<>
          <FieldEditors />
          <SkillsEditor />
          <WorkEditor />
          <details><summary className="cursor-pointer font-medium">Styling</summary><StyleControls /></details>
          <JdBox />
        </>)}
      </section>
      <section className="overflow-y-auto">
        {resume
          ? <ResumePreview resume={resume} style={style} optimize={optimize} />
          : <p className="text-neutral-500">Upload a résumé to see the preview.</p>}
      </section>
    </main>
  );
}
```

- [ ] **Step 5: Verify build + tests pass**

Run: `cd web && npm run build && npm test`
Expected: build succeeds; all Vitest tests pass.

- [ ] **Step 6: Commit**

```bash
git add web/components web/app/page.tsx
git commit -m "feat: editor panels, upload, two-pane page wiring"
```

---

### Task 9: README + end-to-end smoke check

**Files:**
- Create: `web/README.md`
- Modify: none (verification only)

- [ ] **Step 1: Write `web/README.md`**

```markdown
# rho résumé editor (web)

## Run

Backend (Gemini): from repo root, ensure `GEMINI_API_KEY` is in `.env` and
`extraction_backend=gemini`, then:

    uvicorn rho.api.app:app --reload

Frontend:

    cd web && npm install && npm run dev

Open http://localhost:3000. Upload a résumé, edit it, paste a job description,
click **Optimise score**.

## Notes
- Score is the Phase-4 calibrated ATS-match prediction (0–100).
- The optimiser tailors bullet wording to the JD but cannot invent facts: the
  provenance gate blocks any unsourced edit and reports how many it blocked.
- After you edit, "sourced" means "traces to your current résumé", not the
  original upload.
```

- [ ] **Step 2: Start the backend and smoke-test `/parse`**

Run (from repo root):
```bash
.venv/bin/uvicorn rho.api.app:app --port 8000 &
sleep 3
curl -s -F "file=@tests/fixtures/clean.txt" http://localhost:8000/parse | head -c 300
```
Expected: JSON with `structured_resume` and `provenance_map`. (Requires the Gemini key for real extraction; if unset the call errors with a clear message rather than a fake result.)

- [ ] **Step 3: Full backend suite green**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add web/README.md
git commit -m "docs: web README + run instructions"
```

---

## Self-Review

- **Spec coverage:** `/parse` (Task 4), async `/optimize` + polling (Tasks 3–4), `run_from_structured` entering at match (Task 1), edited-résumé provenance rebuild (Task 1), editor content + bullets + skills (Tasks 6, 8), frontend-only styling (Tasks 6–8), before/after bullets + score delta (Tasks 6–7), gaps + fabrication count surfaced (Task 8), every error row from the spec table (Tasks 4, 8), CORS (Task 4), tests for store/api/preview/backend (Tasks 1–8). All covered.
- **Placeholders:** none — every code step carries real content.
- **Type consistency:** `JobStatus`/`OptimizeResult` names identical across Tasks 2–5; `useResumeStore` action names identical across Tasks 6–8; `applyTailored(tailored, score, previousScore)` and `setGaps(gaps, fabricationsBlocked)` signatures consistent between store (Task 6) and JdBox (Task 8); `OptimizeView.originalResume` consumed by ResumePreview (Task 7) matches the store (Task 6).
