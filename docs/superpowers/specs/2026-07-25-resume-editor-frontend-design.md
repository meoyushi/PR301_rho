# Résumé Editor Frontend + API — Design Spec

**Date:** 2026-07-25
**Depends on:** the built `rho` pipeline (Phases 0–7). Shared context:
`../plans/00-SHARED-CONTEXT.md`.

## Goal

A local web app to upload a résumé, see it parsed and displayed, edit every part
of it (text, bullets, skills, section order, and visual styling), and optimise
its ATS-match score against a pasted job description with one button. The button
runs the provenance-verified pipeline: it tailors bullets toward the JD while the
C3 gate guarantees nothing is fabricated.

Two deliverables: **new API endpoints** on the existing FastAPI app, and a **new
Next.js frontend** under `web/`.

## Non-goals (explicit scope boundary)

- No auth, no database, no multi-user. Single-user local tool; job state is
  in-process.
- No PDF/DOCX export of the edited résumé (preview is on-screen only).
- No document-layout backend. Visual styling (fonts, margins, spacing, order)
  is **frontend-only CSS**, never sent to or stored by the backend.
- No live ATS-engine calls from the UI. The score is the Phase-4 calibrated
  prediction.

## What the backend already does (grounding)

- `run_pipeline(file_bytes, filename, jd_text) -> PipelineResponse` runs
  ingest → extract → jd → match → score → rewrite → review.
- The rewriter (`rho.rewrite.llm`) **does tailor bullets to the JD**: it is told
  to reorder, rephrase, select, and emphasise existing content toward the
  JD-derived gaps. It may not invent facts; `rho.rewrite.verifier` strips any
  unsourced hard-content token (the C3 guarantee).
- `StructuredResume` carries per-field `*_prov` provenance id lists.
- `MatchResult` carries `predicted_score` (0–100) and `gaps`.
- Extraction backend is config-selected (`settings.extraction_backend`); the
  Gemini backend runs in seconds and is the target for the async endpoint.

## Architecture

```
Browser (Next.js, web/)                 FastAPI (src/rho/api/)
─────────────────────────               ─────────────────────
upload file ───────────────────────────▶ POST /parse
   ◀─── StructuredResume + ProvenanceMap
edit in browser (no backend calls)
paste JD, click Optimize ──────────────▶ POST /optimize  (edited resume + jd)
   ◀─── {job_id}
poll ──────────────────────────────────▶ GET /optimize/{job_id}
   ◀─── {state, stage, result?}
```

### Backend components (`src/rho/api/`)

`app.py` grows past a single handler, so it is split:

- **`models/api.py`** (extend): `ParseResponse{structured_resume,
  provenance_map}`, `OptimizeJobRequest{resume: StructuredResume, jd_text:
  str}`, `JobStatus{id, state: Literal["queued","running","done","error"],
  stage: str|None, result: OptimizeResult|None, error: str|None}`,
  `OptimizeResult{match_result, tailored_resume, final_score,
  previous_score: float|None}`.
- **`api/jobs.py`**: in-process `JobStore` (dict keyed by uuid). `create(req)`
  spawns a daemon thread that walks the pipeline stages, writing `stage` as it
  goes (`"matching" → "scoring" → "rewriting" → "verifying" → "done"`), and
  sets `state="error"` with the message on any exception — never a silent fail.
- **`api/pipeline_entry.py`** (or a function in `rho.graph`):
  `run_from_structured(resume: StructuredResume, jd_text: str) ->
  PipelineResponse`. Optimize starts from the **edited** résumé, so it must
  enter the graph at `match` (skipping ingest/extract), reusing the existing
  jd/match/score/rewrite/review nodes. The provenance map for the edited résumé
  is rebuilt from its own values via the existing `ingest` path over the
  résumé text (same technique `eval/fabrication_corpus.py` uses), so the gate
  still has spans to verify against.
- **`app.py`**: `POST /parse`, `POST /optimize`, `GET /optimize/{id}`,
  `GET /health`. CORS allows `http://localhost:3000`.

**Provenance for the edited résumé.** The user may have edited values away from
the original document. The rewrite gate verifies against the résumé it is given
as source of truth, so provenance is rebuilt from the edited résumé's own
values. Consequence, stated honestly in the UI: after editing, "sourced" means
"traces to your current résumé", not "traces to the original upload".

### Frontend components (`web/`, Next.js App Router, TypeScript, Tailwind)

- **`lib/types.ts`** — TS mirrors of the backend Pydantic models.
- **`lib/api.ts`** — `parse(file)`, `startOptimize(resume, jd)`,
  `pollOptimize(jobId)` with a poll loop (interval + timeout), typed errors for
  "backend unreachable" vs "job failed".
- **`lib/resumeStore.ts`** — Zustand store: `resume` (editable), `provenance`,
  `style` (fontSize, margin, lineSpacing, sectionOrder, accent), `optimize`
  (result | running | error). Actions for every edit (field set, bullet
  add/remove/edit/reorder, skill add/remove) and style updates.
- **`components/Editor/`** — `FieldEditors` (name, headline, summary, contact),
  `WorkEditor` (per-job title/company/dates + bullet list with add/remove/drag),
  `SkillsEditor` (chips), `EducationEditor`, `StyleControls` (sliders + order +
  accent), `JdBox` (textarea + Optimize button).
- **`components/Preview/ResumePreview.tsx`** — renders store state as a
  paper-like sheet, styled from `style` via CSS variables. When an optimize
  result exists, bullets render before/after (original struck/greyed, tailored
  shown) and a score badge shows the delta.
- **`app/page.tsx`** — two-pane split, upload dropzone, wires store to both
  panes.

## Data flow / state

Single source of truth is the Zustand store. Upload fills it from `/parse`.
Edits mutate it and the preview re-renders live (no backend round-trip).
Optimize sends the **current store `resume`** (edits included) plus the JD;
the returned `tailored_resume` and `match_result` merge back into the store,
and the preview switches to before/after mode. `previous_score` is the score
from the last optimize (or null on first run) so the badge can show a delta.

## Error handling

| Situation | Behaviour |
|---|---|
| Parse fails (bad file, extraction error) | Inline error on dropzone; prior state kept. |
| Optimize job → `error` | Banner with backend message; résumé unchanged. |
| Backend unreachable | "Backend unreachable" notice + the `uvicorn` command to start it. |
| No JD entered | Optimize button disabled with a hint. |
| Fabrications rejected by gate | **Info**, not error: "N unsourced edits blocked" — the C3 feature working as designed. |

## Testing

- **Backend (pytest):** `/parse` response shape; job lifecycle
  (create → poll → done) with the pipeline stubbed via the existing
  `stub_nodes` fixture; `run_from_structured` enters at `match` and never
  re-runs extract; job `error` state on a stubbed exception.
- **Frontend (Vitest + React Testing Library):** store actions (edit/add/
  remove/reorder bullet, skill add/remove, style updates); `api.ts` poll loop
  against a mocked fetch (running → done, and the error path); `ResumePreview`
  renders from a given store state including before/after bullets.
- No end-to-end test hits a real LLM.

## Tech stack additions

- Frontend: Next.js (App Router) + TypeScript + Tailwind + Zustand + Vitest.
- Backend: no new Python deps (threading + FastAPI already present). The Gemini
  backend and its client already exist from Phase 7.

## Definition of done

- `/parse`, `POST /optimize`, `GET /optimize/{id}`, `/health` live with CORS.
- `web/` runs (`npm run dev`), uploads a résumé, displays it, edits persist in
  the preview live, styling controls affect the preview.
- Optimize against a pasted JD returns a score, gaps, and JD-tailored bullets
  with the fabrication count shown.
- Backend and frontend tests pass.
