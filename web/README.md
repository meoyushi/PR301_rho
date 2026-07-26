# rho résumé editor (web)

A two-pane editor over the `rho` provenance-chained résumé pipeline: upload a
résumé, see it parsed on the right, edit every part on the left, then optimise
its ATS-match score against a pasted job description.

## Run

### 1. Backend

From the repo root, make sure `GEMINI_API_KEY` is set (the pipeline reads it from
`.env`). Select the Gemini extraction backend — fast, seconds per résumé — with
the `extraction_backend` **environment variable** when you launch the API. (The
`Settings` object reads OS environment variables but not the `.env` file for this
field, so it must be an env var, not an `.env` line.) The frontend expects the
API on port 8000:

    extraction_backend=gemini .venv/bin/uvicorn rho.api.app:app --port 8000 --reload

Without that variable the backend defaults to the local Ollama 14B model, which
is much slower (see timing table below).

### 2. Frontend

    cd web && npm install && npm run dev

Open http://localhost:3000. Upload a résumé, edit it, paste a job description,
click **Optimise score**.

## Backends and timing (read before uploading a PDF)

| Path | Backend used | Typical time |
|---|---|---|
| `/parse` extraction | `extraction_backend` env var (set it to `gemini`) | ~2 s with Gemini; **~7 min** on the local Ollama 14B (CPU) |
| PDF ingestion (Docling) | always local | **~2–3 min** — Docling OCRs the whole page; first run also downloads OCR weights |
| `.txt` / `.docx` ingestion | always local | instant |
| `/optimize` (JD analysis + rewrite) | always Gemini | seconds |

**Practical guidance:** set `extraction_backend=gemini`, and prefer `.txt` or
`.docx` uploads while iterating — a PDF still takes a couple of minutes in
Docling's OCR regardless of the extraction backend. A real PDF résumé was
verified end-to-end through `/parse` (name, skills, work, education all
recovered correctly); it simply takes minutes, not seconds, on the PDF path.

Note that `/optimize` always runs the Gemini JD-analysis and rewrite path
regardless of `extraction_backend`; only `/parse`'s extraction step follows that
setting.

## Notes

- The **score** is the Phase-4 calibrated ATS-match prediction (0–100), shown
  with a delta from your previous optimise.
- The optimiser tailors bullet **wording** to the JD but cannot invent facts:
  the provenance gate blocks any unsourced edit and reports how many it blocked.
  Changed bullets show the original struck-through next to the tailored text.
- After you edit, **"sourced" means "traces to your current résumé"**, not the
  original upload — provenance is rebuilt from what you send to `/optimize`.
- Visual styling (font size, margins, line spacing, accent) is applied in the
  browser only; it is never sent to the backend.

## Tests

    cd web && npm test          # frontend (Vitest)
    .venv/bin/python -m pytest tests/ -q   # backend (from repo root)
