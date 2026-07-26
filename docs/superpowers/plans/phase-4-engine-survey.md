# Phase 4 — Task 0: Self-Hostable ATS Engine Survey & Go/No-Go

**Date:** 2026-07-20
**Author:** performed as Phase-4 Task 0 (pre-implementation research spike)
**Question:** Can we obtain ≥2 self-hostable ATS engines whose parse and/or match output is harvestable programmatically, to serve as ground truth for the C2 calibration? If not, adopt the parse-injection fallback.

---

## Decision: **GO** ✅

We have **three** viable self-hostable sources plus a fallback. Recommended target set for calibration ground truth:

1. **Resume-Matcher (srbhr)** — primary match-score engine (easy, HTTP).
2. **ats-screener (sunnypatell)** — multi-strategy scorer, 6 platform profiles (medium, port TS logic or drive route).
3. **OpenCATS** — real ATS parse ground truth (for the parse-injection target / parse-ability dimension).

Two independent *match-score* engines (1 + 2) satisfy the C2 requirement of "≥2 engines with harvestable output." OpenCATS + parse-injection is the safety net.

---

## Candidates evaluated

### 1. Resume-Matcher — `github.com/srbhr/Resume-Matcher`  ⭐ primary
- **License:** Apache 2.0 (permissive, safe to depend on + cite).
- **Stack:** FastAPI backend + Next.js frontend, SQLite, **LiteLLM** (100+ LLMs), runs **fully local with Ollama** (no data leaves machine).
- **Output:** numeric match score + *which requirements match / are missing / how to close the gap*. Not just a bare number — structured gap output.
- **Harvestable?** YES, directly. FastAPI, Swagger/OpenAPI at `http://localhost:3000/docs`. Docker image `ghcr.io/srbhr/resume-matcher:latest`. Drive over HTTP, read JSON.
- **Caveats:** score is LLM-backed (semantic) → one engine *family*; **pin the model + temperature=0 via Ollama** for determinism/reproducibility, else calibration target drifts. Exact JSON field names must be read off `/docs` at integration time (README doesn't enumerate them).
- **Effort:** LOW. `docker run`, POST resume+JD, parse JSON `match_score`.

### 2. ats-screener — `github.com/sunnypatell/ats-screener`  ⭐ multi-profile
- **License:** MIT.
- **Stack:** SvelteKit 5 + TypeScript. LLM backend Gemini (Groq fallback).
- **What's special:** simulates **6 named enterprise ATS** (Workday, Taleo, iCIMS, Greenhouse, Lever, SuccessFactors) with *distinct* strategies — Taleo strict literal matching, iCIMS ML-semantic, etc. **5 scoring dimensions** (formatting, keyword, sections, experience, education). This is 6 differentiated pseudo-engines in one repo → great for a multi-engine calibration target *and* a strong paper talking point.
- **Harvestable?** PARTIAL. **Browser-only UI today**; résumé parsing runs client-side in a Web Worker; references an `/api/analyze` route but doesn't document numeric output shape. No CLI / Python lib / documented headless API.
- **Two harvest paths:**
  - (a) **Port the TS scoring modules** (the 6 platform strategies + 5 dimensions are deterministic rule code) into a Node headless runner or reimplement in Python — deterministic, no LLM, fully reproducible. Preferred for the paper.
  - (b) Drive `/api/analyze` via a headless Node process / Playwright — higher friction, pulls in Gemini/Groq (non-local, non-deterministic).
- **Effort:** MEDIUM (path a: read + port ~5 dimension × 6 platform rule modules; worth it — yields 6 reproducible targets).

### 3. OpenCATS — `github.com/opencats/OpenCATS`  (parse ground truth)
- **License:** open source (GPL-family; verify before bundling — fine to run as an external service, don't vendor code into `rho`).
- **Stack:** PHP / LAMP (Apache, MySQL). Self-hostable, Docker images exist in community.
- **What it gives:** a **real ATS parser** (50+ résumé formats → structured candidate fields). This is genuine parser mechanics, not a simulation.
- **Harvestable?** PARTIAL. **No REST API** (documented limitation); extend by editing PHP or **read parsed fields straight from MySQL** after import. No match/rank score exposed (it's tracking, not scoring).
- **Use:** the **parse-injection target** (Task 0b) — inject known fields into a résumé, import, read what OpenCATS recovered from MySQL → recovery rate. Real-parser-grounded, reproducible.
- **Effort:** MEDIUM (stand up LAMP/Docker + MySQL read); LOW-value for match score, HIGH-value for parse-ability dimension.

### Also seen (not selected, noted for completeness)
- "Self-Hosted AI ATS" (open-source AI ATS, FastAPI, LLM screening + CV parsing + candidate matching) — candidate 4th engine if a second *non-srbhr* HTTP match-score is wanted; verify license + output shape at integration.
- Local-First ATS Optimizer (Groq/Ollama/Gemini backends) — optimizer, overlaps our own rewrite; not a clean ground-truth source.

---

## Recommended target definition for C2

Aggregate to `y ∈ [0,100]` from **match-score engines** (Resume-Matcher + ats-screener's 6 profiles):

- `y = mean(resume_matcher_score, mean(ats_screener_6_profile_scores))` — or keep them as separate targets and fit per-engine calibrators + a pooled one (richer results table: "calibrated to engine X" rows).
- Run the **parse-injection recovery** (OpenCATS + open parsers) as a **separate reported dimension** ("parse-ability calibration"), not folded into the match `y`. This gives the paper *two* calibration stories (match + parse) instead of one.

**Determinism rules (must, for reproducibility claim):**
- Resume-Matcher: pin Ollama model + `temperature=0`, record model hash.
- ats-screener: prefer the ported-rule path (no LLM) so scores are deterministic; if driving the LLM route, pin model + temperature and record.
- OpenCATS: deterministic by construction (rule parser).

---

## Impact on Phase-4 plan (adjust the doc accordingly)

- **Task 1 engine adapters** become concrete:
  - `engines/resume_matcher.py` — HTTP client to the dockerized FastAPI (`match_score` + gaps). **Build first (lowest effort).**
  - `engines/ats_screener.py` — wrapper over ported TS scoring rules (Node subprocess or Python reimplementation) exposing the 6 platform scores.
  - `engines/opencats.py` — MySQL-read parse-recovery adapter (feeds Task 0b parse-injection target).
- **Task 0b (parse-injection) is now RECOMMENDED, not just fallback** — OpenCATS makes it cheap and it strengthens the paper (real parser mechanics).
- `to_target` aggregates the match-score engines; parse recovery reported separately.

## Risks / open items for integration time
- Read Resume-Matcher `/docs` to get exact JSON field names for the score.
- Confirm OpenCATS license terms before any code reuse (running as a service is fine).
- ats-screener path (a) requires reading its TS scoring modules — budget time to port faithfully; document any rule you couldn't reproduce.
- Pin all model versions; a drifting LLM target silently invalidates calibration.

## Bottom line
Two harvestable match-score engines (Resume-Matcher, ats-screener) + a real parser for parse-injection (OpenCATS) = **C2 is buildable and reproducible.** Proceed with Phase 4 as written, using the concrete engines above in Task 1.
