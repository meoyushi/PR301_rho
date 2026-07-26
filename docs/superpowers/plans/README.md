# Implementation Plans — Provenance-Chained Resume Optimization

Backend for a résumé parser + ATS optimizer, designed to yield a research paper.
Novelty = a **provenance ID space** threading all 4 pipeline stages (C1), with
**real-engine ATS-calibrated scoring** (C2) and a **provenance-verified rewrite gate** (C3).

**Design spec:** `../specs/2026-07-20-provenance-chained-resume-optimization-design.md`

## How to work these

1. **Every session, read `00-SHARED-CONTEXT.md` first** — thesis, frozen contracts, conventions.
2. Then open the phase doc for the phase you're building. Each is self-contained: it lists what it consumes from prior phases, what it must produce, and TDD steps with real code.
3. Execute a phase in one session (or split by task). Follow TDD: failing test → fail → impl → pass → commit.
4. **Fill in the "Results" section at the bottom of the phase doc** before you finish — the final verification session reads those numbers.
5. Come back to the main chat after each phase (or after all) to verify.

## Order (strict — each depends on earlier)

| # | Doc | Delivers | Paper |
|---|-----|----------|-------|
| 0 | `phase-0-skeleton-and-contracts.md` | FastAPI app + all Pydantic contracts + stubs | — |
| 1 | `phase-1-ingestion-provenance.md` | Docling ingestion + ProvenanceMap | C1 base |
| 2 | `phase-2-extraction-provenance.md` | LLM extraction + prov_id per field | C1 core, first metric |
| 3 | `phase-3-jd-analyzer-matcher.md` | JD → requirements + deterministic matcher | — |
| 4 | `phase-4-ats-harness-calibration.md` | Real-engine ATS calibration | **C2** (hardest) |
| 5 | `phase-5-verified-rewriter.md` | Rewrite + verification gate + fabrication metric | **C3** |
| 6 | `phase-6-orchestration.md` | LangGraph pipeline + reviewer + wired API | — |
| 7 | `phase-7-evaluation-paper.md` | Datasets, metrics, ablations, RESULTS.md | all |

## Known risks (flagged in the docs)

- **Phase 4 is the make-or-break — but Task 0 survey is DONE: decision GO** (see `phase-4-engine-survey.md`). Engines chosen: **Resume-Matcher** (Apache-2.0, HTTP/Docker, local Ollama — build first), **ats-screener** (MIT, 6 platform-profile scores via ported rules), **OpenCATS** (real parser, MySQL parse-recovery for the parse-injection dimension). Determinism: pin model + temperature=0. Start Phase 4 at Task 1.
- **Signature change:** Phase 5 extends `rewrite(resume, gaps)` → `rewrite(resume, gaps, prov)`. Shared context Section 6 gets updated there; Phase 6 caller already passes `prov`.

## Final verification (do in the main chat after all phases)

Bring back, per phase, the filled-in Results section. The paper needs:
- **C1:** extraction field F1 + long-text F1 + provenance-attachment accuracy.
- **C2:** calibrated MAE + Spearman ρ vs real engines, and the cosine-baseline MAE it beats.
- **C3:** unsourced additions shipped gate-OFF vs gate-ON (must be 0 ON), mean fabrication_rate.
- **Ablations:** provenance on/off, calibrated vs cosine, gate on/off.
- All tests green (`pytest -v`), `/optimize` returns a full `PipelineResponse`.
