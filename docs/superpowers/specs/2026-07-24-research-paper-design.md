# Research Paper — Design Spec

**Title (working):** Provenance-Chained Résumé Optimization: A System for Verified ATS-Calibrated Rewriting

**Format:** Short workshop paper, 4–6 pages, NeurIPS-workshop LaTeX style.
**Output:** `docs/paper/paper.tex` + `docs/paper/refs.bib` + vendored `docs/paper/neurips_2024.sty`.

## Sourcing rule

Every number in the paper must trace to one of:
- `docs/superpowers/plans/phase-{0..7}-*.md` — each phase's `## Results` section
- `eval/RESULTS.md`
- `docs/superpowers/specs/2026-07-20-provenance-chained-resume-optimization-design.md` (frozen contracts, architecture)
- `report.md` (background/related-work claims only, not system numbers)

No number is invented or estimated. Where a comparison doesn't exist (e.g. no qwen Table-1
extraction run), the paper states that gap plainly rather than implying data that isn't there.

## Backend framing (explicit user decision)

**Gemini (`gemini-3.1-flash-lite`) is the primary/headline backend for every result.**
This is a deliberate reversal of how the phase docs are written — several phase docs call the
qwen2.5:14b numbers "the number to cite." For the paper: qwen results appear as a **cross-model
generalization check** (a row/column in each results table, or a short "holds across backends"
note), never as the primary citation.

Concretely:
- **C1 (extraction, Table 1):** Gemini-only. No qwen Table-1 run exists (Phase 2 never ran
  `eval/run_all.py`; qwen2.5:14b was only used for JD-analysis/rewrite, not extraction scoring).
  State this gap explicitly — do not imply a comparison that wasn't measured.
- **C2 (calibration):** Gemini headline (MAE 3.25 / ρ 0.333, 836s), qwen as the generalization
  row (MAE 3.17 / ρ 0.328, 39,162s) — near-identical calibration quality, 47x latency difference.
- **C3 (fabrication gate):** Gemini headline (gate-off 40 / gate-on 0), qwen as the generalization
  row (gate-off 31 / gate-on 0) — the structural guarantee (gate-on = 0) holds on both backends,
  which is the point of including qwen at all.

## Limitations section (explicit user decision)

Gemini's 4% (12/300) extraction failure rate gets a **named, mechanistic limitation**, not a
footnote:
- State the failure mode precisely: a deterministic repetition loop on ambiguous multi-line date
  formats, confirmed via direct reproduction and via identical failures across two independent
  full runs on the same document IDs.
- State that it hits `finishReason: MAX_TOKENS` and produces truncated/invalid JSON.
- State that the system's per-document isolation (Phase 7 resilience fix) contained the failure —
  it degrades to "12 documents skipped, logged," not a crashed run.
- This is presented as a real, reproducible finding about the primary backend, not downplayed.

Other limitations to include: hosted-API cost/quota caveat (Gemini is free here only because the
run stayed inside a free tier — contrast with the genuinely-zero-cost-at-any-scale Ollama/vLLM
path used elsewhere), small fabrication-benchmark sample (30 pairs), no fine-tuned small-model
comparison (Phase 2 gap), semantic-similarity thresholds never tuned against a labelled set
(Phase 3 note).

## Structure

1. **Abstract** — one paragraph. Three contributions (C1 continuous provenance chain, C2
   real-engine ATS-calibrated scoring, C3 provenance-verified rewrite gate), headline numbers,
   one line on cross-model validation via qwen2.5:14b.
2. **Introduction** — the résumé-optimization problem; why fabrication and opaque rewriting are
   real risks (grounded in `report.md`'s findings, esp. "Truthfulness is an engineering
   requirement" and the documented failure modes: temporal fabrication, cross-domain
   contamination, invented metrics); state the three contributions plainly.
3. **Related Work** — short, systems-paper-appropriate. From `report.md`: LLM extraction vs.
   traditional NER, constrained/structured decoding as the reliability lever, ATS
   parser+matcher architecture, résumé-rewriting fabrication risks.
4. **System Design** — the LangGraph pipeline (`ingest → extract → {jd, match} → score →
   rewrite+gate → review`), the provenance ID space as the throughline connecting all stages,
   the reviewer node's invariant check. Cite the frozen contracts from the design spec.
5. **Experimental Setup** — datasets (300 gold docs: 120 synthetic + 30 hand-labelled real + 150
   public human-annotated; 199 calibration pairs; 30 fabrication pairs), backends
   (`gemini-3.1-flash-lite` primary, `qwen2.5:14b` via Ollama for cross-validation).
6. **Results** — three subsections, C1/C2/C3, per the backend-framing rule above.
7. **Limitations** — per the section above.
8. **Conclusion.**

## Authorship / template placeholders

Author name(s) and affiliation are left as clearly-marked LaTeX placeholders
(`[Author Name]`, `[Affiliation]`) for the user to fill in before any real submission.

## Compilation

No LaTeX toolchain is installed on this machine (confirmed: no `pdflatex`/`xelatex`/`latexmk`).
The `.tex` will be written carefully using a minimal, well-established package set to keep compile
risk low, but **cannot be compiled or visually verified here**. The user will compile it
(Overleaf is the easiest zero-install path) and report back any errors for a fix pass.

## Out of scope for this spec

- Actually running any new experiments (all numbers come from existing phase docs / `eval/RESULTS.md`).
- Camera-ready formatting for a specific real venue (this is a generic NeurIPS-workshop-style draft).
- Figures beyond a pipeline architecture diagram (no plots/charts unless the user asks after seeing the draft).
