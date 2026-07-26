# Provenance-Chained Resume Optimization — Backend Design & Research Blueprint

**Date:** 2026-07-20
**Status:** Approved design, ready for phased implementation
**Related research report:** `../../../report.md`

---

## 1. Thesis & Contributions

**Working title:** *Provenance-Chained Resume Optimization: End-to-End Source-Grounded Extraction, Real-Engine ATS Calibration, and Verified Truthful Rewriting.*

### 1.1 The single new primitive

A **provenance ID space**. Every atomic value the system extracts or produces — a skill, a date, a job title, a company, a résumé bullet — carries a stable **provenance ID** pointing back to its exact source location: `(char_offset_start, char_offset_end, page, bbox)` in the original uploaded document. That ID survives **unbroken** through all four pipeline stages.

No prior résumé paper threads a single continuous ID space across extraction → scoring → rewriting → verification. Existing work has provenance *inside one stage only*:
- arXiv:2510.09722 (Alibaba SmartResume) — line-index pointers for *re-extraction* only.
- arXiv:2605.05257 (Career-Aware Resume Tailoring) — provenance *tracking* in rewrite, no enforcement gate, no ATS calibration.
- arXiv:2604.02539 (Synapse) — explainable retrieval + genetic optimization, no provenance chain.

### 1.2 Three contributions (mutually dependent)

- **C1 — Continuous provenance chain.** One ID space, extraction → final output. Integration novelty; the connective tissue that makes C2 and C3 possible.
- **C2 — Real-engine ATS-calibrated scoring** *(core novelty).* Predicted 0–100 score calibrated against the **actual parse + match output of self-hostable ATS engines** (multi-engine). First résumé score validated against real parser mechanics rather than a cosine/LLM-judge proxy.
- **C3 — Provenance-verified rewrite gate** *(core novelty).* Every rewrite edit must resolve to a source provenance ID or is **rejected**. Report a benchmarked **fabrication-rate** metric. Enforcement, not just tracking — "fabrication impossible by construction."

### 1.3 Honest framing (for reviewers, stated in the paper)

"Real ATS" = **self-hostable ATS engines** (OpenCATS-class + open résumé parsers + open matcher stacks), NOT closed commercial systems (Workday/Greenhouse/Taleo). Claimed as a **reproducible proxy** for real parser mechanics; the gap to commercial ATS is explicitly acknowledged as a limitation. This trades a slightly weaker "real Workday" claim for full reproducibility — reviewers can rerun every number.

### 1.4 Novelty guard (what keeps each claim publishable)

| Contribution | Prior art | Our delta |
|---|---|---|
| C1 chain | per-stage provenance only | one ID space, all 4 stages, unbroken |
| C2 scoring | cosine / LLM-judge proxy scores | calibrated vs real self-hostable ATS engine output |
| C3 rewrite | provenance *tracking* (2605.05257) | provenance *enforcement gate* + fabrication-rate benchmark |

---

## 2. Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Paper shape | One unifying primitive (provenance chain) | Single defensible thesis, not contribution-soup |
| Core novelty | Both C2 (ATS calibration) + C3 (verified rewrite), chained by C1 | Strongest survivable-against-literature combo |
| ATS ground truth | Self-hostable / open ATS engines | Reproducible, legal, reviewers can rerun |
| v1 scope | Full 4-stage pipeline, thin | End-to-end metrics for every claim |
| Extraction model | Self-hosted open (Qwen3 / Llama) via vLLM + Outlines | Reproducible, on-prem/PII-safe, matches open-ground-truth story |
| Backend stack | Python + FastAPI + LangGraph + Pydantic | Report's recommended stack, richest LLM/agent ecosystem |

---

## 3. Architecture

### 3.1 Data flow (end to end)

```
upload (PDF/DOCX/image)
  │
  ▼
[Stage 0] Ingestion + Provenance Anchoring
  Docling → markdown + per-token/char source map (offset, page, bbox)
  → assigns provenance IDs to every source span
  │  ProvenanceMap  (id → span)
  ▼
[Stage 1] Extraction (open LLM + Outlines constrained decoding)
  markdown → Resume JSON, each field carries prov_id back into ProvenanceMap
  → Pydantic validation; failures → review queue
  │  StructuredResume (+ provenance)          ┌── JD text
  ▼                                           ▼
                                     [Stage 1b] JD Analyzer
                                     JD → RequirementSet (must/nice, years, titles)
  │                                           │
  └──────────────┬────────────────────────────┘
                 ▼
[Stage 2] Scorer + ATS Calibration
  deterministic match (embeddings + keyword coverage + fuzzy)
  → raw component vector
  → CALIBRATION MODEL maps raw vector → predicted ATS score,
    trained/fit against real self-hostable ATS engine outputs
  │  MatchResult (0–100 + component breakdown + gap list, each gap prov-linked)
  ▼
[Stage 3] Verified Rewriter
  grounded LLM rewrite of bullets/sections
  → VERIFICATION GATE: every added skill/keyword/claim must resolve
    to an existing prov_id in source; unresolved → reject edit + log
  → fabrication-rate metric emitted
  │  TailoredResume (+ full provenance chain intact)
  ▼
[Stage 4] Orchestration + Reviewer (LangGraph)
  runs 0→1(+1b)→2→3, parallel resume/JD branches fan-in at scorer,
  reviewer node re-checks provenance chain end-to-end, final ATS score
  │
  ▼
API response: structured_resume, match_result, tailored_resume,
              provenance_chain, fabrication_report, final_score
```

### 3.2 The provenance ID space (C1 — the spine)

- **`ProvenanceMap`**: `prov_id -> SourceSpan{doc_id, char_start, char_end, page, bbox, raw_text}`.
- Every Pydantic model field that holds an extracted value gets a sibling `*_prov` field holding its `prov_id` (or list of IDs for multi-source values).
- IDs are content-addressed + monotonic (`p:<doc>:<seq>`), stable across a run.
- **Invariant (machine-checked):** at Stage 3 output, every token of "hard content" (skills, tools, orgs, numbers, dates) traces to ≥1 `prov_id` whose `raw_text` supports it. Reviewer node asserts this; violations = fabrication.

### 3.3 Components (one purpose each, independently testable)

| Component | Input | Output | Depends on |
|---|---|---|---|
| `ingestion` | file bytes | markdown + ProvenanceMap | Docling |
| `extractor` | markdown, ProvenanceMap | StructuredResume+prov | vLLM+Outlines, Pydantic |
| `jd_analyzer` | JD text | RequirementSet | vLLM+Outlines |
| `matcher` | resume+prov, RequirementSet | raw component vector + gaps | sentence-transformers, KeyBERT, RapidFuzz |
| `ats_harness` | resume file, JD | real-engine parse+match labels | open ATS engines (subprocess/API) |
| `calibrator` | raw vector | predicted 0–100 | fitted model + ats_harness labels |
| `rewriter` | resume+prov, gaps | TailoredResume + rejected-edit log | LLM, verification gate |
| `verifier` | TailoredResume+prov | fabrication_report, pass/fail | ProvenanceMap |
| `graph` | file, JD | full response | LangGraph, all above |
| `api` | HTTP request | JSON response | graph, FastAPI |

### 3.4 Error handling

- Extraction schema-invalid after N Outlines retries → manual-review queue, not silent fill.
- Ingestion structural-check fail (reading-order scramble heuristic) → vision-LLM fallback path.
- ATS harness engine crash on a doc → mark label missing, exclude from calibration fit (don't impute).
- Rewrite verification reject → keep original bullet, record in fabrication_report; never ship unverified edit.

### 3.5 Testing strategy

- Unit: each component against fixtures (golden resume + expected JSON + expected prov IDs).
- Provenance invariant test: property-based — no output hard-content token without a supporting prov_id.
- Calibration eval: held-out resumes, predicted score vs real-engine score (MAE, Spearman ρ).
- Fabrication benchmark: adversarial JD set designed to tempt fabrication; measure fabrication rate with/without gate (ablation).
- End-to-end: sample resume+JD → assert response shape + all invariants.

---

## 4. Datasets & Evaluation (paper deliverables)

- **Gold extraction set:** 100–300 resumes across formats, hand-labeled fields + provenance spans. Per-field F1, long-text F1 tracked separately.
- **ATS calibration set:** resumes × JDs run through ≥2 open ATS engines → (raw component vector, real-engine outcome) pairs for fitting + held-out eval.
- **Fabrication benchmark:** curated JD/resume pairs where keyword pressure tempts fabrication; ground-truth "what is / isn't in source."
- **Metrics:** extraction F1 (per-field + long-text); calibration MAE + Spearman ρ vs real engines; fabrication rate (gate on vs off) + rewrite quality; end-to-end latency.
- **Ablations:** (a) no provenance chain, (b) cosine score vs calibrated score, (c) rewrite gate off vs on.

---

## 5. Non-Goals (YAGNI)

- No frontend / no résumé builder UI (backend + API only).
- No commercial ATS integration (explicit limitation).
- No fine-tuned extraction model in v1 (open model + Outlines is enough for ~0.95 F1; fine-tune only if metrics demand).
- No vector DB / search infra unless a component needs it (matcher works in-memory at this scale).
- No auth / multi-tenant / production hardening in v1.

---

## 6. Phase Breakdown (implementation order)

Each phase is independently runnable and produces something testable. Later phases depend only on earlier ones.

### Phase 0 — Skeleton & contracts
- FastAPI app, config, logging.
- Define **all Pydantic models**: `SourceSpan`, `ProvenanceMap`, `StructuredResume(+prov)`, `RequirementSet`, `MatchResult`, `TailoredResume`, `FabricationReport`, `PipelineResponse`.
- Stub each component with typed signature + `NotImplementedError`.
- One end-to-end test asserting response *shape* (all stubs wired).
- **Deliverable:** app boots, `/health`, `/optimize` returns typed stub. Contracts frozen.

### Phase 1 — Ingestion + provenance anchoring (C1 foundation)
- Integrate Docling; route by file type (digital PDF/DOCX → Docling; image → OCR path).
- Emit markdown **plus** char/page/bbox source map.
- Build `ProvenanceMap`; assign `prov_id`s to source spans.
- Reading-order structural check + heuristic.
- Tests: fixtures for clean PDF, multi-column PDF, DOCX; assert spans map to correct source text.
- **Deliverable:** file → markdown + ProvenanceMap.

### Phase 2 — Extraction with provenance (C1 core)
- vLLM + Outlines constrained decoding, open model (Qwen3 / Llama).
- Prompt + schema (JSON Resume-anchored Pydantic); reasoning-fields-before-answer ordering.
- **Attach `prov_id` to every extracted field** (match extracted value back to source span; fuzzy-align).
- Pydantic validation + Outlines retry + review-queue fallback.
- Tests: gold set per-field F1; **provenance-attachment accuracy** (does each field point to the right span?).
- **Deliverable:** markdown+ProvMap → StructuredResume with provenance. First headline metric.

### Phase 3 — JD analyzer + deterministic matcher
- `jd_analyzer`: JD → RequirementSet (must/nice-to-have, years, titles) via same constrained-decoding stack.
- `matcher`: sentence-transformers (`all-mpnet-base-v2`) + KeyBERT + RapidFuzz → raw component vector (keyword coverage, semantic sim, fuzzy) + gap list; **each gap carries prov_id (or "absent")**.
- Tests: gap-list correctness on fixtures; semantic-equivalence cases (AWS≈cloud).
- **Deliverable:** (resume, JD) → raw component vector + prov-linked gaps.

### Phase 4 — ATS harness + calibration (C2, the hard core)
- Stand up ≥2 **self-hostable ATS engines**; wrapper (`ats_harness`) that feeds resume+JD, harvests real parse + match/rank output.
- Build calibration dataset (raw component vector → real-engine outcome).
- Fit `calibrator` (start simple: linear/logistic or small gradient model) mapping raw vector → predicted 0–100.
- Eval: held-out MAE + Spearman ρ vs real engines. Ablation: calibrated vs raw-cosine score.
- **Deliverable:** predicted score that tracks real engines. **C2 result.**

### Phase 5 — Verified rewriter (C3, the second core)
- Grounded rewrite prompt (master resume = single source of truth, truthfulness instruction).
- **Verification gate** (`verifier`): every added skill/keyword/entity/number in tailored output must resolve to an existing `prov_id` whose `raw_text` supports it; unresolved → reject edit, keep original, log.
- Emit `FabricationReport` + fabrication-rate metric.
- Build fabrication benchmark; ablation gate-on vs gate-off.
- **Deliverable:** truthful tailored resume + fabrication metric. **C3 result.**

### Phase 6 — LangGraph orchestration + reviewer
- Wire Stages 0→1(+1b)→2→3 into a LangGraph graph; parallel resume/JD branches, fan-in barrier at scorer.
- Checkpointing (PostgresSaver), human-in-the-loop pause point at review queue.
- **Reviewer node** re-asserts the end-to-end provenance invariant + final score.
- Tests: end-to-end on sample set, all invariants hold.
- **Deliverable:** full pipeline behind one API call.

### Phase 7 — Evaluation harness & paper tables
- Assemble gold set, calibration set, fabrication benchmark.
- Run all metrics + all ablations; emit result tables/figures.
- Cost/latency logging (cost-per-successful-task).
- **Deliverable:** every number the paper needs, reproducibly.

---

## 7. Open Questions (resolve during phases, not blocking)

- Exact open ATS engines to include (survey in Phase 4 — need ≥2 with harvestable parse+match output).
- Open extraction model + size (Qwen3 vs Llama; decide in Phase 2 by F1 vs latency).
- Provenance-attachment algorithm for extracted values (exact-match first, fuzzy-align fallback) — tune in Phase 2.
- Calibrator model family (linear vs tree) — pick by Phase 4 held-out fit.
