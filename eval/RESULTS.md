# Phase 7 — Evaluation Results

_Generated 2026-07-24T14:18:19+00:00 by `python -m eval.run_all`. Every number below is recomputed from the datasets in `eval/datasets/` and the artifacts of Phases 4–5._

**Extraction backend:** `gemini`


## Table 1 — Extraction quality (C1)

Named-entity fields and long-text fields are reported **separately**: prose is rewritten rather than copied, so averaging it into the entity score would hide that long-text is the hardest field class.


### Table 1c — Public human-annotated gold set (headline C1)

`kens1ang/resume-ner-labelled`, n=143 résumés scored (7 failed extraction, excluded). Annotations are independent of this system and of its authors.

| Field | F1 |
|---|---|
| skills | 0.753 |
| job title | 0.935 |
| institution | 0.917 |
| certification | 0.802 |

- skills precision 0.802 / recall 0.749
- **provenance-attachment accuracy: 0.874**

### Table 1a — Synthetic gold set (upper bound)

n=120 scored. Labels and provenance spans are exact by construction. Templated résumés are cleaner than real ones, so **treat this as an upper bound**, not an estimate of field performance.

| Field | F1 |
|---|---|
| skills | 0.999 |
| work (company+title) | 1.000 |
| work — company only | 1.000 |
| work — title only | 1.000 |
| education (institution) | 1.000 |
| name (exact match) | 1.000 |
| **summary (long-text F1)** | **1.000** |

- **provenance-attachment accuracy: 0.894**

### Table 1b — Hand-labelled real corpus subset (reality check)

n=25 résumés from `Resume.csv` scored (5 failed extraction, excluded). Labelled by the implementing agent, **not** independently verified — the public set (1c) is the trustworthy real-data number. `name` and `work[].company` are unlabelable: the corpus anonymises them.

| Field | F1 |
|---|---|
| skills | 0.443 |
| job title | 0.931 |
| institution | 0.693 |
| **summary (long-text F1)** | **0.979** |

## Table 2 — ATS calibration (C2)

Target `keywordMatch`; 199 usable pairs (139 train / 60 held-out). Held-out target mean 13.36 (sd 4.88).

| Scorer | MAE | Spearman ρ |
|---|---|---|
| **calibrated (rho)** | **3.25** | **0.333** |
| cosine baseline | 27.58 | 0.229 |

_Secondary (composite `overallScore` target): calibrated MAE 4.62 ρ 0.185 vs cosine MAE 12.17 ρ -0.051._

**Read ρ, not MAE, as the headline.** The cosine baseline emits 0–100 while the target band is narrow and low, so most of its MAE is scale mismatch rather than ranking failure.

## Table 3 — Fabrication gate (C3)

Benchmark `fabrication_results_corpus_gemini.json`, 30 pairs scored (0 failed), backend `gemini`.

| Condition | Unsourced additions shipped |
|---|---|
| gate OFF (prompt grounding only) | 40 |
| **gate ON** | **0** |

- mean `fabrication_rate` (gate detections): 0.248
- Gate-ON is 0 **by construction** — that is the claim. The informative number is gate-OFF: how often a grounded prompt alone would have shipped a fabrication.

## Table 4 — Ablations

| Ablation | Condition | Metric | Value |
|---|---|---|---|
| A. provenance chain | ON (span-resolved evidence) | fabrications rejected | 40 |
| A. provenance chain | OFF (source-text substring check) | fabrications rejected | 36 |
| A. provenance chain | OFF | would ship despite the gate (of 40 checked) | 4 |
| A. provenance chain | — | pairs with ≥1 fabrication (of 30) | 10 |
| B. scoring | calibrated (rho) | MAE / Spearman ρ | 3.25 / 0.333 |
| B. scoring | cosine baseline | MAE / Spearman ρ | 27.58 / 0.229 |
| C. rewrite gate | OFF | unsourced shipped | 40 |
| C. rewrite gate | ON | unsourced shipped | 0 |
| C. rewrite gate | ON | mean fabrication_rate detected | 0.248 |

## Latency and cost

- Extraction: median **3.4s** per résumé (n=288, min 1.8s, max 19.4s).
- **Cost per successful task: $0.00 in API spend** (backend `gemini`, free-tier quota). Unlike the Ollama/vLLM path, this backend is a hosted commercial API — free here because the run stayed inside the provider's free-tier daily quota, not because the design has no billing surface. Reproducing this run at larger scale would need either a paid tier or spreading calls across more free-tier accounts.

## Dataset sizes

- gold — public (human-annotated): 150
- gold — synthetic: 120
- gold — hand-labelled real: 30
- calibration pairs: 199
- fabrication pairs: 30
