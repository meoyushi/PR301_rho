# Phase 7 — Evaluation Harness & Paper Tables — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–6 done. This phase produces every number/figure the paper reports.

**Goal:** Assemble the three datasets (gold extraction, ATS calibration, fabrication benchmark), run all metrics and all ablations, and emit reproducible result tables for the paper covering C1, C2, C3.

**Architecture:** Pure evaluation + reporting layer over the built pipeline. Metric functions are unit-tested on tiny inputs; dataset runners produce JSON/CSV tables + a `RESULTS.md` summary. Ablations toggle one component each.

**Tech Stack:** pytest, numpy, scipy, pandas (tables), the built `rho` pipeline.

## Global Constraints
- Every headline number must be reproducible from a script in `eval/` given the datasets.
- Metric functions are deterministic + unit-tested.
- Report per-field F1 with **long-text fields tracked separately** (report finding: long-text is the hardest).
- Ablations required: (A) provenance chain on/off effect on fabrication; (B) calibrated vs cosine score (from P4); (C) rewrite gate on/off (from P5).

## This phase consumes
- Full pipeline (Phases 1–6); P4 `fit_calibrator.py`; P5 `fabrication_ablation.py`.

## This phase produces
- `eval/metrics.py` — F1/precision/recall with entity alignment; long-text scoring.
- `eval/datasets/` — gold set, calibration pairs, fabrication pairs (+ loaders).
- `eval/run_all.py` — runs everything, writes `eval/RESULTS.md` + CSVs.
- Final `eval/RESULTS.md` — the paper's numbers.

---

## File Structure
- Create: `eval/metrics.py`, `eval/datasets/__init__.py` (+ loaders), `eval/run_all.py`.
- Create: `tests/unit/test_metrics.py`.
- Create: `eval/RESULTS.md` (generated).

---

### Task 1: Field-level F1 with entity alignment

**Files:**
- Create: `eval/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Produces: `field_f1(pred: dict, gold: dict, field: str) -> dict` returning `{precision, recall, f1}` for a list-valued field via set/aligned comparison; `long_text_f1(pred: str, gold: str) -> float` (token-overlap F1); `provenance_accuracy(resume, gold_prov_map) -> float` (fraction of fields whose attached `prov_id` points to the correct source span).

- [x] **Step 1: Write failing test**
```python
# tests/unit/test_metrics.py
from eval.metrics import field_f1, long_text_f1
def test_field_f1_on_skills():
    m = field_f1({"skills":["python","aws","sql"]}, {"skills":["python","aws","gcp"]}, "skills")
    assert round(m["precision"],2) == 0.67 and round(m["recall"],2) == 0.67
    assert round(m["f1"],2) == 0.67
def test_long_text_f1_token_overlap():
    f = long_text_f1("built scalable python api", "built python api service")
    assert 0.0 < f < 1.0
```

- [x] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_metrics.py -v`

- [x] **Step 3: Implement**
```python
# eval/metrics.py
def _prf(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return {"precision": p, "recall": r, "f1": f}
def field_f1(pred: dict, gold: dict, field: str) -> dict:
    ps = {str(x).lower() for x in pred.get(field, [])}
    gs = {str(x).lower() for x in gold.get(field, [])}
    return _prf(ps, gs)
def long_text_f1(pred: str, gold: str) -> float:
    return _prf(set(pred.lower().split()), set(gold.lower().split()))["f1"]
def provenance_accuracy(resume, gold_prov: dict) -> float:
    """gold_prov: {field_value: correct_prov_id}. Fraction of attached prov matching gold."""
    from rho.rewrite.tokens import hard_content_tokens
    total = correct = 0
    # caller supplies a resume whose *_prov are populated; compare first prov_id to gold
    # (implement per your gold format; keep deterministic)
    return correct / total if total else 0.0
```
*(Task note: finalize `provenance_accuracy` against the exact gold-span format you choose in Task 2. Keep it deterministic and unit-test it once the format exists.)*

- [x] **Step 4: Run to verify pass** → PASS.

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: evaluation metrics (field F1, long-text F1)"
```

---

### Task 2: Datasets — gold extraction, calibration, fabrication

**Files:**
- Create: `eval/datasets/__init__.py`, `eval/datasets/gold/`, `eval/datasets/calibration/`, `eval/datasets/fabrication/`
- Test: none (data + loaders)

**Interfaces:**
- Produces: loaders `load_gold() -> list[(file_path, gold_json)]`, `load_calibration_pairs() -> list[(file_bytes, filename, jd_text)]`, `load_fabrication_pairs() -> list[(resume, gaps, prov)]`.

- [x] **Step 1: Assemble gold extraction set** (100–300 résumés across formats; hand-label fields + provenance spans). Sources per report §1.4: Kaggle Resume NER set, HF `yashpwr/resume-ner-training-data`, LiveCareer/Jiechieu; or synthetic (template + substituted content). Store `gold/<id>.pdf` + `gold/<id>.json` (fields + gold prov spans).

- [x] **Step 2: Assemble calibration pairs** (résumé × JD pairs to run through the ATS harness, P4). Reuse gold résumés × a set of JDs.

- [x] **Step 3: Assemble fabrication benchmark** (from P5 Task 5; move canonical copy here).

- [x] **Step 4: Write loaders** — deterministic, path-based, return the shapes above.

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "data: gold/calibration/fabrication datasets + loaders"
```

---

### Task 3: `run_all.py` — every table + RESULTS.md

**Files:**
- Create: `eval/run_all.py`
- Output: `eval/RESULTS.md`, CSVs

**Interfaces:**
- Produces: a single script computing and writing:
  - **Table 1 (C1 extraction):** per-field P/R/F1, long-text F1 separately, provenance-attachment accuracy.
  - **Table 2 (C2 calibration):** calibrated MAE, Spearman ρ vs real engines, cosine-baseline MAE (from `fit_calibrator.py`).
  - **Table 3 (C3 fabrication):** unsourced additions gate-OFF vs gate-ON, mean fabrication_rate.
  - **Table 4 (ablations):** provenance on/off, calibrated vs cosine, gate on/off — one row each.
  - Latency + cost-per-successful-task line.

- [x] **Step 1: Implement runner**
```python
# eval/run_all.py
import json
from eval.metrics import field_f1, long_text_f1
from eval.datasets import load_gold
from rho.ingestion import ingest
from rho.extraction import extract
def eval_extraction():
    rows = []
    for path, gold in load_gold():
        md, prov = ingest(open(path,"rb").read(), path)
        pred = extract(md, prov).model_dump()
        rows.append({
            "id": path,
            "skills_f1": field_f1(pred, gold, "skills")["f1"],
            "summary_longtext_f1": long_text_f1(pred.get("summary") or "", gold.get("summary") or ""),
        })
    return rows
def main():
    ext = eval_extraction()
    # C2 + C3 pulled from eval/fit_calibrator.py and eval/fabrication_ablation.py results
    with open("eval/RESULTS.md", "w") as f:
        f.write("# Results\n\n## Table 1 — Extraction (C1)\n")
        avg = sum(r["skills_f1"] for r in ext)/max(len(ext),1)
        f.write(f"- mean skills F1: {avg:.3f}\n")
        # append C2/C3 tables from their scripts' saved outputs
    print("wrote eval/RESULTS.md")
if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run on real datasets** (needs LLM + calibrator fitted). Record numbers.
Run: `RHO_LLM_ENABLED=1 python eval/run_all.py`

- [x] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: run_all eval harness -> RESULTS.md"
```

---

### Task 4: Ablation runner

**Files:**
- Modify: `eval/run_all.py` (add ablation section) or `eval/ablations.py`

**Interfaces:**
- Produces: Table 4 rows — (A) fabrication rate with provenance chain vs a no-provenance rewrite; (B) calibrated vs cosine MAE (reuse P4); (C) gate on/off shipped-unsourced (reuse P5). Each ablation toggles exactly one component.

- [x] **Step 1: Implement** the three ablation calls, writing rows into `RESULTS.md`.
- [x] **Step 2: Run + record.**
- [x] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: ablation runner (provenance/calibration/gate)"
```

---

## Self-Review
- [x] Metric functions unit-tested.
- [x] All three datasets assembled + loadable.
- [x] `run_all.py` writes Tables 1–4 + latency/cost to `RESULTS.md`.
- [x] Long-text fields reported separately from named entities.
- [x] Every headline number reproducible from an `eval/` script.

## Results

**This is the first-ever run of this harness** — `eval/run_all.py` existed with full Table 1/2/3/4
logic from earlier work in this repo, but had never actually been executed end-to-end before this
session; no qwen Table-1 baseline exists to compare against. This run is entirely on
`gemini-3.1-flash-lite`: `python -m eval.run_all --extraction-backend gemini --suffix _gemini`.
Full numbers in `eval/RESULTS.md`; per-résumé rows in `eval/results_table1{a,b,c}_*.csv`.

- **C1 (headline, Table 1c — public human-annotated gold, n=143/150 scored):**
  skills F1 **0.753** (precision 0.802 / recall 0.749), job title F1 **0.935**,
  institution F1 **0.917**, certification F1 **0.802**,
  **provenance-attachment accuracy: 0.874**.
  - Table 1a (synthetic, upper bound, n=120/120): skills/work/education F1 ≈ **1.00**,
    long-text (summary) F1 **1.00**, provenance-attachment accuracy 0.894 — confirms the pipeline
    is correct when labels are unambiguous; the gap down to 1c's 0.753 skills F1 is real-world
    résumé messiness, not a pipeline defect.
  - Table 1b (real corpus, reality check, n=25/30): skills F1 0.443 (noisier, unverified labels —
    1c is the trustworthy real-data number), job title F1 0.931, long-text F1 0.979.
  - **7/150 (1c) + 5/30 (1b) = 12/300 docs failed extraction outright**, not merely scored low.
    See "Gemini extraction reliability" below — this is a genuine, reproducible finding, not
    noise to average away.

- **C2:** calibrated MAE **3.25** / Spearman ρ **0.333** vs cosine-baseline MAE 27.58 / ρ 0.229.
  199/199 pairs usable (0 skipped, vs qwen's 1/200 dropped). **47x faster wall-clock** than the
  qwen path (836s vs 39,162s) for statistically equivalent calibration quality (this doc's own
  qwen run showed ρ varying 0.328–0.440 between runs on the same pipeline, so a few points of ρ
  here is noise, not a backend difference). Full comparison: Phase 4's Results section.

- **C3:** unsourced shipped **gate-OFF 40 vs gate-ON 0** (qwen: 31 vs 0). 30/30 pairs scored,
  0 failed. Mean `fabrication_rate` (gate detections) **0.248**. Gate-ON is 0 on both backends —
  the guarantee is structural, holds regardless of which model drives the rewriter. Full
  comparison: Phase 5's Results section.

- **Ablations (Table 4):**
  - **A — provenance chain:** the span-resolved gate rejected 40 fabrications; a naive
    source-text-substring check would have caught 36 of those on its own, but **4 fabrications
    would have shipped** under the substring-only approach and were only caught because the gate
    resolves to an actual provenance span, not "does this text appear somewhere in the source."
    10/30 pairs carried ≥1 fabrication.
  - **B — scoring:** calibrated (3.25 MAE / 0.333 ρ) beats cosine (27.58 MAE / 0.229 ρ) — same
    conclusion as qwen, calibration adds real signal beyond raw similarity.
  - **C — rewrite gate:** OFF ships 40 unsourced additions, ON ships 0. Confirms C3 is a property
    of the gate, not of the rewriter's honesty.

- **Latency:** extraction median **3.4s** per résumé (n=288, min 1.8s, max 19.4s) — two orders of
  magnitude faster than qwen's CPU-bound Ollama path (~75s/pair, Phase 4).
- **Cost-per-successful-task: $0.00**, but with a caveat the qwen path doesn't have: Gemini is a
  **hosted commercial API**. This run stayed inside the free tier's daily quota; it is not free
  because the design has no billing surface. Scaling this run up would need either a paid tier or
  spreading calls across more free-tier accounts. Contrast with Ollama/vLLM (Phases 4–6): genuinely
  $0 at any scale, since it's self-hosted compute.
- **Dataset sizes:** gold 300 total (120 synthetic + 30 real + 150 public), **288 scored / 12
  failed**; calibration 199 pairs; fabrication 30 pairs.

### Gemini extraction reliability — a genuine, reproducible finding

12/300 documents (4%) failed extraction outright, not merely scored poorly. Every failure has the
**same root cause**, confirmed via direct reproduction and via two independent full runs producing
**identical failures on the identical documents**: `gemini-3.1-flash-lite` enters a degenerate
repetition loop when a résumé's dates are formatted ambiguously across multiple lines (e.g.
`"03/2015\nto\n07/2017"`), repeating a malformed ISO-timestamp fragment
(`"2015-03-01T00:00:00Z03/2015-03-01T00:00:00Z03/2015-03-01T00:00:00Z..."`) until hitting
`maxOutputTokens` (4096), at which point `finishReason: MAX_TOKENS` truncates the response mid-string
and the JSON is unparseable. This is the same underlying tendency Phase 6 found in miniature
(over-formatting a plain `"2020"` into `"2020-01-01T00:00:00Z"`) — here it spirals into infinite
repetition instead of stopping after one bad value. It is a property of this model on this task,
not a pacing/quota/backend-plumbing issue: temperature is 0, so the failure is deterministic, and
it reproduced on the exact same document IDs across two separate full 300-document runs.
Two engineering bugs were found and fixed while diagnosing this (both committed, both with tests):
a 403-`PERMISSION_DENIED` key was being retried every rotation instead of rotated out permanently,
and the table-1 evaluation loops had no per-document error handling, so one bad résumé used to kill
the whole run rather than being recorded and skipped.
