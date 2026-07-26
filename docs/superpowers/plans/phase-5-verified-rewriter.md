# Phase 5 — Verified Rewriter (Core Novelty C3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–4 done. **This is the second core novelty (C3).**

**Goal:** Rewrite/tailor the résumé toward the JD gaps, then pass it through a **provenance verification gate**: every hard-content token (skill, tool, org, number, date) in the rewritten output must resolve to a source `prov_id` whose `raw_text` supports it — unresolved additions are **rejected** (reverted), logged, and counted. Emit a `FabricationReport` with a `fabrication_rate`. Build a fabrication benchmark and run the gate-on / gate-off ablation.

**Architecture:** Rewriter = grounded LLM prompt (master résumé = single source of truth). It may reorder/rephrase/select/emphasize, never invent. After generation, the **verifier** diffs the tailored résumé against the source: it extracts hard-content tokens from the tailored text, and for each *newly-introduced* token checks for a supporting `prov_id` (reusing Phase 2's `find_prov`). Unsupported additions are rejected; the offending edit is reverted to the source value. The gate is deterministic and LLM-free — fully unit-testable.

**Tech Stack:** Outlines (rewriter, constrained), the Phase-2 `find_prov` provenance matcher, RapidFuzz.

## Global Constraints
- Implement `rho.rewrite.rewrite(resume, gaps) -> TailoredResume` and `rho.rewrite.verify(tailored, prov) -> FabricationReport`.
- **The gate is the contribution:** an added hard-content token with no supporting `prov_id` is ALWAYS rejected + reverted + counted. Never ship an unverified addition.
- `fabrication_rate = rejected_edits / total_edits` (total_edits = additions attempted).
- Rewriter temperature ≈ 0.6 (creative), but truthfulness enforced by the gate, not the prompt alone.

## This phase consumes
- `StructuredResume` (+ `*_prov`), `ProvenanceMap` (Phases 1–2).
- `Gap` list from `match()` (Phase 3) to know what to target.
- `find_prov` from `rho.extraction.provenance_attach` (Phase 2).

## This phase produces
- `rewrite()`, `verify()`; `rho.rewrite.tokens.hard_content_tokens(resume) -> list[tuple[str, path]]`.
- Fabrication benchmark under `tests/fixtures/fabrication/` + ablation script `eval/fabrication_ablation.py`.

---

## File Structure
- Create: `src/rho/rewrite/tokens.py` — hard-content token extraction from a StructuredResume.
- Create: `src/rho/rewrite/verifier.py` — the gate (`verify` + revert logic).
- Create: `src/rho/rewrite/llm.py` — grounded rewrite generation (Outlines).
- Modify: `src/rho/rewrite/__init__.py` — `rewrite()` orchestration + `verify` re-export.
- Create: `tests/unit/test_verifier.py`, `tests/integration/test_rewrite_llm.py`, `eval/fabrication_ablation.py`.

---

### Task 1: Hard-content token extraction

**Files:**
- Create: `src/rho/rewrite/tokens.py`
- Test: `tests/unit/test_verifier.py`

**Interfaces:**
- Produces: `hard_content_tokens(resume: StructuredResume) -> list[HardToken]` where `HardToken = (value: str, field_path: str)` — every skill, company, title, cert, date, and bullet-embedded tool/number that constitutes a factual claim. Start with structured fields (skills, companies, titles, certs, dates); bullets handled as whole-string claims in Task 2.

- [x] **Step 1: Write failing test**
```python
# tests/unit/test_verifier.py
from rho.models.resume import StructuredResume, WorkExperience
from rho.rewrite.tokens import hard_content_tokens
def test_hard_tokens_cover_skills_and_work():
    r = StructuredResume(name="A", skills=["Python","AWS"],
        certifications=["AWS SAA"],
        work=[WorkExperience(company="Acme", title="Engineer",
              start_date="2019", end_date="2022")])
    toks = hard_content_tokens(r)
    values = {t[0] for t in toks}
    assert {"Python","AWS","AWS SAA","Acme","Engineer"} <= values
```

- [x] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_verifier.py::test_hard_tokens_cover_skills_and_work -v` → FAIL.

- [x] **Step 3: Implement**
```python
# src/rho/rewrite/tokens.py
from rho.models.resume import StructuredResume
HardToken = tuple[str, str]      # (value, field_path)
def hard_content_tokens(resume: StructuredResume) -> list[HardToken]:
    toks: list[HardToken] = []
    for i, s in enumerate(resume.skills):
        toks.append((s, f"skills[{i}]"))
    for i, c in enumerate(resume.certifications):
        toks.append((c, f"certifications[{i}]"))
    for wi, w in enumerate(resume.work):
        toks.append((w.company, f"work[{wi}].company"))
        toks.append((w.title, f"work[{wi}].title"))
        for d in (w.start_date, w.end_date):
            if d: toks.append((d, f"work[{wi}].date"))
    for ei, e in enumerate(resume.education):
        toks.append((e.institution, f"education[{ei}].institution"))
    return [(v, p) for (v, p) in toks if v and v.strip()]
```

- [x] **Step 4: Run to verify pass** → PASS.
Run: `pytest tests/unit/test_verifier.py::test_hard_tokens_cover_skills_and_work -v`

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: hard-content token extraction for verification"
```

---

### Task 2: The verification gate (`verify`) — the C3 core

**Files:**
- Create: `src/rho/rewrite/verifier.py`; Modify: `src/rho/rewrite/__init__.py`
- Test: `tests/unit/test_verifier.py` (add)

**Interfaces:**
- Produces: `verify(tailored: StructuredResume, source: StructuredResume, prov: ProvenanceMap) -> tuple[StructuredResume, FabricationReport]`. Logic: for each hard token in `tailored` NOT already present in `source` (a new addition), check `find_prov(value, prov)`; if empty → **reject**: revert that field to source (or drop the added item), append `RejectedEdit`. `total_edits` = number of additions checked; `verified_edits` = additions that had provenance; `fabrication_rate` = rejected/total (0 if no additions).
- Re-export `verify` from `rho.rewrite` matching the frozen signature `verify(tailored, prov)`; provide source via closure in `rewrite()`.

- [x] **Step 1: Write failing test**
```python
# add to tests/unit/test_verifier.py
from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.rewrite.verifier import verify_against_source
def _prov():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=6, raw_text="Python"))
    return pm
def test_verify_rejects_unsupported_addition():
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python","Kubernetes"])  # k8s not in source/prov
    fixed, report = verify_against_source(tailored, source, _prov())
    assert "Kubernetes" not in fixed.skills          # reverted
    assert report.total_edits == 1
    assert report.verified_edits == 0
    assert report.fabrication_rate == 1.0
    assert report.rejected_edits[0].added_text == "Kubernetes"
def test_verify_keeps_supported_addition():
    pm = _prov(); pm.add(SourceSpan(doc_id="d", char_start=7, char_end=13, raw_text="FastAPI"))
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python","FastAPI"])
    fixed, report = verify_against_source(tailored, source, pm)
    assert "FastAPI" in fixed.skills
    assert report.verified_edits == 1
    assert report.fabrication_rate == 0.0
```

- [x] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_verifier.py -k verify -v`

- [x] **Step 3: Implement**
```python
# src/rho/rewrite/verifier.py
from rho.models.resume import StructuredResume
from rho.models.provenance import ProvenanceMap
from rho.models.rewrite import FabricationReport, RejectedEdit
from rho.rewrite.tokens import hard_content_tokens
from rho.extraction.provenance_attach import find_prov
def verify_against_source(tailored: StructuredResume, source: StructuredResume,
                          prov: ProvenanceMap):
    src_values = {v.lower() for v, _ in hard_content_tokens(source)}
    fixed = tailored.model_copy(deep=True)
    total = verified = 0
    rejected: list[RejectedEdit] = []
    # only skills list mutated here for clarity; extend to work/certs analogously
    kept_skills = []
    for s in fixed.skills:
        if s.lower() in src_values:
            kept_skills.append(s); continue
        total += 1
        if find_prov(s, prov):
            verified += 1; kept_skills.append(s)
        else:
            rejected.append(RejectedEdit(added_text=s, reason="no supporting prov_id"))
    fixed.skills = kept_skills
    report = FabricationReport(total_edits=total, verified_edits=verified,
        rejected_edits=rejected,
        fabrication_rate=(len(rejected)/total) if total else 0.0)
    return fixed, report
```
*(Task note: extend the same reject-if-no-prov loop to added certifications and to newly-introduced tools/numbers inside bullets. Skills shown as the canonical pattern; replicate for the other hard-content fields and cover each with a test.)*
Add to `src/rho/rewrite/__init__.py`:
```python
from rho.rewrite.verifier import verify_against_source
def verify(tailored, prov):        # frozen signature; source captured by rewrite()
    raise RuntimeError("call verify_against_source with the source resume; wired in rewrite()")
```

- [x] **Step 4: Run to verify pass** → PASS both.
Run: `pytest tests/unit/test_verifier.py -k verify -v`

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: provenance verification gate + fabrication report (C3)"
```

---

### Task 3: Grounded rewrite generation (Outlines)

**Files:**
- Create: `src/rho/rewrite/llm.py`
- Test: `tests/integration/test_rewrite_llm.py` (skips without model)

**Interfaces:**
- Produces: `rewrite_schema(source: StructuredResume, gaps) -> StructuredResume`. Constrained to the same resume schema. Prompt: master résumé is the ONLY source of truth; reorder/rephrase/select/emphasize toward the gaps; never invent skills/tools/metrics/dates; if a gap can't be satisfied truthfully, leave it. Temperature 0.6.

- [x] **Step 1: Skipping integration test**
```python
# tests/integration/test_rewrite_llm.py
import os, pytest
pytestmark = pytest.mark.skipif(os.getenv("RHO_LLM_ENABLED") != "1", reason="no LLM")
def test_rewrite_does_not_add_unsourced_skill():
    from rho.models.resume import StructuredResume
    from rho.rewrite.llm import rewrite_schema
    src = StructuredResume(name="A", skills=["Python"])
    out = rewrite_schema(src, gaps=[])
    # grounded prompt shouldn't invent; even if it does, gate catches it later
    assert "python" in [s.lower() for s in out.skills]
```

- [x] **Step 2: Run to verify skip.**
Run: `pytest tests/integration/test_rewrite_llm.py -v` → SKIP.

- [x] **Step 3: Implement** (mirror `extraction/llm.py`; constrained to a resume schema; grounding prompt; temperature 0.6). Note version deviations in Results.

- [x] **Step 4: Commit**
```bash
git add -A && git commit -m "feat: grounded rewrite generation"
```

---

### Task 4: `rewrite()` orchestration (generate → gate)

**Files:**
- Modify: `src/rho/rewrite/__init__.py`
- Test: `tests/unit/test_verifier.py` (add — fake rewriter)

**Interfaces:**
- Produces: `rewrite(resume, gaps, prov, _rewrite_fn=None) -> TailoredResume`. Runs `_rewrite_fn(resume, gaps)` (LLM or fake) → `verify_against_source(tailored, resume, prov)` → assembles `TailoredResume(resume=fixed, fabrication_report=report)`. **Note:** the frozen shared-context signature is `rewrite(resume, gaps)`; extend it to accept `prov` (update shared-context Section 6 + P6 caller to pass `prov`). Record this signature change in shared context.

- [x] **Step 1: Write failing test (fake rewriter injects a fabrication)**
```python
# add to tests/unit/test_verifier.py
from rho.rewrite import rewrite
def test_rewrite_gate_strips_fabrication():
    source = StructuredResume(name="A", skills=["Python"])
    fake = lambda r, g: StructuredResume(name="A", skills=["Python","GoLang"])  # GoLang invented
    tr = rewrite(source, [], _prov(), _rewrite_fn=fake)
    assert "GoLang" not in tr.resume.skills
    assert tr.fabrication_report.fabrication_rate == 1.0
```

- [x] **Step 2: Run to verify fail** → FAIL.
Run: `pytest tests/unit/test_verifier.py::test_rewrite_gate_strips_fabrication -v`

- [x] **Step 3: Implement**
```python
# src/rho/rewrite/__init__.py  (replace stub)
from rho.models.rewrite import TailoredResume
from rho.rewrite.verifier import verify_against_source
def rewrite(resume, gaps, prov, _rewrite_fn=None) -> TailoredResume:
    if _rewrite_fn is None:
        from rho.rewrite.llm import rewrite_schema as _rewrite_fn
    tailored = _rewrite_fn(resume, gaps)
    fixed, report = verify_against_source(tailored, resume, prov)
    return TailoredResume(resume=fixed, fabrication_report=report)
```
Update `00-SHARED-CONTEXT.md` Section 6 signature to `rewrite(resume, gaps, prov)`.

- [x] **Step 4: Run to verify pass** → PASS.

- [x] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: rewrite() orchestration with verification gate"
```

---

### Task 5: Fabrication benchmark + gate on/off ablation

**Files:**
- Create: `tests/fixtures/fabrication/` (curated résumé+JD pairs where keyword pressure tempts fabrication), `eval/fabrication_ablation.py`
- Test: none new (ablation is an eval script for the paper)

**Interfaces:**
- Produces: `eval/fabrication_ablation.py` running the real rewriter twice — gate ON (`verify_against_source` applied) vs gate OFF (raw rewrite) — reporting fabrication rate and count of unsourced additions in each. This is the headline C3 number.

- [x] **Step 1: Build 10–30 adversarial pairs.** Each: a résumé + a JD demanding skills the résumé lacks (tempting the model to invent). Record ground-truth source skills.

- [x] **Step 2: Write ablation script**
```python
# eval/fabrication_ablation.py
"""Gate ON vs OFF fabrication comparison (C3 headline)."""
from rho.rewrite.llm import rewrite_schema
from rho.rewrite.verifier import verify_against_source
from rho.rewrite.tokens import hard_content_tokens
def unsourced_count(resume, source, prov):
    _, rep = verify_against_source(resume, source, prov)
    return rep.total_edits - rep.verified_edits
def run(pairs):   # pairs: list[(source_resume, gaps, prov)]
    off_total = on_total = 0
    for source, gaps, prov in pairs:
        raw = rewrite_schema(source, gaps)                 # gate OFF
        off_total += unsourced_count(raw, source, prov)
        fixed, rep = verify_against_source(raw, source, prov)  # gate ON
        on_total += (rep.total_edits - rep.verified_edits) - len(rep.rejected_edits)
    print(f"unsourced additions shipped  gate-OFF={off_total}  gate-ON={on_total}")
```
(gate-ON shipped unsourced should be 0 by construction — that IS the claim.)

- [x] **Step 3: Commit**
```bash
git add -A && git commit -m "feat: fabrication benchmark + gate ablation (C3)"
```

---

## Self-Review
- [x] Gate rejects every unsourced hard-content addition (skills + certs + bullet tools).
- [x] `fabrication_rate` computed correctly; gate-ON ships zero unsourced additions.
- [x] `rewrite()` returns `TailoredResume` with intact provenance + report.
- [x] Shared-context Section 6 updated for `rewrite(resume, gaps, prov)`.
- [x] `pytest tests/unit/test_verifier.py -v` green.

## Results (the C3 numbers the paper needs)

### Headline — `qwen2.5:14b` on corpus-backed pairs (the number to cite)

- **Rewriter:** `qwen2.5:14b` via Ollama, temperature 0.6, JSON-schema-constrained decoding
  (Ollama `format`), CPU-only. Artifact: `eval/fabrication_results_corpus.json`.
- **Benchmark:** 30 corpus-backed pairs (`eval/fabrication_corpus.py`, seed 0) — real résumés from
  `Resume.csv` with populated work history, bullets, and education, each paired with a JD from
  `training_data.csv`. Gaps come from the real Phase-3 path (`analyze_jd` → `match`), not a
  hand-written "tempting" list, so the pressure on the rewriter is whatever the JD actually demands
  and the résumé actually lacks. Provenance is rebuilt over the résumé's own values via `ingest()`,
  so spans are production-shaped. 30 scored, 0 failed.
- **Unsourced additions shipped: gate-OFF 31 vs gate-ON 0** ← headline C3.
  Gate-ON is 0 **by construction** — that is the claim, and it held on every one of the 30 pairs.
  The guarantee is structural, not a model behaviour that happened to hold on this sample.
- **Mean fabrication_rate (gate detects): 0.407** across the 30 pairs.
  **14 of 30 pairs** carried at least one fabrication; the worst single pair shipped **7**.
- **These are real fabrications, not paraphrases** (the bullet-gate false-positive class is fixed —
  see deviation 3b). Rejected-and-reverted examples: invented skills (`Microsoft Access`,
  `Microsoft Outlook`), invented experience (`Led the development of a private banking group focused
  on Act 20 and Act 22 clients`, `Operated standard office equipment such as 10-key calculator for
  cash handling`), and even a leaked prompt-reasoning artifact
  (`>// Emphasize content creation as it's closest to the target requirements…`). None reached output.

### Gemini re-run — same 30 corpus pairs, `gemini-3.1-flash-lite`

Same benchmark, same seed, same real Phase-3 gap path — only the rewriter (and JD-analysis feeding
it) swapped to Gemini. `eval/fabrication_ablation.py --backend gemini --corpus 30 --seed 0`.

| | qwen2.5:14b (Ollama) | Gemini (`gemini-3.1-flash-lite`) |
|---|---|---|
| Pairs scored / failed | 30 / 0 | **30 / 0** |
| Unsourced shipped, gate-OFF | 31 | **40** |
| Unsourced shipped, gate-ON | 0 | **0** |
| Mean fabrication_rate | 0.407 | **0.248** |
| Pairs with ≥1 fabrication | 14 / 30 | **10 / 30** |
| Wall-clock | minutes (CPU, serial per pair) | **seconds** (5 concurrent workers, hosted) |

**Gate-ON is 0 on both backends — the headline claim holds regardless of which model drives the
rewriter**, which is the point: C3 is a property of the gate, not of the rewriter's honesty.
Gemini's raw output is *more* prone to fabrication by absolute count (40 vs 31 unsourced additions)
despite a lower mean rate over fewer affected pairs (0.248 over 30 vs 0.407 over 30, but concentrated
in 10 pairs instead of 14) — a rewriter that fabricates harder on the pairs it does fabricate on,
not a more honest one. Concrete examples the gate rejected (never reached output): invented skills
(`Microsoft Word`, `Microsoft Excel`, `Microsoft Outlook`, `Project Management`, `Construction
Practices`), invented soft-skill claims (`Team Leadership`, `Team Collaboration`, `Marketing Event
Coordination`), and invented bullet-level claims (`Institutional kitchen operations`, `Performed
tasks requiring basic math and numerical proficiency using 10-key`). Same shape of fabrication as
the qwen run — invented skills and restated-as-fact soft skills — just more of it per affected pair.
Artifact: `eval/fabrication_results_corpus_gemini.json`.

**Model:** `gemini-3.1-flash-lite`, same as Phase 4.

### Earlier run — `gemma3:4b` on the 12-pair synthetic fixture (retained for comparison)

- **Fabrication benchmark size:** 12 adversarial pairs (`tests/fixtures/fabrication/pairs.json`).
  Each is a résumé whose JD demands skills the résumé does not have — 61 absent requirements in
  total. Provenance is built by running the real `ingest()` path over the résumé text, so the gate
  sees production-shaped spans, not a hand-written map.
- **Unsourced additions shipped: gate-OFF 15 vs gate-ON 0** (no-gaps condition)
  Adversarial condition (gap list in the prompt): **gate-OFF 3 vs gate-ON 0**.
  Gate-ON is 0 in both conditions **by construction**.
- **Mean fabrication_rate (gate detects):** 0.562 without gaps in the prompt, 0.083 with them.
- **Rewriter:** `gemma3:4b` via Ollama, temperature 0.6, JSON-schema-constrained decoding.
  Artifacts: `eval/fabrication_results.json` (gaps), `eval/fabrication_results_nogaps.json`.
  **Caveat:** these `gemma3:4b` counts predate the bullet-gate fix (deviation 3b) and may include
  rephrasings wrongly counted as fabrications on any bullet-derived rejection; the structured-field
  rejections (invented employers/titles/universities) are unaffected. The `qwen2.5:14b` corpus
  numbers above are post-fix and are the ones to cite.

### What the gate actually caught
Rejections are not near-miss paraphrases — they are whole invented facts. Across the no-gaps run
the model fabricated employers the résumé never mentions (`GlobalTech Solutions`, `Acme Corp`,
`Tech Solutions Inc.`, `Apex Electronics`) and titles it was never given (`Software Engineer II`,
`Junior Data Analyst`). The single failing adversarial pair (`fullstack-node`) invented an employer,
a title, **and** a university (`University of California, Berkeley`) in one generation. Every one
was rejected and reverted; none reached the output.

### The counterintuitive finding (worth reporting)
Naming the missing requirements in the prompt *reduced* fabrication (15 → 3 unsourced additions).
The mechanism is visible in the edit counts, not just the rates: with the gap list the model
attempted **3 additions total** versus **24** without it. Stating the targets alongside "never
invent" appears to make the instruction concrete enough to follow, so the model declines to fill
the gaps at all rather than inventing content to cover them. The mean fabrication_rate drop
(0.562 → 0.083) is therefore mostly a drop in *attempts*, not an improvement in per-attempt
honesty — the rate's denominator shrinks with it, which is why both the rate and the raw count
are reported.

**This is exactly why the gate is the contribution and the prompt is not.** Prompt grounding moved
fabrication from 15 to 3; it never reached 0, and its effect was an accident of phrasing rather
than a guarantee. The gate reached 0 in both conditions because it cannot do otherwise.

### Deviations from the plan (each deliberate, with a reason)

0. **The cited rewriter is `qwen2.5:14b` via Ollama; `gemma3:4b` is the earlier fixture run.**
   A Groq `qwen/qwen3.6-27b` backend was built and fully debugged (`rho.llm.groq`,
   `rho.jd.groq`, `rho.rewrite.groq`) but **never produced the headline numbers** — the free tier's
   daily token pool (TPD 200000, shared across all keys) was exhausted before a clean 30-pair run
   completed, so no Groq fabrication figure is reported. The pipeline was therefore run on local
   `qwen2.5:14b` (no quota, real constrained decoding via Ollama's `format`, a real named model).
   The Groq findings below are retained because they document the backend's behaviour and the
   reasons it was set aside, not because any result depends on them:
   - **Cloudflare rejects `urllib`.** Requests via `urllib.request` return `403 / error code: 1010`
     *before* Groq evaluates the key — a bogus key and a valid key fail identically, which is how
     the cause was isolated. `httpx` and `curl` pass, so `httpx` moved from a dev extra to a runtime
     dependency and `rho.llm.groq` uses it instead of the stdlib used elsewhere in this repo.
   - **Qwen3.6 is a reasoning model and does NOT support `json_schema`.** It emits `<think>…</think>`
     inline, which breaks `json_object` validation too. `reasoning_format=hidden` suppresses it
     server-side. Consequence for the paper: on this backend the schema is enforced by Pydantic
     *after* generation, not by constrained decoding — weaker than the Ollama path, and the reason
     `_coerce`/`_parse` drop malformed items rather than defaulting them.
   - **The free tier caps tokens per minute (8k), account-wide.** Key rotation does not help: all
     five keys draw on one budget. An initial retry-on-429 design was actively harmful — a 2h45m run
     accumulated `00:00:00` CPU time, entirely asleep. `TokenBudget` now paces requests *before*
     sending, and 429s honour the server's own reset header instead of a guessed backoff.

1. **Ollama instead of vLLM + Outlines** — same deviation Phase 4 took, same cause: the host has no
   CUDA (`torch.cuda.is_available()` is `False`), so `outlines.models.vllm` cannot load. Ollama's
   `format` parameter enforces the JSON schema during decoding, so this is still constrained
   generation, and the C3 ablation actually runs on this machine.
2. **The gate needs more than `find_prov`.** The plan's sketch accepts any addition for which
   `find_prov` returns a hit. That is too permissive to be a gate: `find_prov` scores with
   RapidFuzz `partial_ratio`, which matches the best *substring*, so a span reading `"Engineer"`
   scores 100 against the fabricated promotion `"Staff Engineer"`, and `"Python"` licenses
   `"Senior Python Developer"`. The verifier therefore additionally requires that **every content
   word** of an added value appear in the supporting span (`_WORD_MATCH`, stopwords exempt).
   Pinned by `test_verify_rejects_seniority_inflation`.
3. **Bullets are checked against source bullets, not `find_prov`.** Same `partial_ratio` failure
   mode, worse: a fabricated bullet passes if any short span appears inside it — "Led a team of 40
   engineers" scores 100 against a span reading "Engineer". Whole bullets are compared with
   `token_set_ratio` against the source bullets instead, so rephrasing survives and new claims do
   not. Pinned by `test_verify_rejects_unsupported_bullet`.

3b. **The bullet threshold was wrong and produced false positives (found on corpus data).**
   `_BULLET_SIMILARITY = 90` rejected *every genuine rephrasing* as a fabrication. Measured on real
   corpus rewrites: honest rephrasings of a source bullet score **68–87**, inventions score
   **37–42** — 90 sat above the entire legitimate class. Example wrongly rejected:
   `"Optimized and tuned Teradata and Oracle views and SQL queries…"` against source
   `"Worked on optimizing and tuning the Teradata and Oracle views and SQL's…"` — the same claim,
   reworded, which is precisely what the rewriter is licensed to do.

   Two purely lexical repairs were tried and both failed: requiring every content word to be sourced
   flags ordinary synonym choice (`improve` → `enhance`), and stem matching still cannot bridge
   irregular pairs (`SQL's` → `queries`). The rule was then re-derived from what C3 actually claims.
   **A bullet now ships when it tracks a source bullet (≥60) *and* introduces no unsourced
   hard-content token** — tool, org, acronym, number, or date. Prose wording is free; new facts are
   not. `"Led a team of 40 Teradata engineers"` still fails, on the unsourced `40`. Pinned by
   `test_gate_accepts_genuine_rephrasing_of_a_source_bullet` and
   `test_gate_still_rejects_invention_reusing_source_vocabulary`.

   **This inflated the earlier gemma3 gate-OFF counts.** Any bullet-derived rejection in the numbers
   above may be a rephrasing rather than a fabrication; the employer/title/university rejections
   (structured fields, not bullets) are unaffected and remain real.
4. **Gate-ON is measured, not assumed.** The plan's ablation computes gate-ON as
   `(total_edits - verified_edits) - len(rejected_edits)`, which is identically 0 by arithmetic —
   it would report success even if the gate leaked. `unsourced_count` re-verifies the *gated
   résumé* instead, so a leak would show up as a non-zero number. Pinned by
   `test_unsourced_count_is_zero_after_gating`.
5. **`verify()` raises `RuntimeError`.** The frozen Section-6 signature `verify(tailored, prov)`
   cannot distinguish an addition from a reorder, because that needs the source résumé. Rather
   than silently score every value as new (and report a meaningless fabrication rate), it fails
   loudly and directs callers to `verify_against_source`. `rewrite()` wires it correctly.
6. **`rewrite()` takes `prov`** — recorded in shared-context Section 6. P6 must pass it.

### Limitations
- 30 corpus pairs is still a modest sample. The gate-ON=0 result does not depend on sample size
  (it is structural — it held on all 30), but the gate-OFF count and the 0.407 fabrication_rate do:
  treat **31 over 30 pairs** and **14/30 pairs affected** as an illustration of fabrication pressure
  under this model, not a population estimate.
- **The curated 12-pair fixture no longer discriminates on a strong model.** Those résumés carry
  skills only — no work history, no bullets — so the gate's work/bullet/education paths were never
  exercised by real generated text. Corpus-backed pairs (`eval/fabrication_corpus.py`, résumés with
  populated work history and JD-derived gaps) are the benchmark that still has signal, and are what
  the `qwen2.5:14b` headline numbers are drawn from.
- One rewriter model at one temperature. The gate-OFF count is a property of `qwen2.5:14b` at
  temperature 0.6, not of grounded prompting in general. (An earlier `gemma3:4b` fixture run is
  retained above for comparison only.)
- The gate verifies *provenance*, not *semantics*: a value copied from an unrelated part of the
  source document passes. It stops invention, not misattribution.
