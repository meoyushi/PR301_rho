# Phase 3 — JD Analyzer + Deterministic Matcher — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–2 done.

**Goal:** (a) Turn a job description into a structured `RequirementSet`. (b) Match a `StructuredResume` against it, producing a raw `ComponentVector` (keyword coverage, semantic similarity, fuzzy coverage, must/nice coverage) and a prov-linked `Gap` list. `predicted_score` stays `0.0` here — Phase 4 sets it.

**Architecture:** JD analyzer reuses the constrained-decoding stack (Outlines) to emit a `RequirementSet`. The matcher is deterministic and LLM-free: embed résumé skills/text and each requirement with sentence-transformers, compute cosine similarity; keyword coverage via KeyBERT-extracted JD terms present in résumé; fuzzy coverage via RapidFuzz; each requirement classified present/weak/absent with `evidence_prov` = the résumé skill's `prov_id`s that satisfy it.

**Tech Stack:** sentence-transformers (`all-mpnet-base-v2`), KeyBERT, RapidFuzz, Outlines (JD analyzer).

## Global Constraints
- Implement `rho.jd.analyze_jd(jd_text) -> RequirementSet` and `rho.matching.match(resume, reqs) -> MatchResult`.
- Matcher is deterministic + fully unit-testable without an LLM.
- Every `Gap` with status `present`/`weak` must carry `evidence_prov` (the résumé value's prov_ids) — provenance chain continues here.
- `MatchResult.predicted_score` left `0.0` (Phase 4 owns it).

## This phase consumes
- `StructuredResume` (+ `*_prov`) from Phase 2.
- Models `RequirementSet`, `Requirement`, `ComponentVector`, `Gap`, `MatchResult` (Phase 0).

## This phase produces
- `analyze_jd()`; `match()`.
- `rho.matching.embed.Embedder` (cached sentence-transformer wrapper).
- Reusable semantic-similarity + coverage helpers for Phase 4 calibration features.

---

## File Structure
- Create: `src/rho/jd/schema.py` — JD LLM output schema.
- Modify: `src/rho/jd/__init__.py` — `analyze_jd`.
- Create: `src/rho/matching/embed.py` — Embedder (lazy, cached).
- Create: `src/rho/matching/coverage.py` — keyword + fuzzy coverage helpers.
- Modify: `src/rho/matching/__init__.py` — `match`.
- Create: `tests/unit/test_matching.py`, `tests/unit/test_jd.py`.

---

### Task 1: JD analyzer schema + `analyze_jd`

**Files:**
- Create: `src/rho/jd/schema.py`; Modify: `src/rho/jd/__init__.py`
- Test: `tests/unit/test_jd.py`

**Interfaces:**
- Produces: `analyze_jd(jd_text, _schema_fn=None) -> RequirementSet`. Injectable `_schema_fn` for LLM-free tests. LLM path constrained to a `JDSchema` (reasoning-first) mapped to `RequirementSet`.

- [ ] **Step 1: Write failing test (fake LLM)**
```python
# tests/unit/test_jd.py
from rho.jd import analyze_jd
from rho.jd.schema import JDSchema
def test_analyze_jd_maps_requirements():
    fake = lambda t: JDSchema(reasoning="x", title="Backend Engineer",
        requirements=[
            {"text":"Python","kind":"skill","priority":"must","years":None},
            {"text":"AWS","kind":"skill","priority":"nice","years":None},
            {"text":"5 years backend","kind":"experience","priority":"must","years":5.0},
        ])
    rs = analyze_jd("...", _schema_fn=fake)
    assert rs.title == "Backend Engineer"
    musts = [r for r in rs.requirements if r.priority == "must"]
    assert len(musts) == 2
    assert any(r.years == 5.0 for r in rs.requirements)
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_jd.py -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**
```python
# src/rho/jd/schema.py
from pydantic import BaseModel
from typing import Literal
from rho.models.jd import RequirementSet, Requirement
class ReqItem(BaseModel):
    text: str
    kind: Literal["skill","tool","title","cert","experience"]
    priority: Literal["must","nice"]
    years: float | None = None
class JDSchema(BaseModel):
    reasoning: str
    title: str | None = None
    requirements: list[ReqItem] = []
def to_requirement_set(js: JDSchema) -> RequirementSet:
    return RequirementSet(title=js.title,
        requirements=[Requirement(text=r.text, kind=r.kind, priority=r.priority, years=r.years)
                      for r in js.requirements])
```
```python
# src/rho/jd/__init__.py
from rho.models.jd import RequirementSet
from rho.jd.schema import to_requirement_set
def analyze_jd(jd_text: str, _schema_fn=None) -> RequirementSet:
    if _schema_fn is None:
        from rho.jd.llm import analyze_jd_schema as _schema_fn   # optional, mirrors extraction/llm.py
    js = _schema_fn(jd_text)
    return to_requirement_set(js)
```
Create `src/rho/jd/llm.py` mirroring `extraction/llm.py` but constrained to `JDSchema` with a JD-extraction prompt (reasoning first; classify must vs nice; pull years). Same skip-if-no-model discipline.

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_jd.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: JD analyzer -> RequirementSet"
```

---

### Task 2: Embedder + semantic similarity

**Files:**
- Create: `src/rho/matching/embed.py`
- Test: `tests/unit/test_matching.py`

**Interfaces:**
- Produces: `Embedder` with `.encode(list[str]) -> np.ndarray` and `cosine(a, b) -> float`. Model `all-mpnet-base-v2`, lazy-loaded, singleton.

- [ ] **Step 1: Add deps**
Add `sentence-transformers>=3`, `keybert>=0.8`, `numpy` to `pyproject.toml`; install.

- [ ] **Step 2: Write failing test**
```python
# tests/unit/test_matching.py
from rho.matching.embed import Embedder
def test_semantic_similarity_high_for_synonyms():
    e = Embedder()
    v = e.encode(["machine learning model development", "AWS cloud platform"])
    sim = e.cosine(e.encode(["ML model building"])[0], v[0])
    assert sim > 0.4         # synonym-ish should beat unrelated
```

- [ ] **Step 3: Run to verify fail**
Run: `pytest tests/unit/test_matching.py::test_semantic_similarity_high_for_synonyms -v`
Expected: FAIL — not defined.

- [ ] **Step 4: Implement**
```python
# src/rho/matching/embed.py
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer("all-mpnet-base-v2")
class Embedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return _model().encode(texts, normalize_embeddings=True)
    def cosine(self, a, b) -> float:
        return float(np.dot(a, b))     # already normalized
```

- [ ] **Step 5: Run to verify pass**
Run: `pytest tests/unit/test_matching.py::test_semantic_similarity_high_for_synonyms -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: sentence-transformer Embedder + cosine"
```

---

### Task 3: Coverage helpers (keyword + fuzzy)

**Files:**
- Create: `src/rho/matching/coverage.py`
- Test: `tests/unit/test_matching.py` (add)

**Interfaces:**
- Produces: `keyword_coverage(req_terms: list[str], resume_skills: list[str]) -> float` (fraction of req terms literally present, case-insensitive substring). `fuzzy_coverage(req_terms, resume_skills, threshold=85) -> float` (RapidFuzz).

- [ ] **Step 1: Write failing test**
```python
# add to tests/unit/test_matching.py
from rho.matching.coverage import keyword_coverage, fuzzy_coverage
def test_keyword_and_fuzzy_coverage():
    reqs = ["Python", "Kubernetes", "AWS"]
    skills = ["python", "aws", "kubernets"]   # last is a typo
    assert keyword_coverage(reqs, skills) == 2/3     # Python, AWS exact; Kubernetes no
    assert fuzzy_coverage(reqs, skills) == 1.0       # typo caught by fuzzy
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_matching.py -k coverage -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**
```python
# src/rho/matching/coverage.py
from rapidfuzz import fuzz
def keyword_coverage(req_terms: list[str], resume_skills: list[str]) -> float:
    if not req_terms: return 1.0
    blob = " ".join(s.lower() for s in resume_skills)
    hit = sum(1 for t in req_terms if t.lower() in blob)
    return hit / len(req_terms)
def fuzzy_coverage(req_terms: list[str], resume_skills: list[str], threshold: int = 85) -> float:
    if not req_terms: return 1.0
    low = [s.lower() for s in resume_skills]
    hit = sum(1 for t in req_terms
              if any(fuzz.ratio(t.lower(), s) >= threshold or t.lower() in s for s in low))
    return hit / len(req_terms)
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_matching.py -k coverage -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: keyword + fuzzy coverage helpers"
```

---

### Task 4: `match()` — component vector + prov-linked gaps

**Files:**
- Modify: `src/rho/matching/__init__.py`
- Test: `tests/unit/test_matching.py` (add)

**Interfaces:**
- Produces: `match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult`. For each requirement: decide present/weak/absent (present = keyword or fuzzy or semantic ≥ thresholds against résumé skills; weak = semantic-only mid-band; absent otherwise). `evidence_prov` = prov_ids of the résumé skills that matched. `predicted_score` left `0.0`.

- [ ] **Step 1: Write failing test**
```python
# add to tests/unit/test_matching.py
from rho.matching import match
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet, Requirement
def test_match_builds_vector_and_prov_gaps():
    resume = StructuredResume(name="A",
        skills=["Python","AWS"], skills_prov=[["p:d:1"],["p:d:2"]])
    reqs = RequirementSet(requirements=[
        Requirement(text="Python", kind="skill", priority="must"),
        Requirement(text="Kubernetes", kind="skill", priority="must"),
    ])
    mr = match(resume, reqs)
    assert mr.predicted_score == 0.0
    assert 0.0 <= mr.component_vector.must_have_coverage <= 1.0
    py_gap = next(g for g in mr.gaps if g.requirement.text == "Python")
    assert py_gap.status == "present"
    assert py_gap.evidence_prov == ["p:d:1"]      # provenance chain preserved
    k8s_gap = next(g for g in mr.gaps if g.requirement.text == "Kubernetes")
    assert k8s_gap.status == "absent"
    assert k8s_gap.evidence_prov == []
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_matching.py::test_match_builds_vector_and_prov_gaps -v`
Expected: FAIL — `match` still stub.

- [ ] **Step 3: Implement**
```python
# src/rho/matching/__init__.py
from rho.models.resume import StructuredResume
from rho.models.jd import RequirementSet
from rho.models.scoring import MatchResult, ComponentVector, Gap
from rho.matching.embed import Embedder
from rho.matching.coverage import keyword_coverage, fuzzy_coverage
from rapidfuzz import fuzz
def _skill_evidence(term: str, resume: StructuredResume, emb: Embedder,
                    kw=True, sem_hi=0.65, sem_lo=0.45):
    """returns (status, prov_ids)"""
    tl = term.lower()
    for skill, prov in zip(resume.skills, resume.skills_prov):
        sl = skill.lower()
        if tl in sl or sl in tl or fuzz.ratio(tl, sl) >= 85:
            return "present", prov
    if resume.skills:
        tv = emb.encode([term])[0]
        sv = emb.encode(resume.skills)
        best_i = max(range(len(resume.skills)), key=lambda i: emb.cosine(tv, sv[i]))
        best = emb.cosine(tv, sv[best_i])
        if best >= sem_hi:
            return "present", resume.skills_prov[best_i]
        if best >= sem_lo:
            return "weak", resume.skills_prov[best_i]
    return "absent", []
def match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult:
    emb = Embedder()
    req_terms = [r.text for r in reqs.requirements]
    must = [r for r in reqs.requirements if r.priority == "must"]
    nice = [r for r in reqs.requirements if r.priority == "nice"]
    gaps = []
    present_must = present_nice = 0
    for r in reqs.requirements:
        status, prov = _skill_evidence(r.text, resume, emb)
        gaps.append(Gap(requirement=r, status=status, evidence_prov=prov))
        if status in ("present", "weak"):
            if r.priority == "must": present_must += 1
            else: present_nice += 1
    cv = ComponentVector(
        keyword_coverage=keyword_coverage(req_terms, resume.skills),
        semantic_similarity=sum(1 for g in gaps if g.status in ("present","weak"))/max(len(gaps),1),
        fuzzy_coverage=fuzzy_coverage(req_terms, resume.skills),
        must_have_coverage=(present_must/len(must)) if must else 1.0,
        nice_have_coverage=(present_nice/len(nice)) if nice else 1.0,
    )
    return MatchResult(component_vector=cv, predicted_score=0.0, gaps=gaps)
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_matching.py -v`
Expected: PASS all.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: deterministic matcher with prov-linked gaps"
```

---

## Self-Review
- [ ] `analyze_jd` maps must/nice/years correctly (fake-LLM test).
- [ ] `match` fills all 5 `ComponentVector` fields in [0,1].
- [ ] Present/weak gaps carry `evidence_prov`; absent gaps carry `[]` — chain preserved.
- [ ] `predicted_score == 0.0` (Phase 4 will set it).
- [ ] `pytest tests/unit/test_matching.py tests/unit/test_jd.py -v` green.

## Results (filled in)
- **Embedding model + version:** `all-mpnet-base-v2` via `sentence-transformers` **5.6.0** (CPU). First test run downloads ~420MB and takes ~138s; cached runs ~7s.
- **Semantic thresholds:** sem_hi=**0.65**, sem_lo=**0.45** — still the plan's proposed defaults, **not tuned**. No sweep was run because no labelled match set exists yet. They are now `settings.sem_hi` / `settings.sem_lo` rather than hardcoded, so P7 can sweep them without touching matcher code. Treat the values as provisional.
- **Tests passing:** 34 passed / 1 skipped (full suite). Skip is still the P2 LLM integration test (`RHO_LLM_ENABLED=1`).
- **JD LLM path never executed.** `src/rho/jd/llm.py` mirrors `extraction/llm.py` (same pre-0.2 Outlines API, same `[llm]` extra, still uninstalled). Written but unrun — same caveat as P2.

### Resolved: `ComponentVector.semantic_similarity` now a real cosine
Originally implemented per the plan as `count(present|weak) / count(gaps)` — a **match-rate** that never touched a cosine. A 3-requirement probe (skills `Python/AWS/Docker`) returned `semantic_similarity 1.0` while only 1 of 3 requirements was literally covered. Since P4 feeds this vector straight to the calibrator, that would have propagated into the headline score.

**Fixed** (`fix: semantic_similarity is mean best-match cosine, not match-rate`): the field is now the mean best-match cosine across requirements, clamped to [0,1]. Field name and type are unchanged, so the P0 frozen contract holds — only the value semantics changed. `_skill_evidence` returns `(status, prov_ids, best_cosine)`; requirements below `sem_lo` still contribute their real cosine to the mean rather than a zero. Regression-tested by `test_semantic_similarity_is_mean_cosine_not_match_rate` (asserts `< 1.0` on the exact case that used to return 1.0) and `test_semantic_similarity_zero_when_no_skills`.

### Deviations from plan
- **`_skill_evidence` no longer `zip`s `skills` with `skills_prov`.** The plan's `zip` silently drops trailing skills when `skills_prov` is shorter (legal — prov is only filled by P2 attachment, hand-built resumes often omit it), meaning a skill could fail to match purely for lacking provenance. Replaced with an index-guarded `_prov_for()` that returns `[]` for a missing prov and keeps the skill matchable.
- **`tests/unit/test_stubs.py` rewritten to be phase-agnostic.** It had needed hand-editing for three phases running (off `ingest`, then `analyze_jd`, then `match`). It now enumerates the Section-6 surface in a `SECTION_6` table, detects stub-ness by inspecting each body for `raise NotImplementedError`, and asserts only the still-stubbed ones raise. As each phase lands, its function drops out of the stub set with no edit to the file. Verified the detector classifies correctly: `ingest`/`extract`/`analyze_jd`/`match` → IMPLEMENTED, `harvest_ats`/`Calibrator`/`rewrite`/`verify`/`run_pipeline` → STUB. A `checked > 0` guard means the test tells you to delete it once P6 lands rather than silently passing on an empty set.
- **KeyBERT is now actually used.** It was installed per Task 2's dep list but dead — `keyword_coverage` is plain substring matching over pre-split terms, and nothing extracted terms from raw JD text. Added `rho.matching.coverage.extract_jd_terms(jd_text, top_n)` (KeyBERT, 1–2gram keyphrases, English stopwords) reusing the cached mpnet instance so no second model loads. Covered by `test_extract_jd_terms_pulls_keyphrases` and an empty-input case. Note it is not yet *wired into* `match()` — `match` still uses `Requirement.text` as its terms, which is the correct source there; `extract_jd_terms` exists for P4 calibration features and for JD analysis that bypasses the LLM.
