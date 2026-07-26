# Phase 2 — Extraction with Provenance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phases 0–1 done.

**Goal:** Convert ingestion markdown into a `StructuredResume` where every value-bearing field also has its `*_prov` list filled with the `prov_id`s of the source spans that support it. This is the first headline metric of the paper (extraction F1 + provenance-attachment accuracy).

**Architecture:** Two stages. (1) **Constrained extraction** — an open LLM (Qwen3/Llama) via **vLLM + Outlines** emits JSON conforming to a Pydantic schema (schema derived from `StructuredResume` minus the `*_prov` fields; reasoning fields before answer fields). (2) **Provenance attachment** — for each extracted value, locate the supporting span(s) in the `ProvenanceMap` (exact match first, fuzzy align fallback) and record their `prov_id`s into the `*_prov` fields. Validation via Pydantic; on failure, Outlines retry, then route to a review queue (return partial + flag).

**Tech Stack:** vLLM, Outlines, Pydantic v2, RapidFuzz (for provenance fuzzy alignment).

## Global Constraints
- Implement `rho.extraction.extract(markdown: str, prov: ProvenanceMap) -> StructuredResume` (frozen signature).
- Extraction temperature ≈ `settings.temperature` (0.2). Seed for test stability where the backend allows.
- **No silent fill:** if a required-looking field is absent in source, leave empty/`None` — never invent to satisfy the schema.
- Provenance attachment must not fabricate: a `*_prov` id is added only if its span's `raw_text` actually supports the value (exact or high-fuzzy match ≥ threshold).

## This phase consumes
- `ingest()` output `(markdown, ProvenanceMap)` (Phase 1).
- Models `StructuredResume`, `WorkExperience`, `Education` (Phase 0).

## This phase produces
- `extract()` filling values + `*_prov`.
- `rho.extraction.attach_provenance(resume, prov) -> StructuredResume` (public, unit-testable without an LLM).
- `rho.extraction.schema.ExtractionSchema` (the LLM output Pydantic schema, no prov fields).
- Provenance-attachment accuracy metric helper for Phase 7.

---

## File Structure
- Create: `src/rho/extraction/schema.py` — LLM output schema (reasoning-first).
- Create: `src/rho/extraction/llm.py` — vLLM+Outlines call, returns `ExtractionSchema`.
- Create: `src/rho/extraction/provenance_attach.py` — value → prov_id matching.
- Modify: `src/rho/extraction/__init__.py` — `extract()` orchestrating llm + attach + validate.
- Create: `tests/unit/test_extraction_provenance.py`, `tests/integration/test_extract_llm.py`.

---

### Task 1: Extraction output schema (reasoning-first, no prov fields)

**Files:**
- Create: `src/rho/extraction/schema.py`
- Test: `tests/unit/test_extraction_provenance.py`

**Interfaces:**
- Produces: `ExtractionSchema` (Pydantic) — mirrors `StructuredResume` value fields, plus a leading `reasoning: str`. NO `*_prov` fields (LLM doesn't emit provenance; we attach it deterministically).

- [ ] **Step 1: Write failing test**
```python
# tests/unit/test_extraction_provenance.py
from rho.extraction.schema import ExtractionSchema, to_structured
def test_schema_maps_to_structured_resume():
    es = ExtractionSchema(reasoning="found name+skill",
        name="Alice", skills=["python"],
        work=[{"company":"Acme","title":"Eng","start_date":"2019","end_date":"2022","bullets":["Built API"]}])
    sr = to_structured(es)
    assert sr.name == "Alice"
    assert sr.skills == ["python"]
    assert sr.work[0].company == "Acme"
    # prov fields exist but empty until attach step
    assert sr.name_prov == []
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_extraction_provenance.py::test_schema_maps_to_structured_resume -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement schema + mapper**
```python
# src/rho/extraction/schema.py
from pydantic import BaseModel
from rho.models.resume import StructuredResume, WorkExperience, Education
class WorkItem(BaseModel):
    company: str; title: str
    start_date: str | None = None; end_date: str | None = None
    bullets: list[str] = []
class EduItem(BaseModel):
    institution: str; degree: str | None = None
    field: str | None = None; end_year: str | None = None
class ExtractionSchema(BaseModel):
    reasoning: str                       # FIRST: LLMs generate left-to-right
    name: str
    headline: str | None = None
    summary: str | None = None
    emails: list[str] = []; phones: list[str] = []; urls: list[str] = []
    work: list[WorkItem] = []
    education: list[EduItem] = []
    skills: list[str] = []
    certifications: list[str] = []
def to_structured(es: ExtractionSchema) -> StructuredResume:
    return StructuredResume(
        name=es.name, headline=es.headline, summary=es.summary,
        emails=es.emails, phones=es.phones, urls=es.urls,
        work=[WorkExperience(company=w.company, title=w.title,
              start_date=w.start_date, end_date=w.end_date, bullets=w.bullets) for w in es.work],
        education=[Education(institution=e.institution, degree=e.degree,
              field=e.field, end_year=e.end_year) for e in es.education],
        skills=es.skills, certifications=es.certifications,
    )
```

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_extraction_provenance.py::test_schema_maps_to_structured_resume -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: extraction output schema + StructuredResume mapper"
```

---

### Task 2: Provenance attachment (LLM-free, fully unit-testable)

**Files:**
- Create: `src/rho/extraction/provenance_attach.py`
- Test: `tests/unit/test_extraction_provenance.py` (add)

**Interfaces:**
- Consumes: `StructuredResume` (values filled, prov empty), `ProvenanceMap`.
- Produces: `attach_provenance(resume, prov, threshold=90) -> StructuredResume` — for each value, find span(s) whose `raw_text` contains or fuzzy-matches the value (RapidFuzz `partial_ratio >= threshold`); set the matching `prov_id`s. Also `find_prov(value, prov, threshold) -> list[str]` helper.

- [ ] **Step 1: Add dep**
Add `rapidfuzz>=3` to `pyproject.toml`; `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing test**
```python
# add to tests/unit/test_extraction_provenance.py
from rho.models.provenance import SourceSpan, ProvenanceMap
from rho.models.resume import StructuredResume, WorkExperience
from rho.extraction.provenance_attach import attach_provenance, find_prov
def _pm():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=13, raw_text="Alice Johnson"))
    pm.add(SourceSpan(doc_id="d", char_start=14, char_end=45, raw_text="Skills: Python, FastAPI, AWS"))
    pm.add(SourceSpan(doc_id="d", char_start=46, char_end=70, raw_text="Acme Corp Backend Engineer"))
    return pm
def test_find_prov_exact_and_fuzzy():
    pm = _pm()
    assert find_prov("Alice Johnson", pm) == ["p:d:0"]
    assert find_prov("Python", pm) == ["p:d:1"]     # substring of skills line
def test_attach_sets_prov_fields():
    pm = _pm()
    r = StructuredResume(name="Alice Johnson", skills=["Python","AWS"],
        work=[WorkExperience(company="Acme Corp", title="Backend Engineer")])
    r2 = attach_provenance(r, pm)
    assert r2.name_prov == ["p:d:0"]
    assert r2.skills_prov[0] == ["p:d:1"]
    assert "p:d:2" in r2.work[0].company_prov
def test_attach_leaves_empty_when_no_support():
    pm = _pm()
    r = StructuredResume(name="Nonexistent Person")
    r2 = attach_provenance(r, pm)
    assert r2.name_prov == []          # never invent provenance
```

- [ ] **Step 3: Run to verify fail**
Run: `pytest tests/unit/test_extraction_provenance.py -k "find_prov or attach" -v`
Expected: FAIL — not defined.

- [ ] **Step 4: Implement**
```python
# src/rho/extraction/provenance_attach.py
from rapidfuzz import fuzz
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
def find_prov(value: str, prov: ProvenanceMap, threshold: int = 90) -> list[str]:
    if not value or not value.strip():
        return []
    v = value.strip().lower()
    hits = []
    for pid, span in prov.spans.items():
        raw = span.raw_text.lower()
        if v in raw or fuzz.partial_ratio(v, raw) >= threshold:
            hits.append(pid)
    return hits
def attach_provenance(resume: StructuredResume, prov: ProvenanceMap, threshold: int = 90) -> StructuredResume:
    r = resume.model_copy(deep=True)
    r.name_prov = find_prov(r.name, prov, threshold)
    r.contact_prov = []
    for c in (r.emails + r.phones + r.urls):
        r.contact_prov += find_prov(c, prov, threshold)
    r.skills_prov = [find_prov(s, prov, threshold) for s in r.skills]
    for w in r.work:
        w.company_prov = find_prov(w.company, prov, threshold)
        w.title_prov = find_prov(w.title, prov, threshold)
        w.date_prov = find_prov((w.start_date or "") + " " + (w.end_date or ""), prov, threshold)
        w.bullet_prov = [find_prov(b, prov, threshold) for b in w.bullets]
    for e in r.education:
        e.institution_prov = find_prov(e.institution, prov, threshold)
        e.edu_prov = find_prov((e.degree or "") + " " + (e.field or ""), prov, threshold)
    return r
```

- [ ] **Step 5: Run to verify pass**
Run: `pytest tests/unit/test_extraction_provenance.py -k "find_prov or attach" -v`
Expected: PASS all three.

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: deterministic provenance attachment (exact+fuzzy)"
```

---

### Task 3: LLM extraction via vLLM + Outlines

**Files:**
- Create: `src/rho/extraction/llm.py`
- Test: `tests/integration/test_extract_llm.py` (guarded by env — skips if no model server)

**Interfaces:**
- Produces: `extract_schema(markdown: str) -> ExtractionSchema`. Uses Outlines-guided generation constrained to `ExtractionSchema`. Model id from `settings.extraction_model`.
- **Prompt** instructs: extract only what is present; do not invent; put reasoning first; use ISO-8601 dates; empty `""`/`[]` when absent.

**Note:** vLLM+Outlines needs a GPU/model. Test is an integration test that SKIPS unless `RHO_LLM_ENABLED=1`. Unit correctness of the pipeline is covered by Task 2 (attachment) + Task 4 (orchestration with a fake extractor).

- [ ] **Step 1: Add deps**
Add `outlines>=0.1`, `vllm>=0.6` to `pyproject.toml` (or an `extra` `[llm]`). Install per your GPU setup.

- [ ] **Step 2: Write integration test (skips without model)**
```python
# tests/integration/test_extract_llm.py
import os, pytest
pytestmark = pytest.mark.skipif(os.getenv("RHO_LLM_ENABLED") != "1", reason="no LLM backend")
def test_extract_schema_on_simple_markdown():
    from rho.extraction.llm import extract_schema
    md = "Alice Johnson\nSenior Python Engineer\nSkills: Python, FastAPI, AWS\nAcme Corp Backend Engineer 2019-2022"
    es = extract_schema(md)
    assert es.name.startswith("Alice")
    assert any("python" in s.lower() for s in es.skills)
```

- [ ] **Step 3: Run to verify skip/fail**
Run: `pytest tests/integration/test_extract_llm.py -v`
Expected: SKIP (no env) — becomes runnable once a model is available.

- [ ] **Step 4: Implement**
```python
# src/rho/extraction/llm.py
import outlines
from rho.config import settings
from rho.extraction.schema import ExtractionSchema
_PROMPT = """You extract structured data from a resume. Rules:
- Extract ONLY information present in the text. Never invent, infer, or fill.
- If a field is absent, leave it empty ("" or []).
- Dates in ISO-8601 (2019, 2019-06). Use "" for present.
- Fill the `reasoning` field first, briefly, then the data fields.
Resume:
---
{markdown}
---"""
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = outlines.models.vllm(settings.extraction_model)
    return _model
def extract_schema(markdown: str) -> ExtractionSchema:
    model = _get_model()
    generator = outlines.generate.json(model, ExtractionSchema)
    return generator(_PROMPT.format(markdown=markdown),
                     temperature=settings.temperature)
```
*(Outlines API differs across versions — if `generate.json` signature changed in the installed version, adapt while keeping: constrained to `ExtractionSchema`, temperature from settings. Note deviation in Results.)*

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: vLLM+Outlines constrained extraction"
```

---

### Task 4: `extract()` orchestration (LLM + attach + validate + fallback)

**Files:**
- Modify: `src/rho/extraction/__init__.py`
- Test: `tests/unit/test_extraction_provenance.py` (add — uses a fake schema extractor, no real LLM)

**Interfaces:**
- Produces final `extract(markdown, prov) -> StructuredResume`. Internally: `extract_schema(markdown)` → `to_structured` → `attach_provenance(., prov)`. Accepts an injectable extractor for testing: `extract(markdown, prov, _schema_fn=None)`.

- [ ] **Step 1: Write failing test (fake LLM)**
```python
# add to tests/unit/test_extraction_provenance.py
from rho.extraction import extract
from rho.extraction.schema import ExtractionSchema
def test_extract_end_to_end_with_fake_llm():
    pm = _pm()
    md = "Alice Johnson\nSkills: Python, FastAPI, AWS\nAcme Corp Backend Engineer"
    fake = lambda m: ExtractionSchema(reasoning="x", name="Alice Johnson",
        skills=["Python","AWS"],
        work=[{"company":"Acme Corp","title":"Backend Engineer"}])
    r = extract(md, pm, _schema_fn=fake)
    assert r.name == "Alice Johnson"
    assert r.name_prov == ["p:d:0"]
    assert r.work[0].company_prov == ["p:d:2"]
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_extraction_provenance.py::test_extract_end_to_end_with_fake_llm -v`
Expected: FAIL — `extract` still stub.

- [ ] **Step 3: Implement**
```python
# src/rho/extraction/__init__.py
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.extraction.schema import to_structured
from rho.extraction.provenance_attach import attach_provenance
def extract(markdown: str, prov: ProvenanceMap, _schema_fn=None) -> StructuredResume:
    if _schema_fn is None:
        from rho.extraction.llm import extract_schema as _schema_fn_default
        _schema_fn = _schema_fn_default
    es = _schema_fn(markdown)            # ExtractionSchema (validated by Pydantic already)
    resume = to_structured(es)
    return attach_provenance(resume, prov)
```
*(Validation note: Outlines guarantees schema-valid JSON, so `ExtractionSchema(...)` construction is the validation. A retry/review-queue wrapper can be added around `_schema_fn` in P6 orchestration; keep P2 focused.)*

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_extraction_provenance.py -v`
Expected: PASS all.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: extract() orchestration with provenance attachment"
```

---

## Self-Review
- [ ] `extract()` returns `StructuredResume` with `*_prov` filled from real spans only.
- [ ] Attachment never invents a prov_id (test `test_attach_leaves_empty_when_no_support` passes).
- [ ] LLM path constrained to `ExtractionSchema`; reasoning field first.
- [ ] Unit tests green without a GPU (fake extractor); LLM integration test skips cleanly.

## Results (filled in)
- **Model used + version:** none run. `settings.extraction_model` is still the P0 default `Qwen/Qwen3-0.6B`. `outlines`/`vllm` were added as an optional `[llm]` extra in `pyproject.toml` and **not installed** (GPU deps, no GPU in this env). The LLM path (`rho.extraction.llm.extract_schema`) is written but has never been executed.
- **Extraction per-field F1:** deferred to P7 — no gold set exists yet, and no LLM was run, so there is nothing to score.
- **Provenance-attachment accuracy on fixtures:** not measured as a rate. There is no labelled fixture set for attachment yet; the P2 tests are hand-built assertions (4 unit tests over a 3-span synthetic `ProvenanceMap`), not a corpus. The metric helper that shared-context lists under "This phase produces" was **not built** — it needs the P7 gold set to be meaningful. Flagged as a carry-forward to P7.
- **Outlines/vLLM API deviations:** none observed — but this is untested. `outlines.models.vllm(...)` + `outlines.generate.json(...)` is the pre-0.2 API and is likely to need adapting against whatever version actually gets installed. Verify before trusting.
- **Tests passing:** 16 passed / 1 skipped (full `pytest` run). Of these, 5 are new in P2: 1 schema-mapper, 3 provenance-attachment (incl. `test_attach_leaves_empty_when_no_support`), 1 end-to-end via injected fake extractor. The skip is `tests/integration/test_extract_llm.py` (gated on `RHO_LLM_ENABLED=1`).

### Deviations from plan
- **`vllm`/`outlines` placed in an optional `[llm]` extra**, not in `dependencies` as Task 3 Step 1 suggested. Keeps `pip install -e ".[dev]"` working without a GPU.
- **`tests/unit/test_stubs.py` was edited.** P1 had already repointed its `NotImplementedError` assertion from `ingest` to `extract("", None)`; implementing `extract` in this phase broke it (it now reaches the real LLM path and raises `ModuleNotFoundError: No module named 'outlines'`). Assertion moved on to `analyze_jd("")`, still a genuine P3 stub. `extract` is covered by the new P2 tests instead.
- **`extract()` signature gained `_schema_fn=None`** (per Task 4), a test seam. The frozen 2-arg call from shared-context Section 6 is unchanged for callers.
- No retry / review-queue wrapper around `_schema_fn` — plan defers it to P6.
