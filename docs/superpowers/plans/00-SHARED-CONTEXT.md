# Shared Context — Provenance-Chained Resume Optimization

> **Read this first, every session.** Every phase plan assumes this file. It carries the thesis, the frozen contracts, conventions, and the map of what each phase hands the next. You should not need the original brainstorming chat to execute any phase.

**Design spec:** `../specs/2026-07-20-provenance-chained-resume-optimization-design.md`
**Research report:** `../../../report.md`

---

## 1. What we are building (one paragraph)

A **backend** that takes a résumé file + a job description and returns: a structured résumé (JSON), a 0–100 ATS-match score calibrated against real self-hostable ATS engines, and a truthfully-rewritten résumé where every edit is provably grounded in the original document. The research novelty is a **provenance ID space** — every value the system touches carries a stable ID pointing to its exact source location, and that ID survives unbroken through all four pipeline stages. No frontend. API only.

## 2. The three contributions (what the paper claims)

- **C1 — Continuous provenance chain.** One ID space, extraction → final output. (Phases 1, 2 build it; every phase preserves it.)
- **C2 — Real-engine ATS-calibrated scoring.** Predicted score calibrated vs actual output of self-hostable ATS engines. (Phase 4.)
- **C3 — Provenance-verified rewrite gate.** Every rewrite edit must resolve to a source provenance ID or is rejected; fabrication-rate metric reported. (Phase 5.)

"Real ATS" = **self-hostable open engines**, not Workday/Greenhouse. Stated as a reproducible proxy; the gap is an acknowledged limitation.

## 3. Tech stack (fixed)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Data models | Pydantic v2 |
| Ingestion | Docling |
| Extraction LLM | Self-hosted open model (Qwen3 / Llama) via **vLLM + Outlines** (constrained decoding) |
| Matching | sentence-transformers (`all-mpnet-base-v2`), KeyBERT, RapidFuzz |
| Orchestration | LangGraph |
| Tests | pytest (+ hypothesis for property tests) |
| Package/deps | `uv` (or `pip` + `requirements.txt`) |

## 4. Repo layout (target, built up across phases)

```
tf/
  pyproject.toml
  src/rho/                     # "rho" = Resume Holistic Optimizer (package name)
    __init__.py
    config.py                  # settings (P0)
    models/                    # Pydantic contracts (P0)
      __init__.py
      provenance.py            # SourceSpan, ProvenanceMap, prov_id helpers
      resume.py                # StructuredResume + *_prov fields
      jd.py                    # RequirementSet
      scoring.py               # MatchResult, component vector
      rewrite.py               # TailoredResume, FabricationReport
      api.py                   # PipelineResponse, request models
    ingestion/                 # P1
    extraction/                # P2
    jd/                        # P3
    matching/                  # P3
    ats/                       # P4  (harness + calibrator)
    rewrite/                   # P5  (rewriter + verifier)
    graph/                     # P6  (LangGraph orchestration)
    api/                       # P0 app, grows each phase
      app.py
  tests/
    fixtures/                  # sample resumes, JDs, golden JSON
    unit/
    integration/
  eval/                        # P7 datasets + metric scripts
  docs/superpowers/{specs,plans}/
```

Package name **`rho`** (Resume Holistic Optimizer). Use it everywhere.

## 5. Frozen contracts (the spine — do NOT change signatures after Phase 0)

These are the interfaces every phase depends on. Phase 0 creates them as real Pydantic models. Later phases fill behavior but must not change field names/types without updating this file and every dependent phase.

```python
# models/provenance.py
class SourceSpan(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None   # x0,y0,x1,y1
    raw_text: str

class ProvenanceMap(BaseModel):
    doc_id: str
    spans: dict[str, SourceSpan] = {}          # prov_id -> span
    def add(self, span: SourceSpan) -> str: ... # returns new prov_id "p:<doc>:<seq>"
    def get(self, prov_id: str) -> SourceSpan: ...

# models/resume.py  (each value-bearing field has a sibling *_prov: list[str])
class WorkExperience(BaseModel):
    company: str;               company_prov: list[str] = []
    title: str;                 title_prov: list[str] = []
    start_date: str | None = None; end_date: str | None = None
    date_prov: list[str] = []
    bullets: list[str] = [];    bullet_prov: list[list[str]] = []  # per-bullet
class Education(BaseModel):
    institution: str; institution_prov: list[str] = []
    degree: str | None = None; field: str | None = None
    end_year: str | None = None; edu_prov: list[str] = []
class StructuredResume(BaseModel):
    name: str; name_prov: list[str] = []
    headline: str | None = None; summary: str | None = None
    emails: list[str] = []; phones: list[str] = []; urls: list[str] = []
    contact_prov: list[str] = []
    work: list[WorkExperience] = []
    education: list[Education] = []
    skills: list[str] = []; skills_prov: list[list[str]] = []
    certifications: list[str] = []

# models/jd.py
class Requirement(BaseModel):
    text: str
    kind: Literal["skill","tool","title","cert","experience"]
    priority: Literal["must","nice"]
    years: float | None = None
class RequirementSet(BaseModel):
    title: str | None = None
    requirements: list[Requirement] = []

# models/scoring.py
class ComponentVector(BaseModel):     # raw signals, pre-calibration
    keyword_coverage: float           # 0..1
    semantic_similarity: float        # 0..1
    fuzzy_coverage: float             # 0..1
    must_have_coverage: float         # 0..1
    nice_have_coverage: float         # 0..1
class Gap(BaseModel):
    requirement: Requirement
    status: Literal["present","absent","weak"]
    evidence_prov: list[str] = []     # prov_ids supporting "present"/"weak"
class MatchResult(BaseModel):
    component_vector: ComponentVector
    predicted_score: float            # 0..100  (set by calibrator, P4)
    gaps: list[Gap] = []

# models/rewrite.py
class RejectedEdit(BaseModel):
    added_text: str
    reason: str                       # e.g. "no supporting prov_id"
class FabricationReport(BaseModel):
    total_edits: int
    verified_edits: int
    rejected_edits: list[RejectedEdit] = []
    fabrication_rate: float           # rejected / total
class TailoredResume(BaseModel):
    resume: StructuredResume          # rewritten, prov chain intact
    fabrication_report: FabricationReport

# models/api.py
class OptimizeRequest(BaseModel):
    jd_text: str
    # file arrives as multipart upload, not in this model
class PipelineResponse(BaseModel):
    structured_resume: StructuredResume
    provenance_map: ProvenanceMap
    match_result: MatchResult
    tailored_resume: TailoredResume
    final_score: float
```

## 6. Component signatures (each phase implements one; keep these exact)

```python
# ingestion/  (P1)
def ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]:
    """file -> (markdown, ProvenanceMap)"""

# extraction/ (P2)
def extract(markdown: str, prov: ProvenanceMap) -> StructuredResume:
    """markdown+prov -> StructuredResume with *_prov filled"""

# jd/ (P3)
def analyze_jd(jd_text: str) -> RequirementSet: ...

# matching/ (P3)
def match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult:
    """fills component_vector + gaps; predicted_score left 0.0 until P4"""

# ats/ (P4)
def harvest_ats(file_bytes: bytes, filename: str, jd_text: str) -> dict:
    """run self-hostable ATS engines -> real parse+match labels"""
class Calibrator:
    def fit(self, X: list[ComponentVector], y: list[float]) -> None: ...
    def predict(self, cv: ComponentVector) -> float: ...   # 0..100

# rewrite/ (P5)
# SIGNATURE CHANGE (P5): `prov` added. The gate cannot verify without the
# provenance map, and C3 forbids shipping an ungated rewrite. P6 must pass it.
def rewrite(resume: StructuredResume, gaps: list[Gap], prov: ProvenanceMap) -> TailoredResume: ...
# `verify` keeps its frozen signature but raises RuntimeError: distinguishing an
# *addition* from a reorder needs the source résumé, which this signature cannot
# carry. Call `verify_against_source(tailored, source, prov)` instead — that is
# what `rewrite()` uses internally.
def verify(tailored: StructuredResume, prov: ProvenanceMap) -> FabricationReport: ...
def verify_against_source(tailored: StructuredResume, source: StructuredResume,
                          prov: ProvenanceMap) -> tuple[StructuredResume, FabricationReport]: ...

# graph/ (P6)
def run_pipeline(file_bytes: bytes, filename: str, jd_text: str) -> PipelineResponse: ...
```

## 7. The provenance invariant (machine-checked, the heart of the paper)

> At the output of Stage 3 (rewrite), every "hard-content" token — skills, tools, org names, numbers, dates — must trace to at least one `prov_id` whose `raw_text` supports it.

Phase 5 implements the check; Phase 6 reviewer re-asserts it; Phase 7 measures it. Any violation = a fabrication and must be counted, never silently allowed.

## 8. Conventions

- **TDD always:** write failing test → run (see it fail) → minimal impl → run (pass) → commit. One behavior per test.
- **Commits:** conventional prefix (`feat:`, `test:`, `fix:`, `chore:`). Frequent, small.
- **No silent fills:** when data is missing, leave `None`/empty + log; never invent to satisfy a required field.
- **Fixtures:** every phase adds its fixtures under `tests/fixtures/`; later phases reuse them.
- **Determinism:** set model temperature low for extraction (~0.2); seed where possible so tests are stable.
- **Provenance discipline:** whenever you create or copy a value, carry its `*_prov`. Losing provenance = breaking C1.

## 9. Phase dependency graph

```
P0 contracts ─┬─> P1 ingestion ──> P2 extraction ─┐
              │                                    ├─> P3 match ─> P4 ATS calib ─┐
              └────────────> P3 jd_analyzer ───────┘                            ├─> P6 graph ─> P7 eval
                                              P2 ─> P5 verified rewrite ─────────┘
```

Execute in number order 0→7. Each phase plan lists exactly what it consumes from prior phases and what it must hand forward.

## 10. Definition of Done (per phase)

A phase is done when: all its tests pass, its deliverable in the phase doc is demonstrably produced, provenance discipline is preserved, and work is committed. Record actual metric numbers in the phase's "Results" section (bottom of each phase doc) so the final verification session can read them.
