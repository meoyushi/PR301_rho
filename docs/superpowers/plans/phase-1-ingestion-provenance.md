# Phase 1 — Ingestion + Provenance Anchoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.
> **Read `00-SHARED-CONTEXT.md` first.** Confirm Phase 0 done (models + stubs exist, tests green).

**Goal:** Turn an uploaded résumé (PDF/DOCX/text/image) into clean markdown **plus** a populated `ProvenanceMap` where every source span has a stable `prov_id` with char offsets (and page/bbox when available).

**Architecture:** Route by file type. Digital PDF/DOCX → Docling → markdown + element geometry. Plain text → trivial passthrough. Image/scanned → Docling OCR path. As we emit markdown, record each source element as a `SourceSpan` (char offsets into the markdown, plus page/bbox from Docling) and register it in the `ProvenanceMap`. A reading-order structural heuristic flags likely-scrambled multi-column docs.

**Tech Stack:** Docling, Pydantic v2. (`pip install docling` — downloads 1–2 GB model weights first run.)

## Global Constraints
- Package `rho`. Implement `rho.ingestion.ingest` (replace the P0 stub body only; signature frozen).
- `ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]`.
- Char offsets in every `SourceSpan` must index into the returned markdown string.
- No silent data loss: if geometry (page/bbox) unavailable (e.g. plain text), leave those `None` but still record char offsets.

## This phase consumes
- `rho.models.provenance.{SourceSpan, ProvenanceMap}` (Phase 0).

## This phase produces
- Working `ingest()` → `(markdown, ProvenanceMap)`.
- `rho.ingestion.structural_check(markdown, prov) -> bool` (True = likely reading-order OK).
- Fixtures under `tests/fixtures/`: `clean.txt`, `clean.pdf`, `multicolumn.pdf`, `simple.docx` (add what you can generate; see Task notes).

---

## File Structure
- Modify: `src/rho/ingestion/__init__.py` — real `ingest`, route logic.
- Create: `src/rho/ingestion/docling_adapter.py` — Docling call + span extraction.
- Create: `src/rho/ingestion/text_adapter.py` — plain-text passthrough with offsets.
- Create: `src/rho/ingestion/structural.py` — reading-order heuristic.
- Create: `tests/unit/test_ingestion.py`, `tests/fixtures/*`.

---

### Task 1: Plain-text ingestion with real offsets (no external deps)

**Files:**
- Create: `src/rho/ingestion/text_adapter.py`
- Modify: `src/rho/ingestion/__init__.py`
- Test: `tests/unit/test_ingestion.py`; fixture `tests/fixtures/clean.txt`

**Interfaces:**
- Produces: `text_adapter.ingest_text(text: str, doc_id: str) -> tuple[str, ProvenanceMap]` — one `SourceSpan` per non-empty line, char offsets into the markdown (which for text == the text).

- [ ] **Step 1: Create fixture**
`tests/fixtures/clean.txt`:
```
Alice Johnson
Senior Python Engineer
Skills: Python, FastAPI, AWS
Acme Corp — Backend Engineer 2019-2022
```

- [ ] **Step 2: Write failing test**
```python
# tests/unit/test_ingestion.py
from pathlib import Path
from rho.ingestion import ingest
FIX = Path(__file__).parent.parent / "fixtures"
def test_text_ingest_offsets_map_back():
    data = (FIX / "clean.txt").read_bytes()
    md, pm = ingest(data, "clean.txt")
    assert "Alice Johnson" in md
    assert len(pm.spans) >= 4
    # every span's raw_text equals the markdown slice at its offsets
    for span in pm.spans.values():
        assert md[span.char_start:span.char_end] == span.raw_text
```

- [ ] **Step 3: Run to verify fail**
Run: `pytest tests/unit/test_ingestion.py -v`
Expected: FAIL — `ingest` still raises `NotImplementedError`.

- [ ] **Step 4: Implement text adapter + route for .txt**
```python
# src/rho/ingestion/text_adapter.py
from rho.models.provenance import SourceSpan, ProvenanceMap
def ingest_text(text: str, doc_id: str) -> tuple[str, ProvenanceMap]:
    pm = ProvenanceMap(doc_id=doc_id)
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.strip():
            start = offset + (len(stripped) - len(stripped.lstrip()))
            content = stripped.strip()
            end = start + len(content)
            pm.add(SourceSpan(doc_id=doc_id, char_start=start, char_end=end, raw_text=content))
        offset += len(line)
    return text, pm
```
```python
# src/rho/ingestion/__init__.py
from pathlib import Path
from rho.models.provenance import ProvenanceMap
from rho.ingestion.text_adapter import ingest_text
def ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]:
    doc_id = Path(filename).stem
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return ingest_text(file_bytes.decode("utf-8", errors="replace"), doc_id)
    from rho.ingestion.docling_adapter import ingest_docling   # lazy import (heavy)
    return ingest_docling(file_bytes, filename, doc_id)
```

- [ ] **Step 5: Run to verify pass**
Run: `pytest tests/unit/test_ingestion.py::test_text_ingest_offsets_map_back -v`
Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: plain-text ingestion with provenance offsets"
```

---

### Task 2: Docling adapter (PDF/DOCX) with span geometry

**Files:**
- Create: `src/rho/ingestion/docling_adapter.py`
- Test: `tests/unit/test_ingestion.py` (add), fixture `tests/fixtures/simple.docx` or `clean.pdf`

**Interfaces:**
- Produces: `docling_adapter.ingest_docling(file_bytes, filename, doc_id) -> tuple[str, ProvenanceMap]`. Markdown from Docling; one `SourceSpan` per text element, char offsets located in the exported markdown, page/bbox from the Docling item when present.

**Note on fixtures:** generate `simple.docx` with `python-docx` (add to dev deps) or drop any small real résumé PDF into `tests/fixtures/`. Keep it tiny.

- [ ] **Step 1: Add dep**
Add `docling>=2.0` to `pyproject.toml` dependencies, `python-docx>=1.1` to dev. Run `pip install -e ".[dev]"`.

- [ ] **Step 2: Write failing test**
```python
# add to tests/unit/test_ingestion.py
import pytest
def test_docling_ingest_produces_markdown_and_spans():
    docx = (FIX / "simple.docx")
    if not docx.exists():
        pytest.skip("fixture simple.docx missing")
    md, pm = ingest(docx.read_bytes(), "simple.docx")
    assert isinstance(md, str) and len(md) > 0
    assert len(pm.spans) >= 1
    for span in pm.spans.values():
        # offsets are valid indices into md
        assert 0 <= span.char_start <= span.char_end <= len(md)
```

- [ ] **Step 3: Run to verify fail/skip**
Run: `pytest tests/unit/test_ingestion.py::test_docling_ingest_produces_markdown_and_spans -v`
Expected: FAIL (import error) — then after impl, PASS or SKIP if no fixture.

- [ ] **Step 4: Implement adapter**
```python
# src/rho/ingestion/docling_adapter.py
import tempfile, os
from rho.models.provenance import SourceSpan, ProvenanceMap
from docling.document_converter import DocumentConverter
def ingest_docling(file_bytes: bytes, filename: str, doc_id: str) -> tuple[str, ProvenanceMap]:
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes); path = tmp.name
    try:
        result = DocumentConverter().convert(path)
        doc = result.document
        md = doc.export_to_markdown()
    finally:
        os.unlink(path)
    pm = ProvenanceMap(doc_id=doc_id)
    # Locate each text item's string in the exported markdown to get char offsets.
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        idx = md.find(text.strip())
        if idx == -1:
            continue
        page = None; bbox = None
        prov = getattr(item, "prov", None)
        if prov:
            p0 = prov[0]
            page = getattr(p0, "page_no", None)
            bb = getattr(p0, "bbox", None)
            if bb is not None:
                bbox = (bb.l, bb.t, bb.r, bb.b)
        pm.add(SourceSpan(doc_id=doc_id, char_start=idx, char_end=idx+len(text.strip()),
                          page=page, bbox=bbox, raw_text=text.strip()))
    return md, pm
```
*(Docling API surface may shift; if `iterate_items`/`prov` differ in the installed version, adapt to the equivalent — the invariant to preserve: char offsets index into `md`, and page/bbox come from item geometry when available. Document any deviation in Results.)*

- [ ] **Step 5: Run to verify**
Run: `pytest tests/unit/test_ingestion.py -v`
Expected: PASS (or SKIP if fixture absent).

- [ ] **Step 6: Commit**
```bash
git add -A && git commit -m "feat: Docling ingestion adapter with span geometry"
```

---

### Task 3: Reading-order structural check

**Files:**
- Create: `src/rho/ingestion/structural.py`
- Modify: `src/rho/ingestion/__init__.py` (export)
- Test: `tests/unit/test_ingestion.py` (add)

**Interfaces:**
- Produces: `structural_check(markdown: str, prov: ProvenanceMap) -> bool`. Heuristic: if spans carry bboxes, check that reading order (span order) is broadly top-to-bottom, left-to-right per page; return False if a large fraction of consecutive spans jump upward/backward (multi-column scramble signal). If no bboxes, return True (can't tell).

- [ ] **Step 1: Write failing test**
```python
# add to tests/unit/test_ingestion.py
from rho.ingestion import structural_check
from rho.models.provenance import SourceSpan, ProvenanceMap
def test_structural_check_true_when_no_geometry():
    pm = ProvenanceMap(doc_id="d")
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=3, raw_text="abc"))
    assert structural_check("abc", pm) is True
def test_structural_check_flags_backward_jumps():
    pm = ProvenanceMap(doc_id="d")
    # y decreasing then jumping = out-of-order columns
    pm.add(SourceSpan(doc_id="d", char_start=0, char_end=1, page=1, bbox=(0,700,1,710), raw_text="a"))
    pm.add(SourceSpan(doc_id="d", char_start=1, char_end=2, page=1, bbox=(0,100,1,110), raw_text="b"))
    pm.add(SourceSpan(doc_id="d", char_start=2, char_end=3, page=1, bbox=(300,690,301,700), raw_text="c"))
    assert structural_check("abc", pm) is False
```

- [ ] **Step 2: Run to verify fail**
Run: `pytest tests/unit/test_ingestion.py -k structural -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**
```python
# src/rho/ingestion/structural.py
from rho.models.provenance import ProvenanceMap
def structural_check(markdown: str, prov: ProvenanceMap) -> bool:
    spans = [s for s in prov.spans.values() if s.bbox is not None and s.page is not None]
    if len(spans) < 3:
        return True
    backward = 0
    prev_page, prev_top = None, None
    for s in spans:
        top = s.bbox[1]
        if prev_page == s.page and prev_top is not None and top > prev_top + 50:
            backward += 1        # note: PDF y grows upward; big upward jump = new column
        prev_page, prev_top = s.page, top
    return (backward / max(len(spans) - 1, 1)) < 0.34
```
Export in `__init__.py`: `from rho.ingestion.structural import structural_check`.

- [ ] **Step 4: Run to verify pass**
Run: `pytest tests/unit/test_ingestion.py -k structural -v`
Expected: PASS both.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: reading-order structural heuristic"
```

---

## Self-Review
- [ ] `ingest` routes .txt/.md, .pdf, .docx; returns markdown + populated ProvenanceMap.
- [ ] Every `SourceSpan.raw_text` equals `markdown[char_start:char_end]` (text path) or is `.find`-located (Docling path).
- [ ] `structural_check` exported and tested.
- [ ] Full run: `pytest tests/unit/test_ingestion.py -v` green.

## Results (fill in)
- Docling version used: **2.113.0**. `iterate_items()` and `item.prov` worked as the plan assumed — no API adaptation needed.
- Fixtures created: `clean.txt`, `simple.docx` (36K, generated by `tests/fixtures/make_simple_docx.py`, committed so the suite runs without regenerating). No PDF fixture yet — `multicolumn.pdf` / `clean.pdf` still outstanding, see Notes.
- Tests passing: **11 / 11** full suite (`pytest -v`); 5 of them ingestion-specific.
- Notes:
  - **Deviation from plan (Task 2), deliberate:** the plan's adapter calls `md.find(text.strip())` from index 0 on every item, so a line appearing twice maps *both* spans to the first occurrence. Verified against the real fixture: two identical bullets both resolved to `char_start=128`. Since C1 depends on prov_ids pointing at the right source location, that is silent provenance corruption. Added a `search_from` cursor that advances past each match, with a global-find fallback when the ordered search misses. Both bullets now resolve distinctly (128 and 215).
  - Added `test_docling_repeated_lines_get_distinct_offsets` to lock this in; confirmed it fails (`assert 128 != 128`) against the plan's original logic and passes with the fix. The plan's own range-only assertion (`0 <= start <= end <= len(md)`) passes in both cases, i.e. it would not have caught this.
  - Also strengthened `test_docling_ingest_produces_markdown_and_spans` to assert `md[start:end] == raw_text` rather than only that offsets are in range.
  - **Modified a Phase 0 test:** `tests/unit/test_stubs.py` asserted `ingest(b"", "x.pdf")` raises `NotImplementedError`, which stopped being true once P1 implemented `ingest`. Dropped `ingest` from that test; it now asserts on `extract`, still stubbed. Ingestion coverage lives in `test_ingestion.py`.
  - DOCX carries no page/bbox geometry, so those spans have `page=None, bbox=None` (allowed by shared-context §"no silent data loss" — char offsets still recorded). Consequence: `structural_check` returns `True` for DOCX by construction, since it needs geometry to judge. It is only meaningfully exercised on PDFs.
  - **Not yet validated on PDF.** The geometry path (`page`, `bbox` from `item.prov`) and the multi-column scramble heuristic have no real-PDF fixture behind them — `structural_check`'s PDF behavior is currently covered only by synthetic bboxes in unit tests. Worth adding `clean.pdf` + `multicolumn.pdf` before relying on either.
  - Env: Docling pulled a large dependency tree (torch et al); first install took ~15 min. First `convert()` call downloads model weights, so the initial docling test run took ~31s.
