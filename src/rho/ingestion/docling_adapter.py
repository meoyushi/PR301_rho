import os
import tempfile
from functools import lru_cache

from rho.models.provenance import ProvenanceMap, SourceSpan


@lru_cache(maxsize=1)
def _converter():
    """The Docling converter, built once and reused across requests.

    `DocumentConverter()` loads layout/OCR model weights on construction, so
    rebuilding it per call reloaded them on every ingest. Cached here, the
    weights load once (warmed at app startup via `warm_up`), and each request
    reuses the resident models.
    """
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


# Smallest valid single-page PDF with a text layer — converting it forces the
# converter to load every model it lazily initialises on first real use (layout
# *and* OCR), so the first user PDF does not pay that cost.
_WARMUP_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]"
    b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 44>>stream\n"
    b"BT /F1 12 Tf 20 100 Td (warm up) Tj ET\n"
    b"endstream endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)


def warm_up() -> None:
    """Force the converter and its model weights to load now.

    Called from the API startup hook so the first real request does not pay the
    one-time model-load cost. Runs a tiny PDF through the full path so the OCR
    model (loaded lazily on the first PDF, not at converter construction) is
    resident too.
    """
    import tempfile

    conv = _converter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(_WARMUP_PDF)
            path = tmp.name
        conv.convert(path)
        os.unlink(path)
    except Exception:
        # A warm-up conversion failure is not fatal: the converter is still
        # cached, so requests just pay the OCR load on the first real PDF.
        pass


def ingest_docling(file_bytes: bytes, filename: str, doc_id: str) -> tuple[str, ProvenanceMap]:
    """PDF/DOCX/image -> (markdown, ProvenanceMap) via Docling.

    Char offsets index into the exported markdown; page/bbox come from the
    Docling item geometry when present.
    """
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        result = _converter().convert(path)
        doc = result.document
        md = doc.export_to_markdown()
    finally:
        os.unlink(path)

    pm = ProvenanceMap(doc_id=doc_id)
    # Locate each text item's string in the exported markdown to get char offsets.
    # search_from advances so repeated strings map to successive occurrences
    # rather than all collapsing onto the first.
    search_from = 0
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        content = text.strip()
        idx = md.find(content, search_from)
        if idx == -1:
            idx = md.find(content)
            if idx == -1:
                continue
        else:
            search_from = idx + len(content)
        page = None
        bbox = None
        prov = getattr(item, "prov", None)
        if prov:
            p0 = prov[0]
            page = getattr(p0, "page_no", None)
            bb = getattr(p0, "bbox", None)
            if bb is not None:
                bbox = (bb.l, bb.t, bb.r, bb.b)
        pm.add(
            SourceSpan(
                doc_id=doc_id,
                char_start=idx,
                char_end=idx + len(content),
                page=page,
                bbox=bbox,
                raw_text=content,
            )
        )
    return md, pm
