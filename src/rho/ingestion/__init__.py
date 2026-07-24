from pathlib import Path

from rho.ingestion.structural import structural_check
from rho.ingestion.text_adapter import ingest_text
from rho.models.provenance import ProvenanceMap

__all__ = ["ingest", "structural_check"]


def ingest(file_bytes: bytes, filename: str) -> tuple[str, ProvenanceMap]:
    """file -> (markdown, ProvenanceMap)"""
    doc_id = Path(filename).stem
    ext = Path(filename).suffix.lower()
    if ext in {".txt", ".md"}:
        return ingest_text(file_bytes.decode("utf-8", errors="replace"), doc_id)
    from rho.ingestion.docling_adapter import ingest_docling  # lazy import (heavy)

    return ingest_docling(file_bytes, filename, doc_id)
