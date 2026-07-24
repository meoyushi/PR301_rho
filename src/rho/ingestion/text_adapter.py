from rho.models.provenance import ProvenanceMap, SourceSpan


def ingest_text(text: str, doc_id: str) -> tuple[str, ProvenanceMap]:
    """Plain text passthrough: markdown == text, one span per non-empty line."""
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
