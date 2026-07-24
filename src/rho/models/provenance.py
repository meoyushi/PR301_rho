from pydantic import BaseModel


class SourceSpan(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    raw_text: str


class ProvenanceMap(BaseModel):
    doc_id: str
    spans: dict[str, SourceSpan] = {}

    def add(self, span: SourceSpan) -> str:
        pid = f"p:{self.doc_id}:{len(self.spans)}"
        self.spans[pid] = span
        return pid

    def get(self, prov_id: str) -> SourceSpan:
        return self.spans[prov_id]
