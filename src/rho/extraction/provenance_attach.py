from rapidfuzz import fuzz

from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume


def find_prov(value: str, prov: ProvenanceMap, threshold: int = 90) -> list[str]:
    """Return prov_ids whose span raw_text supports `value` (exact substring or fuzzy)."""
    if not value or not value.strip():
        return []
    v = value.strip().lower()
    hits = []
    for pid, span in prov.spans.items():
        raw = span.raw_text.lower()
        if v in raw or fuzz.partial_ratio(v, raw) >= threshold:
            hits.append(pid)
    return hits


def attach_provenance(
    resume: StructuredResume, prov: ProvenanceMap, threshold: int = 90
) -> StructuredResume:
    r = resume.model_copy(deep=True)
    r.name_prov = find_prov(r.name, prov, threshold)
    r.contact_prov = []
    for c in r.emails + r.phones + r.urls:
        r.contact_prov += find_prov(c, prov, threshold)
    r.skills_prov = [find_prov(s, prov, threshold) for s in r.skills]
    r.achievements_prov = [find_prov(a, prov, threshold) for a in r.achievements]
    for w in r.work:
        w.company_prov = find_prov(w.company, prov, threshold)
        w.title_prov = find_prov(w.title, prov, threshold)
        w.date_prov = find_prov(
            (w.start_date or "") + " " + (w.end_date or ""), prov, threshold
        )
        w.bullet_prov = [find_prov(b, prov, threshold) for b in w.bullets]
    for e in r.education:
        e.institution_prov = find_prov(e.institution, prov, threshold)
        e.edu_prov = find_prov((e.degree or "") + " " + (e.field or ""), prov, threshold)
    for p in r.projects:
        p.name_prov = find_prov(p.name, prov, threshold)
        p.url_prov = find_prov(p.url or "", prov, threshold)
        p.tech_prov = [find_prov(t, prov, threshold) for t in p.tech]
        p.bullet_prov = [find_prov(b, prov, threshold) for b in p.bullets]
    return r
