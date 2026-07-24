from rho.models.provenance import ProvenanceMap


def structural_check(markdown: str, prov: ProvenanceMap) -> bool:
    """Heuristic reading-order sanity check. True = order looks OK.

    PDF y grows upward, so a large upward jump between consecutive spans on the
    same page signals a column break emitted out of order. No geometry = can't
    tell = True.
    """
    spans = [s for s in prov.spans.values() if s.bbox is not None and s.page is not None]
    if len(spans) < 3:
        return True
    backward = 0
    prev_page, prev_top = None, None
    for s in spans:
        top = s.bbox[1]
        if prev_page == s.page and prev_top is not None and top > prev_top + 50:
            backward += 1
        prev_page, prev_top = s.page, top
    return (backward / max(len(spans) - 1, 1)) < 0.34
