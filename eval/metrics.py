"""Evaluation metrics for C1 — extraction quality and provenance attachment.

Three metric families, deliberately kept separate because they answer different
questions and the report's finding depends on not averaging them together:

  `field_f1`        — named entities (skills, work, education). Set-based, so
                      order and duplication do not affect the score.
  `long_text_f1`    — free prose (summary, headline). Token-overlap only; exact
                      match is not a meaningful target for generated prose, and
                      mixing these into the entity average hides that long-text
                      is the hardest field class (Phase 7 global constraint).
  `provenance_accuracy` — C1's own metric: does the attached prov_id point at
                      the *correct* source span, not merely at some span.

Everything here is pure and deterministic: no LLM, no I/O, no randomness. The
paper's Table 1 is reproducible from these functions plus the gold set.
"""

from typing import Any, Iterable, Mapping, Sequence

from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume

Metrics = dict[str, float]


def _prf(pred_set: set, gold_set: set) -> Metrics:
    """Precision/recall/F1 over two sets.

    Both empty scores 1.0: the extractor correctly reported that the section is
    absent. Scoring that 0.0 would punish honest empties, and shared context
    Section 8 ("no silent fills") requires empty to be a legitimate answer.
    """
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def _norm(value: Any) -> str:
    """Casing and surrounding whitespace are formatting, not extraction errors."""
    return str(value).strip().lower()


def _entity_key(item: Any, keys: Sequence[str] | None) -> Any:
    """Comparable key for one list element.

    Scalars compare by their normalised text. Dicts/models compare on `keys`
    only — the labelled fields — so an extra unlabelled field (a start_date the
    gold set never recorded) does not count against the extractor.
    """
    if isinstance(item, Mapping):
        if keys is None:
            return tuple(sorted((k, _norm(v)) for k, v in item.items()))
        return tuple(_norm(item.get(k, "")) for k in keys)
    if hasattr(item, "model_dump"):
        return _entity_key(item.model_dump(), keys)
    return _norm(item)


def field_f1(
    pred: Mapping[str, Any],
    gold: Mapping[str, Any],
    field: str,
    keys: Sequence[str] | None = None,
) -> Metrics:
    """P/R/F1 for one list-valued field, aligned as a set of entities.

    `keys` selects which sub-fields identify an entity for list-of-dict fields
    such as `work` and `education`; ignored for lists of strings.
    """
    ps = {_entity_key(x, keys) for x in (pred.get(field) or [])}
    gs = {_entity_key(x, keys) for x in (gold.get(field) or [])}
    return _prf(ps, gs)


def long_text_f1(pred: str | None, gold: str | None) -> float:
    """Token-overlap F1 for a free-prose field.

    Reported separately from entity F1 (never averaged in): prose is rewritten
    rather than copied, so a low score here is a different phenomenon from a
    missed entity.
    """
    return _prf(set((pred or "").lower().split()), set((gold or "").lower().split()))["f1"]


def _attached_prov(resume: StructuredResume) -> dict[str, list[str]]:
    """field_path -> attached prov_ids, over the same paths as `hard_content_tokens`.

    Paths mirror `rho.rewrite.tokens.hard_content_tokens` so that what C1 scores
    here is exactly what the Phase-5 gate and Phase-6 reviewer check.
    """
    out: dict[str, list[str]] = {}
    for i, _ in enumerate(resume.skills):
        out[f"skills[{i}]"] = list(resume.skills_prov[i]) if i < len(resume.skills_prov) else []
    out["name"] = list(resume.name_prov)
    for wi, w in enumerate(resume.work):
        out[f"work[{wi}].company"] = list(w.company_prov)
        out[f"work[{wi}].title"] = list(w.title_prov)
        out[f"work[{wi}].date"] = list(w.date_prov)
        for bi, _ in enumerate(w.bullets):
            out[f"work[{wi}].bullets[{bi}]"] = (
                list(w.bullet_prov[bi]) if bi < len(w.bullet_prov) else []
            )
    for ei, e in enumerate(resume.education):
        out[f"education[{ei}].institution"] = list(e.institution_prov)
        out[f"education[{ei}].edu"] = list(e.edu_prov)
    return out


def _resolves_to(prov_id: str, target: Any, prov: ProvenanceMap) -> bool:
    """Does `prov_id` name the span the gold entry points at?

    Gold may identify the correct span either by prov_id (when the gold set was
    built alongside the ProvenanceMap) or by a `(char_start, char_end)` range
    (when it was hand-labelled against the document text). Both are supported so
    the synthetic and hand-labelled gold sets can share one metric.
    """
    if isinstance(target, str):
        return prov_id == target
    if isinstance(target, (tuple, list)) and len(target) == 2:
        span = prov.spans.get(prov_id)
        if span is None:
            return False
        start, end = target
        # Overlap, not equality: a span may be a whole line while the gold range
        # marks the value inside it. Containment either way counts as correct.
        return not (span.char_end <= start or span.char_start >= end)
    return False


def provenance_accuracy(
    resume: StructuredResume,
    gold_prov: Mapping[str, Any],
    prov: ProvenanceMap | None = None,
) -> float:
    """Fraction of gold-labelled fields whose attached prov points at the right span.

    `gold_prov` maps a field path (`"skills[0]"`, `"work[0].company"`) to the
    correct source location, given as a prov_id or a `(char_start, char_end)`
    range. A field with no attached provenance counts as wrong, not skipped: an
    unattached value is a broken chain, which is precisely what C1 claims cannot
    happen.
    """
    if not gold_prov:
        return 0.0
    attached = _attached_prov(resume)
    prov = prov or ProvenanceMap(doc_id="")
    correct = 0
    for path, target in gold_prov.items():
        ids = attached.get(path) or []
        if any(_resolves_to(pid, target, prov) for pid in ids):
            correct += 1
    return correct / len(gold_prov)


def mean(values: Iterable[float]) -> float:
    """Mean that treats an empty sequence as 0.0 rather than raising."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
