from rapidfuzz import fuzz

from rho.config import settings
from rho.matching.coverage import fuzzy_coverage, keyword_coverage, resume_text_terms
from rho.matching.embed import Embedder
from rho.models.jd import RequirementSet
from rho.models.resume import StructuredResume
from rho.models.scoring import ComponentVector, Gap, MatchResult


def _prov_for(resume: StructuredResume, i: int) -> list[str]:
    """skills_prov may be shorter than skills; missing prov is [], never a dropped skill."""
    return resume.skills_prov[i] if i < len(resume.skills_prov) else []


def _best_cosine(term: str, resume: StructuredResume, emb: Embedder) -> float:
    """Highest cosine between `term` and any resume skill. 0.0 when there are no skills."""
    if not resume.skills:
        return 0.0
    tv = emb.encode([term])[0]
    sv = emb.encode(resume.skills)
    return max(emb.cosine(tv, sv[i]) for i in range(len(resume.skills)))


def _skill_evidence(
    term: str,
    resume: StructuredResume,
    emb: Embedder,
    sem_hi: float | None = None,
    sem_lo: float | None = None,
) -> tuple[str, list[str], float]:
    """returns (status, prov_ids, best_cosine)"""
    sem_hi = settings.sem_hi if sem_hi is None else sem_hi
    sem_lo = settings.sem_lo if sem_lo is None else sem_lo
    tl = term.lower()
    for i, skill in enumerate(resume.skills):
        sl = skill.lower()
        if tl in sl or sl in tl or fuzz.ratio(tl, sl) >= 85:
            return "present", _prov_for(resume, i), _best_cosine(term, resume, emb)
    if resume.skills:
        tv = emb.encode([term])[0]
        sv = emb.encode(resume.skills)
        best_i = max(range(len(resume.skills)), key=lambda i: emb.cosine(tv, sv[i]))
        best = emb.cosine(tv, sv[best_i])
        if best >= sem_hi:
            return "present", _prov_for(resume, best_i), best
        if best >= sem_lo:
            return "weak", _prov_for(resume, best_i), best
        return "absent", [], best
    return "absent", [], 0.0


def match(resume: StructuredResume, reqs: RequirementSet) -> MatchResult:
    """fills component_vector + gaps; predicted_score left 0.0 until P4"""
    emb = Embedder()
    req_terms = [r.text for r in reqs.requirements]
    must = [r for r in reqs.requirements if r.priority == "must"]
    nice = [r for r in reqs.requirements if r.priority == "nice"]
    gaps = []
    cosines = []
    present_must = present_nice = 0
    for r in reqs.requirements:
        status, prov, best_cos = _skill_evidence(r.text, resume, emb)
        gaps.append(Gap(requirement=r, status=status, evidence_prov=prov))
        cosines.append(best_cos)
        if status in ("present", "weak"):
            if r.priority == "must":
                present_must += 1
            else:
                present_nice += 1
    # mean best-match cosine across requirements — a true semantic signal, distinct
    # from the coverage fields. Clamped to [0,1]: cosines can go slightly negative.
    mean_cos = (sum(cosines) / len(cosines)) if cosines else 1.0
    # Coverage searches the whole résumé, matching _skill_evidence above:
    # requirement evidence lives in bullets and titles, not only the skills list.
    haystack = resume_text_terms(resume)
    cv = ComponentVector(
        keyword_coverage=keyword_coverage(req_terms, haystack),
        semantic_similarity=min(1.0, max(0.0, mean_cos)),
        fuzzy_coverage=fuzzy_coverage(req_terms, haystack),
        must_have_coverage=(present_must / len(must)) if must else 1.0,
        nice_have_coverage=(present_nice / len(nice)) if nice else 1.0,
    )
    return MatchResult(component_vector=cv, predicted_score=0.0, gaps=gaps)
