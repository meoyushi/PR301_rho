"""Reviewer node internals — re-assert the provenance invariant end-to-end.

Phase 5's gate already rejects unsupported edits at rewrite time. This module
re-checks the *shipped* résumé independently (shared context Section 7): every
hard-content token must trace to at least one prov_id. A violation here means
the gate leaked, so it is reported, never silently dropped, and never fatal —
the caller still gets a response with `invariant_ok=False`.
"""

from rho.extraction.provenance_attach import find_prov
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport
from rho.models.scoring import MatchResult
from rho.rewrite.tokens import hard_content_tokens


def check_provenance_invariant(
    tailored: StructuredResume, prov: ProvenanceMap
) -> tuple[bool, list[str]]:
    """(ok, violations) — values with no supporting prov_id, in field order."""
    violations: list[str] = []
    for value, _path in hard_content_tokens(tailored):
        if not find_prov(value, prov):
            violations.append(value)
    return (len(violations) == 0, violations)


def compute_final_score(
    match_result: MatchResult, fabrication_report: FabricationReport
) -> float:
    """`predicted_score` is already the calibrated ATS score.

    Fabrication is prevented by the Phase-5 gate rather than priced in here, so
    applying a penalty would double-count it. The report is taken as a parameter
    to keep the weighting hook available without changing callers later.
    """
    return match_result.predicted_score
