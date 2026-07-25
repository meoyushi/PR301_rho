from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, TailoredResume
from rho.models.scoring import Gap
from rho.rewrite.verifier import verify_against_source

__all__ = ["rewrite", "verify", "verify_against_source"]


def rewrite(
    resume: StructuredResume,
    gaps: list[Gap],
    prov: ProvenanceMap,
    _rewrite_fn=None,
) -> TailoredResume:
    """Tailor `resume` toward `gaps`, then gate the result against `prov`.

    Signature extends the frozen Section-6 `rewrite(resume, gaps)` with `prov`:
    the verification gate cannot run without the provenance map, and C3 requires
    that nothing ships ungated. Shared context Section 6 records the change.
    """
    if _rewrite_fn is None:
        from rho.rewrite.llm import rewrite_schema as _rewrite_fn
    tailored = _rewrite_fn(resume, gaps)
    fixed, report = verify_against_source(tailored, resume, prov)
    return TailoredResume(resume=fixed, fabrication_report=report)


def verify(tailored: StructuredResume, prov: ProvenanceMap) -> FabricationReport:
    """Frozen Section-6 signature, deliberately not callable.

    The gate must distinguish an *addition* from a reorder of values the source
    already claimed, which needs the source résumé — something this frozen
    signature cannot carry. `rewrite()` therefore calls `verify_against_source`
    directly with the source in hand. Failing loudly beats silently scoring
    every value as new and reporting a meaningless fabrication rate.
    """
    raise RuntimeError(
        "call verify_against_source(tailored, source, prov); wired in rewrite()"
    )
