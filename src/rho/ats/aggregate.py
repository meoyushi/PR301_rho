"""Aggregate per-engine outputs into the single calibration target `y`.

This is the one documented place where engine outputs collapse to a number, so
the paper can state the target definition precisely.
"""


# ats-screener scores five dimensions, but only keywordMatch varies with the
# job description; formatting/sections/experience/education score the résumé on
# its own. Averaging all five into `overallScore` therefore produces a target
# dominated by résumé-intrinsic quality — on the calibration corpus keywordMatch
# contributed only ~2-11 of ~45 points, so a match-feature calibration fitted
# against it would mostly be predicting résumé quality.
JD_DEPENDENT_DIMENSIONS = ("keywordMatch",)


def to_match_target(engine_outputs: dict) -> float:
    """-> y in 0..100 from the JD-dependent dimensions only.

    Raises ValueError when no engine reported a usable breakdown, so the caller
    skips the doc rather than imputing a value.
    """
    scores = []
    for out in engine_outputs.values():
        breakdown = (out.get("raw") or {}).get("breakdown") or {}
        for dims in breakdown.values():
            present = [dims[d] for d in JD_DEPENDENT_DIMENSIONS if d in dims]
            if present:
                scores.append(sum(present) / len(present))
    if not scores:
        raise ValueError("no engine reported a JD-dependent breakdown; exclude this doc")
    return sum(scores) / len(scores)


def to_target(engine_outputs: dict) -> float:
    """-> y in 0..100: mean of the engines that produced a score.

    Raises ValueError when no engine scored, so the caller skips the doc
    rather than imputing a value.
    """
    scores = [
        o["match_score"] for o in engine_outputs.values() if o.get("match_score") is not None
    ]
    if not scores:
        raise ValueError("no engine produced a score; exclude this doc from fit")
    return sum(scores) / len(scores)
