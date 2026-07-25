"""Phase 4 — ATS harness + calibration (contribution C2)."""

import logging

from rho.ats.calibrator import Calibrator
from rho.models.resume import StructuredResume
from rho.models.scoring import MatchResult

logger = logging.getLogger(__name__)

__all__ = ["Calibrator", "harvest_ats", "score_with_calibrator"]


def harvest_ats(resume: StructuredResume, jd_text: str) -> dict:
    """Run every configured ATS engine -> {engine_name: engine_output}.

    An engine that raises is logged and omitted rather than failing the whole
    harvest; a doc that ends up with no scores is skipped at dataset-build time.
    """
    from rho.ats.registry import ENGINES

    outputs = {}
    for engine in ENGINES:
        try:
            outputs[engine.name] = engine.run(resume, jd_text)
        except Exception as exc:
            logger.warning("engine %s failed: %s", engine.name, exc)
    return outputs


def score_with_calibrator(match_result: MatchResult, calibrator: Calibrator) -> MatchResult:
    """Fill `predicted_score` from the fitted calibrator.

    Lives here rather than in `matching` so the matcher stays engine-free.
    """
    mr = match_result.model_copy(deep=True)
    mr.predicted_score = calibrator.predict(mr.component_vector)
    return mr
