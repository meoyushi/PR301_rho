"""LangGraph node functions — thin wrappers over the Phase 1-5 components.

Each node takes the whole state and returns only the keys it produced, which is
what lets `extract` and `jd` run in parallel without clobbering each other.

Components are imported into this module's namespace on purpose: tests
monkeypatch `rho.graph.nodes.extract` (etc.) to run the graph without an LLM.
"""

import logging
import os

from rho.ats import Calibrator, score_with_calibrator
from rho.extraction import extract
from rho.graph.review import check_provenance_invariant, compute_final_score
from rho.ingestion import ingest
from rho.jd import analyze_jd
from rho.matching import match
from rho.rewrite import rewrite

logger = logging.getLogger(__name__)

# Resolved at call time, not import time, so tests can monkeypatch this name.
CALIBRATOR_PATH = "eval/calibrator.joblib"

# `CALIBRATOR_PATH` is relative, so it only resolved when the process happened to
# start in the repo root — in the container (WORKDIR /app) it never did, and every
# score silently fell back to 0.0. Fall back to a path anchored on this file so
# the calibrator is found regardless of CWD: src/rho/graph/ -> ../../../eval.
_REPO_CALIBRATOR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "eval", "calibrator.joblib"
)


def _calibrator_path() -> str | None:
    """First existing calibrator location, or None when none is installed."""
    for path in (CALIBRATOR_PATH, _REPO_CALIBRATOR):
        if os.path.exists(path):
            return path
    return None


def ingest_node(state):
    md, prov = ingest(state["file_bytes"], state["filename"])
    return {"markdown": md, "prov": prov}


def extract_node(state):
    return {"resume": extract(state["markdown"], state["prov"])}


def jd_node(state):
    return {"reqs": analyze_jd(state["jd_text"])}


def match_node(state):
    return {"match_result": match(state["resume"], state["reqs"])}


def score_node(state):
    """Apply the Phase-4 calibrator when one has been fitted.

    Without `eval/calibrator.joblib` there is nothing calibrated to say, so
    `predicted_score` stays at the raw-vector fallback the matcher set (0.0)
    rather than being invented here — shared context Section 8, no silent fills.
    """
    mr = state["match_result"]
    path = _calibrator_path()
    if path is not None:
        cal = Calibrator().load(path)
        mr = score_with_calibrator(mr, cal)
    else:
        logger.warning(
            "%s missing; predicted_score left at raw fallback", CALIBRATOR_PATH
        )
    return {"match_result": mr}


def rewrite_node(state):
    return {
        "tailored": rewrite(
            state["resume"], state["match_result"].gaps, state["prov"]
        )
    }


def review_node(state):
    ok, viol = check_provenance_invariant(state["tailored"].resume, state["prov"])
    if not ok:
        logger.warning("provenance invariant violated: %s", viol)
    final = compute_final_score(
        state["match_result"], state["tailored"].fabrication_report
    )
    return {"invariant_ok": ok, "invariant_violations": viol, "final_score": final}
