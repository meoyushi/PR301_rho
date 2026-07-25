"""Build (ComponentVector, y) pairs for calibration.

Docs are dropped — never imputed — when no engine scores them or when
featurisation fails, so the fit only sees real observations.
"""

import logging

from rho.ats.aggregate import to_target

logger = logging.getLogger(__name__)


def build_calibration_dataset(pairs, harvest_fn, feature_fn, target_fn=to_target, on_progress=None):
    """pairs: [(resume, jd_text), ...] -> (X, y).

    `target_fn` selects the target definition — `to_target` for the engines'
    composite score, `to_match_target` for the JD-dependent dimensions only.

    `on_progress(index, total, status, kept)` fires once per pair, including
    skipped ones, so a caller can report accurate remaining counts on a run
    that takes hours.
    """
    X, y = [], []
    total = len(pairs)

    def _report(index, status):
        if on_progress is not None:
            on_progress(index=index, total=total, status=status, kept=len(X))

    for index, (resume, jd_text) in enumerate(pairs, start=1):
        outs = harvest_fn(resume, jd_text)
        try:
            target = target_fn(outs)
        except ValueError:
            logger.info("no engine score; skipping doc")
            _report(index, "skipped_no_score")
            continue
        try:
            features = feature_fn(resume, jd_text)
        except Exception as exc:
            logger.warning("featurisation failed; skipping doc: %s", exc)
            _report(index, "skipped_no_features")
            continue
        X.append(features)
        y.append(target)
        _report(index, "ok")
    return X, y
