"""Human-readable rescaling of the calibrated ATS score.

The calibrated `predicted_score` is a faithful proxy for a real ATS engine's
`keywordMatch` dimension, fit on a corpus where that target sits in a narrow,
low band (mean ~13, sd ~4.9). Its practical output range is roughly 7 (a poor
JD match) to 27 (an excellent one) — so a strong résumé scores ~21/100, which
reads as a failing grade to anyone who does not know the scale.

`to_display_score` maps that real band onto a 0-100 scale for the UI, so the
number moves the way a user expects while the underlying calibrated value is
kept unchanged for reproducibility. This is a presentation transform, NOT a new
model: it never invents signal, it only stretches the axis.

The anchors below were measured from the shipped calibrator
(`eval/calibrator.joblib`) over the feature cube — see the band derivation in
the conversation/PR that introduced this. If the calibrator is refit, re-measure
and update `_FLOOR`/`_CEIL`.
"""

# Calibrated-score band observed from the fitted Ridge model: a poor match
# floors near 7, an excellent one ceils near 27.
_FLOOR = 7.0
_CEIL = 27.0


def to_display_score(calibrated: float) -> float:
    """Map a calibrated score (~7-27) onto a readable 0-100, clamped.

    Linear: `_FLOOR` -> 0, `_CEIL` -> 100. Values outside the band clamp to the
    ends rather than exceeding [0, 100].
    """
    span = _CEIL - _FLOOR
    if span <= 0:  # guard a degenerate reconfiguration
        return max(0.0, min(100.0, calibrated))
    scaled = (calibrated - _FLOOR) / span * 100.0
    return max(0.0, min(100.0, round(scaled, 1)))
