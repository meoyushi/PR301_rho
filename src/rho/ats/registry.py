"""Configured match-score engines.

Construction is lazy and failure-tolerant so importing `rho.ats` never requires
a live engine at unit-test time. OpenCATS is deliberately absent: it feeds the
separate parse-recovery path, not the match-score target.
"""

import logging

logger = logging.getLogger(__name__)


def _build_engines() -> list:
    engines = []
    try:
        from rho.ats.engines.ats_screener import ATSScreener

        engines.append(ATSScreener())
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("ats_screener unavailable: %s", exc)
    return engines


ENGINES = _build_engines()
