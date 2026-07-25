from typing import Protocol

from rho.models.resume import StructuredResume


class ATSEngine(Protocol):
    """A self-hostable ATS engine we can harvest real scoring output from.

    Engines take rho's already-extracted `StructuredResume` rather than raw
    file bytes: the résumé has been parsed once in P1/P2, and re-parsing it per
    engine would make the calibration target depend on each engine's parser
    quirks instead of its scoring rules.
    """

    name: str

    def run(self, resume: StructuredResume, jd_text: str) -> dict:
        """-> {"engine": str, "parse_fields": dict|None, "match_score": float|None, "raw": Any}"""
        ...
