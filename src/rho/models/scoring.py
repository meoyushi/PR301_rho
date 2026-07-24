from typing import Literal

from pydantic import BaseModel

from rho.models.jd import Requirement


class ComponentVector(BaseModel):
    """Raw signals, pre-calibration."""

    keyword_coverage: float  # 0..1
    semantic_similarity: float  # 0..1
    fuzzy_coverage: float  # 0..1
    must_have_coverage: float  # 0..1
    nice_have_coverage: float  # 0..1


class Gap(BaseModel):
    requirement: Requirement
    status: Literal["present", "absent", "weak"]
    evidence_prov: list[str] = []  # prov_ids supporting "present"/"weak"


class MatchResult(BaseModel):
    component_vector: ComponentVector
    predicted_score: float  # 0..100 (set by calibrator, P4)
    gaps: list[Gap] = []
