from typing import Literal

from pydantic import BaseModel

from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import TailoredResume
from rho.models.scoring import MatchResult


class OptimizeRequest(BaseModel):
    jd_text: str
    # file arrives as multipart upload, not in this model


class PipelineResponse(BaseModel):
    structured_resume: StructuredResume
    provenance_map: ProvenanceMap
    match_result: MatchResult
    tailored_resume: TailoredResume
    final_score: float


JobState = Literal["queued", "running", "done", "error"]


class ParseResponse(BaseModel):
    structured_resume: StructuredResume
    provenance_map: ProvenanceMap


class OptimizeJobRequest(BaseModel):
    resume: StructuredResume
    jd_text: str


class ExportDocxRequest(BaseModel):
    resume: StructuredResume
    section_order: list[str] | None = None
    accent: str = "#b5482a"
    hidden_sections: list[str] | None = None


class ScoreComponent(BaseModel):
    """One before/after component pair, for the UI's improvement breakdown."""

    label: str
    before: float  # 0..1
    after: float  # 0..1


class OptimizeResult(BaseModel):
    match_result: MatchResult
    tailored_resume: TailoredResume
    final_score: float  # calibrated proxy (research value)
    display_score: float = 0.0  # final_score rescaled to a readable 0..100
    baseline_score: float | None = None  # calibrated score of the ORIGINAL résumé
    baseline_display_score: float | None = None  # its rescaled 0..100
    components: list[ScoreComponent] = []  # before/after per matching signal
    previous_score: float | None = None


class JobStatus(BaseModel):
    id: str
    state: JobState = "queued"
    stage: str | None = None
    result: OptimizeResult | None = None
    error: str | None = None
