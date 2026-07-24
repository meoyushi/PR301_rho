from pydantic import BaseModel

from rho.models.resume import StructuredResume


class RejectedEdit(BaseModel):
    added_text: str
    reason: str  # e.g. "no supporting prov_id"


class FabricationReport(BaseModel):
    total_edits: int
    verified_edits: int
    rejected_edits: list[RejectedEdit] = []
    fabrication_rate: float  # rejected / total


class TailoredResume(BaseModel):
    resume: StructuredResume  # rewritten, prov chain intact
    fabrication_report: FabricationReport
