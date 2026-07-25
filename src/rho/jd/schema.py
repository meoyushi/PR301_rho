from typing import Literal

from pydantic import BaseModel

from rho.models.jd import Requirement, RequirementSet


class ReqItem(BaseModel):
    text: str
    kind: Literal["skill", "tool", "title", "cert", "experience"]
    priority: Literal["must", "nice"]
    years: float | None = None


class JDSchema(BaseModel):
    reasoning: str  # FIRST: LLMs generate left-to-right
    title: str | None = None
    requirements: list[ReqItem] = []


def to_requirement_set(js: JDSchema) -> RequirementSet:
    return RequirementSet(
        title=js.title,
        requirements=[
            Requirement(text=r.text, kind=r.kind, priority=r.priority, years=r.years)
            for r in js.requirements
        ],
    )
