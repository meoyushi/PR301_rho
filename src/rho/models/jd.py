from typing import Literal

from pydantic import BaseModel


class Requirement(BaseModel):
    text: str
    kind: Literal["skill", "tool", "title", "cert", "experience"]
    priority: Literal["must", "nice"]
    years: float | None = None


class RequirementSet(BaseModel):
    title: str | None = None
    requirements: list[Requirement] = []
