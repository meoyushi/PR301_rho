from pydantic import BaseModel


class WorkExperience(BaseModel):
    company: str
    company_prov: list[str] = []
    title: str
    title_prov: list[str] = []
    start_date: str | None = None
    end_date: str | None = None
    date_prov: list[str] = []
    bullets: list[str] = []
    bullet_prov: list[list[str]] = []  # per-bullet


class Project(BaseModel):
    name: str
    name_prov: list[str] = []
    url: str | None = None
    url_prov: list[str] = []
    tech: list[str] = []
    tech_prov: list[list[str]] = []  # per-tech, like skills_prov
    bullets: list[str] = []
    bullet_prov: list[list[str]] = []  # per-bullet, like work


class Education(BaseModel):
    institution: str
    institution_prov: list[str] = []
    degree: str | None = None
    field: str | None = None
    end_year: str | None = None
    edu_prov: list[str] = []


class StructuredResume(BaseModel):
    name: str
    name_prov: list[str] = []
    headline: str | None = None
    summary: str | None = None
    emails: list[str] = []
    phones: list[str] = []
    urls: list[str] = []
    contact_prov: list[str] = []
    work: list[WorkExperience] = []
    education: list[Education] = []
    projects: list[Project] = []
    skills: list[str] = []
    skills_prov: list[list[str]] = []
    certifications: list[str] = []
    achievements: list[str] = []
    achievements_prov: list[list[str]] = []
