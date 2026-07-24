from pydantic import BaseModel

from rho.models.resume import Education, Project, StructuredResume, WorkExperience


class WorkItem(BaseModel):
    company: str
    title: str
    start_date: str | None = None
    end_date: str | None = None
    bullets: list[str] = []


class ProjectItem(BaseModel):
    name: str
    url: str | None = None
    tech: list[str] = []
    bullets: list[str] = []


class EduItem(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    end_year: str | None = None


class ExtractionSchema(BaseModel):
    reasoning: str  # FIRST: LLMs generate left-to-right
    name: str
    headline: str | None = None
    summary: str | None = None
    emails: list[str] = []
    phones: list[str] = []
    urls: list[str] = []
    work: list[WorkItem] = []
    education: list[EduItem] = []
    projects: list[ProjectItem] = []
    skills: list[str] = []
    certifications: list[str] = []
    achievements: list[str] = []


def to_structured(es: ExtractionSchema) -> StructuredResume:
    return StructuredResume(
        name=es.name,
        headline=es.headline,
        summary=es.summary,
        emails=es.emails,
        phones=es.phones,
        urls=es.urls,
        work=[
            WorkExperience(
                company=w.company,
                title=w.title,
                start_date=w.start_date,
                end_date=w.end_date,
                bullets=w.bullets,
            )
            for w in es.work
        ],
        education=[
            Education(
                institution=e.institution,
                degree=e.degree,
                field=e.field,
                end_year=e.end_year,
            )
            for e in es.education
        ],
        projects=[
            Project(
                name=p.name,
                url=p.url,
                tech=p.tech,
                bullets=p.bullets,
            )
            for p in es.projects
        ],
        skills=es.skills,
        certifications=es.certifications,
        achievements=es.achievements,
    )
