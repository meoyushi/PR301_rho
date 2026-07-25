"""build_docx produces a valid .docx carrying every résumé section."""

import io

from docx import Document

from rho.api.docx_export import build_docx
from rho.models.resume import Education, Project, StructuredResume, WorkExperience


def _resume() -> StructuredResume:
    return StructuredResume(
        name="Jane Doe",
        headline="Staff Engineer",
        summary="Builds reliable systems.",
        emails=["jane@example.com"],
        phones=[],
        urls=["github.com/jane"],
        skills=["Python", "Kubernetes"],
        certifications=[],
        achievements=["Winner, ACM ICPC Regionals 2021"],
        work=[WorkExperience(company="Acme", title="Engineer",
                             start_date="2020", end_date="2024",
                             bullets=["Cut latency 40% by adding a Redis cache"])],
        education=[Education(institution="MIT", degree="BS", field="CS", end_year="2019")],
        projects=[Project(name="CredVault", url="https://x", tech=["Python", "Redis"],
                          bullets=["Built auth handling 2TB/day"])],
    )


def _text(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def test_docx_is_valid_and_contains_all_sections():
    data = build_docx(_resume())
    assert data[:2] == b"PK"  # docx is a zip
    text = _text(data)
    for expected in ["Jane Doe", "Staff Engineer", "Builds reliable systems.",
                     "Python, Kubernetes", "Acme", "Cut latency 40%",
                     "CredVault", "Built auth handling 2TB/day",
                     "Winner, ACM ICPC Regionals 2021", "MIT"]:
        assert expected in text, f"missing {expected!r}"


def test_section_order_is_honoured():
    data = build_docx(_resume(), section_order=["skills", "summary"])
    text = _text(data)
    assert text.index("Python, Kubernetes") < text.index("Builds reliable systems.")


def test_omitted_sections_still_render_after_ordered_ones():
    # order lists only skills; work/projects/education must not be dropped
    data = build_docx(_resume(), section_order=["skills"])
    text = _text(data)
    assert "CredVault" in text and "Acme" in text and "MIT" in text


def test_hidden_sections_are_dropped_entirely():
    # projects hidden via toggle: gone from output even though data exists and
    # even though the omitted-section fallback would otherwise re-add it.
    data = build_docx(_resume(), section_order=["summary", "skills"],
                      hidden_sections=["projects"])
    text = _text(data)
    assert "CredVault" not in text and "Built auth handling 2TB/day" not in text
    assert "Acme" in text  # non-hidden fallback section still present
