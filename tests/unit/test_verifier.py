"""Phase 5 (C3): hard-content tokens + the provenance verification gate."""

from rho.models.provenance import ProvenanceMap, SourceSpan
from rho.models.resume import Education, Project, StructuredResume, WorkExperience
from rho.models.rewrite import TailoredResume
from rho.rewrite import rewrite
from rho.rewrite.tokens import hard_content_tokens
from rho.rewrite.verifier import verify_against_source


def _prov(*texts: str) -> ProvenanceMap:
    """A ProvenanceMap whose spans support exactly `texts` (default: "Python")."""
    pm = ProvenanceMap(doc_id="d")
    cursor = 0
    for t in texts or ("Python",):
        pm.add(
            SourceSpan(
                doc_id="d", char_start=cursor, char_end=cursor + len(t), raw_text=t
            )
        )
        cursor += len(t) + 1
    return pm


def test_hard_tokens_cover_skills_and_work():
    r = StructuredResume(
        name="A",
        skills=["Python", "AWS"],
        certifications=["AWS SAA"],
        work=[
            WorkExperience(
                company="Acme", title="Engineer", start_date="2019", end_date="2022"
            )
        ],
    )
    toks = hard_content_tokens(r)
    values = {t[0] for t in toks}
    assert {"Python", "AWS", "AWS SAA", "Acme", "Engineer"} <= values


def test_hard_tokens_cover_education_and_dates():
    r = StructuredResume(
        name="A",
        work=[
            WorkExperience(
                company="Acme", title="Engineer", start_date="2019", end_date="2022"
            )
        ],
        education=[Education(institution="MIT")],
    )
    values = {t[0] for t in hard_content_tokens(r)}
    assert {"2019", "2022", "MIT"} <= values


def test_hard_tokens_cover_project_name_and_tech():
    r = StructuredResume(
        name="A",
        projects=[Project(name="CredVault", tech=["Python", "Redis"], bullets=["Built X"])],
    )
    paths = {t[0]: t[1] for t in hard_content_tokens(r)}
    assert paths["CredVault"] == "projects[0].name"
    assert paths["Python"] == "projects[0].tech[0]"
    assert paths["Redis"] == "projects[0].tech[1]"
    # bullets are prose, checked by the bullet path, not tokenised here
    assert "Built X" not in paths


def test_verify_keeps_sourced_project_drops_fabricated():
    source = StructuredResume(
        name="Dana Whitfield",
        projects=[Project(name="CredVault", tech=["Python"], bullets=["Built auth service"])],
    )
    tailored = StructuredResume(
        name="Dana Whitfield",
        projects=[
            Project(name="CredVault", tech=["Python", "Kubernetes"], bullets=["Built auth service"]),
            Project(name="Falconry Ledger", tech=["Go"], bullets=["Invented project"]),
        ],
    )
    prov = _prov("Dana Whitfield", "CredVault", "Python", "Built auth service")
    fixed, report = verify_against_source(tailored, source, prov)
    assert [p.name for p in fixed.projects] == ["CredVault"]  # fabricated project dropped
    assert fixed.projects[0].tech == ["Python"]  # invented tech reverted
    rejected = {r.added_text for r in report.rejected_edits}
    assert "Kubernetes" in rejected and "Falconry Ledger" in rejected


def test_verify_keeps_rephrased_project_bullet():
    source = StructuredResume(
        name="Dana Whitfield",
        projects=[Project(name="CredVault", tech=["Python"], bullets=["Built the authentication service in Python"])],
    )
    tailored = StructuredResume(
        name="Dana Whitfield",
        projects=[Project(name="CredVault", tech=["Python"], bullets=["Built authentication service using Python"])],
    )
    prov = _prov("Dana Whitfield", "CredVault", "Python", "Built the authentication service in Python")
    fixed, report = verify_against_source(tailored, source, prov)
    # a genuine rephrasing of a source project bullet survives the gate
    assert len(fixed.projects[0].bullets) == 1


def test_hard_tokens_skip_blank_values():
    r = StructuredResume(name="A", skills=["Python", "", "   "])
    values = [t[0] for t in hard_content_tokens(r)]
    assert values == ["Python"]


def test_hard_tokens_field_paths_are_addressable():
    r = StructuredResume(
        name="A",
        skills=["Python"],
        work=[WorkExperience(company="Acme", title="Engineer")],
    )
    paths = {t[0]: t[1] for t in hard_content_tokens(r)}
    assert paths["Python"] == "skills[0]"
    assert paths["Acme"] == "work[0].company"
    assert paths["Engineer"] == "work[0].title"


# --- the gate: skills ---------------------------------------------------


def test_verify_rejects_unsupported_addition():
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python", "Kubernetes"])
    fixed, report = verify_against_source(tailored, source, _prov())
    assert "Kubernetes" not in fixed.skills  # reverted
    assert report.total_edits == 1
    assert report.verified_edits == 0
    assert report.fabrication_rate == 1.0
    assert report.rejected_edits[0].added_text == "Kubernetes"


def test_verify_keeps_supported_addition():
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python", "FastAPI"])
    fixed, report = verify_against_source(tailored, source, _prov("Python", "FastAPI"))
    assert "FastAPI" in fixed.skills
    assert report.verified_edits == 1
    assert report.fabrication_rate == 0.0


def test_verify_counts_no_edits_when_unchanged():
    """Reordering/reselecting existing values is not an edit to verify."""
    source = StructuredResume(name="A", skills=["Python", "SQL"])
    tailored = StructuredResume(name="A", skills=["SQL", "Python"])
    fixed, report = verify_against_source(tailored, source, _prov())
    assert fixed.skills == ["SQL", "Python"]
    assert report.total_edits == 0
    assert report.fabrication_rate == 0.0


# --- the gate: certifications ------------------------------------------


def test_verify_rejects_unsupported_certification():
    source = StructuredResume(name="A", certifications=["AWS SAA"])
    tailored = StructuredResume(name="A", certifications=["AWS SAA", "CISSP"])
    fixed, report = verify_against_source(tailored, source, _prov("AWS SAA"))
    assert fixed.certifications == ["AWS SAA"]
    assert report.rejected_edits[0].added_text == "CISSP"
    assert report.fabrication_rate == 1.0


def test_verify_keeps_supported_certification():
    source = StructuredResume(name="A", certifications=[])
    tailored = StructuredResume(name="A", certifications=["AWS SAA"])
    fixed, report = verify_against_source(tailored, source, _prov("AWS SAA"))
    assert fixed.certifications == ["AWS SAA"]
    assert report.verified_edits == 1


# --- the gate: achievements --------------------------------------------


def test_verify_rejects_unsupported_achievement():
    source = StructuredResume(name="A", achievements=["Dean's List 2020"])
    tailored = StructuredResume(name="A", achievements=["Dean's List 2020", "Nobel Prize"])
    fixed, report = verify_against_source(tailored, source, _prov("Dean's List 2020"))
    assert fixed.achievements == ["Dean's List 2020"]
    assert report.rejected_edits[0].added_text == "Nobel Prize"


def test_verify_keeps_supported_achievement():
    source = StructuredResume(name="A", achievements=[])
    tailored = StructuredResume(name="A", achievements=["Dean's List 2020"])
    fixed, report = verify_against_source(tailored, source, _prov("Dean's List 2020"))
    assert fixed.achievements == ["Dean's List 2020"]
    assert report.verified_edits == 1


# --- the gate: work bullets --------------------------------------------


def test_verify_rejects_unsupported_bullet():
    source = StructuredResume(
        name="A",
        work=[WorkExperience(company="Acme", title="Engineer", bullets=["Built APIs"])],
    )
    tailored = StructuredResume(
        name="A",
        work=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                bullets=["Built APIs", "Led a team of 40 engineers"],
            )
        ],
    )
    fixed, report = verify_against_source(
        tailored, source, _prov("Acme", "Engineer", "Built APIs")
    )
    assert fixed.work[0].bullets == ["Built APIs"]
    assert report.rejected_edits[0].added_text == "Led a team of 40 engineers"


def test_verify_keeps_rephrased_bullet_with_provenance():
    """The rewriter may rephrase a bullet as long as the source supports it."""
    source = StructuredResume(
        name="A",
        work=[
            WorkExperience(
                company="Acme", title="Engineer", bullets=["Built REST APIs in Python"]
            )
        ],
    )
    tailored = StructuredResume(
        name="A",
        work=[
            WorkExperience(
                company="Acme", title="Engineer", bullets=["Built REST APIs in Python"]
            )
        ],
    )
    fixed, report = verify_against_source(
        tailored, source, _prov("Acme", "Engineer", "Built REST APIs in Python")
    )
    assert fixed.work[0].bullets == ["Built REST APIs in Python"]
    assert report.total_edits == 0


def test_verify_rejects_fabricated_employer():
    source = StructuredResume(
        name="A", work=[WorkExperience(company="Acme", title="Engineer")]
    )
    tailored = StructuredResume(
        name="A",
        work=[
            WorkExperience(company="Acme", title="Engineer"),
            WorkExperience(company="Google", title="Staff Engineer"),
        ],
    )
    fixed, report = verify_against_source(tailored, source, _prov("Acme", "Engineer"))
    assert [w.company for w in fixed.work] == ["Acme"]
    assert {r.added_text for r in report.rejected_edits} == {"Google", "Staff Engineer"}


def test_verify_preserves_provenance_on_kept_values():
    """C1: surviving values keep their prov chain."""
    source = StructuredResume(name="A", skills=["Python"], skills_prov=[["p:d:0"]])
    tailored = StructuredResume(name="A", skills=["Python"], skills_prov=[["p:d:0"]])
    fixed, _ = verify_against_source(tailored, source, _prov())
    assert fixed.skills_prov == [["p:d:0"]]


def test_verify_rejects_seniority_inflation():
    """A span reading "Engineer" must not license the promotion "Staff Engineer"."""
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python", "Senior Python Developer"])
    fixed, report = verify_against_source(tailored, source, _prov("Python"))
    assert fixed.skills == ["Python"]
    assert report.rejected_edits[0].added_text == "Senior Python Developer"


def test_verify_keeps_addition_whose_every_word_is_sourced():
    source = StructuredResume(name="A", skills=[])
    tailored = StructuredResume(name="A", skills=["Senior Python Developer"])
    fixed, report = verify_against_source(
        tailored, source, _prov("Senior Python Developer at Acme")
    )
    assert fixed.skills == ["Senior Python Developer"]
    assert report.verified_edits == 1


def test_verify_reports_rejection_reason():
    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python", "Kubernetes"])
    _, report = verify_against_source(tailored, source, _prov())
    assert report.rejected_edits[0].reason == "no supporting prov_id"


# --- rewrite() orchestration: generate -> gate --------------------------


def test_rewrite_gate_strips_fabrication():
    source = StructuredResume(name="A", skills=["Python"])

    def fake(r, g):
        return StructuredResume(name="A", skills=["Python", "GoLang"])  # invented

    tr = rewrite(source, [], _prov(), _rewrite_fn=fake)
    assert "GoLang" not in tr.resume.skills
    assert tr.fabrication_report.fabrication_rate == 1.0


def test_rewrite_returns_tailored_resume_with_report():
    source = StructuredResume(name="A", skills=["Python"])
    tr = rewrite(source, [], _prov(), _rewrite_fn=lambda r, g: r)
    assert isinstance(tr, TailoredResume)
    assert tr.resume.skills == ["Python"]
    assert tr.fabrication_report.total_edits == 0


def test_rewrite_keeps_truthful_tailoring():
    """The gate must not punish a rewrite that only surfaces sourced content."""
    source = StructuredResume(name="A", skills=["Python"])

    def fake(r, g):
        return StructuredResume(name="A", skills=["Python", "FastAPI"])

    tr = rewrite(source, [], _prov("Python", "FastAPI"), _rewrite_fn=fake)
    assert "FastAPI" in tr.resume.skills
    assert tr.fabrication_report.fabrication_rate == 0.0


# --- ablation accounting ------------------------------------------------


def test_unsourced_count_is_zero_after_gating():
    """The gate-ON metric must re-measure the gated résumé, not assume success.

    The plan's sketch computed `(total-verified) - len(rejected)`, which is
    identically 0 by arithmetic and so would report success even if the gate
    leaked. This measures the output instead.
    """
    from eval.fabrication_ablation import unsourced_count

    source = StructuredResume(name="A", skills=["Python"])
    tailored = StructuredResume(name="A", skills=["Python", "Kubernetes", "Rust"])
    prov = _prov()

    assert unsourced_count(tailored, source, prov) == 2  # gate OFF ships both
    fixed, _ = verify_against_source(tailored, source, prov)
    assert unsourced_count(fixed, source, prov) == 0  # gate ON ships neither


# --- bullet rephrasing vs invention (corpus-driven regression) -----------

_SRC_BULLET = (
    "Worked on optimizing and tuning the Teradata and Oracle views and SQL's "
    "to improve the performance of batch and response times"
)


def _bullet_case(tailored_bullet: str):
    source = StructuredResume(
        name="A",
        work=[WorkExperience(company="Acme", title="Engineer", bullets=[_SRC_BULLET])],
    )
    tailored = StructuredResume(
        name="A",
        work=[
            WorkExperience(company="Acme", title="Engineer", bullets=[tailored_bullet])
        ],
    )
    return verify_against_source(
        tailored, source, _prov("Acme", "Engineer", _SRC_BULLET)
    )


def test_gate_accepts_genuine_rephrasing_of_a_source_bullet():
    """Rewriting a bullet is the rewriter's job; only new claims are fabrication."""
    fixed, report = _bullet_case(
        "Optimized and tuned Teradata and Oracle views and SQL queries to "
        "enhance batch processing performance and user data response times."
    )
    assert fixed.work[0].bullets != []  # not a fabrication
    assert report.rejected_edits == []


def test_gate_still_rejects_invention_reusing_source_vocabulary():
    """Sharing words with the source must not launder a new factual claim."""
    fixed, report = _bullet_case(
        "Led a team of 40 Teradata and Oracle engineers across three continents."
    )
    assert fixed.work[0].bullets == []
    assert report.rejected_edits[0].added_text.startswith("Led a team of 40")
