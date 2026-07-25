"""ats-screener engine adapter — deterministic, rule-based, no LLM/network."""

from rho.models.resume import Education, StructuredResume, WorkExperience


def _resume() -> StructuredResume:
    return StructuredResume(
        name="Jane Smith",
        headline="Frontend Web Developer",
        skills=["HTML", "CSS", "JavaScript", "Bootstrap", "WordPress", "jQuery"],
        work=[
            WorkExperience(
                company="Acme Web Studio",
                title="Frontend Web Developer",
                bullets=[
                    "Built responsive UI with Bootstrap for multiple screen sizes",
                    "Developed WordPress themes, increased traffic by 40%",
                ],
            )
        ],
        education=[Education(institution="State University", degree="BS Computer Science")],
    )


WEBDEV_JD = (
    "frontend web developer with strong HTML CSS JavaScript Bootstrap and "
    "WordPress experience. responsive design and typography skills required."
)
CHEF_JD = (
    "executive chef required, menu planning, food safety, kitchen management, "
    "culinary arts, sous vide, banquet catering experience."
)


def test_run_returns_six_profiles_and_match_score():
    from rho.ats.engines.ats_screener import ATSScreener

    out = ATSScreener().run(_resume(), WEBDEV_JD)

    assert out["engine"] == "ats_screener"
    assert out["parse_fields"] is None
    assert 0.0 <= out["match_score"] <= 100.0
    per_platform = out["raw"]["per_platform"]
    assert len(per_platform) == 6
    # match_score is the mean of the six platform scores
    assert out["match_score"] == sum(per_platform.values()) / 6


def test_score_discriminates_relevant_from_irrelevant_jd():
    from rho.ats.engines.ats_screener import ATSScreener

    engine = ATSScreener()
    assert engine.run(_resume(), WEBDEV_JD)["match_score"] > engine.run(_resume(), CHEF_JD)["match_score"]


def test_scoring_is_deterministic():
    from rho.ats.engines.ats_screener import ATSScreener

    engine = ATSScreener()
    assert engine.run(_resume(), WEBDEV_JD) == engine.run(_resume(), WEBDEV_JD)
