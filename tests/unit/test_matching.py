from rho.matching import match
from rho.models.jd import RequirementSet, Requirement
from rho.models.resume import StructuredResume, WorkExperience
from rho.matching.coverage import keyword_coverage, fuzzy_coverage, resume_text_terms
from rho.matching.embed import Embedder


def test_semantic_similarity_high_for_synonyms():
    e = Embedder()
    v = e.encode(["machine learning model development", "AWS cloud platform"])
    sim = e.cosine(e.encode(["ML model building"])[0], v[0])
    assert sim > 0.4  # synonym-ish should beat unrelated


def test_keyword_and_fuzzy_coverage():
    reqs = ["Python", "Kubernetes", "AWS"]
    skills = ["python", "aws", "kubernets"]  # last is a typo
    assert keyword_coverage(reqs, skills) == 2 / 3  # Python, AWS exact; Kubernetes no
    assert fuzzy_coverage(reqs, skills) == 1.0  # typo caught by fuzzy


def test_keyword_coverage_credits_partial_phrase_overlap():
    """Whole-string containment never fires for phrasal requirements: a résumé
    saying "key account management" does not contain the literal string
    "account project management experience" even though the skill is present."""
    skills = ["key account management", "market planning", "SQL"]
    assert keyword_coverage(["account project management experience"], skills) > 0.5
    # An unrelated requirement still scores zero.
    assert keyword_coverage(["kubernetes cluster orchestration"], skills) == 0.0


def test_fuzzy_coverage_credits_partial_phrase_overlap():
    """Same whole-string problem as keyword_coverage: fuzz.ratio on a full
    phrase against a full skill never clears the threshold."""
    skills = ["key account management", "market planning"]
    assert fuzzy_coverage(["account project managment experience"], skills) > 0.5
    assert fuzzy_coverage(["kubernetes cluster orchestration"], skills) == 0.0


def test_fuzzy_coverage_still_catches_single_token_typos():
    """Phase 3 behaviour for skill tokens is unchanged."""
    assert fuzzy_coverage(["Python", "Kubernetes", "AWS"], ["python", "aws", "kubernets"]) == 1.0


def test_keyword_coverage_still_exact_for_single_tokens():
    """Phase 3 behaviour for skill tokens is unchanged."""
    reqs = ["Python", "Kubernetes", "AWS"]
    skills = ["python", "aws", "kubernets"]
    assert keyword_coverage(reqs, skills) == 2 / 3


def _ranked(lead: list[str], filler_count: int) -> list[str]:
    """`lead` followed by enough filler to push later entries into the tail band."""
    return lead + [f"unrelated skill {i}" for i in range(filler_count)]


def test_keyword_coverage_rewards_leading_the_requested_skills():
    """Reordering must move the score.

    The rewriter's only permitted skills edit is reordering, so when coverage
    flattened the résumé into an unordered set the optimise pass could not change
    any component: every before/after pair in the UI came back identical.
    """
    reqs = ["Docker", "Kubernetes"]
    buried = _ranked(["Java", "C++", "SQL", "Python", "JavaScript"], 12) + ["Docker", "Kubernetes"]
    led = _ranked(["Docker", "Kubernetes"], 0) + ["Java", "C++", "SQL", "Python", "JavaScript"]

    assert keyword_coverage(reqs, led) > keyword_coverage(reqs, buried)
    assert fuzzy_coverage(reqs, led) > fuzzy_coverage(reqs, buried)


def test_buried_skill_still_outscores_a_missing_one():
    """Position discounts evidence; it does not erase it.

    A résumé that lists a requirement late is still a better match than one that
    never claims it, so the tail weight stays well above zero.
    """
    reqs = ["Kubernetes"]
    buried = _ranked(["Java"], 20) + ["Kubernetes"]
    missing = _ranked(["Java"], 20)

    assert keyword_coverage(reqs, buried) > keyword_coverage(reqs, missing)
    assert keyword_coverage(reqs, missing) == 0.0


def test_keyword_coverage_ignores_stopwords_in_requirements():
    """"and"/"of" must not manufacture overlap against unrelated résumés."""
    assert keyword_coverage(["search and ecommerce experience"], ["welding and pipefitting"]) == 0.0


def test_coverage_searches_whole_resume_not_only_skills():
    """Evidence for a requirement often lives in a bullet, not the skills list."""
    resume = StructuredResume(
        name="A",
        skills=["Excel"],
        work=[
            WorkExperience(
                company="Acme",
                title="Engineer",
                bullets=["Built Python services deployed on AWS"],
            )
        ],
    )
    assert keyword_coverage(["Python", "AWS"], resume_text_terms(resume)) == 1.0


def test_coverage_matches_multiword_requirement_against_resume():
    """LLM requirements are phrases; substring-matching them must still work."""
    resume = StructuredResume(
        name="A",
        skills=["Bootstrap", "WordPress"],
        work=[
            WorkExperience(
                company="Acme",
                title="Developer",
                bullets=["Translated designs responsively for multiple screen sizes"],
            )
        ],
    )
    terms = resume_text_terms(resume)
    assert keyword_coverage(["responsively for multiple screen sizes"], terms) == 1.0
    assert keyword_coverage(["kubernetes cluster administration"], terms) == 0.0


def test_match_builds_vector_and_prov_gaps():
    resume = StructuredResume(
        name="A", skills=["Python", "AWS"], skills_prov=[["p:d:1"], ["p:d:2"]]
    )
    reqs = RequirementSet(
        requirements=[
            Requirement(text="Python", kind="skill", priority="must"),
            Requirement(text="Kubernetes", kind="skill", priority="must"),
        ]
    )
    mr = match(resume, reqs)
    assert mr.predicted_score == 0.0
    assert 0.0 <= mr.component_vector.must_have_coverage <= 1.0
    py_gap = next(g for g in mr.gaps if g.requirement.text == "Python")
    assert py_gap.status == "present"
    assert py_gap.evidence_prov == ["p:d:1"]  # provenance chain preserved
    k8s_gap = next(g for g in mr.gaps if g.requirement.text == "Kubernetes")
    assert k8s_gap.status == "absent"
    assert k8s_gap.evidence_prov == []


def test_semantic_similarity_is_mean_cosine_not_match_rate():
    """semantic_similarity must reflect actual embedding similarity, not gap counts."""
    resume = StructuredResume(
        name="A", skills=["Python", "AWS", "Docker"], skills_prov=[["p:d:1"], ["p:d:2"], ["p:d:3"]]
    )
    reqs = RequirementSet(
        requirements=[
            Requirement(text="Python", kind="skill", priority="must"),
            Requirement(text="Kubernetes", kind="skill", priority="must"),
            Requirement(text="containerization", kind="skill", priority="nice"),
        ]
    )
    mr = match(resume, reqs)
    sem = mr.component_vector.semantic_similarity
    # all three reqs score present/weak, so the old match-rate formula returned exactly 1.0
    assert sem < 1.0, "semantic_similarity is still the present/weak match-rate"
    # exact-match req ("Python") pins the mean well above zero
    assert sem > 0.3


def test_semantic_similarity_zero_when_no_skills():
    resume = StructuredResume(name="A")
    reqs = RequirementSet(
        requirements=[Requirement(text="Python", kind="skill", priority="must")]
    )
    assert match(resume, reqs).component_vector.semantic_similarity == 0.0


def test_extract_jd_terms_pulls_keyphrases():
    from rho.matching.coverage import extract_jd_terms

    jd = (
        "We are looking for a backend engineer with strong Python experience. "
        "You will build REST APIs and deploy to Kubernetes clusters on AWS."
    )
    terms = extract_jd_terms(jd, top_n=10)
    assert terms, "no terms extracted"
    blob = " ".join(terms).lower()
    assert "python" in blob
    assert "kubernetes" in blob


def test_extract_jd_terms_empty_text():
    from rho.matching.coverage import extract_jd_terms

    assert extract_jd_terms("") == []


def test_semantic_thresholds_come_from_settings(monkeypatch):
    """Thresholds must be tunable (P7 sweep), not hardcoded in the matcher."""
    from rho.config import settings

    resume = StructuredResume(name="A", skills=["Docker"], skills_prov=[["p:d:1"]])
    reqs = RequirementSet(
        requirements=[Requirement(text="Kubernetes", kind="skill", priority="must")]
    )
    # Docker~Kubernetes sits in the weak band under defaults
    assert match(resume, reqs).gaps[0].status == "weak"
    # raising sem_lo above that cosine must demote it to absent
    monkeypatch.setattr(settings, "sem_lo", 0.99)
    monkeypatch.setattr(settings, "sem_hi", 0.995)
    assert match(resume, reqs).gaps[0].status == "absent"
