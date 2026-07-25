"""Build résumé×JD pairs from the local corpus.

Resume.csv (2484 résumés, 24 `Category` buckets) and training_data.csv (853 JDs
with free-text `position_title`) share no join key, so pairs are constructed.
We deliberately sample both same-domain and cross-domain pairs: calibration
needs a *spread* of match quality, not only good matches, or the fit sees no
contrast and Spearman ρ is uninformative.
"""

import random
import re

import pandas as pd

from rho.models.resume import Education, StructuredResume, WorkExperience

# Coarse résumé Category -> keywords appearing in JD position_title.
CATEGORY_TITLE_HINTS = {
    "INFORMATION-TECHNOLOGY": ["developer", "engineer", "it ", "systems", "software", "web"],
    "ENGINEERING": ["engineer", "mechanical", "electrical", "civil"],
    "SALES": ["sales", "account executive", "retail"],
    "ACCOUNTANT": ["account", "bookkeep", "audit", "financial analyst"],
    "FINANCE": ["financial", "finance", "analyst", "bank"],
    "HR": ["human resources", "recruit", "hr "],
    "HEALTHCARE": ["nurse", "medical", "health", "clinical"],
    "TEACHER": ["teacher", "instructor", "tutor", "professor"],
    "CHEF": ["chef", "cook", "culinary", "kitchen"],
    "DESIGNER": ["designer", "design", "graphic", "ux", "ui"],
    "ADVOCATE": ["advocate", "attorney", "legal", "lawyer", "paralegal", "counsel"],
    "AGRICULTURE": ["agricultur", "farm", "crop", "horticultur", "livestock"],
    "APPAREL": ["apparel", "fashion", "garment", "textile", "merchandis"],
    "ARTS": ["art", "creative", "photograph", "illustrat", "musician"],
    "AUTOMOBILE": ["automotive", "vehicle", "mechanic", "auto ", "technician"],
    "AVIATION": ["aviation", "pilot", "flight", "aircraft", "airline"],
    "BANKING": ["bank", "teller", "loan", "credit", "mortgage"],
    "BPO": ["call center", "customer service", "support agent", "representative"],
    "BUSINESS-DEVELOPMENT": ["business development", "partnership", "account executive"],
    "CONSTRUCTION": ["construction", "contractor", "foreman", "carpenter", "site manager"],
    "CONSULTANT": ["consultant", "consulting", "advisor", "strategist"],
    "DIGITAL-MEDIA": ["digital", "social media", "content", "seo", "marketing"],
    "FITNESS": ["fitness", "trainer", "coach", "gym", "wellness"],
    "PUBLIC-RELATIONS": ["public relations", "communications", "pr ", "media relations"],
}

_BULLET_SPLIT = re.compile(r"\s{2,}|\n")

# Section headers that introduce skill lists in this corpus's formatting.
_SKILL_HEADER = re.compile(
    r"^(skills?|highlights?|core competenc(y|ies)|technical skills?|areas of expertise)\b",
    re.I,
)
# Headers that end the skills section.
_OTHER_HEADER = re.compile(
    r"^(summary|experience|education|professional|work history|accomplishments|"
    r"certifications?|affiliations?|interests?|references?)\b",
    re.I,
)


def _extract_skills(lines: list[str], limit: int = 25) -> list[str]:
    """Pull skills from the résumé's own skills/highlights section.

    Length-based heuristics pick up section headers and job titles instead of
    skills, which flattens the feature vector across résumés and leaves the
    calibrator regressing on noise.
    """
    skills: list[str] = []
    in_section = False
    for ln in lines:
        if _SKILL_HEADER.match(ln):
            in_section = True
            continue
        if _OTHER_HEADER.match(ln):
            in_section = False
            continue
        if not in_section:
            continue
        # Skill sections are comma- or bullet-delimited runs of short phrases.
        for token in re.split(r"[,;•|]", ln):
            token = token.strip(" .\t-")
            if 2 <= len(token) <= 40 and not token.isdigit():
                skills.append(token)
    # Deduplicate, preserving order.
    seen, out = set(), []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out[:limit]


def _to_structured(resume_str: str, category: str) -> StructuredResume:
    """Cheap heuristic parse of the corpus text into StructuredResume.

    Phase 2's LLM extractor is the real path; for calibration we only need the
    fields ats-screener scores over, and a deterministic parse keeps the whole
    dataset build reproducible.
    """
    lines = [ln.strip() for ln in _BULLET_SPLIT.split(resume_str) if ln.strip()]
    headline = lines[0] if lines else category

    bullets = [ln for ln in lines if 40 <= len(ln) <= 300][:20]
    skills = _extract_skills(lines)

    edu_text = " ".join(
        ln for ln in lines if re.search(r"\b(BS|BA|MS|MBA|PhD|Bachelor|Master|University|College)\b", ln)
    )[:400]

    return StructuredResume(
        name="Candidate",
        headline=headline,
        summary=" ".join(lines[1:4])[:600] or None,
        skills=skills,
        work=[WorkExperience(company="Employer", title=headline, bullets=bullets)] if bullets else [],
        education=[Education(institution=edu_text)] if edu_text else [],
    )


def _matches_category(title: str, category: str) -> bool:
    hints = CATEGORY_TITLE_HINTS.get(category, [])
    low = str(title).lower()
    return any(h in low for h in hints)


def build_pairs(
    n_pairs: int = 200,
    resume_csv: str = "Resume.csv",
    jd_csv: str = "training_data.csv",
    # Weighted toward same-domain so literal coverage has real overlap to find;
    # the cross-domain remainder keeps a low-match tail in the target range.
    same_domain_ratio: float = 0.8,
    seed: int = 0,
) -> list[tuple[StructuredResume, str]]:
    rng = random.Random(seed)
    resumes = pd.read_csv(resume_csv)
    jds = pd.read_csv(jd_csv)

    # Drop length outliers: truncated stubs and multi-résumé blobs.
    resumes = resumes[resumes.Resume_str.str.len().between(800, 12000)]
    jds = jds[jds.job_description.str.len().between(400, 8000)]

    # Categories the JD corpus can actually serve; résumés outside these can
    # never be same-domain paired and would silently dilute the ratio.
    pairable = {
        c
        for c in resumes.Category.unique()
        if jds.position_title.apply(lambda t: _matches_category(t, c)).any()
    }
    same_pool = resumes[resumes.Category.isin(pairable)]

    pairs = []
    n_same = int(n_pairs * same_domain_ratio)
    for i in range(n_pairs):
        pool = same_pool if (i < n_same and len(same_pool)) else resumes
        r = pool.sample(1, random_state=seed + i).iloc[0]
        candidates = jds[jds.position_title.apply(lambda t: _matches_category(t, r.Category))]
        if i < n_same and len(candidates):
            jd = candidates.sample(1, random_state=seed + i).iloc[0]
        else:
            # Cross-domain: prefer a JD that does NOT match the résumé category.
            others = jds[~jds.position_title.apply(lambda t: _matches_category(t, r.Category))]
            pool = others if len(others) else jds
            jd = pool.sample(1, random_state=seed + i).iloc[0]
        pairs.append((_to_structured(r.Resume_str, r.Category), jd.job_description))

    rng.shuffle(pairs)
    return pairs
