"""Hard-content token extraction — what the verification gate is allowed to check.

A "hard-content token" is a factual claim the résumé makes: a skill, a tool, an
employer, a title, a certification, a date, an institution. Prose (summary,
headline, bullet phrasing) is deliberately excluded: the rewriter is *supposed*
to rephrase those, and only the facts embedded in them are checkable. Bullets
are handled by the verifier as whole-string claims, not tokenised here.
"""

from rho.models.resume import StructuredResume

HardToken = tuple[str, str]  # (value, field_path)


def hard_content_tokens(resume: StructuredResume) -> list[HardToken]:
    """Every checkable factual claim in `resume`, paired with its field path."""
    toks: list[HardToken] = []
    for i, s in enumerate(resume.skills):
        toks.append((s, f"skills[{i}]"))
    for i, c in enumerate(resume.certifications):
        toks.append((c, f"certifications[{i}]"))
    for i, a in enumerate(resume.achievements):
        toks.append((a, f"achievements[{i}]"))
    for wi, w in enumerate(resume.work):
        toks.append((w.company, f"work[{wi}].company"))
        toks.append((w.title, f"work[{wi}].title"))
        for d in (w.start_date, w.end_date):
            if d:
                toks.append((d, f"work[{wi}].date"))
    for ei, e in enumerate(resume.education):
        toks.append((e.institution, f"education[{ei}].institution"))
        if e.end_year:
            toks.append((e.end_year, f"education[{ei}].end_year"))
    for pi, p in enumerate(resume.projects):
        toks.append((p.name, f"projects[{pi}].name"))
        for ti, t in enumerate(p.tech):
            toks.append((t, f"projects[{pi}].tech[{ti}]"))
    return [(v, p) for (v, p) in toks if v and v.strip()]
