"""ats-screener adapter: 6 enterprise ATS profiles, rule-based, deterministic.

Drives the vendored scoring rules (`vendor/ats_screener/`) through a headless
Node runner. No LLM and no network, so scores are reproducible from the pinned
upstream commit recorded in `vendor/ats_screener/COMMIT.txt`.
"""

import json
import subprocess
from pathlib import Path

from rho.models.resume import StructuredResume

_RUNNER_DIR = Path(__file__).resolve().parents[4] / "vendor" / "ats_screener"
_RUNNER = _RUNNER_DIR / "run_scorer.mjs"
_TIMEOUT_SECONDS = 60


class ATSScreenerError(RuntimeError):
    """The Node runner failed; caller should exclude this doc from the fit."""


def _to_scoring_input(resume: StructuredResume, jd_text: str) -> dict:
    """Map a StructuredResume onto ats-screener's `ScoringInput`."""
    bullets = [b for w in resume.work for b in w.bullets]

    sections = []
    if resume.summary or resume.headline:
        sections.append("summary")
    if resume.work:
        sections.append("experience")
    if resume.education:
        sections.append("education")
    if resume.skills:
        sections.append("skills")
    if resume.certifications:
        sections.append("certifications")
    if resume.achievements:
        sections.append("achievements")

    education_text = " ".join(
        " ".join(filter(None, [e.degree, e.field, e.institution, e.end_year]))
        for e in resume.education
    )

    text_parts = [resume.name, resume.headline or "", resume.summary or ""]
    text_parts += [f"{w.title} {w.company}" for w in resume.work]
    text_parts += bullets + resume.skills + [education_text]
    resume_text = "\n".join(p for p in text_parts if p)

    return {
        "resumeText": resume_text,
        "resumeSkills": resume.skills,
        "resumeSections": sections,
        "experienceBullets": bullets,
        "educationText": education_text,
        # rho ingests to markdown, so layout quirks are already normalised away.
        "hasMultipleColumns": False,
        "hasTables": False,
        "hasImages": False,
        "pageCount": max(1, round(len(resume_text) / 3000)),
        "wordCount": len(resume_text.split()),
        "jobDescription": jd_text,
    }


class ATSScreener:
    name = "ats_screener"

    def run(self, resume: StructuredResume, jd_text: str) -> dict:
        payload = json.dumps(_to_scoring_input(resume, jd_text))
        try:
            proc = subprocess.run(
                ["node", "--experimental-strip-types", str(_RUNNER)],
                input=payload,
                capture_output=True,
                text=True,
                cwd=_RUNNER_DIR,
                timeout=_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ATSScreenerError(f"ats-screener runner failed: {exc}") from exc

        if proc.returncode != 0:
            raise ATSScreenerError(f"ats-screener runner exited {proc.returncode}: {proc.stderr[:500]}")

        results = json.loads(proc.stdout)
        per_platform = {r["system"]: float(r["overallScore"]) for r in results}

        return {
            "engine": self.name,
            "parse_fields": None,
            "match_score": sum(per_platform.values()) / len(per_platform),
            "raw": {
                "per_platform": per_platform,
                "passes_filter": {r["system"]: bool(r["passesFilter"]) for r in results},
                "breakdown": {
                    r["system"]: {k: v["score"] for k, v in r["breakdown"].items()} for r in results
                },
            },
        }
