"""Synthetic gold-résumé generator (C1 Table 1a).

Why synthetic at all: the public résumé-NER sets named in the research report are
not reachable from this host, and the local corpus (`Resume.csv`) ships raw text
with a `Category` label and **no field annotations** — nothing to score against.
Hand-labelling is done too (`eval/datasets/real/`, Table 1b), but only at a size
one annotator can reach.

What synthesis buys, beyond volume: the gold **provenance spans are exact by
construction**. The generator writes each value at a known character offset and
records it, so `provenance_accuracy` compares against ground truth rather than
against another heuristic's guess. Hand-labelling spans on real résumés cannot
reach that standard.

What it costs, stated plainly and repeated in RESULTS.md: templated résumés are
cleaner than real ones — consistent section headers, no multi-column layout, no
OCR noise. **Table 1a is an upper bound on extraction quality, not an estimate of
field performance.** Table 1b is the reality check, and the gap between them is
itself a reportable finding.

Determinism: every document is a pure function of `seed` and its index, so the
whole dataset rebuilds byte-identically.
"""

import io
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "gold"

FIRST = [
    "Dana", "Rowan", "Priya", "Malik", "Ingrid", "Tomas", "Aiko", "Nadia",
    "Ellis", "Omar", "Beatriz", "Soren", "Leilani", "Viktor", "Amara", "Idris",
]
LAST = [
    "Whitfield", "Okonkwo", "Lindqvist", "Ramachandran", "Delgado", "Novak",
    "Fairbanks", "Castellanos", "Hyun", "Aubert", "Mbeki", "Kowalski",
]
COMPANIES = [
    "Northwind Systems", "Brightline Analytics", "Cobalt Interactive",
    "Harborview Logistics", "Meridian Health Group", "Ironwood Robotics",
    "Silverpine Media", "Trellis Financial", "Copperfield Retail",
    "Blue Harbor Energy", "Vantage Point Consulting", "Lantern Software",
]
INSTITUTIONS = [
    "State University", "Riverside Institute of Technology", "Northgate College",
    "Lakeshore University", "Ashford Polytechnic", "Fairmont State College",
]
DEGREES = [
    ("B.S.", "Computer Science"), ("B.A.", "Economics"), ("M.S.", "Data Science"),
    ("B.S.", "Mechanical Engineering"), ("M.B.A.", "Business Administration"),
    ("B.S.", "Information Systems"),
]

# Role families: (title, skills pool, bullet templates). Bullets deliberately
# carry numbers and tool names — the hard-content tokens the C3 gate checks —
# so the same documents exercise provenance end to end.
ROLES = [
    (
        "Data Engineer",
        ["Python", "SQL", "Airflow", "Spark", "dbt", "Snowflake", "Kafka", "AWS"],
        [
            "Built streaming ingestion pipelines processing {n} TB of event data daily.",
            "Migrated {n} legacy ETL jobs from cron to Airflow, cutting failures by {p}%.",
            "Modelled the warehouse layer in dbt across {n} source systems.",
        ],
    ),
    (
        "Backend Engineer",
        ["Java", "Go", "PostgreSQL", "Redis", "Kubernetes", "gRPC", "Docker", "Kafka"],
        [
            "Designed a gRPC service handling {n} thousand requests per second.",
            "Reduced p99 latency by {p}% by introducing a Redis cache layer.",
            "Led migration of {n} services onto Kubernetes.",
        ],
    ),
    (
        "Financial Analyst",
        ["Excel", "SQL", "Tableau", "SAP", "Forecasting", "Variance Analysis"],
        [
            "Owned the quarterly forecast for a {n} million dollar product line.",
            "Automated variance reporting in SQL, saving {n} analyst hours per month.",
            "Presented monthly results to a committee of {n} executives.",
        ],
    ),
    (
        "Registered Nurse",
        ["Patient Care", "Triage", "EPIC", "Phlebotomy", "IV Therapy", "ACLS"],
        [
            "Managed a {n} bed unit across rotating night shifts.",
            "Trained {n} new graduate nurses on EPIC charting workflows.",
            "Maintained medication accuracy above {p}% across audits.",
        ],
    ),
    (
        "Marketing Manager",
        ["SEO", "Google Analytics", "HubSpot", "Copywriting", "A/B Testing", "Salesforce"],
        [
            "Grew organic traffic {p}% year over year through an SEO content programme.",
            "Ran {n} A/B tests on lifecycle email, lifting conversion {p}%.",
            "Managed a {n} million dollar annual paid media budget.",
        ],
    ),
    (
        "Mechanical Engineer",
        ["SolidWorks", "AutoCAD", "FEA", "GD&T", "MATLAB", "Six Sigma"],
        [
            "Designed {n} injection-moulded assemblies released to production.",
            "Cut part cost {p}% through a design-for-manufacture review.",
            "Ran FEA validation on {n} structural components.",
        ],
    ),
]

SUMMARIES = [
    "{title} with {yrs} years of experience delivering production systems in "
    "regulated environments. Focused on reliability, measurement, and clear handover.",
    "Hands-on {title} who has spent {yrs} years shipping and maintaining systems "
    "used daily by large teams. Comfortable owning work end to end.",
    "{title} with {yrs} years across startups and enterprise. Known for turning "
    "vague requirements into scoped, measurable delivery.",
]

# Header wording varies per document: an extractor that keys on one literal
# spelling of "SKILLS" would otherwise score far higher here than in reality.
SKILL_HEADERS = ["SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES", "SKILL HIGHLIGHTS"]
WORK_HEADERS = ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT HISTORY"]
EDU_HEADERS = ["EDUCATION", "EDUCATION AND TRAINING", "ACADEMIC BACKGROUND"]
SUMMARY_HEADERS = ["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE"]


class _Doc:
    """Accumulates lines while recording the char span of tagged values.

    Offsets index the plain-text rendering, which is exactly what `ingest_text`
    consumes, so a `.txt` gold document's recorded spans line up with the spans
    the real ingest path produces.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.spans: dict[str, tuple[int, int]] = {}
        self._pos = 0

    def add(self, text: str, path: str | None = None, value: str | None = None) -> None:
        """Append `text`; if `path` given, record the span of `value` within it."""
        start = self._pos
        if path is not None:
            target = value if value is not None else text
            off = text.find(target)
            # A value must be locatable in its own line, or the gold span would
            # be a guess — fail loudly rather than record a wrong offset.
            if off < 0:
                raise ValueError(f"value {target!r} not found in line {text!r}")
            self.spans[path] = (start + off, start + off + len(target))
        self.lines.append(text)
        self._pos += len(text) + 1  # +1 for the newline join

    def text(self) -> str:
        return "\n".join(self.lines)


def _make_record(idx: int, seed: int) -> dict:
    """One synthetic résumé: document text, gold fields, and gold prov spans."""
    rng = random.Random(f"{seed}:{idx}")
    title, skill_pool, bullet_templates = rng.choice(ROLES)
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    skills = rng.sample(skill_pool, k=rng.randint(4, min(6, len(skill_pool))))
    yrs = rng.randint(3, 18)
    summary = rng.choice(SUMMARIES).format(title=title, yrs=yrs)

    n_jobs = rng.randint(1, 3)
    end = rng.randint(2021, 2025)
    jobs = []
    for _ in range(n_jobs):
        span_years = rng.randint(1, 4)
        start = end - span_years
        jobs.append(
            {
                "company": rng.choice(COMPANIES),
                "title": title if not jobs else f"{rng.choice(['Junior', 'Associate'])} {title}",
                "start_date": str(start),
                "end_date": str(end),
                "bullets": [
                    t.format(n=rng.randint(2, 60), p=rng.randint(10, 45))
                    for t in rng.sample(bullet_templates, k=rng.randint(2, len(bullet_templates)))
                ],
            }
        )
        end = start - rng.randint(0, 1)

    degree, field = rng.choice(DEGREES)
    edu = {
        "institution": rng.choice(INSTITUTIONS),
        "degree": degree,
        "field": field,
        "end_year": str(jobs[-1]["start_date"]),
    }

    doc = _Doc()
    doc.add(name, "name")
    doc.add(title)
    doc.add(f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com")
    doc.add("")

    doc.add(rng.choice(SUMMARY_HEADERS))
    doc.add(summary)
    doc.add("")

    doc.add(rng.choice(SKILL_HEADERS))
    # Skills on one delimited line, the dominant real-résumé layout. Each skill
    # gets its own gold span inside the shared line.
    skill_line = ", ".join(skills)
    start = doc._pos
    for i, s in enumerate(skills):
        off = skill_line.find(s)
        doc.spans[f"skills[{i}]"] = (start + off, start + off + len(s))
    doc.add(skill_line)
    doc.add("")

    doc.add(rng.choice(WORK_HEADERS))
    for wi, job in enumerate(jobs):
        header = f"{job['company']} — {job['title']} ({job['start_date']}–{job['end_date']})"
        doc.add(header, f"work[{wi}].company", job["company"])
        # Title and dates live on the same line; record them against that line too.
        line_start = doc._pos - len(header) - 1
        toff = header.find(job["title"])
        doc.spans[f"work[{wi}].title"] = (line_start + toff, line_start + toff + len(job["title"]))
        doff = header.find(job["start_date"])
        doc.spans[f"work[{wi}].date"] = (line_start + doff, line_start + doff + len(job["start_date"]))
        for bi, b in enumerate(job["bullets"]):
            doc.add(f"- {b}", f"work[{wi}].bullets[{bi}]", b)
        doc.add("")

    doc.add(rng.choice(EDU_HEADERS))
    edu_line = f"{edu['institution']} — {edu['degree']} {edu['field']}, {edu['end_year']}"
    doc.add(edu_line, "education[0].institution", edu["institution"])

    # The text each gold path should trace back to. Offsets alone are not
    # portable: Docling re-exports .docx to markdown with slightly different
    # spacing, so a source-text offset drifts by the time the pipeline sees the
    # document. The loader re-resolves these values against the *ingested* text,
    # which keeps one gold format valid for both .txt and .docx.
    gold_values = {
        "name": name,
        **{f"skills[{i}]": s for i, s in enumerate(skills)},
    }
    for wi, job in enumerate(jobs):
        gold_values[f"work[{wi}].company"] = job["company"]
        gold_values[f"work[{wi}].title"] = job["title"]
        gold_values[f"work[{wi}].date"] = job["start_date"]
        for bi, b in enumerate(job["bullets"]):
            gold_values[f"work[{wi}].bullets[{bi}]"] = b
    gold_values["education[0].institution"] = edu["institution"]

    return {
        "id": f"g{idx:03d}",
        "text": doc.text(),
        "gold_values": gold_values,
        "gold": {
            "name": name,
            "headline": title,
            "summary": summary,
            "skills": skills,
            "work": [
                {
                    "company": j["company"],
                    "title": j["title"],
                    "start_date": j["start_date"],
                    "end_date": j["end_date"],
                    "bullets": j["bullets"],
                }
                for j in jobs
            ],
            "education": [edu],
        },
        "gold_prov": {k: list(v) for k, v in doc.spans.items()},
    }


def _write_docx(text: str, path: Path) -> None:
    """Render `text` as a one-paragraph-per-line .docx."""
    import docx

    d = docx.Document()
    for line in text.split("\n"):
        d.add_paragraph(line)
    buf = io.BytesIO()
    d.save(buf)
    path.write_bytes(buf.getvalue())


def build(n: int = 120, seed: int = 0, out_dir: Path = DATA_DIR, docx_every: int = 4) -> list[dict]:
    """Generate `n` gold résumés into `out_dir`.

    Every `docx_every`-th document is written as .docx (exercising the Docling
    path) and the rest as .txt (the text adapter). Both produce per-line
    provenance spans; see the module docstring in `eval/datasets/__init__.py`
    for why PDF is not used for the gold set.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        rec = _make_record(i, seed)
        as_docx = (i % docx_every) == (docx_every - 1)
        doc_path = out_dir / f"{rec['id']}.{'docx' if as_docx else 'txt'}"
        if as_docx:
            _write_docx(rec["text"], doc_path)
        else:
            doc_path.write_text(rec["text"], encoding="utf-8")
        (out_dir / f"{rec['id']}.json").write_text(
            json.dumps(
                {
                    "gold": rec["gold"],
                    # Source-text offsets: exact for .txt, and kept for .docx as a
                    # record of where the generator wrote each value.
                    "gold_prov": rec["gold_prov"],
                    # The portable form the loader actually scores against.
                    "gold_prov_values": rec["gold_values"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.append({"id": rec["id"], "file": doc_path.name})
    (out_dir / "manifest.json").write_text(
        json.dumps({"seed": seed, "n": n, "items": manifest}, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build(n=args.n, seed=args.seed)
    print(f"wrote {len(m)} gold résumés to {DATA_DIR}")
