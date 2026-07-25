"""Fetch the public human-annotated résumé-NER gold set (C1 Table 1c).

Source: `kens1ang/resume-ner-labelled` on the Hugging Face Hub — 2,260 real
résumés (the LiveCareer-style corpus the research report §1.4 points at) with
**human entity annotations carrying character offsets**:

    Skills · Job title · Institution Name · Degree · Certifications

Why this set matters more than the other two gold sources:

  * The annotations are **independent** — not written by the system under test
    and not by the agent implementing it, which is the weakness of the
    hand-labelled subset in `eval/datasets/real/`.
  * The offsets are exact (verified: every span slices to a clean entity), so
    they double as **gold provenance spans** — the thing `provenance_accuracy`
    needs and that no other available corpus supplies.

Known limitations, carried into RESULTS.md rather than smoothed over:

  * `Skills` is annotated exhaustively including every repeat mention (median 58
    entities/doc, 149k of the 174k total), and overlapping spans are common
    (`"Macros"` is annotated both alone and inside a longer span). The loader
    therefore **deduplicates by normalised surface form** — the metric asks
    "which skills does the résumé claim", not "how many times".
  * No `name`, `summary`, employer, or date labels, so Table 1c scores skills,
    titles, institutions, degrees and certifications only. Long-text F1 and
    company extraction stay with the synthetic set.
  * Documents are the flattened single-line form (like `Resume.csv`), so the
    loader segments them before ingest for honest per-line provenance spans —
    same treatment, same reason, as `eval/fabrication_corpus.py`.

The download is cached under `eval/datasets/public/`; the raw file is gitignored
(19 MB) and rebuilt on demand by rerunning this module.
"""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "public"
RAW_PATH = DATA_DIR / "resume_ner_labelled.json"
PREPARED_PATH = DATA_DIR / "gold_public.json"

URL = (
    "https://huggingface.co/datasets/kens1ang/resume-ner-labelled/"
    "resolve/main/train_data.json"
)

# Annotation label -> the StructuredResume field it evaluates.
LABEL_FIELDS = {
    "Skills": "skills",
    "Job title": "work_titles",
    "Institution Name": "institutions",
    "Degree": "degrees",
    "Certifications": "certifications",
}


def download(force: bool = False) -> Path:
    """Fetch the raw annotation file into the cache (idempotent)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and not force:
        return RAW_PATH
    req = urllib.request.Request(URL, headers={"User-Agent": "rho-eval/0.1"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(RAW_PATH, "wb") as fh:
        fh.write(resp.read())
    return RAW_PATH


def _dedup(values: list[str]) -> list[str]:
    """Drop repeat mentions and case variants, preserving first-seen order."""
    seen, out = set(), []
    for v in values:
        key = " ".join(v.split()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(" ".join(v.split()))
    return out


def prepare(n: int = 150, seed: int = 0, min_chars: int = 1500, max_chars: int = 8000) -> list[dict]:
    """Convert raw annotations into gold records and write `gold_public.json`.

    Selection is deterministic (sorted by a hash of the document text, then
    truncated), so the sample is stable across machines without depending on
    dict or file ordering.

    The length window drops truncated stubs and multi-résumé blobs, matching the
    filter `eval/corpus.py` already applies to `Resume.csv`.
    """
    import hashlib

    raw = json.loads(download().read_text())

    records = []
    for text, ann in raw:
        if not (min_chars <= len(text) <= max_chars):
            continue
        by_field: dict[str, list[str]] = defaultdict(list)
        # first_span keeps the earliest offset per value: the gold provenance
        # target. Later repeats of the same skill point at the same claim.
        first_span: dict[str, tuple[int, int]] = {}
        for start, end, label in ann.get("entities", []):
            field = LABEL_FIELDS.get(label)
            if field is None:
                continue
            surface = " ".join(text[start:end].split())
            if not surface:
                continue
            by_field[field].append(surface)
            key = f"{field}:{surface.lower()}"
            if key not in first_span or start < first_span[key][0]:
                first_span[key] = (start, end)

        if not by_field.get("skills"):
            continue  # nothing to score

        gold = {f: _dedup(v) for f, v in by_field.items()}
        # Gold provenance: field_path -> the source text it must trace to.
        gold_prov_values = {}
        for i, s in enumerate(gold.get("skills", [])):
            gold_prov_values[f"skills[{i}]"] = s
        records.append(
            {
                "digest": hashlib.sha1(text.encode()).hexdigest(),
                "text": text,
                "gold": gold,
                "gold_prov_values": gold_prov_values,
            }
        )

    records.sort(key=lambda r: r["digest"])
    picked = records[:n]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_PATH.write_text(
        json.dumps({"n": len(picked), "source": URL, "items": picked}, indent=1),
        encoding="utf-8",
    )
    return picked


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    recs = prepare(n=args.n)
    counts = {f: sum(len(r["gold"].get(f, [])) for r in recs) for f in set(LABEL_FIELDS.values())}
    print(f"wrote {len(recs)} public gold résumés to {PREPARED_PATH}")
    print("entity counts:", counts)
