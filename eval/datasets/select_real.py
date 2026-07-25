"""Select the real-corpus subset for hand labelling (C1 Table 1b).

Picks a deterministic, category-stratified sample from `Resume.csv` and writes
each résumé as a `.txt` document plus an empty label stub. Labels are then filled
in by hand in `eval/datasets/real/<id>.json`.

Stratification is by `Category` so the subset is not dominated by whichever
vertical the corpus happens to over-represent; extraction difficulty varies a lot
between a software résumé (dense named tools) and a nursing one (prose duties).

Selection is a pure function of `seed`, so the subset is reproducible and the
labelling effort is never invalidated by a reshuffle.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "real"

# Length window: below ~1200 chars the cell is usually a truncated stub, above
# ~4000 it is often several résumés concatenated. Same reasoning as eval/corpus.py.
MIN_CHARS, MAX_CHARS = 1200, 4000


def select(n: int = 30, seed: int = 3, resume_csv: str = "Resume.csv") -> list[dict]:
    """Write `n` corpus résumés (stratified by Category) as .txt + label stubs."""
    import pandas as pd

    from eval.fabrication_corpus import segment_corpus_text

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(resume_csv)
    df = df[df.Resume_str.str.len().between(MIN_CHARS, MAX_CHARS)]

    cats = sorted(df.Category.unique())
    per_cat = max(1, n // len(cats))
    picked = []
    for cat in cats:
        sub = df[df.Category == cat]
        take = min(per_cat, len(sub))
        picked.extend(sub.sample(take, random_state=seed).to_dict("records"))
    # Top up to exactly n from whatever remains, deterministically.
    if len(picked) < n:
        chosen_ids = {r["ID"] for r in picked}
        rest = df[~df.ID.isin(chosen_ids)]
        picked.extend(rest.sample(n - len(picked), random_state=seed).to_dict("records"))
    picked = sorted(picked, key=lambda r: r["ID"])[:n]

    manifest = []
    for i, row in enumerate(picked):
        rid = f"r{i:03d}"
        text = segment_corpus_text(row["Resume_str"])
        (DATA_DIR / f"{rid}.txt").write_text(text, encoding="utf-8")
        stub = DATA_DIR / f"{rid}.json"
        if not stub.exists():  # never clobber labels already written by hand
            stub.write_text(
                json.dumps(
                    {
                        "source_id": int(row["ID"]),
                        "category": row["Category"],
                        "labelled": False,
                        "gold": {"name": None, "summary": None, "skills": [], "work": [], "education": []},
                        "gold_prov_values": {},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        manifest.append({"id": rid, "file": f"{rid}.txt", "category": row["Category"]})

    (DATA_DIR / "manifest.json").write_text(
        json.dumps({"seed": seed, "n": len(manifest), "items": manifest}, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    m = select(n=args.n, seed=args.seed)
    print(f"wrote {len(m)} corpus résumés to {DATA_DIR} (label stubs need filling)")
