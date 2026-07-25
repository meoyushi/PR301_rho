"""Dataset loaders for the Phase-7 evaluation harness.

Three datasets, three loaders:

  `load_gold()`              — synthetic gold extraction set (Table 1a). Fields
                               and provenance spans are exact by construction.
  `load_real_gold()`         — hand-labelled real-corpus subset (Table 1b).
  `load_calibration_pairs()` — résumé × JD pairs for the ATS calibrator (C2).
  `load_fabrication_pairs()` — adversarial pairs for the rewrite gate (C3).

**Why the gold set has no PDFs.** The plan calls for `gold/<id>.pdf`. Docling's
PDF pipeline on this host requires OCR weights it cannot download (offline), and
with OCR disabled it returns the entire page as a *single* text item — one span
covering the whole document. A one-span provenance map "supports" nearly any
value by substring match, which would make `provenance_accuracy` meaningless and
hollow out C1's claim to an exact source location (the same trap
`eval/fabrication_corpus.py` documents for flattened corpus text). The gold set
therefore uses `.txt` (text adapter) and `.docx` (Docling), both of which yield
honest per-line spans. PDF ingestion is still exercised by the Phase-1 tests.

Every loader raises rather than returning an empty list when its dataset is
missing: a silent empty dataset yields a results table full of zeros that reads
exactly like a successful run.
"""

import json
from pathlib import Path
from typing import Any

GOLD_DIR = Path(__file__).parent / "gold"
REAL_DIR = Path(__file__).parent / "real"
PUBLIC_DIR = Path(__file__).parent / "public"

__all__ = [
    "load_gold",
    "load_real_gold",
    "load_public_gold",
    "load_calibration_pairs",
    "load_fabrication_pairs",
]


def load_gold(limit: int | None = None, data_dir: Path | None = None) -> list[tuple[Path, dict]]:
    """Synthetic gold set as `(document_path, gold_record)`, sorted by id.

    `gold_record` carries the labelled fields plus `gold_prov_values`
    (field_path -> the source text the field should trace to). The loader keeps
    provenance in *value* form rather than character offsets because Docling
    re-exports .docx to markdown with different spacing, so a source offset no
    longer indexes what the pipeline actually sees.
    """
    d = data_dir or GOLD_DIR
    manifest = d / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no gold manifest at {manifest}. Build it with: "
            f"python -m eval.datasets.synthetic --n 120"
        )
    items = json.loads(manifest.read_text())["items"]
    out: list[tuple[Path, dict]] = []
    for item in items:
        doc = d / item["file"]
        rec = json.loads((d / f"{item['id']}.json").read_text())
        gold = dict(rec["gold"])
        gold["gold_prov_values"] = rec.get("gold_prov_values", {})
        gold["id"] = item["id"]
        out.append((doc, gold))
    return out[:limit] if limit else out


def load_real_gold(limit: int | None = None, data_dir: Path | None = None) -> list[tuple[Path, dict]]:
    """Hand-labelled real-corpus subset as `(document_path, labels)`.

    Only `skills`, `work_titles`, `institutions` and `summary` are labelled;
    `name` and `work[].company` are unlabelable because the corpus anonymises
    them (every employer is the literal string "Company Name"). See the
    `_README` key in `real/labels.json`.
    """
    d = data_dir or REAL_DIR
    labels_path = d / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"no labels at {labels_path}")
    labels = json.loads(labels_path.read_text())
    out = []
    for rid in sorted(k for k in labels if not k.startswith("_")):
        doc = d / f"{rid}.txt"
        if not doc.exists():
            raise FileNotFoundError(
                f"labels reference {doc.name} but the document is missing; "
                f"rebuild with python -m eval.datasets.select_real"
            )
        rec = dict(labels[rid])
        rec["id"] = rid
        out.append((doc, rec))
    return out[:limit] if limit else out


def load_public_gold(limit: int | None = None) -> list[tuple[str, dict]]:
    """Public human-annotated gold set as `(document_text, gold_record)`.

    Returns text rather than a path: the source corpus stores each résumé as a
    flattened single line, so the caller segments it (`segment_corpus_text`)
    before ingest — writing it to disk first would only add a temp file.

    Labelled fields are `skills`, `work_titles`, `institutions`, `degrees`,
    `certifications`. See `eval/datasets/fetch_public.py` for provenance of the
    data and its limitations.
    """
    path = PUBLIC_DIR / "gold_public.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no public gold set at {path}. Build it with: "
            f"python -m eval.datasets.fetch_public --n 150"
        )
    items = json.loads(path.read_text())["items"]
    out = []
    for it in items:
        gold = dict(it["gold"])
        gold["gold_prov_values"] = it.get("gold_prov_values", {})
        gold["id"] = it["digest"][:8]
        out.append((it["text"], gold))
    return out[:limit] if limit else out


def load_calibration_pairs(n_pairs: int = 200, seed: int = 0) -> list[tuple[Any, str]]:
    """Résumé × JD pairs for the C2 calibrator.

    Delegates to `eval.corpus.build_pairs`, the same builder Phase 4 fitted the
    shipped calibrator with, so the calibration dataset reported in Table 2 is
    the one the artifact was actually trained on rather than a re-derivation.
    """
    from eval.corpus import build_pairs

    return build_pairs(n_pairs=n_pairs, seed=seed)


def load_fabrication_pairs(limit: int | None = None, corpus: int = 0, seed: int = 0) -> list[dict]:
    """Adversarial pairs for the C3 gate.

    `corpus=0` loads the curated 12-pair fixture; `corpus=N` draws N
    corpus-backed pairs (real work history, JD-derived gaps), which is what the
    Phase-5 headline numbers were measured on.
    """
    if corpus:
        from eval.fabrication_corpus import build_corpus_pairs

        pairs = build_corpus_pairs(n_pairs=corpus, seed=seed)
    else:
        from eval.fabrication_ablation import load_pairs

        pairs = load_pairs()
    if not pairs:
        raise RuntimeError("fabrication benchmark is empty; refusing to report on 0 pairs")
    return pairs[:limit] if limit else pairs
