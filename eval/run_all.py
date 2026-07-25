"""Phase 7 evaluation harness — computes every number the paper reports.

Writes `eval/RESULTS.md` plus per-table CSVs. Each table is reproducible on its
own; nothing here invents a number that is not computed from a dataset in
`eval/datasets/` or an artifact produced by an earlier phase's script.

Tables
  1a  C1 extraction, synthetic gold (exact labels + exact provenance spans)
  1b  C1 extraction, hand-labelled real corpus subset
  1c  C1 extraction, public human-annotated gold set  <- the headline C1 number
  2   C2 calibration (from eval/fit_calibrator.py output)
  3   C3 fabrication (from eval/fabrication_ablation.py output)
  4   Ablations (eval/ablations.py)

Extraction is an LLM call per résumé (~4 min on this CPU-only host), so results
are cached per document in `eval/cache/extraction/`. A re-run reuses the cache
and only extracts what is missing, which makes a long run resumable after an
interruption instead of restarting from zero.

Usage:
    python -m eval.run_all --limit 20          # quick pass
    python -m eval.run_all                     # full run, all datasets
    python -m eval.run_all --tables 1c,2,3     # selected tables only
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

# Must precede any torch import (sentence-transformers sizes its thread pool at
# import time and oversubscribes this 16-core box otherwise).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval.datasets import (  # noqa: E402
    load_gold,
    load_public_gold,
    load_real_gold,
)
from eval.metrics import field_f1, long_text_f1, mean, provenance_accuracy  # noqa: E402

EVAL_DIR = Path(__file__).parent
CACHE_DIR = EVAL_DIR / "cache" / "extraction"
RESULTS_MD = EVAL_DIR / "RESULTS.md"


# --------------------------------------------------------------------------
# extraction with an on-disk cache
# --------------------------------------------------------------------------


def _extract_cached(text: str, filename: str, use_cache: bool = True) -> tuple[dict, float, int]:
    """(resume_dict, seconds, n_spans) for one document, memoised on disk.

    The cache key covers the document bytes *and* the extraction backend, so
    switching models does not silently reuse another model's outputs.
    """
    from rho.config import settings

    backend = getattr(settings, "extraction_backend", "unknown")
    key = hashlib.sha1(f"{backend}\x00{text}".encode()).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if use_cache and cache_path.exists():
        rec = json.loads(cache_path.read_text())
        return rec["resume"], rec["seconds"], rec["n_spans"]

    from rho.extraction import extract
    from rho.ingestion import ingest

    started = time.monotonic()
    md, prov = ingest(text.encode(), filename)
    resume = extract(md, prov)
    seconds = time.monotonic() - started

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "resume": resume.model_dump(),
        "seconds": seconds,
        "n_spans": len(prov.spans),
        "backend": backend,
    }
    cache_path.write_text(json.dumps(payload))
    return payload["resume"], seconds, payload["n_spans"]


def _extract_doc(path: Path, use_cache: bool = True):
    """Extract from a file on disk, preserving its real ingest path (.txt/.docx).

    Returns `(resume_dict, seconds, prov)` — the ProvenanceMap is rebuilt from
    the document (ingest is deterministic and fast) so provenance accuracy can
    be scored even on a cache hit.
    """
    from rho.ingestion import ingest

    data = path.read_bytes()
    md, prov = ingest(data, path.name)

    from rho.config import settings

    backend = getattr(settings, "extraction_backend", "unknown")
    key = hashlib.sha1(f"{backend}\x00{path.name}\x00".encode() + data).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if use_cache and cache_path.exists():
        rec = json.loads(cache_path.read_text())
        return rec["resume"], rec["seconds"], prov

    from rho.extraction import extract

    started = time.monotonic()
    resume = extract(md, prov)
    seconds = time.monotonic() - started
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"resume": resume.model_dump(), "seconds": seconds, "backend": backend})
    )
    return resume.model_dump(), seconds, prov


# --------------------------------------------------------------------------
# Table 1 — extraction quality (C1)
# --------------------------------------------------------------------------


def eval_synthetic(limit: int | None, use_cache: bool = True) -> dict:
    """Table 1a — synthetic gold: every field labelled, provenance exact.

    One doc's extraction failing (LLM outage, a permanently denied key, a
    malformed response) must not lose the whole table's progress — the failure
    is recorded and printed, never silently dropped, but the loop continues.
    """
    rows = []
    failures: list[str] = []
    for path, gold in load_gold(limit=limit):
        try:
            pred, seconds, prov = _extract_doc(path, use_cache)
        except Exception as exc:
            failures.append(f"{gold['id']}: {type(exc).__name__}: {exc}")
            print(f"  [1a {len(rows) + len(failures)}] {gold['id']}: FAILED ({exc})", flush=True)
            continue
        from rho.models.resume import StructuredResume

        resume_obj = StructuredResume(**pred)
        rows.append(
            {
                "id": gold["id"],
                "format": path.suffix.lstrip("."),
                "name_exact": float(
                    (pred.get("name") or "").strip().lower() == gold["name"].strip().lower()
                ),
                "skills_f1": field_f1(pred, gold, "skills")["f1"],
                "work_f1": field_f1(pred, gold, "work", keys=("company", "title"))["f1"],
                "work_company_f1": field_f1(pred, gold, "work", keys=("company",))["f1"],
                "work_title_f1": field_f1(pred, gold, "work", keys=("title",))["f1"],
                "education_f1": field_f1(pred, gold, "education", keys=("institution",))["f1"],
                "summary_longtext_f1": long_text_f1(pred.get("summary"), gold.get("summary")),
                "prov_accuracy": provenance_accuracy(
                    resume_obj, _resolve_gold_prov(gold["gold_prov_values"], prov), prov
                ),
                "seconds": round(seconds, 1),
            }
        )
        print(
            f"  [1a {len(rows)}] {gold['id']}: skills={rows[-1]['skills_f1']:.2f} "
            f"work={rows[-1]['work_f1']:.2f} prov={rows[-1]['prov_accuracy']:.2f} "
            f"({rows[-1]['seconds']:.0f}s)",
            flush=True,
        )
    if failures:
        print(f"  WARNING: {len(failures)} doc(s) failed extraction: {failures[:3]}", flush=True)
    return {"rows": rows, "summary": _summarise(rows), "failures": failures}


def _resolve_gold_prov(gold_values: dict, prov) -> dict:
    """field_path -> prov_id, by locating each gold value's source span.

    Gold provenance is stored as the *text* a field must trace to (offsets do
    not survive Docling's markdown re-export). The correct prov_id is the span
    that actually contains that text in the ingested document; resolving it here
    keeps `provenance_accuracy` an exact-span check rather than a fuzzy one.
    """
    resolved = {}
    for path, value in gold_values.items():
        needle = str(value).strip().lower()
        if not needle:
            continue
        for pid, span in prov.spans.items():
            if needle in span.raw_text.lower():
                resolved[path] = pid
                break
    return resolved


def eval_public(limit: int | None, use_cache: bool = True) -> dict:
    """Table 1c — public human-annotated gold set (the headline C1 number)."""
    from eval.fabrication_corpus import segment_corpus_text
    from rho.models.resume import StructuredResume

    rows = []
    failures: list[str] = []
    for text, gold in load_public_gold(limit=limit):
        doc = segment_corpus_text(text)
        try:
            pred, seconds, prov = _extract_text(doc, f"pub{gold['id']}.txt", use_cache)
        except Exception as exc:
            failures.append(f"{gold['id']}: {type(exc).__name__}: {exc}")
            print(f"  [1c {len(rows) + len(failures)}] {gold['id']}: FAILED ({exc})", flush=True)
            continue
        resume_obj = StructuredResume(**pred)
        pred_titles = {"work_titles": [w.get("title", "") for w in pred.get("work", [])]}
        pred_insts = {"institutions": [e.get("institution", "") for e in pred.get("education", [])]}
        rows.append(
            {
                "id": gold["id"],
                "skills_f1": field_f1(pred, gold, "skills")["f1"],
                "skills_p": field_f1(pred, gold, "skills")["precision"],
                "skills_r": field_f1(pred, gold, "skills")["recall"],
                "title_f1": field_f1(pred_titles, gold, "work_titles")["f1"],
                "institution_f1": field_f1(pred_insts, gold, "institutions")["f1"],
                "certification_f1": field_f1(pred, gold, "certifications")["f1"],
                "prov_accuracy": provenance_accuracy(
                    resume_obj, _resolve_gold_prov(gold["gold_prov_values"], prov), prov
                ),
                "seconds": round(seconds, 1),
            }
        )
        print(
            f"  [1c {len(rows)}] {gold['id']}: skills={rows[-1]['skills_f1']:.2f} "
            f"title={rows[-1]['title_f1']:.2f} prov={rows[-1]['prov_accuracy']:.2f} "
            f"({rows[-1]['seconds']:.0f}s)",
            flush=True,
        )
    if failures:
        print(f"  WARNING: {len(failures)} doc(s) failed extraction: {failures[:3]}", flush=True)
    return {"rows": rows, "summary": _summarise(rows), "failures": failures}


def _extract_text(doc: str, filename: str, use_cache: bool = True):
    """Extract from in-memory text; returns `(resume_dict, seconds, prov)`."""
    from rho.ingestion import ingest

    md, prov = ingest(doc.encode(), filename)
    pred, seconds, _ = _extract_cached(doc, filename, use_cache)
    return pred, seconds, prov


def eval_real(limit: int | None, use_cache: bool = True) -> dict:
    """Table 1b — hand-labelled real corpus subset.

    Only the labelable fields are scored: the corpus anonymises candidate names
    and every employer ("Company Name"), so name and company are excluded rather
    than scored against a placeholder.
    """
    rows = []
    failures: list[str] = []
    for path, gold in load_real_gold(limit=limit):
        try:
            pred, seconds, _prov = _extract_doc(path, use_cache)
        except Exception as exc:
            failures.append(f"{gold['id']}: {type(exc).__name__}: {exc}")
            print(f"  [1b {len(rows) + len(failures)}] {gold['id']}: FAILED ({exc})", flush=True)
            continue
        pred_titles = {"work_titles": [w.get("title", "") for w in pred.get("work", [])]}
        pred_insts = {"institutions": [e.get("institution", "") for e in pred.get("education", [])]}
        rows.append(
            {
                "id": gold["id"],
                "category": gold.get("category", ""),
                "skills_f1": field_f1(pred, gold, "skills")["f1"],
                "title_f1": field_f1(pred_titles, gold, "work_titles")["f1"],
                "institution_f1": field_f1(pred_insts, gold, "institutions")["f1"],
                "summary_longtext_f1": long_text_f1(pred.get("summary"), gold.get("summary")),
                "seconds": round(seconds, 1),
            }
        )
        print(
            f"  [1b {len(rows)}] {gold['id']}: skills={rows[-1]['skills_f1']:.2f} "
            f"title={rows[-1]['title_f1']:.2f} ({rows[-1]['seconds']:.0f}s)",
            flush=True,
        )
    if failures:
        print(f"  WARNING: {len(failures)} doc(s) failed extraction: {failures[:3]}", flush=True)
    return {"rows": rows, "summary": _summarise(rows), "failures": failures}


def _summarise(rows: list[dict]) -> dict:
    """Mean of every numeric column, plus n."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)):
            out[key] = mean(r[key] for r in rows)
    return out


# --------------------------------------------------------------------------
# Tables 2 and 3 — read the artifacts produced by the P4/P5 scripts
# --------------------------------------------------------------------------


def load_c2(suffix: str = "") -> dict | None:
    """Calibration metrics from `eval/fit_calibrator.py`'s saved progress file.

    `suffix` selects a backend-specific artifact (e.g. `_gemini` reads
    `progress_gemini.json`) without disturbing the qwen-baseline default.
    """
    path = EVAL_DIR / f"progress{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("metrics")


def load_c3(suffix: str = "") -> dict | None:
    """Fabrication summary from `eval/fabrication_ablation.py`'s output.

    Prefers the corpus-backed run (the Phase-5 headline) over the fixture run.
    """
    for name in (f"fabrication_results_corpus{suffix}.json", f"fabrication_results{suffix}.json"):
        path = EVAL_DIR / name
        if path.exists():
            d = json.loads(path.read_text())
            d["_source"] = name
            return d
    return None


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def write_results(res: dict) -> None:
    """Render every computed table into eval/RESULTS.md."""
    L: list[str] = []
    add = L.append
    add("# Phase 7 — Evaluation Results\n")
    add(
        f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by "
        f"`python -m eval.run_all`. Every number below is recomputed from the datasets "
        f"in `eval/datasets/` and the artifacts of Phases 4–5._\n"
    )
    backend = res.get("backend", "unknown")
    add(f"**Extraction backend:** `{backend}`\n")

    # ---- Table 1
    add("\n## Table 1 — Extraction quality (C1)\n")
    add(
        "Named-entity fields and long-text fields are reported **separately**: prose is "
        "rewritten rather than copied, so averaging it into the entity score would hide "
        "that long-text is the hardest field class.\n"
    )

    pub = res.get("public", {}).get("summary")
    if pub and pub.get("n"):
        pub_failed = len(res.get("public", {}).get("failures") or [])
        add("\n### Table 1c — Public human-annotated gold set (headline C1)\n")
        add(
            f"`kens1ang/resume-ner-labelled`, n={pub['n']} résumés scored"
            + (f" ({pub_failed} failed extraction, excluded)" if pub_failed else "")
            + ". Annotations are independent of this system and of its authors.\n"
        )
        add("| Field | F1 |")
        add("|---|---|")
        add(f"| skills | {_fmt(pub.get('skills_f1'))} |")
        add(f"| job title | {_fmt(pub.get('title_f1'))} |")
        add(f"| institution | {_fmt(pub.get('institution_f1'))} |")
        add(f"| certification | {_fmt(pub.get('certification_f1'))} |")
        add(
            f"\n- skills precision {_fmt(pub.get('skills_p'))} / recall {_fmt(pub.get('skills_r'))}"
        )
        add(f"- **provenance-attachment accuracy: {_fmt(pub.get('prov_accuracy'))}**")

    syn = res.get("synthetic", {}).get("summary")
    if syn and syn.get("n"):
        syn_failed = len(res.get("synthetic", {}).get("failures") or [])
        add("\n### Table 1a — Synthetic gold set (upper bound)\n")
        add(
            f"n={syn['n']} scored"
            + (f" ({syn_failed} failed extraction, excluded)" if syn_failed else "")
            + ". Labels and provenance spans are exact by construction. "
            "Templated résumés are cleaner than real ones, so **treat this as an upper "
            "bound**, not an estimate of field performance.\n"
        )
        add("| Field | F1 |")
        add("|---|---|")
        add(f"| skills | {_fmt(syn.get('skills_f1'))} |")
        add(f"| work (company+title) | {_fmt(syn.get('work_f1'))} |")
        add(f"| work — company only | {_fmt(syn.get('work_company_f1'))} |")
        add(f"| work — title only | {_fmt(syn.get('work_title_f1'))} |")
        add(f"| education (institution) | {_fmt(syn.get('education_f1'))} |")
        add(f"| name (exact match) | {_fmt(syn.get('name_exact'))} |")
        add(f"| **summary (long-text F1)** | **{_fmt(syn.get('summary_longtext_f1'))}** |")
        add(f"\n- **provenance-attachment accuracy: {_fmt(syn.get('prov_accuracy'))}**")

    real = res.get("real", {}).get("summary")
    if real and real.get("n"):
        real_failed = len(res.get("real", {}).get("failures") or [])
        add("\n### Table 1b — Hand-labelled real corpus subset (reality check)\n")
        add(
            f"n={real['n']} résumés from `Resume.csv` scored"
            + (f" ({real_failed} failed extraction, excluded)" if real_failed else "")
            + ". Labelled by the implementing agent, "
            "**not** independently verified — the public set (1c) is the trustworthy real-data "
            "number. `name` and `work[].company` are unlabelable: the corpus anonymises them.\n"
        )
        add("| Field | F1 |")
        add("|---|---|")
        add(f"| skills | {_fmt(real.get('skills_f1'))} |")
        add(f"| job title | {_fmt(real.get('title_f1'))} |")
        add(f"| institution | {_fmt(real.get('institution_f1'))} |")
        add(f"| **summary (long-text F1)** | **{_fmt(real.get('summary_longtext_f1'))}** |")

    # ---- Table 2
    add("\n## Table 2 — ATS calibration (C2)\n")
    c2 = res.get("c2")
    if c2:
        add(
            f"Target `{c2.get('target')}`; {c2.get('n_usable')} usable pairs "
            f"({c2.get('n_train')} train / {c2.get('n_heldout')} held-out). "
            f"Held-out target mean {_fmt(c2.get('y_mean'), 2)} (sd {_fmt(c2.get('y_std'), 2)}).\n"
        )
        add("| Scorer | MAE | Spearman ρ |")
        add("|---|---|---|")
        add(f"| **calibrated (rho)** | **{_fmt(c2.get('mae'), 2)}** | **{_fmt(c2.get('spearman'))}** |")
        add(f"| cosine baseline | {_fmt(c2.get('cosine_mae'), 2)} | {_fmt(c2.get('cosine_spearman'))} |")
        ov = c2.get("overall_target") or {}
        if ov:
            add(
                f"\n_Secondary (composite `overallScore` target): calibrated MAE "
                f"{_fmt(ov.get('mae'), 2)} ρ {_fmt(ov.get('spearman'))} vs cosine MAE "
                f"{_fmt(ov.get('cosine_mae'), 2)} ρ {_fmt(ov.get('cosine_spearman'))}._"
            )
        add(
            "\n**Read ρ, not MAE, as the headline.** The cosine baseline emits 0–100 while the "
            "target band is narrow and low, so most of its MAE is scale mismatch rather than "
            "ranking failure."
        )
    else:
        add("_Not available — run `python -m eval.fit_calibrator`._")

    # ---- Table 3
    add("\n## Table 3 — Fabrication gate (C3)\n")
    c3 = res.get("c3")
    if c3:
        add(
            f"Benchmark `{c3.get('_source')}`, {c3.get('pairs_scored')} pairs scored "
            f"({c3.get('pairs_failed')} failed), backend `{c3.get('backend')}`.\n"
        )
        add("| Condition | Unsourced additions shipped |")
        add("|---|---|")
        add(f"| gate OFF (prompt grounding only) | {c3.get('unsourced_shipped_gate_off')} |")
        add(f"| **gate ON** | **{c3.get('unsourced_shipped_gate_on')}** |")
        add(f"\n- mean `fabrication_rate` (gate detections): {_fmt(c3.get('mean_fabrication_rate'))}")
        add(
            "- Gate-ON is 0 **by construction** — that is the claim. The informative number is "
            "gate-OFF: how often a grounded prompt alone would have shipped a fabrication."
        )
    else:
        add("_Not available — run `python -m eval.fabrication_ablation`._")

    # ---- Table 4
    add("\n## Table 4 — Ablations\n")
    abl = res.get("ablations")
    if abl:
        add("| Ablation | Condition | Metric | Value |")
        add("|---|---|---|---|")
        for row in abl:
            add(
                f"| {row['ablation']} | {row['condition']} | {row['metric']} | "
                f"{_fmt(row['value']) if isinstance(row['value'], float) else row['value']} |"
            )
    else:
        add("_Not available — run `python -m eval.ablations`._")

    # ---- latency / cost
    add("\n## Latency and cost\n")
    lat = res.get("latency") or {}
    if lat:
        add(f"- Extraction: median **{_fmt(lat.get('median'), 1)}s** per résumé "
            f"(n={lat.get('n')}, min {_fmt(lat.get('min'), 1)}s, max {_fmt(lat.get('max'), 1)}s).")
    backend = res.get("backend", "unknown")
    if backend in ("ollama", "vllm"):
        add(
            "- **Cost per successful task: $0.00 in API spend.** The whole pipeline runs on "
            "self-hosted open models (Ollama/vLLM) on this host; the cost is wall-clock "
            "compute, not per-token billing. This is a deliberate property of the design — "
            "every contribution is reproducible without a commercial API account."
        )
    else:
        add(
            f"- **Cost per successful task: $0.00 in API spend** (backend `{backend}`, "
            "free-tier quota). Unlike the Ollama/vLLM path, this backend is a hosted "
            "commercial API — free here because the run stayed inside the provider's "
            "free-tier daily quota, not because the design has no billing surface. "
            "Reproducing this run at larger scale would need either a paid tier or "
            "spreading calls across more free-tier accounts."
        )

    add("\n## Dataset sizes\n")
    sizes = res.get("dataset_sizes", {})
    for k, v in sizes.items():
        add(f"- {k}: {v}")

    RESULTS_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {RESULTS_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap résumés per table")
    ap.add_argument("--tables", default="1a,1b,1c,2,3,4", help="comma-separated table ids")
    ap.add_argument("--no-cache", action="store_true", help="ignore the extraction cache")
    ap.add_argument(
        "--extraction-backend",
        default=None,
        choices=["ollama", "vllm", "gemini"],
        help="override rho.config.settings.extraction_backend for table 1 (default: config's own setting)",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="read backend-specific C2/C3 artifacts, e.g. --suffix _gemini reads "
        "progress_gemini.json / fabrication_results_corpus_gemini.json instead of "
        "the qwen-baseline files",
    )
    args = ap.parse_args()

    want = {t.strip() for t in args.tables.split(",")}
    use_cache = not args.no_cache
    res: dict = {}

    from rho.config import settings

    if args.extraction_backend:
        settings.extraction_backend = args.extraction_backend
    res["backend"] = getattr(settings, "extraction_backend", "unknown")

    all_seconds: list[float] = []
    if "1a" in want:
        print("Table 1a — synthetic gold...", flush=True)
        res["synthetic"] = eval_synthetic(args.limit, use_cache)
        _write_csv(EVAL_DIR / "results_table1a_synthetic.csv", res["synthetic"]["rows"])
        all_seconds += [r["seconds"] for r in res["synthetic"]["rows"]]
    if "1b" in want:
        print("Table 1b — hand-labelled real subset...", flush=True)
        res["real"] = eval_real(args.limit, use_cache)
        _write_csv(EVAL_DIR / "results_table1b_real.csv", res["real"]["rows"])
        all_seconds += [r["seconds"] for r in res["real"]["rows"]]
    if "1c" in want:
        print("Table 1c — public human-annotated gold...", flush=True)
        res["public"] = eval_public(args.limit, use_cache)
        _write_csv(EVAL_DIR / "results_table1c_public.csv", res["public"]["rows"])
        all_seconds += [r["seconds"] for r in res["public"]["rows"]]

    if "2" in want:
        res["c2"] = load_c2(args.suffix)
    if "3" in want:
        res["c3"] = load_c3(args.suffix)
    if "4" in want:
        from eval.ablations import run_ablations

        res["ablations"] = run_ablations(args.suffix)

    if all_seconds:
        res["latency"] = {
            "n": len(all_seconds),
            "median": statistics.median(all_seconds),
            "min": min(all_seconds),
            "max": max(all_seconds),
        }

    res["dataset_sizes"] = {
        "gold — public (human-annotated)": len(load_public_gold()),
        "gold — synthetic": len(load_gold()),
        "gold — hand-labelled real": len(load_real_gold()),
        "calibration pairs": (res.get("c2") or {}).get("n_usable", "—"),
        "fabrication pairs": (res.get("c3") or {}).get("pairs_scored", "—"),
    }

    write_results(res)


if __name__ == "__main__":
    main()
