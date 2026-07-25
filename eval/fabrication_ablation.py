"""Gate ON vs OFF fabrication comparison (C3 headline).

Each benchmark pair is a résumé whose JD demands skills the résumé does not
have — maximum pressure to invent. The rewriter runs once per pair; we then
count how many unsourced hard-content values would *ship* under each condition:

  gate OFF — the raw rewriter output, prompt-grounding only.
  gate ON  — the same output after `verify_against_source` strips rejections.

Gate-ON is 0 by construction, and that is exactly the claim: the guarantee is
structural, not a model behaviour that happens to hold on this sample. The
number that carries information is gate-OFF — how often a grounded prompt alone
would have shipped a fabrication.

Usage: python -m eval.fabrication_ablation [--limit N] [--out results.json]
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rho.ingestion import ingest
from rho.models.jd import Requirement
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.scoring import Gap
from rho.rewrite.verifier import verify_against_source

PAIRS_PATH = Path(__file__).parent.parent / "tests/fixtures/fabrication/pairs.json"


def get_rewriter(backend: str):
    """Resolve the rewrite function for `backend` ("groq", "gemini", or "ollama")."""
    if backend == "groq":
        from rho.rewrite.groq import rewrite_schema_groq

        return rewrite_schema_groq
    if backend == "gemini":
        from rho.rewrite.gemini import rewrite_schema_gemini

        return rewrite_schema_gemini
    from rho.rewrite.llm import rewrite_schema

    return rewrite_schema


def load_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    """Each pair carries a résumé parsed into (resume, prov) via real ingestion.

    Provenance comes from the real ingest path rather than a hand-built map, so
    the gate is exercised against the same span shapes it sees in production.
    """
    raw = json.loads(path.read_text())
    pairs = []
    for item in raw:
        text = item["resume"]
        _, prov = ingest(text.encode(), f"{item['id']}.txt")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        resume = StructuredResume(
            name=lines[0],
            headline=lines[1] if len(lines) > 1 else None,
            skills=list(item["source_skills"]),
        )
        # The absent requirements ARE the adversarial pressure: naming them as
        # targets is what tempts the model to invent. Passing gaps=[] would test
        # the rewriter in an easy condition the benchmark was built to avoid.
        gaps = [
            Gap(
                requirement=Requirement(text=t, kind="skill", priority="must"),
                status="absent",
            )
            for t in item["tempting_absent"]
        ]
        pairs.append(
            {
                "id": item["id"],
                "resume": resume,
                "prov": prov,
                "jd": item["jd"],
                "gaps": gaps,
                "tempting_absent": item["tempting_absent"],
            }
        )
    return pairs


def unsourced_count(resume: StructuredResume, source: StructuredResume, prov: ProvenanceMap) -> int:
    """How many hard-content additions in `resume` lack supporting provenance."""
    _, rep = verify_against_source(resume, source, prov)
    return rep.total_edits - rep.verified_edits


def _score_pair(pair: dict, rewriter) -> dict:
    """Rewrite one pair and score it under both conditions."""
    source, prov = pair["resume"], pair["prov"]
    started = time.monotonic()
    try:
        raw = rewriter(source, pair["gaps"])  # gate OFF: ship as generated
    except Exception as exc:  # a dead model must not look like a clean run
        return {"id": pair["id"], "error": str(exc)}

    off = unsourced_count(raw, source, prov)
    fixed, rep = verify_against_source(raw, source, prov)  # gate ON
    on = unsourced_count(fixed, source, prov)  # what actually survives the gate
    return {
        "id": pair["id"],
        "unsourced_off": off,
        "unsourced_on": on,
        "total_edits": rep.total_edits,
        "fabrication_rate": rep.fabrication_rate,
        "rejected": [r.added_text for r in rep.rejected_edits],
        "seconds": round(time.monotonic() - started, 1),
    }


def run(
    pairs: list[dict],
    verbose: bool = True,
    backend: str = "groq",
    workers: int = 5,
) -> dict:
    """Score every pair, `workers` at a time.

    Groq requests round-robin across the available API keys, so concurrency
    spreads load across quotas rather than hammering one. Results are collected
    into input order regardless of completion order, keeping runs comparable.
    """
    rewriter = get_rewriter(backend)
    results: dict[int, dict] = {}
    done = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_score_pair, pair, rewriter): i for i, pair in enumerate(pairs)
        }
        for future in as_completed(futures):
            i = futures[future]
            record = future.result()
            results[i] = record
            if verbose:
                with lock:
                    done += 1
                    if "error" in record:
                        print(f"  [{done}/{len(pairs)}] {record['id']}: FAILED ({record['error'][:80]})")
                    else:
                        print(
                            f"  [{done}/{len(pairs)}] {record['id']}: "
                            f"OFF={record['unsourced_off']} ON={record['unsourced_on']} "
                            f"rate={record['fabrication_rate']:.2f} ({record['seconds']:.0f}s)"
                        )

    per_pair = [results[i] for i in sorted(results)]
    scored = [p for p in per_pair if "error" not in p]
    off_total = sum(p["unsourced_off"] for p in scored)
    on_total = sum(p["unsourced_on"] for p in scored)
    rates = [p["fabrication_rate"] for p in scored]

    mean_rate = sum(rates) / len(rates) if rates else 0.0
    summary = {
        "backend": backend,
        "pairs": len(pairs),
        "pairs_scored": len(rates),
        "pairs_failed": len(per_pair) - len(scored),
        "unsourced_shipped_gate_off": off_total,
        "unsourced_shipped_gate_on": on_total,
        "mean_fabrication_rate": mean_rate,
        "per_pair": per_pair,
    }
    if not scored:
        # "gate-OFF=0 over 0 pairs" is not a result — it is the shape a totally
        # failed run takes, and it reads like a clean one. Never let it pass.
        raise RuntimeError(
            f"no pairs scored ({len(per_pair)} attempted, all failed). "
            "The ablation produced no data; do not report these numbers."
        )
    print(
        f"\nunsourced additions shipped  gate-OFF={off_total}  gate-ON={on_total}"
        f"\nmean fabrication_rate = {mean_rate:.3f}  over {len(rates)} pairs"
        f"  ({len(per_pair) - len(scored)} failed)"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default="ollama", choices=["groq", "gemini", "ollama"])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument(
        "--corpus",
        type=int,
        default=0,
        metavar="N",
        help="draw N pairs from Resume.csv x training_data.csv instead of the "
        "curated fixture (real work history, bullets, and JD-derived gaps)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "fabrication_results.json")
    args = ap.parse_args()

    if args.corpus:
        from eval.fabrication_corpus import build_corpus_pairs

        print(f"building {args.corpus} corpus pairs (JD analysis via LLM)...")
        pairs = build_corpus_pairs(
            n_pairs=args.corpus,
            seed=args.seed,
            workers=args.workers,
            backend=args.backend,
        )
    else:
        pairs = load_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(
        f"running fabrication ablation on {len(pairs)} pairs "
        f"[backend={args.backend}, workers={args.workers}]..."
    )
    summary = run(pairs, backend=args.backend, workers=args.workers)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
