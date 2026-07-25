"""Table 4 — the three ablations, each toggling exactly one component.

  A. Provenance chain on/off  — does the gate depend on provenance, or would
     prompt grounding alone have produced the same output? Measured as unsourced
     hard-content additions shipped with the gate applied vs. bypassed.
  B. Calibrated vs cosine     — reuses the Phase-4 held-out fit (`progress.json`).
  C. Rewrite gate on/off      — reuses the Phase-5 run (`fabrication_results_*.json`).

B and C read the artifacts their own phase scripts produced rather than re-running
them: re-running would draw a different train/test split and a different set of
generations, so the ablation row would no longer match the headline number it is
supposed to explain.

Ablation A is computed here because it is the one comparison no earlier phase
made — it isolates the *provenance chain* specifically, rather than the gate as a
whole.
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def _load(name: str) -> dict | None:
    path = EVAL_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def ablation_provenance(suffix: str = "") -> list[dict]:
    """A — does the gate need the *provenance chain*, or just the source text?

    C and A must not be the same measurement wearing two labels. Ablation C
    toggles the whole gate (ship the raw generation vs ship the gated one).
    Ablation A holds the gate *on* and removes only the provenance chain,
    replacing span-resolved verification with the obvious cheaper substitute:
    check whether the added value appears anywhere in the source résumé's text.

    That substitute is what a system without a provenance ID space would have to
    use, so the gap between the two is the measured value of C1 to C3. It is a
    real toggle of one component: same generations, same gate logic, different
    evidence source.

    Both conditions are recomputed here from the stored per-pair rejections, so
    the row does not silently inherit C's numbers.
    """
    data = _load(f"fabrication_results_corpus{suffix}.json") or _load(
        f"fabrication_results{suffix}.json"
    )
    if not data:
        return []
    scored = [p for p in data.get("per_pair", []) if "error" not in p]
    if not scored:
        return []

    pairs_by_id = _fabrication_sources(suffix)
    chain_blocked = 0  # rejected by the real, provenance-backed gate
    text_blocked = 0  # would also be caught by a plain source-text search
    checked = 0
    for rec in scored:
        source_text = pairs_by_id.get(rec["id"])
        for added in rec.get("rejected", []):
            chain_blocked += 1
            if source_text is None:
                continue
            checked += 1
            # The naive alternative: is the invented value present in the source
            # document at all? If yes, a text-only gate would have let it ship.
            if _appears_in(added, source_text):
                text_blocked += 1

    rows = [
        {
            "ablation": "A. provenance chain",
            "condition": "ON (span-resolved evidence)",
            "metric": "fabrications rejected",
            "value": chain_blocked,
        }
    ]
    if checked:
        leaked = checked - text_blocked  # caught by chain, missed by text-only search
        rows.append(
            {
                "ablation": "A. provenance chain",
                "condition": "OFF (source-text substring check)",
                "metric": "fabrications rejected",
                "value": leaked,
            }
        )
        rows.append(
            {
                "ablation": "A. provenance chain",
                "condition": "OFF",
                "metric": f"would ship despite the gate (of {checked} checked)",
                "value": text_blocked,
            }
        )
    else:
        # Never let "0 rejected" stand in for "no source documents to compare
        # against" — that reads like a finding and is only missing data.
        rows.append(
            {
                "ablation": "A. provenance chain",
                "condition": "OFF (source-text substring check)",
                "metric": "not computed — source documents unavailable",
                "value": "n/a",
            }
        )
    affected = sum(1 for p in scored if p["unsourced_off"] > 0)
    rows.append(
        {
            "ablation": "A. provenance chain",
            "condition": "—",
            "metric": f"pairs with ≥1 fabrication (of {len(scored)})",
            "value": affected,
        }
    )
    return rows


def _appears_in(value: str, source_text: str) -> bool:
    """Would a provenance-free gate accept `value`?

    Mirrors what such a gate could cheaply do: case-insensitive containment of
    the added value, or of every one of its content words, anywhere in the
    source document. No span identity, no offsets — which is precisely the
    capability the provenance chain adds.
    """
    hay = source_text.lower()
    v = " ".join(value.split()).lower()
    if not v:
        return False
    if v in hay:
        return True
    words = [w for w in v.replace(",", " ").split() if len(w) > 3]
    return bool(words) and all(w in hay for w in words)


def _fabrication_sources(suffix: str = "") -> dict[str, str]:
    """pair_id -> source résumé text, for the benchmark the results came from.

    Two benchmarks, two id spaces. The curated fixture carries its résumés
    inline (`fab-*` ids). The corpus benchmark builds pairs at run time
    (`corpus-<seed>-<i>`) and the Phase-5 results file stores only rejections,
    not the source documents — so those are rebuilt from the same deterministic
    builder (`build_pairs`, seed 0) and cached in `fabrication_sources<suffix>.json`.
    The cache is suffixed too: different backends can draw different corpus
    seeds/sizes, so a qwen-run cache must not silently answer a Gemini lookup.

    Rebuilding uses `build_pairs`, not `build_corpus_pairs`: only the résumé
    text is needed, and `build_corpus_pairs` would additionally re-run JD
    analysis through the LLM for every pair.
    """
    sources: dict[str, str] = {}
    sources_cache = EVAL_DIR / f"fabrication_sources{suffix}.json"

    fixture = EVAL_DIR.parent / "tests/fixtures/fabrication/pairs.json"
    if fixture.exists():
        try:
            for item in json.loads(fixture.read_text()):
                if "id" in item and "resume" in item:
                    sources[item["id"]] = item["resume"]
        except (OSError, json.JSONDecodeError):
            pass

    if sources_cache.exists():
        try:
            sources.update(json.loads(sources_cache.read_text()))
            return sources
        except (OSError, json.JSONDecodeError):
            pass

    data = _load(f"fabrication_results_corpus{suffix}.json")
    if not data:
        return sources

    # Rebuild the corpus résumés the Phase-5 run scored. `_trim` is applied for
    # the same reason it is in the benchmark: the gate saw the trimmed document,
    # so the ablation must compare against exactly that text.
    try:
        from eval.corpus import build_pairs
        from eval.fabrication_corpus import _trim
    except ImportError:
        return sources

    seeds = {
        int(p["id"].split("-")[1]) for p in data.get("per_pair", []) if p["id"].startswith("corpus-")
    }
    n_pairs = len(data.get("per_pair", []))
    rebuilt: dict[str, str] = {}
    for seed in seeds:
        try:
            pairs = build_pairs(n_pairs=n_pairs, seed=seed)
        except Exception:
            continue
        for i, (resume, _jd) in enumerate(pairs):
            r = _trim(resume)
            lines = [r.name, r.headline or "", r.summary or "", *r.skills]
            for w in r.work:
                lines += [w.company, w.title, *w.bullets]
            for e in r.education:
                lines += [e.institution, e.degree or "", e.field or ""]
            rebuilt[f"corpus-{seed}-{i}"] = "\n".join(ln for ln in lines if ln and ln.strip())

    if rebuilt:
        sources_cache.write_text(json.dumps(rebuilt, indent=1), encoding="utf-8")
        sources.update(rebuilt)
    return sources


def ablation_calibration(suffix: str = "") -> list[dict]:
    """B — calibrated score vs raw cosine similarity (Phase-4 held-out split)."""
    progress = _load(f"progress{suffix}.json") or {}
    m = progress.get("metrics")
    if not m:
        return []
    return [
        {
            "ablation": "B. scoring",
            "condition": "calibrated (rho)",
            "metric": "MAE / Spearman ρ",
            "value": f"{m['mae']:.2f} / {m['spearman']:.3f}",
        },
        {
            "ablation": "B. scoring",
            "condition": "cosine baseline",
            "metric": "MAE / Spearman ρ",
            "value": f"{m['cosine_mae']:.2f} / {m['cosine_spearman']:.3f}",
        },
    ]


def ablation_gate(suffix: str = "") -> list[dict]:
    """C — rewrite gate on/off (Phase-5 run)."""
    data = _load(f"fabrication_results_corpus{suffix}.json") or _load(
        f"fabrication_results{suffix}.json"
    )
    if not data:
        return []
    return [
        {
            "ablation": "C. rewrite gate",
            "condition": "OFF",
            "metric": "unsourced shipped",
            "value": data["unsourced_shipped_gate_off"],
        },
        {
            "ablation": "C. rewrite gate",
            "condition": "ON",
            "metric": "unsourced shipped",
            "value": data["unsourced_shipped_gate_on"],
        },
        {
            "ablation": "C. rewrite gate",
            "condition": "ON",
            "metric": "mean fabrication_rate detected",
            "value": float(data["mean_fabrication_rate"]),
        },
    ]


def run_ablations(suffix: str = "") -> list[dict]:
    """All three ablations as flat rows for Table 4."""
    return (
        ablation_provenance(suffix) + ablation_calibration(suffix) + ablation_gate(suffix)
    )


if __name__ == "__main__":
    rows = run_ablations()
    if not rows:
        raise SystemExit(
            "no ablation inputs found. Run eval.fit_calibrator and "
            "eval.fabrication_ablation first."
        )
    for r in rows:
        print(f"{r['ablation']:24} {r['condition']:28} {r['metric']:40} {r['value']}")
