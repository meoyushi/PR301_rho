"""Build the calibration dataset, fit, and report held-out metrics (C2).

Reports calibrated MAE + Spearman ρ against the cosine-similarity baseline,
which is the ablation the paper needs: does calibrating against real engine
output beat using raw cosine as the score?
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from eval.corpus import build_pairs
from rho.ats import Calibrator, harvest_ats
from rho.ats.aggregate import to_match_target, to_target
from rho.ats.dataset import build_calibration_dataset
from rho.jd import analyze_jd
from rho.matching import match

# JD analysis runs through an LLM path (temperature 0). The KeyBERT fallback
# tried earlier extracted company names and boilerplate ("nashville office")
# rather than requirements, leaving keyword_coverage and fuzzy_coverage
# identically 0.0 across every pair.


def _jd_schema_fn(backend: str):
    """Resolve the JD-analysis schema function for `backend`."""
    if backend == "gemini":
        from rho.jd.gemini import analyze_jd_schema_gemini

        return analyze_jd_schema_gemini
    if backend == "groq":
        from rho.jd.groq import analyze_jd_schema_groq

        return analyze_jd_schema_groq
    from rho.jd.ollama import analyze_jd_schema as _ollama_schema_fn

    return _ollama_schema_fn


def make_feature_fn(backend: str = "ollama"):
    """resume+jd -> ComponentVector (rho's own raw signals, pre-calibration).

    Backend functions differ in what they hand back: Ollama's returns a raw
    `JDSchema` meant to be converted via `analyze_jd(_schema_fn=...)`, while
    Groq's and Gemini's call sites already do that conversion internally and
    return a `RequirementSet` directly — so only the Ollama branch routes
    through `analyze_jd`.
    """
    if backend == "gemini" or backend == "groq":
        schema_fn = _jd_schema_fn(backend)

        def feature_fn(resume, jd_text):
            return match(resume, schema_fn(jd_text)).component_vector

        return feature_fn

    schema_fn = _jd_schema_fn(backend)

    def feature_fn(resume, jd_text):
        return match(resume, analyze_jd(jd_text, _schema_fn=schema_fn)).component_vector

    return feature_fn


PROGRESS_PATH = "eval/progress.json"


def _progress_writer(path: str, started: float):
    """Write run progress to `path` after every pair so a viewer can poll it."""

    def write(index: int, total: int, status: str, kept: int) -> None:
        elapsed = time.time() - started
        rate = elapsed / index if index else 0.0
        payload = {
            "index": index,
            "total": total,
            "kept": kept,
            "skipped": index - kept,
            "last_status": status,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_pair": round(rate, 1),
            "eta_seconds": round(rate * (total - index), 1),
            "done": index >= total,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, path)  # atomic: a poller never reads a half-written file

    return write


def _fit_and_score(X, y, seed: int, out: str | None) -> dict:
    """Fit a calibrator on one target definition and score it held-out."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed)
    cal = Calibrator()
    cal.fit(Xtr, ytr)

    preds = np.array([cal.predict(x) for x in Xte])
    yte_arr = np.array(yte)

    # Ablation: raw cosine (semantic_similarity * 100) as the score.
    cos = np.array([x.semantic_similarity * 100 for x in Xte])

    if out:
        cal.save(out)
    return {
        "n_train": len(Xtr),
        "n_heldout": len(Xte),
        "mae": float(np.mean(np.abs(preds - yte_arr))),
        "spearman": float(spearmanr(preds, yte_arr).statistic),
        "cosine_mae": float(np.mean(np.abs(cos - yte_arr))),
        "cosine_spearman": float(spearmanr(cos, yte_arr).statistic),
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
    }


def main(
    n_pairs: int = 200,
    seed: int = 0,
    out: str = "eval/calibrator.joblib",
    progress_path: str = PROGRESS_PATH,
    backend: str = "ollama",
) -> dict:
    pairs = build_pairs(n_pairs=n_pairs, seed=seed)
    feature_fn = make_feature_fn(backend)

    # Harvest once, derive both targets from the same engine outputs so the two
    # calibrations are directly comparable on identical features and pairs.
    # `both_targets` rides along on the kept rows only, so it stays aligned with
    # X even when a pair is dropped for failing featurisation.
    def both_targets(engine_outputs: dict) -> tuple[float, float]:
        return to_target(engine_outputs), to_match_target(engine_outputs)

    X, paired = build_calibration_dataset(
        pairs,
        harvest_ats,
        feature_fn,
        target_fn=both_targets,
        on_progress=_progress_writer(progress_path, time.time()),
    )
    if len(X) < 10:
        raise SystemExit(f"only {len(X)} usable pairs; need more data to fit")

    y_overall = [t[0] for t in paired]
    y_keyword = [t[1] for t in paired]

    # Primary (headline C2): the JD-dependent dimension. rho's features are
    # résumé-vs-JD match signals, so the match dimension is the target they can
    # actually predict — on overallScore they correlate negatively because the
    # composite is ~80% résumé-intrinsic quality.
    primary = _fit_and_score(X, y_keyword, seed, out)
    # Secondary ablation: the composite overallScore. Reported to show *why*
    # keywordMatch is the right target, not as a competing headline.
    overall = _fit_and_score(X, y_overall, seed, None)

    metrics = {
        "backend": backend,
        "n_pairs_requested": n_pairs,
        "n_usable": len(X),
        "target": "keywordMatch",
        **primary,
        "overall_target": overall,
    }
    print(json.dumps(metrics, indent=2))
    print(
        f"\nkeywordMatch target (PRIMARY): calibrated MAE={primary['mae']:.2f} "
        f"Spearman={primary['spearman']:.3f} | cosine MAE={primary['cosine_mae']:.2f} "
        f"Spearman={primary['cosine_spearman']:.3f}  (y mean {primary['y_mean']:.1f})"
    )
    print(
        f"overallScore target (ablation): calibrated MAE={overall['mae']:.2f} "
        f"Spearman={overall['spearman']:.3f} | cosine MAE={overall['cosine_mae']:.2f} "
        f"Spearman={overall['cosine_spearman']:.3f}  (y mean {overall['y_mean']:.1f})"
    )

    # Fold the results into the progress file so a viewer shows them on finish.
    try:
        with open(progress_path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload.update({"done": True, "metrics": metrics})
    with open(progress_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="eval/calibrator.joblib")
    p.add_argument("--progress-path", default=PROGRESS_PATH)
    p.add_argument("--backend", default="ollama", choices=["ollama", "groq", "gemini"])
    a = p.parse_args()
    main(
        n_pairs=a.n_pairs,
        seed=a.seed,
        out=a.out,
        progress_path=a.progress_path,
        backend=a.backend,
    )
