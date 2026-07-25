"""Corpus-backed pairs for the fabrication benchmark (C3, option 2).

The synthetic benchmark in `tests/fixtures/fabrication/pairs.json` is 12 curated
résumés carrying skills only, so the gate's work/education/bullet paths never
face real generated text. This module draws instead from the Phase-4 corpus
(`Resume.csv` × `training_data.csv`), giving résumés with populated work history,
bullets, and education — and far more of them.

The one thing the corpus does not give is provenance. `Resume_str` is stored as a
single line with multi-space runs standing in for the original line breaks, so
`ingest()` produces one span covering the whole document. That span "supports"
almost any value by substring match, which would hollow out the gate and break
C1's claim to an *exact* source location. `segment_corpus_text` restores the line
structure first; provenance is then built by the real ingest path over the
segmented text, so offsets stay honest.
"""

import os
import re

# Must precede any torch import: sentence-transformers otherwise sizes its
# intra-op pool to all 16 cores per calling thread and thrashes.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rho.ingestion import ingest  # noqa: E402
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume

# The corpus flattens line breaks into runs of 2+ spaces. `eval.corpus` splits on
# the same signal to recover bullets, so the two views of the document agree.
_SEGMENT_BREAK = re.compile(r"\s{2,}")


def segment_corpus_text(text: str) -> str:
    """Restore the line structure the corpus flattened into multi-space runs."""
    return "\n".join(seg.strip() for seg in _SEGMENT_BREAK.split(text) if seg.strip())


def corpus_prov(text: str, doc_id: str) -> ProvenanceMap:
    """ProvenanceMap over `text`, one span per recovered line.

    Offsets index the *segmented* text (what `segment_corpus_text` returns), not
    the raw CSV cell, so `raw_text` and the char range stay consistent.
    """
    _, prov = ingest(segment_corpus_text(text).encode(), f"{doc_id}.txt")
    return prov


def _jd_analyzer(backend: str):
    """Resolve the JD-analysis function for `backend` ("ollama", "groq", "gemini")."""
    if backend == "groq":
        from rho.jd.groq import analyze_jd_schema_groq

        return analyze_jd_schema_groq
    if backend == "gemini":
        from rho.jd.gemini import analyze_jd_schema_gemini

        return analyze_jd_schema_gemini
    # Ollama path: reuse the frozen analyze_jd contract with the Ollama schema fn.
    from rho.jd import analyze_jd
    from rho.jd.ollama import analyze_jd_schema as _ollama_schema_fn

    return lambda jd: analyze_jd(jd, _schema_fn=_ollama_schema_fn)


def build_corpus_pairs(
    n_pairs: int = 30,
    seed: int = 0,
    resume_csv: str = "Resume.csv",
    jd_csv: str = "training_data.csv",
    workers: int = 5,
    backend: str = "ollama",
) -> list[dict]:
    """Corpus résumé × JD pairs shaped for `eval.fabrication_ablation.run`.

    Gaps come from the real Phase-3 path — `analyze_jd` then `match` — rather
    than a hand-written "tempting" list, so the pressure on the rewriter is
    whatever the JD actually demands and the résumé actually lacks.
    """
    from concurrent.futures import ThreadPoolExecutor

    from eval.corpus import build_pairs
    from rho.matching import match

    analyze = _jd_analyzer(backend)
    # Ollama runs one CPU model; concurrent calls thrash. Groq/Gemini are network-bound.
    jd_workers = workers if backend in ("groq", "gemini") else 1

    raw_pairs = [
        (_trim(resume), jd_text)
        for resume, jd_text in build_pairs(
            n_pairs=n_pairs, seed=seed, resume_csv=resume_csv, jd_csv=jd_csv
        )
    ]

    failures: list[str] = []

    def _analyze(indexed):
        """Run JD analysis for one pair (backend chosen above)."""
        i, (_, jd_text) = indexed
        try:
            return i, analyze(jd_text)
        except Exception as exc:
            # Never drop silently: a swallowed 429 would shrink the benchmark
            # without saying so, which reads as a clean run on fewer pairs.
            failures.append(f"corpus-{seed}-{i}: {type(exc).__name__}: {exc}")
            return i, None

    # Phase 1 — JD analysis is a network round-trip per pair; keys round-robin
    # underneath and `TokenBudget` paces the account-wide TPM cap.
    with ThreadPoolExecutor(max_workers=jd_workers) as pool:
        analyzed = dict(pool.map(_analyze, enumerate(raw_pairs)))

    # Phase 2 — `match()` runs sentence-transformers on CPU. Torch spawns its own
    # intra-op pool per caller, so calling it from N worker threads oversubscribes
    # the machine (measured: 98 threads on 16 cores, all futex-blocked, throughput
    # near zero). Embedding is therefore done serially, on the main thread.
    kept = []
    for i, (resume, jd_text) in enumerate(raw_pairs):
        reqs = analyzed.get(i)
        if reqs is None:
            continue
        gaps = [g for g in match(resume, reqs).gaps if g.status != "present"]
        kept.append(
            {
                "id": f"corpus-{seed}-{i}",
                "resume": resume,
                "prov": _prov_for(resume, i),
                "jd": jd_text,
                "gaps": gaps,
                "tempting_absent": [g.requirement.text for g in gaps],
            }
        )

    if failures:
        print(
            f"  WARNING: {len(failures)}/{len(raw_pairs)} pairs dropped during JD "
            f"analysis. First: {failures[0][:160]}"
        )
    if not kept:
        raise RuntimeError(
            f"JD analysis failed for all {len(raw_pairs)} pairs — benchmark is "
            f"empty, not clean. First failure: {failures[0][:200] if failures else 'n/a'}"
        )
    return kept


# Groq's free tier caps *tokens per minute* (8k), not just requests, and a full
# corpus résumé plus its JD runs ~2.5k tokens — five parallel workers exhaust the
# budget immediately. Trimming keeps every field type represented (so the gate's
# work/education/bullet paths still get exercised) at a third of the payload.
_MAX_SKILLS = 12
_MAX_BULLETS = 6
_MAX_BULLET_CHARS = 220


def _trim(resume: StructuredResume) -> StructuredResume:
    """Shrink a corpus résumé to fit the token budget, keeping all field types.

    Applied before provenance is built, so `_prov_for` sees exactly the values
    that survive — a value trimmed away is simply not in the source document, and
    the gate stays consistent with what it is shown.
    """
    trimmed = resume.model_copy(deep=True)
    trimmed.skills = trimmed.skills[:_MAX_SKILLS]
    trimmed.summary = (trimmed.summary or "")[:400] or None
    for work in trimmed.work:
        work.bullets = [b[:_MAX_BULLET_CHARS] for b in work.bullets[:_MAX_BULLETS]]
    for edu in trimmed.education:
        edu.institution = edu.institution[:200]
    return trimmed


def _prov_for(resume: StructuredResume, idx: int) -> ProvenanceMap:
    """Provenance over the résumé's own values.

    `build_pairs` hands back a parsed `StructuredResume`, not the raw cell, so
    the source document is reconstructed from the values it kept. Anything the
    parse dropped is genuinely absent from the source as far as the gate is
    concerned — which is the conservative direction: the gate can only be
    stricter than reality, never more permissive.
    """
    lines = [resume.name, resume.headline or "", resume.summary or ""]
    lines += resume.skills
    for w in resume.work:
        lines += [w.company, w.title, *w.bullets]
    for e in resume.education:
        lines += [e.institution, e.degree or "", e.field or ""]
    doc = "\n".join(ln.strip() for ln in lines if ln and ln.strip())
    _, prov = ingest(doc.encode(), f"corpus{idx}.txt")
    return prov
