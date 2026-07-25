"""Run the pipeline from an already-structured (edited) résumé.

The graph in `rho.graph` starts from file bytes: ingest -> extract -> ... The
editor sends back a résumé the user has already parsed and edited, so this entry
skips ingest/extract and composes the remaining stages (jd, match, score,
rewrite, review) directly from the same node functions the graph uses.

Provenance is rebuilt from the edited résumé's own values via the real ingest
path (the technique `eval/fabrication_corpus.py` uses), so the C3 gate still has
spans to verify against. Consequence, surfaced in the UI: "sourced" now means
"traces to the current résumé", not "traces to the original upload".

LLM legs use the Gemini backend explicitly: `analyze_jd`/`rewrite` default to
CUDA-only backends otherwise.
"""

from rho.graph import nodes as N
from rho.ingestion import ingest
from rho.jd import analyze_jd
from rho.models.api import PipelineResponse
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.rewrite import rewrite


def build_prov_from_resume(resume: StructuredResume) -> ProvenanceMap:
    """ProvenanceMap over the résumé's own values, one span per line."""
    lines = [resume.name, resume.headline or "", resume.summary or "", *resume.skills, *resume.certifications, *resume.achievements]
    for w in resume.work:
        lines += [w.company, w.title, w.start_date or "", w.end_date or "", *w.bullets]
    for e in resume.education:
        lines += [e.institution, e.degree or "", e.field or "", e.end_year or ""]
    for p in resume.projects:
        lines += [p.name, p.url or "", *p.tech, *p.bullets]
    doc = "\n".join(ln.strip() for ln in lines if ln and ln.strip())
    _, prov = ingest(doc.encode(), "edited.txt")
    return prov


def _gemini_jd_fn():
    from rho.jd.gemini import analyze_jd_schema_gemini
    return analyze_jd_schema_gemini


def _gemini_rewrite_fn():
    from rho.rewrite.gemini import rewrite_schema_gemini
    return rewrite_schema_gemini


def run_from_structured(
    resume: StructuredResume,
    jd_text: str,
    *,
    jd_fn=None,
    rewrite_fn=None,
    on_stage=None,
) -> PipelineResponse:
    """Score and tailor an edited résumé against `jd_text`.

    `on_stage(name)` is called before each stage so a caller (the job worker)
    can report progress. `jd_fn`/`rewrite_fn` override the Gemini defaults in
    tests.
    """
    def stage(name):
        if on_stage:
            on_stage(name)

    prov = build_prov_from_resume(resume)

    stage("analyzing_jd")
    reqs = analyze_jd(jd_text, _schema_fn=jd_fn or _gemini_jd_fn())

    stage("matching")
    state = {"resume": resume, "reqs": reqs, "prov": prov}
    state.update(N.match_node(state))

    stage("scoring")
    state.update(N.score_node(state))

    stage("rewriting")
    tailored = rewrite(resume, state["match_result"].gaps, prov, _rewrite_fn=rewrite_fn or _gemini_rewrite_fn())
    state["tailored"] = tailored

    stage("reviewing")
    state.update(N.review_node(state))

    return PipelineResponse(
        structured_resume=resume,
        provenance_map=prov,
        match_result=state["match_result"],
        tailored_resume=tailored,
        final_score=state["final_score"],
    )


_COMPONENT_LABELS = [
    ("keyword_coverage", "Keyword match"),
    ("must_have_coverage", "Must-have coverage"),
    ("semantic_similarity", "Semantic match"),
    ("nice_have_coverage", "Nice-to-have coverage"),
]


def run_optimize(
    resume: StructuredResume,
    jd_text: str,
    *,
    jd_fn=None,
    rewrite_fn=None,
    on_stage=None,
) -> "OptimizeResult":
    """Full optimise result: baseline score, tailored score, and the component
    before/after breakdown the UI shows.

    One pass: the original résumé is matched+scored (baseline), rewritten, then
    the tailored résumé is matched+scored (final). Both scores are rescaled to a
    readable 0-100 via `to_display_score`; the raw calibrated values are kept as
    `final_score`/`baseline_score` for reproducibility.
    """
    from rho.ats.display import to_display_score
    from rho.models.api import OptimizeResult, ScoreComponent

    def stage(name):
        if on_stage:
            on_stage(name)

    prov = build_prov_from_resume(resume)

    stage("analyzing_jd")
    reqs = analyze_jd(jd_text, _schema_fn=jd_fn or _gemini_jd_fn())

    stage("matching")
    base_state = {"resume": resume, "reqs": reqs, "prov": prov}
    base_state.update(N.match_node(base_state))
    stage("scoring")
    base_state.update(N.score_node(base_state))
    baseline_mr = base_state["match_result"]

    stage("rewriting")
    tailored = rewrite(resume, baseline_mr.gaps, prov, _rewrite_fn=rewrite_fn or _gemini_rewrite_fn())

    stage("reviewing")
    tail_state = {"resume": tailored.resume, "reqs": reqs, "prov": prov}
    tail_state.update(N.match_node(tail_state))
    tail_state.update(N.score_node(tail_state))
    tail_state["tailored"] = tailored
    tail_state.update(N.review_node(tail_state))
    tailored_mr = tail_state["match_result"]

    before_cv = baseline_mr.component_vector
    after_cv = tailored_mr.component_vector
    components = [
        ScoreComponent(label=label, before=getattr(before_cv, attr), after=getattr(after_cv, attr))
        for attr, label in _COMPONENT_LABELS
    ]

    return OptimizeResult(
        match_result=tailored_mr,
        tailored_resume=tailored,
        final_score=tailored_mr.predicted_score,
        display_score=to_display_score(tailored_mr.predicted_score),
        baseline_score=baseline_mr.predicted_score,
        baseline_display_score=to_display_score(baseline_mr.predicted_score),
        components=components,
    )
