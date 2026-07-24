# Résumé Optimization Using Generative AI: A Provenance-Verified Rewrite Gate

**Author.** [Name], [Affiliation], [email]

---

## Abstract

Large language models make résumé tailoring trivial to prototype and dangerous to deploy: a rewriter that invents a skill, an employer, or a metric produces a document that scores better and lies about the candidate. We argue that this is not a prompt-engineering problem but a structural one, because the optimization objective and the failure mode point in the same direction — an invented keyword is precisely the keyword the job description asked for, so any match-based metric ranks a fabricating system above an honest one. We present a résumé optimization pipeline built around a *provenance chain*: every value extracted from a source document carries a stable identifier pointing at the exact span that supports it, and that identifier survives through matching, scoring, and rewriting. A deterministic, LLM-free **verification gate** then checks every value the rewriter proposes against that chain and deletes whatever cannot be traced. On 30 résumé/job-description pairs, prompt grounding alone shipped 40 unsourced claims (33% of documents affected; 44% of proposed edits unsupported); with the gate enabled, zero unsourced claims reached the output, a result reproduced on a second, independently hosted model. We report the gate's cost as well as its benefit, and argue that the central finding is not the zero — which is true by construction — but the measured inadequacy of prompt grounding that motivates it.

**Keywords:** generative AI, résumé optimization, hallucination mitigation, provenance, applicant tracking systems, verified generation.

---

## 1. Introduction

Résumé optimization — taking a candidate's résumé and a target job description, producing a tailored version plus a predicted match score — is now routinely built as an LLM pipeline. Document parsing has largely shifted from named-entity recognition to LLM extraction under schema-constrained decoding [1, 4, 5], which buys reliability on the *structure* of the output.

It does not buy reliability on the *content*. A rewriting stage asked to "tailor" a résumé toward a set of requirements is exactly the setting in which language models fabricate: invented metrics, employers that never existed, skills the candidate never claimed. In a hiring context this is not cosmetic. A résumé fabricated by the tool the candidate trusted misrepresents them to an employer, with consequences they did not choose and may not be aware of.

The core difficulty is that the obvious objective is adversarial to truthfulness. If a résumé lacks *Kubernetes* and the posting demands it, the single most effective way to raise any keyword- or embedding-based match score is to add *Kubernetes*. **Optimizing for match score actively rewards fabrication.** Evaluation that measures only score improvement therefore cannot distinguish a good system from a dishonest one.

We take the position that truthfulness here has a specific, checkable shape rather than being a prompt-engineering aspiration. Concretely: *every hard-content token in a rewritten résumé — skill, tool, organization, number, date — must trace to at least one span in the original document that supports it.* We call this the **provenance invariant**. A rewrite that cannot demonstrate the invariant for a value does not ship that value.

**Contributions.**

- **C1.** A continuous provenance chain established at ingestion and threaded unbroken into every downstream representation of the résumé.
- **C2.** A match score calibrated against a real, rule-based ATS engine rather than reported as a raw embedding-similarity proxy.
- **C3.** A deterministic, LLM-free rewrite gate that rejects unsupported values before they ship, with a fabrication rate reported as a first-class metric rather than assumed away.

---

## 2. Related Work

**Layout-aware extraction.** Digital-native PDF parsers scramble reading order on multi-column layouts, the largest reported cause of downstream extraction error. Docling [1] unifies PDF, DOCX and image inputs into a single representation preserving reading order and table structure; specialised layout models [2] and OCR-free transformers [3] represent a heavier alternative better suited to bespoke extractors than to systems built on general LLMs.

**Constrained decoding.** Three reliability tiers are commonly distinguished: prompt-only JSON requests (unreliable), function-calling or JSON mode (syntactically valid, not schema-guaranteed), and constrained decoding, which masks invalid tokens during generation. Outlines [4] and vLLM [5] implement this for self-hosted models; hosted APIs increasingly expose an equivalent schema mechanism. Constrained decoding guarantees output *shape*, never output *truth* — a distinction central to this work.

**ATS matching.** Applicant tracking systems are typically a parser stage followed by a matcher combining keyword coverage with semantic similarity [6]. Systems reporting a match score from raw cosine similarity conflate two different quantities: how similar two embeddings are, and how an actual ATS would score the pair. We calibrate against the latter directly.

**Fabrication in LLM rewriting.** Résumé rewriters without an explicit truthfulness mechanism exhibit documented failure modes: invented metrics, fabricated employers and titles, temporal fabrication, and cross-domain contamination between unrelated sections. Prompt grounding measurably reduces but does not eliminate this, as our gate-OFF measurements confirm (§5). This is our central argument for a deterministic verification gate rather than a stronger prompt.

---

## 3. System Architecture

The system is a seven-node directed graph [7] with two parallel branches that fan in at matching.

```
                    ┌──────────┐      ┌──────────┐
      ┌────────────▶│ extract  │─────▶│          │
      │             │  (LLM)   │      │          │
┌─────┴────┐        └──────────┘      │  match   │
│  ingest  │                          │(fan-in ∥ │
│ Docling  │                          │ barrier) │
└─────┬────┘        ┌──────────┐      │          │
      │             │    jd    │      │          │
      └────────────▶│  (LLM)   │─────▶│          │
                    └──────────┘      └─────┬────┘
                                            │
                                            ▼
   ┌─────────┐    ┌───────────────┐    ┌─────────┐
   │ review  │◀───│   rewrite     │◀───│  score  │
   │invariant│    │ (LLM) + GATE  │    │  Ridge  │
   │ re-check│    └───────────────┘    └─────────┘
   └─────────┘
        │
        ▼
   tailored résumé + fabrication report + calibrated score

   ═══ deterministic ═══   ┄┄┄ stochastic ┄┄┄
   ingest, match, score,   extract, jd,
   review, GATE            rewrite
```

**Figure 1.** The pipeline. Résumé parsing and job-description analysis are independent and overlap in wall-clock time; `match` is a deferred fan-in barrier rather than a first-input-wins trigger, so it never observes a partially populated state. Every stochastic stage is immediately followed by a deterministic stage that checks it.

**Provenance chain (C1).** At ingestion, each text item becomes a `SourceSpan` recording its document, character offsets into the exported markdown, page, bounding box, and raw text. Every value-bearing field produced by extraction carries a sibling `*_prov` list of supporting span identifiers. This is attached *during* extraction, not reconstructed afterward, and is what the gate later checks against.

**Calibrated scoring (C2).** The matcher emits a five-dimensional feature vector — keyword coverage, semantic similarity, fuzzy coverage, must-have coverage, nice-to-have coverage — computed over the whole résumé rather than the skills list alone, since requirement evidence usually sits in an experience bullet. Rather than reporting a hand-weighted function of this vector, a Ridge regression is fit against the output of a self-hostable rule-based ATS simulator [8]. The regression target is the simulator's *job-description-dependent* dimension only: its composite score is dominated by résumé-intrinsic signals (formatting, section completeness) that do not vary with the posting, and calibrating against the composite yields a target our match features correlate negatively with.

---

## 4. The Verification Gate (C3)

The gate is the paper's central mechanism. It is **subtractive** (it can only delete, never rewrite, so it cannot itself introduce content), **deterministic** (pure string matching and regular expressions — no model, no randomness), and **post-hoc** (it inspects finished text, so it is independent of which model produced it).

The rewriter is prompted with the source résumé as sole source of truth and instructed to reorder, rephrase and emphasise but never invent, and to leave a requirement honestly unsatisfied rather than fabricate coverage. **This prompt is deliberately not the safety mechanism** — §5 measures how often it fails.

The gate distinguishes two content classes, which fail differently.

**Short factual values** (skills, tools, titles, employers, certifications). Two conditions apply in sequence. First, candidate evidence is located by fuzzy substring search over all spans (`partial_ratio ≥ 90`), tolerant enough to survive formatting noise — a bullet glyph or a line break mid-phrase — that exact matching would reject as fabrication. Second, and critically, *locating a span is necessary but not sufficient*: because substring scoring returns the best-matching window, the span `"Engineer"` scores 100 against the fabricated promotion `"Staff Engineer"`. Every content word of an added value must therefore appear in the supporting span (`≥ 88`), or the unmatched words are unsourced claims.

**Bullets** (prose). Rephrasing is legitimate here, so exact tests are inappropriate. Bullets are compared whole-string against source bullets using order-independent token-set overlap. Similarity alone is insufficient, however: a long invention that recycles the source's vocabulary scores well — *"Led a team of 40 Teradata engineers"*. A bullet is therefore accepted only if it (i) closely tracks some source bullet **and** (ii) introduces no unsourced *hard-content token*, where hard content is capitalised terms mid-sentence, all-caps acronyms, and digit runs. Prose may change freely; facts may not appear from nowhere. Inflections are collapsed by a crude stemmer so that ordinary rewording (*optimising* → *optimised*) does not read as a new claim.

**Threshold derivation.** The bullet threshold was measured rather than chosen: on corpus rewrites, genuine rephrasings scored 68–87 and inventions 37–42, leaving an empty band. A threshold of 90 rejected every genuine rephrasing; 60 sits in the gap. Notably, none of the gate's thresholds are cosine similarities — the gate is deliberately embedding-free, since determinism is what makes the reported fabrication rate a reproducible measurement rather than one model's opinion of another.

**Independent re-check.** A final node re-verifies the *shipped* résumé from scratch against the provenance map, duplicating the gate on purpose. If it fires, the gate leaked. It reports the violation rather than raising, so a leak is surfaced and logged rather than silently swallowed.

---

## 5. Results

**Table 1.** Fabrication gate, 30 corpus-backed pairs. "Gate OFF" is the rewriter's raw output under prompt grounding alone.

| Backend (rewriter) | Pairs | Gate OFF | Gate ON | Mean fab. rate | Docs affected |
|---|---|---|---|---|---|
| gemini-3.1-flash-lite | 30/30 | 40 | **0** | 0.248 | 10/30 |
| qwen2.5:14b (local) | 30/30 | 31 | **0** | 0.407 | 14/30 |

Expressed as accuracy: gate-OFF, 66.7% of documents shipped completely clean and 44% of individual proposed edits (40 of 72) were unsourced. Gate-ON, both figures are 100% and 0% respectively. The 32 supported edits still shipped — the gate removed the unsupported subset rather than suppressing rewriting.

**Table 2.** Extraction quality and calibration, primary backend.

| Metric | Value |
|---|---|
| Skills F1 (public human-annotated, n=143) | 0.753 |
| Job title / institution F1 | 0.935 / 0.917 |
| Provenance-attachment accuracy | 0.874 |
| Calibrated score MAE / Spearman ρ | 3.25 / 0.333 |
| Cosine baseline MAE / ρ | 27.58 / 0.229 |

**Ablation.** Replacing span-resolved provenance with the cheaper substitute a provenance-free system would use — a case-insensitive substring search anywhere in the source text — catches 36 of 40 rejections but **misses 4**: additions appearing somewhere in the document yet unsupported where the rewriter used them. Those four are the measured value of resolving to an actual span.

**Interpretation.** Gate-ON = 0 is true *by construction* and is not offered as an empirical finding; the gate's mechanism is deleting exactly those claims. The informative result is gate-OFF: an explicitly grounded prompt, instructing the model that "an honest gap is the correct output," still shipped 40 unsourced claims. That the guarantee holds identically on a second, independently hosted model supports the claim that it is a property of the deterministic gate rather than of one model's generation style.

---

## 6. Limitations

The scorer is also the evaluator: improvement is measured against a calibrator this work fitted, and ρ = 0.333 is a modest predictor, not a faithful ATS simulation. Guarantees inherit provenance accuracy — at 0.874, roughly one span in eight is imperfect, so a *truthful* claim whose span was mis-attached is wrongly rejected; this false-positive rate is the gate's real cost to a user and is currently unmeasured. The gate checks that nothing is invented but not that nothing is silently *dropped*; omission is enforced only by prompt instruction. The benchmark is 30 pairs — sufficient to demonstrate the effect, insufficient for a confidence interval on the mean fabrication rate. Semantic bands used for requirement classification are provisional defaults, never swept against a labelled match set. Finally, the hard-token heuristic treats Latin capitalisation as a factual-content signal and would not transfer to a caseless script.

---

## 7. Conclusion

We presented a résumé optimization system in which a provenance identifier space survives unbroken from ingestion through rewriting, a match score is calibrated against a real ATS engine rather than an embedding proxy, and a deterministic gate rejects any value the provenance chain cannot support. The measured result is that prompt grounding alone is inadequate — a third of documents shipped at least one fabricated claim — while a subtractive, model-independent gate reduces this to zero on two backends. We note that truthful tailoring has a low ceiling set by what the candidate has actually done, and argue that making that ceiling visible is more useful than optimizing past it.

**Future work.** Retained gain (what fraction of ungated improvement survives gating); the gate's false-positive rate; an adversarial rewriter explicitly instructed to evade the gate; and human evaluation against recruiter judgement.

---

## References

[1] Docling: An Efficient Open-Source Toolkit for PDF Document Conversion. 2024.
[2] Huang et al. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. 2022.
[3] Kim et al. OCR-free Document Understanding Transformer (Donut). ECCV 2022.
[4] Willard & Louf. Efficient Guided Generation for Large Language Models (Outlines). 2023.
[5] Kwon et al. Efficient Memory Management for LLM Serving with PagedAttention (vLLM). SOSP 2023.
[6] Reimers & Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv:1908.10084, 2019.
[7] LangGraph: Building Stateful Multi-Actor Applications with LLMs. 2024.
[8] ats-screener: rule-based applicant tracking system simulator. Pinned commit 4105f77.

---

*Draft — §5 figures are reproduced from `eval/RESULTS.md`. The coverage function was modified after the shipped calibrator was fitted; the calibrator must be refit and ablations re-run before these figures are submitted.*
