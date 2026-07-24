# Building a Resume Parsing & ATS-Optimization System with GenAI / Agentic AI: A Technical Blueprint

## TL;DR
- **Build a hybrid, two-stage pipeline, not a single monolith.** Stage 1: convert documents to clean text/markdown with a layout-aware parser (Docling or a cloud OCR), then extract structured JSON with a strong LLM under *constrained decoding / schema-forced output* (Instructor + Pydantic for APIs; Outlines/vLLM for self-hosted). Stage 2: run a deterministic matching layer (embeddings + keyword coverage) plus an LLM rewriter, orchestrated as a small multi-step agent graph (LangGraph).
- **Prompting a strong general LLM beats training a custom model for most builders** — GPT-4o/Claude/Gemini hit ~0.95 field-level F1 on resume→JSON out of the box. A custom fine-tuned model is only worth it at high volume: a fine-tuned Qwen3-0.6B matched/beat frontier models on real resumes (F1 0.964, surpassing Claude-4's 0.959) at 3–4× lower latency, but that requires a labeled dataset and MLOps.
- **ATS reality should drive design:** modern ATS run a *parser* (structured fields) then a *matcher* (keyword + increasingly semantic). Optimize for truthful keyword coverage in the right sections, single-column parse-safe formatting, and a 0–100 match score with concrete gap suggestions — never fabrication.

## Key Findings

1. **Resume parsing is now an LLM extraction problem, not an NER problem.** The dominant 2025–2026 architecture is: (a) text/layout extraction → (b) LLM with a strict JSON schema → (c) Pydantic/JSON-Schema validation → (d) human-in-the-loop fallback for failures. Traditional NER pipelines (spaCy, pyresparser) are now baselines/fallbacks, not the core.
2. **Constrained/structured decoding is the single biggest reliability lever.** Prompt-only "return JSON" fails a meaningful fraction of the time in production; native structured outputs and constrained decoding guarantee schema-valid JSON.
3. **Layout handling matters most for messy/multi-column/scanned resumes.** Reading-order scrambling is the #1 cause of downstream errors — the same failure real ATS parsers exhibit.
4. **A custom trained model is not required.** No high-quality general open-source resume parser dominates; the best path for a builder is prompting a strong LLM, with optional fine-tuning of a small model only at scale.
5. **ATS matching = keyword coverage + semantic similarity + knockout filters.** Best implementation blends deterministic scoring (embeddings + fuzzy keyword match) with LLM reasoning for gap analysis and rewriting.
6. **Truthfulness is an engineering requirement, not a disclaimer.** LLM resume rewriting has documented, dangerous failure modes (temporal fabrication, cross-domain contamination, invented metrics) that must be designed out with grounding + validation.

## Details

### STAGE 1 — Resume → Structured JSON

#### 1.1 Document/text extraction & layout handling
The pipeline's first job is turning PDF/DOCX/image into clean, reading-order-correct text or markdown. Options, roughly in order of build-effort:

- **Digital PDFs / DOCX (fast path):** `PyMuPDF` (fitz) and `pdfplumber` for PDFs; `python-docx` for Word. Cheap and instant, but they scramble multi-column layouts and drop reading order — exactly the failure that breaks downstream extraction.
- **Docling (IBM Research, now Linux Foundation LF AI & Data)** — the strongest open-source default in 2025–2026. It parses PDF, DOCX, PPTX, XLSX, HTML and images into a unified `DoclingDocument` with layout, reading order, and table structure, exporting to Markdown/JSON/HTML/DocTags. It runs locally (good for PII), has native LangChain/LlamaIndex/CrewAI/Haystack integrations and an MCP server, and crossed ~37k+ GitHub stars. Setup cost: downloads 1–2 GB of model weights on first run. Its VLM, **Granite-Docling-258M** (Apache 2.0), does one-shot image→DocTags. IBM reports its layout model was trained on ~81,000 manually labeled pages (patents, manuals, 10-K filings) and came within five percentage points of human accuracy on page-element identification.
- **Cloud OCR / Document AI:** AWS Textract, Google Document AI, Google Cloud Vision (~94% accuracy on complex layouts in one arXiv benchmark, vs ~91% for DocTR and ~85% for Tesseract v4). Best raw accuracy on degraded scans and complex tables; costs per page and sends data off-prem.
- **`unstructured.io`** — good general-purpose preprocessor that emits semantically labeled elements to drive chunking.
- **Layout-aware ML models:** **LayoutLMv3** (needs upstream OCR; 92.1 F1 on FUNSD, 96.6 on CORD, ~35 ms/receipt) and **Donut** (OCR-free, end-to-end but slower ~110 ms and cannot exploit a strong external OCR). These are for teams building a specialized in-house extractor, not most builders.
- **Vision-LLM "attach the image and ask" path:** GPT-4o / Gemini 2.5 / Claude can read a resume image directly and emit JSON in one pass — simplest to build and layout-aware by nature. **Caveat:** vision LLMs are weaker at pure OCR precision than specialized OCR (documented cases of GPT-4o vision returning wrong characters), and slower (GPT-4o ~5–15 s/page, Gemini 2.5 Pro ~10–30 s per Parsli's 2026 benchmark). For scanned/degraded resumes, OCR-then-LLM is more reliable; for clean digital resumes, direct vision or text-then-LLM both work.

**Recommendation:** Route by file type. Digital PDF/DOCX → Docling (or PyMuPDF for speed) → markdown. Scanned/image → cloud OCR or Docling OCR → markdown. Keep a vision-LLM fallback for resumes that fail structural checks.

#### 1.2 Extraction: LLM + schema-forced output (the core)
Feed the clean markdown to an LLM instructed to emit strict JSON conforming to your schema. The critical technique is **structured output enforcement**, which has three tiers:

- **Prompt-only** ("return JSON with these fields") — unreliable; fails a meaningful fraction of production calls, especially on deeply nested schemas (4+ levels).
- **JSON mode / function calling** — guarantees syntactically valid JSON and *likely* schema adherence.
- **Constrained decoding (schema-guaranteed)** — masks invalid tokens during generation so output *cannot* violate the schema (100% structural validity), and can even speed generation.

**Libraries (the practical landscape):**
- **Instructor** — most popular (11k+ GitHub stars, 3M+ monthly downloads). Wraps OpenAI/Anthropic/Gemini/Ollama/vLLM with Pydantic models, automatic retries with validation feedback, streaming. Reported ~95–98% success on API providers. *Start here for API-based builds.*
- **Outlines** — constrained decoding via finite-state machines for self-hosted models (transformers/vLLM/llama.cpp); ~99.9% structural validity, no retry cost. *Best for local models.*
- **Pydantic AI** — agent abstraction + typed structured output; clean for multi-step logic.
- **OpenAI native Structured Outputs** — `response_format` with a Pydantic/JSON schema guarantees schema conformance for OpenAI models (supports a subset of JSON Schema; first call per schema has compile latency).
- **BAML** — schema-first DSL with cross-language codegen and forgiving "schema-aligned parsing"; good for polyglot teams.
- **vLLM / SGLang / Ollama** — all support guided/`guided_json` decoding via XGrammar or Outlines for self-hosted schema enforcement.

**Schema design tips that measurably improve accuracy:** put reasoning fields *before* answer fields (LLMs generate left-to-right); make genuinely-optional fields `Optional`/nullable to avoid forced hallucination; add field descriptions; avoid 4+ levels of nesting and 50+ field mega-schemas; wrap lists in a container model for multi-record extraction.

#### 1.3 The JSON schema for resumes
Don't invent from scratch — anchor on the community **JSON Resume** standard (`jsonresume.org`, MIT, `@jsonresume/schema`), which defines `basics`, `work`, `education`, `skills`, `projects`, `certificates`, `awards`, `publications`, `languages`, `volunteer`, `interests`, `references`, `meta`. It uses ISO-8601 dates with optional precision (`2014`, `2014-06`, `2014-06-29`). There's a companion **JSON Job Description schema** (JSONJob) for Stage 2. A JSON-LD variant (`schema-resume.org`) adds Schema.org semantic mapping and XSD.

Represent this as **Pydantic models** so validation is automatic. A minimal, battle-tested shape (seen across production repos):
```python
class ContactInfo(BaseModel):
    location: Optional[str]; phone: Optional[str]
    email: Optional[str]; urls: list[str] = []
class WorkExperience(BaseModel):
    company: str; title: str
    start_date: Optional[str]; end_date: Optional[str]  # ISO-8601, "" = present
    bullets: list[str] = []
class Education(BaseModel):
    institution: str; degree: Optional[str]
    field: Optional[str]; end_year: Optional[str]
class Resume(BaseModel):
    name: str; headline: Optional[str]; summary: Optional[str]
    contact: ContactInfo
    work: list[WorkExperience] = []
    education: list[Education] = []
    skills: list[str] = []
    certifications: list[str] = []
```
Validation stack: **Pydantic v2** for types/constraints; add validators for date-range sanity (start ≤ end), email/phone format, and enum-constrained fields. On validation failure, Instructor/PydanticAI auto-retry with the error message; after N retries, route to a manual-review queue.

**Messy/multi-column handling:** the LLM layer is far more robust than regex/NER here — a strong LLM reconstructs sections even from imperfectly-ordered text. The key is fixing reading order upstream (Docling/OCR) so the skills column doesn't get interleaved into work history.

#### 1.4 Custom-trained models, datasets & open-source parsers
- **Existing open-source parsers:** `pyresparser` (spaCy + NLTK, rule/regex-based — now dated, best as a baseline; its successor migrated to a local Qwen2.5-1.5B), `resume-parser` libraries, and various GitHub repos. Affinda notes there is **no dominant high-quality open-source resume parser** — resume parsing has a "mind-boggling" number of edge cases, which is why commercial APIs (Affinda, Sovren/Textkernel Tx, RChilli, Daxtra — Affinda extracts 100+ fields across 50+ languages with trained ML, *not* LLMs) exist. HuggingFace has **no canonical resume-extraction model**, though community NER models exist (e.g. `yashpwr/resume-ner-bert-v2`, reported 90.87% F1 on 22,542 samples; a RoBERTa resume-NER variant).
- **Datasets for fine-tuning/eval:** Kaggle "Resume NER Training Dataset" (5,960 standardized samples combining ATS/HF/Corpus/Doccano sets); `yashpwr/resume-ner-training-data` on HF; the LiveCareer (2,484) + Jiechieu (29,780) resume sets used in research; Kaggle Resume Dataset (2,400 resumes, 24 professions). Synthetic-resume generation (template + LLM-substituted content) is a common way to build labeled corpora.
- **Prompt vs. fine-tune — the decision, with hard numbers.** The Alibaba framework (Zhu et al., arXiv:2510.09722, Oct 2025) is the best public evidence. Inside a layout-aware pipeline, **zero-shot frontier models scored ~0.95 field-level F1** on resume→JSON: on their SynthResume set, **GPT-4o 0.952 F1 / 0.952 Acc**, Gemini-2.5-flash 0.951, DeepSeek-v3 0.950, Qwen-max 0.946, Claude-4 0.946; on the harder real-world **RealResume** set GPT-4o 0.954 F1, Claude-4 0.959. A **fine-tuned Qwen3-0.6B-SFT** (trained on 15,500 resumes / 59,500 instruction samples) was the top real-world performer: per the paper, *"supervised fine-tuning boosts the base Qwen3-0.6B model's F1-score on RealResume from 0.641 to 0.964 — surpassing even Claude-4 (0.959) while reducing latency to just 1.54 seconds, achieving a 3–4× speedup"* (large in-pipeline models took 4.6–13.7 s/resume). On synthetic data the small model trailed the frontier models (0.917 F1). The same paper's SmartResume pipeline reports 92.1% mAP layout detection and 93.1% overall extraction accuracy on RealResume. **Interpretation:** prompt a frontier LLM to ship fast at ~0.95 F1; only invest in fine-tuning a small model when volume/latency/cost or on-prem/PII constraints justify the labeling + MLOps effort. A separate 2025 MDPI study (Kurek et al., *Applied Sciences* 16(1):217; 2,280 multilingual CVs) reinforces this: using a frozen GPT-4o as reference (100% by construction), open models **GPT-OSS-120B reached 73.52% completeness / 72.44% content similarity** and **Llama-3.1-8B-Instruct 79.21% completeness / 58.66% similarity**, concluding *"GPT-4o remains the only model ensuring strict schema coverage suitable for high-integrity extraction tasks."*

#### 1.5 Accuracy, evaluation & failure modes
- **Evaluation method:** field-level Precision/Recall/F1 with entity alignment (the Alibaba paper uses Hungarian alignment + a novel "accuracy = correct/aligned" metric to separate alignment errors from field-match errors). Build a human-labeled gold set of 100–300 resumes across formats; measure per-field F1; track "long text" fields (job descriptions/summaries) separately — they're the hardest. Per arXiv:2510.09722, on RealResume Long Text fields *"the naïve LLM baseline (Claude-4) achieves only an F1-score of 0.548,"* rising to 0.854 with the full pipeline, while Qwen3-0.6B's Long Text F1 *"jumps from 0.136 to 0.846 after SFT"* — vs 0.95+ for named entities and dates.
- **Common failure modes:** reading-order scrambling in multi-column/table layouts; contact info in headers/footers dropped; date parsing ("present", overlapping roles, gaps); skills bleeding across sections; hallucinated/filled fields when the schema forces a required field that isn't present; OCR character errors on scans.

### STAGE 2 — Parsed JSON + JD → ATS-Optimized Output

#### 2.1 How ATS actually work (and myths)
Modern ATS are two layers: **(1) a parser** that extracts structured fields (name, contact, titles, employers, dates, skills, education), and **(2) a matcher/ranker** that scores the resume against the job requisition, plus **knockout filters** (min years, required certs, location, work authorization). Per Jobscan's 2025 ATS Usage Report (Fortune 500 data gathered June 2, 2025), 98.4% of Fortune 500 companies use a detectable ATS (492 of 500), with over 39% using Workday for talent acquisition and SuccessFactors at 13.2%; Greenhouse, Lever, Taleo, and iCIMS are also common. Older systems do keyword-frequency matching; newer ones add semantic matching, skills inference, title-seniority scoring, and increasingly an LLM layer on top.

**Established truths for design:**
- Single-column, standard section headings, standard fonts, DOCX or text-based PDF (never image PDF, `.pages`, `.rtf`). Two-column layouts failed in 7 of 8 systems in one hands-on test; tables were partially or fully dropped in 5.
- Keyword *placement* matters — terms in a labeled Skills section carry more weight than the same words buried in prose; include both acronym and expansion ("SEO"/"Search Engine Optimization").
- **Myths to reject:** "ATS auto-reject 75% of resumes" (conflates parse failures, filter misses, and recruiter passes); "more keywords = better" (keyword stuffing adds noise and can trip spam heuristics); hidden white-text keywords; chasing a 100% match. Coverage of must-have skills *in context* beats density. Reported thresholds: many employers set 60–75%; 75+ generally advances, 85+ is top-tier; below 60 usually signals a parsing/formatting problem, not a content one.

#### 2.2 Matching the resume JSON against a specific JD
Blend deterministic and semantic signals:
- **JD keyword/skill extraction:** `KeyBERT` (BERT-embedding keyphrase extraction, cosine similarity to the document, with MMR for diversity) and/or an LLM to pull required vs. preferred skills, tools, titles, and years. `spaCy`/`scikit-learn CountVectorizer` for n-grams; open-source `Keywords4CV` does TF-IDF + section-weighting + WordNet synonym expansion + fuzzy matching specifically for this task.
- **Semantic similarity:** encode resume sections and JD requirements with **Sentence-Transformers** (`all-mpnet-base-v2` for quality, `all-MiniLM-L6-v2` for speed, `multi-qa-MiniLM-L6-cos-v1` for query-style matching) and compute cosine similarity. This catches synonyms ("ML model development" ≈ "machine learning", "AWS/Azure" ≈ "cloud platforms") that literal matching misses. Handle length mismatch by chunking/averaging per section.
- **Fuzzy string matching:** `RapidFuzz` (MIT, C++-fast) for near-miss skill/title matching and deduping variants (typos handled by fuzz ratio; semantic equivalents like "Linux"≈"Ubuntu" handled by embeddings).
- **Skill-gap analysis:** set-difference of JD must-have skills vs. resume skills, weighted (required > preferred), with semantic matching so equivalent skills count.
- **Scoring:** a weighted composite typically resembles real ATS: keyword/skill coverage (30–40%), formatting/parse-ability (25–35%), and experience/title/seniority relevance. Produce a 0–100 score plus a component breakdown and a ranked list of missing must-have keywords.

#### 2.3 GenAI rewriting — truthful by construction
Use an LLM to rewrite bullets and tailor content to the JD, but **grounding is mandatory**. Documented failure modes in resume rewriting (arXiv:2607.01457, "Grounded Optimization"): **temporal fabrication** (injecting tools that postdate the role, e.g. LangChain into a 2018 job), **cross-domain contamination** (adding Azure/GCP to an AWS-only role to match keywords), **structural mutation** (silently deleting real achievements), and **content fabrication** (invented companies, inflated metrics, fake certs).

Engineering guardrails:
- Set the master resume as the single source of truth with an explicit instruction: *"never invent, inflate, or imply experience, skills, tools, or outcomes not in the source; you may reorder, rephrase, select, and emphasize, never fabricate; if the JD requires something absent, flag it rather than write around it."*
- Constrain rewrites to preserve tools/tech mentioned and truthful scope (no "led" unless true).
- Add a verification pass: check that every skill/keyword added to the tailored resume exists in the source JSON; reject or flag any new entity. Schema-lock the output.
- Keep a human-in-the-loop review step for anything customer-facing.

#### 2.4 Agentic orchestration
Decompose into specialized steps rather than one mega-prompt (separation of concerns; different temperatures — factual extraction at ~0.2, creative rewriting at ~0.6):
1. **Parser agent** — document → resume JSON (Stage 1).
2. **JD analyzer agent** — JD → structured requirements JSON (keywords, must/nice-to-have, years, title).
3. **Matcher/scorer agent** — deterministic embeddings + keyword coverage → score + gap list (a tool, not necessarily an LLM).
4. **Rewriter agent** — grounded bullet rewriting + tailoring, constrained to truthful edits.
5. (Optional) **Reviewer/verifier agent** — fabrication check + final ATS score.

**Framework choice (2025–2026):**
- **LangGraph** — the production default. Graph/state-machine model with typed shared state, native checkpointing (PostgresSaver), human-in-the-loop pauses, parallel branches with a sync barrier (needed so the matcher waits for *both* resume and JD nodes), streaming, and time-travel debugging via LangSmith/LangGraph Studio. Reached v1.0 in late 2025. **Best for this pipeline** — the resume/JD fan-in and iterative rewrite loop map cleanly to a graph.
- **CrewAI** — fastest to prototype (role/task "crew" metaphor, 2–4 hr prototypes, per-agent temperature). Great for a first version; weaker on checkpointing/state persistence. Real repos exist: `tonykipkemboi/resume-optimization-crew`, `ramansrivastava/resume-tailoring-agent`.
- **AutoGen (→ Microsoft Agent Framework, Oct 2025; v0.2 now maintenance)** — conversational multi-agent; good for debate/critique loops but token-expensive and now in transition.
- **Pydantic AI** — low-abstraction, typed, clean structured I/O; favored for control without heavy framework.
- **LlamaIndex / LlamaParse** — strong on the ingestion side (layout-aware "agentic document parsing" to JSON with confidence metadata).

A representative LangGraph state: `resume_doc`, `jd_doc`, `structured_resume`, `jd_requirements`, `match_result`, `tailored_resume`, `final_feedback`, with the JD-search and resume-parse branches running in parallel and converging at the scorer.

### Implementation Stack Recommendation
- **Ingestion:** Docling (default, local) + cloud OCR fallback (Textract/Document AI) for scans; PyMuPDF for the fast digital path.
- **Extraction:** GPT-4o / GPT-4.1 or Claude Sonnet (API) via **Instructor + Pydantic**; or self-hosted Qwen3/Llama via **vLLM + Outlines** for constrained decoding. Frontier model to ship; consider a fine-tuned small model at scale.
- **Matching:** Sentence-Transformers (`all-mpnet-base-v2`) + KeyBERT + RapidFuzz; store embeddings in a vector DB (pgvector/Milvus) if you need search.
- **Rewriting:** frontier LLM with grounding guardrails + verification pass.
- **Orchestration:** LangGraph (prototype in CrewAI if speed matters).
- **Serving:** FastAPI; validation with Pydantic; queue + human-review fallback.

### Cost / Latency Tradeoffs
- **APIs are cheaper and simpler below roughly 100M–500M tokens/month.** Cheap extraction-grade models in 2026: GPT-4o-mini ($0.15/$0.60 per 1M in/out), Gemini 2.5 Flash (~$0.30/$2.50), Claude Haiku 4.5 (~$1/$5). Frontier models (GPT-5.x, Claude Opus, Gemini 3 Pro) cost 5–12× more and are usually unnecessary for extraction.
- **Self-hosting open models (Llama, Qwen, DeepSeek, Mistral) wins at high, sustained volume** — one analysis puts break-even for a fine-tuned 14B model around $7,400 vs ~$42,500 (GPT-5) per 1M documents at 10k in/3k out; another cites self-hosting Llama breaking even only above ~500M tokens/day at 70%+ GPU utilization. Below ~$3,000/month API spend, self-hosting overhead isn't worth it; above ~$10,000/month it usually is.
- **Latency:** text-then-LLM extraction of a 300–400-token resume is ~1–6 s depending on model; vision-LLM on page images is 5–30 s/page. A fine-tuned 0.6B model does ~1.2–1.5 s/resume. Use prompt caching (30–90% input savings) and batch APIs (another ~50%) for bulk processing.
- **Cost hygiene:** the real metric is *cost per successful task*, not per token — a cheap model that fails 40% of the time and retries is more expensive than a reliable one. Build a provider-abstraction layer and route easy docs to cheap models, hard ones to frontier.

## Recommendations
1. **Ship a v1 in days, API-first.** Docling → GPT-4o/Claude via Instructor+Pydantic (JSON Resume schema) → deterministic matcher (Sentence-Transformers + KeyBERT + RapidFuzz) → grounded LLM rewriter, wired as a linear script. This alone reaches ~0.95 extraction F1.
2. **Add the agent graph and guardrails next.** Move to LangGraph for parallel resume/JD branches, the fabrication-verification node, and human-in-the-loop. Implement the truthfulness constraints and the post-hoc "every added keyword must exist in source" check.
3. **Instrument and evaluate.** Build a 100–300-resume labeled gold set; track per-field F1 (especially long-text fields), parse-failure rate, and match-score calibration. Log cost-per-successful-task.
4. **Decide on fine-tuning by thresholds, not instinct.** Only fine-tune a small model (Qwen3-0.6B/Llama-3.1-8B via LoRA + Outlines) when you cross **~$3–10k/month API spend, or need on-prem/PII isolation, or need sub-2s latency at scale** — the evidence shows a fine-tuned 0.6B can match/beat frontier F1 on real resumes (0.964 vs Claude-4's 0.959) at 3–4× lower latency, but only after investing in ~15k labeled samples.
5. **Format the ATS output for action.** Return a 0–100 score with component breakdown (keyword coverage, parse-ability, experience relevance), a ranked missing-keyword list, and specific truthful rewrite suggestions — plus a parse-simulation view so users see what an ATS actually extracts.

## Caveats
- **Vendor benchmarks are self-interested.** Affinda (sells a parser) argues open-source can't scale; LlamaIndex/Ragie/Unstract promote their own APIs. Treat their accuracy claims as directional. The Alibaba and MDPI papers are peer-style research, but the Alibaba results tables contain at least one internal arithmetic inconsistency (a Recall/F1 pair that doesn't compute), and the MDPI study measures fidelity *relative to a frozen GPT-4o reference* (100% by construction), not absolute ground-truth accuracy — so treat "GPT-4o = 100%" as "the baseline," not "perfect."
- **Every ATS is different.** Workday, Greenhouse, Taleo, iCIMS parse and score differently; any single "ATS score" is a proxy, not a guarantee. Scores from different checker tools aren't comparable.
- **The AI-screening layer is moving fast.** Per SHRM's 2025 Talent Trends report (survey of 2,040 HR professionals, fielded Feb 3–12, 2025), just over half of organizations (51%) now use AI to support recruiting efforts, and overall AI use in HR tasks climbed to 43% in 2025 (up from 26% in 2024). LLM layers increasingly sit atop traditional parsers — the keyword/format fundamentals still apply but semantic matching is rising in weight.
- **Truthfulness is a legal/ethical line.** Optimization must never cross into fabrication; candidates bear the consequences (disqualification, termination). Design grounding and verification in from the start.
- **Model names/prices churn monthly.** Specific 2026 model versions and per-token prices cited here will drift; re-benchmark on your own data before committing.