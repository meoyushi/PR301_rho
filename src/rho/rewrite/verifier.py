"""The provenance verification gate (C3).

Every hard-content value the rewriter *introduces* must resolve to a source
`prov_id` whose `raw_text` supports it. Unsupported additions are rejected —
reverted to the source value or dropped — and counted in a `FabricationReport`.

Two things make an addition acceptable:
1. the value is already in the source résumé (a reorder/reselect, not an edit), or
2. `find_prov` locates a span in the original document that supports it.

Anything else ships nothing. The gate is deterministic and LLM-free: given the
same (tailored, source, prov) it always renders the same verdict, which is what
makes the fabrication rate a reproducible metric rather than a model opinion.
"""

import re

from rapidfuzz import fuzz

from rho.extraction.provenance_attach import find_prov
from rho.models.provenance import ProvenanceMap
from rho.models.resume import StructuredResume
from rho.models.rewrite import FabricationReport, RejectedEdit
from rho.rewrite.tokens import hard_content_tokens

_NO_PROV = "no supporting prov_id"

# Bullets are prose, and `find_prov` is the wrong test for them: it scores with
# `partial_ratio`, which finds the best-matching *substring*, so a short span
# ("Engineer") scores 100 against any long bullet that happens to contain it —
# a fabricated bullet would pass on the strength of one incidental word. Whole
# strings are therefore compared with `token_set_ratio` against the source
# bullets: rephrasing survives, new claims do not.
#
# Threshold calibrated on corpus rewrites (`eval/fabrication_corpus.py`): genuine
# rephrasings of a source bullet score 68-87, while inventions score 37-42. 90
# was above *every* real rephrasing and rejected them all as fabrications; 60
# sits in the empty band between the two classes.
_BULLET_SIMILARITY = 60

# Similarity alone is not enough — a long invention that reuses the source's
# vocabulary scores well ("Led a team of 40 Teradata engineers"). But requiring
# every *word* to be sourced is too strict in the other direction: rephrasing
# legitimately swaps synonyms ("improve" -> "enhance") and irregular inflections
# ("tuning" -> "tuned"), none of which assert a new fact.
#
# What C3 actually protects is *hard content*: tools, orgs, acronyms, numbers,
# dates. So a bullet is checked for newly-introduced hard tokens, not new prose.
# Capitalised terms mid-sentence, all-caps acronyms, and digit runs qualify; the
# leading word is skipped because sentence case capitalises any opening verb.
_HARD_IN_BULLET = re.compile(r"\b([A-Z][A-Za-z0-9+#.]+|[A-Z]{2,}|\d[\d,.$%/-]*)\b")

# `find_prov` locating a span is necessary but not sufficient for the gate. It is
# tuned for Phase-2 extraction (tolerant, substring-based), so a span reading
# "Engineer" also "supports" the fabricated promotion "Staff Engineer", and
# "Python" supports "Senior Python Developer". Every *word* of an added value
# must appear in the supporting span, or the added words are unsourced claims.
_WORD_MATCH = 88
# Function words carry no factual claim; requiring provenance for them would
# reject truthful rephrasings like "Engineer of Platform" over "Platform Engineer".
_STOPWORDS = frozenset({"of", "and", "in", "at", "the", "a", "an", "for", "to", "&"})


def _stem(word: str) -> str:
    """Crude suffix stripper so inflections of a source word compare equal.

    Deliberately not a real stemmer: no new dependency, and the gate only needs
    "optimizing"/"optimized" to collide, not linguistic correctness.
    """
    word = word.strip("'.-")
    for suffix in ("ations", "ation", "ingly", "ing", "edly", "ed", "es", "s", "ly"):
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class _Ledger:
    """Accumulates verdicts so every field type shares one counting rule."""

    def __init__(self, source: StructuredResume, prov: ProvenanceMap):
        self._src_values = {v.lower() for v, _ in hard_content_tokens(source)}
        self._src_bullets = [
            b.strip().lower() for w in source.work for b in w.bullets if b.strip()
        ] + [
            b.strip().lower() for p in source.projects for b in p.bullets if b.strip()
        ]
        # Stems of every word the source states anywhere — a bullet may
        # legitimately pull a term from the skills list or a job title.
        blob = " ".join(
            [v.lower() for v, _ in hard_content_tokens(source)]
            + self._src_bullets
            + [(source.summary or "").lower(), (source.headline or "").lower()]
        )
        self._src_stems = {
            _stem(w) for w in re.findall(r"[a-z0-9][a-z0-9'+#.-]*", blob)
        }
        self._prov = prov
        self.total = 0
        self.verified = 0
        self.rejected: list[RejectedEdit] = []

    def accept_bullet(self, bullet: str) -> bool:
        """True if `bullet` restates a source bullet closely enough to ship.

        Two conditions, both required: it must closely track some source bullet,
        *and* introduce no unsourced hard-content token. Similarity alone would
        let a long invention pass on borrowed vocabulary; requiring every word to
        be sourced would reject ordinary rewording.
        """
        if not bullet or not bullet.strip():
            return True
        b = bullet.strip().lower()
        nearest = max(
            (fuzz.token_set_ratio(b, s) for s in self._src_bullets), default=0
        )
        if nearest >= _BULLET_SIMILARITY and not self._unsourced_hard_tokens(bullet):
            return True  # a rephrasing of something the source already claims
        self.total += 1
        self.rejected.append(RejectedEdit(added_text=bullet, reason=_NO_PROV))
        return False

    def _unsourced_hard_tokens(self, bullet: str) -> list[str]:
        """Hard-content tokens the bullet introduces without support.

        Compared by stem so ordinary inflection ("optimizing" -> "optimized")
        does not read as a new fact; a genuinely new tool, org, or number has no
        stem in the source and no supporting span.
        """
        unsourced = []
        # Skip the leading word: sentence case capitalises whatever verb opens it.
        body = bullet.split(None, 1)[1] if " " in bullet.strip() else ""
        for match in _HARD_IN_BULLET.finditer(body):
            token = match.group(1)
            low = token.lower()
            if low in _STOPWORDS or len(low) < 2:
                continue
            if _stem(low) in self._src_stems:
                continue
            if find_prov(token, self._prov):
                continue
            unsourced.append(token)
        return unsourced

    def accept(self, value: str) -> bool:
        """True if `value` may ship. Counts it as an edit only when it is new."""
        if not value or not value.strip():
            return True
        if value.strip().lower() in self._src_values:
            return True  # not an edit: the source already claims this
        self.total += 1
        if self._fully_supported(value):
            self.verified += 1
            return True
        self.rejected.append(RejectedEdit(added_text=value, reason=_NO_PROV))
        return False

    def _fully_supported(self, value: str) -> bool:
        """True when some span supports *every* content word of `value`."""
        words = [
            w for w in value.strip().lower().split() if w and w not in _STOPWORDS
        ]
        if not words:
            return False
        for pid in find_prov(value, self._prov):
            raw = self._prov.get(pid).raw_text.lower()
            if all(
                w in raw or fuzz.partial_ratio(w, raw) >= _WORD_MATCH for w in words
            ):
                return True
        return False

    def report(self) -> FabricationReport:
        return FabricationReport(
            total_edits=self.total,
            verified_edits=self.verified,
            rejected_edits=self.rejected,
            fabrication_rate=(len(self.rejected) / self.total) if self.total else 0.0,
        )


def _keep_supported(values: list[str], provs: list, accept):
    """Filter a value list and its parallel *_prov list together (C1 discipline)."""
    kept_values: list[str] = []
    kept_provs: list = []
    for i, v in enumerate(values):
        if not accept(v):
            continue
        kept_values.append(v)
        if i < len(provs):
            kept_provs.append(provs[i])
    return kept_values, kept_provs


def verify_against_source(
    tailored: StructuredResume, source: StructuredResume, prov: ProvenanceMap
) -> tuple[StructuredResume, FabricationReport]:
    """Strip every unsourced addition from `tailored`; report what was rejected."""
    fixed = tailored.model_copy(deep=True)
    ledger = _Ledger(source, prov)

    fixed.skills, fixed.skills_prov = _keep_supported(
        fixed.skills, fixed.skills_prov, ledger.accept
    )
    fixed.certifications, _ = _keep_supported(fixed.certifications, [], ledger.accept)
    fixed.achievements, fixed.achievements_prov = _keep_supported(
        fixed.achievements, fixed.achievements_prov, ledger.accept
    )

    kept_work = []
    for w in fixed.work:
        # An employer or title with no provenance is a fabricated job. Judge both
        # fields before deciding, so a wholly-invented entry is counted as the two
        # fabrications it is, then drop the entry rather than ship a half-real one.
        company_ok = ledger.accept(w.company)
        title_ok = ledger.accept(w.title)
        if not (company_ok and title_ok):
            continue
        w.bullets, w.bullet_prov = _keep_supported(
            w.bullets, w.bullet_prov, ledger.accept_bullet
        )
        kept_work.append(w)
    fixed.work = kept_work

    kept_edu = []
    for e in fixed.education:
        if ledger.accept(e.institution):
            kept_edu.append(e)
    fixed.education = kept_edu

    kept_projects = []
    for p in fixed.projects:
        # A project name with no provenance is a fabricated project; drop the
        # whole entry rather than ship a real-looking shell.
        if not ledger.accept(p.name):
            continue
        p.tech, p.tech_prov = _keep_supported(p.tech, p.tech_prov, ledger.accept)
        p.bullets, p.bullet_prov = _keep_supported(
            p.bullets, p.bullet_prov, ledger.accept_bullet
        )
        kept_projects.append(p)
    fixed.projects = kept_projects

    return fixed, ledger.report()
