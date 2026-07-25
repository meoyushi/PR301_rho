"""Corpus pairing — the calibration set's résumé×JD construction."""

from eval.corpus import CATEGORY_TITLE_HINTS, _matches_category


def test_every_corpus_category_has_title_hints():
    """A category without hints can never be same-domain paired: it silently
    falls through to the cross-domain branch and dilutes the requested ratio."""
    import pandas as pd

    categories = set(pd.read_csv("Resume.csv").Category.unique())
    assert categories - set(CATEGORY_TITLE_HINTS) == set()


def test_matches_category_is_case_insensitive():
    assert _matches_category("Senior Web Developer", "INFORMATION-TECHNOLOGY")
    assert _matches_category("SOUS CHEF", "CHEF")


def test_matches_category_rejects_unrelated_titles():
    assert not _matches_category("Executive Chef", "INFORMATION-TECHNOLOGY")
    assert not _matches_category("Systems Administrator", "CHEF")
