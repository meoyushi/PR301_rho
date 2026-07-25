"""Tests for the Table 4 ablation runner (Phase 7, Task 4).

The property under test is that each ablation reports a *measurement* or says it
has no data — never a zero standing in for missing inputs. That failure mode
reads like a clean result, which is how a paper ends up reporting a number
nobody computed.
"""

import json

from eval.ablations import (
    _appears_in,
    ablation_calibration,
    ablation_gate,
    ablation_provenance,
    run_ablations,
)


def test_appears_in_accepts_verbatim_source_text():
    assert _appears_in("Python", "skills: python, sql")


def test_appears_in_accepts_scattered_content_words():
    """The naive check's weakness, made explicit: words from anywhere count."""
    assert _appears_in("Microsoft Access", "Microsoft Word\nAccess control procedures")


def test_appears_in_rejects_genuinely_absent_value():
    assert not _appears_in("Kubernetes", "skills: python, sql")


def test_appears_in_rejects_empty():
    assert not _appears_in("", "anything")


def test_ablation_a_and_c_are_not_the_same_measurement():
    """A isolates the provenance chain; C toggles the whole gate.

    If A ever becomes a relabelling of C, the ablation table claims a result it
    did not measure.
    """
    a = {(r["condition"], r["metric"]) for r in ablation_provenance()}
    c = {(r["condition"], r["metric"]) for r in ablation_gate()}
    assert a and c
    assert not (a & c)


def test_ablation_a_reports_na_rather_than_zero_without_sources(monkeypatch):
    """No source documents must surface as 'n/a', never as '0 rejected'."""
    monkeypatch.setattr("eval.ablations._fabrication_sources", lambda suffix="": {})
    rows = ablation_provenance()
    off = [r for r in rows if "OFF" in r["condition"]]
    assert off, "the OFF condition must always be reported"
    assert any(r["value"] == "n/a" for r in off)


def test_ablation_rows_have_the_shape_the_table_renders():
    for row in run_ablations():
        assert {"ablation", "condition", "metric", "value"} <= set(row)


def test_ablation_calibration_matches_the_saved_artifact():
    """Table 4 row B must equal the Phase-4 numbers, not a re-derivation."""
    rows = ablation_calibration()
    if not rows:
        return  # calibrator not yet fitted on this checkout
    metrics = json.loads(open("eval/progress.json").read())["metrics"]
    calibrated = next(r for r in rows if "calibrated" in r["condition"])
    assert calibrated["value"].startswith(f"{metrics['mae']:.2f}")


def test_ablation_calibration_reads_suffixed_artifact_when_given():
    """A backend re-run (e.g. Gemini) writes progress_gemini.json rather than
    overwriting the qwen baseline; `suffix` must read that file instead."""
    rows = ablation_calibration(suffix="_gemini")
    if not rows:
        return  # gemini calibrator not yet fitted on this checkout
    metrics = json.loads(open("eval/progress_gemini.json").read())["metrics"]
    calibrated = next(r for r in rows if "calibrated" in r["condition"])
    assert calibrated["value"].startswith(f"{metrics['mae']:.2f}")


def test_ablation_gate_reads_suffixed_artifact_when_given():
    rows = ablation_gate(suffix="_gemini")
    if not rows:
        return
    data = json.loads(open("eval/fabrication_results_corpus_gemini.json").read())
    off_row = next(r for r in rows if r["condition"] == "OFF")
    assert off_row["value"] == data["unsourced_shipped_gate_off"]


def test_ablation_default_suffix_is_unchanged():
    """No suffix must keep reading the original qwen-baseline filenames."""
    assert ablation_calibration() == ablation_calibration(suffix="")
