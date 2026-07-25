"""The calibrator must be findable regardless of the process's CWD.

`CALIBRATOR_PATH` is relative ("eval/calibrator.joblib"), so it only resolved
when the process happened to start in the repo root. In the container (WORKDIR
/app) it never did: `score_node` took its documented fallback and left
`predicted_score` at 0.0 for every résumé, which reached the UI as
"MATCH SCORE 0/100" with no error anywhere in the response.
"""

import os

from rho.graph import nodes


def test_calibrator_found_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert nodes._calibrator_path() is not None, (
        "calibrator must resolve outside the repo root — this is the container case"
    )


def test_calibrator_prefers_the_overridable_relative_path(tmp_path, monkeypatch):
    """A caller pointing CALIBRATOR_PATH at a real file still wins."""
    override = tmp_path / "custom.joblib"
    override.write_bytes(b"stub")
    monkeypatch.setattr(nodes, "CALIBRATOR_PATH", str(override))

    assert nodes._calibrator_path() == str(override)


def test_calibrator_path_is_none_when_nothing_is_installed(monkeypatch):
    """No calibrator anywhere -> None, so score_node warns instead of inventing."""
    monkeypatch.setattr(nodes, "CALIBRATOR_PATH", "/nonexistent/calibrator.joblib")
    monkeypatch.setattr(nodes, "_REPO_CALIBRATOR", "/nonexistent/repo.joblib")

    assert nodes._calibrator_path() is None


def test_repo_calibrator_constant_points_at_the_real_file():
    """Guards the ../../.. hop against a future move of nodes.py."""
    assert os.path.exists(nodes._REPO_CALIBRATOR)
