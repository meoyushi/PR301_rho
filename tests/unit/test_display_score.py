"""Display-score rescaling: calibrated band (~7-27) -> readable 0-100."""

from rho.ats.display import to_display_score


def test_floor_maps_to_zero():
    assert to_display_score(7.0) == 0.0


def test_ceil_maps_to_hundred():
    assert to_display_score(27.0) == 100.0


def test_midpoint_maps_to_fifty():
    assert to_display_score(17.0) == 50.0


def test_a_good_resume_reads_as_a_good_score():
    # 20.9 (the optimised Cred.pdf value) should read well above the midpoint.
    assert to_display_score(20.9) > 65.0


def test_below_floor_clamps_to_zero():
    assert to_display_score(3.0) == 0.0


def test_above_ceil_clamps_to_hundred():
    assert to_display_score(40.0) == 100.0


def test_direction_is_preserved():
    # A higher calibrated score must never produce a lower display score.
    assert to_display_score(18.8) < to_display_score(20.9)
