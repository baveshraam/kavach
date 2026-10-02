"""Choosing the voice threshold and the inconclusive band from measured scores.

The threshold answers "how often may a stranger get through" (set from the impostor scores);
the band answers "how often may the owner be asked twice" (set from the owner's own scores).
Both come from data, with the uncertainty reported, and the result says when it is provisional.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kavach.calibrate_voice import MIN_THRESHOLD, choose_operating_point


def gauss(mean, sd, n, seed):
    return np.random.default_rng(seed).normal(mean, sd, n)


def test_well_separated_scores_give_a_threshold_between_the_classes_and_a_band_under_it():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.20, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    assert op.threshold > np.percentile(i, 99.9) - 1e-9
    assert op.floor < op.threshold
    assert op.frr_at_threshold == 0.0
    assert op.far_at_threshold <= 0.001
    assert op.ready


def test_the_threshold_is_the_impostor_quantile_not_a_midpoint():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.30, 0.08, 5000, 2)
    op = choose_operating_point(g, i, far_target=0.001)
    assert op.threshold == pytest.approx(np.quantile(i, 0.999), abs=1e-9)
    assert op.threshold > MIN_THRESHOLD  # so the quantile, not the floor on the threshold, decided it


def test_the_threshold_never_drops_below_the_minimum_however_low_the_impostors_are():
    g, i = gauss(0.9, 0.03, 200, 1), gauss(0.05, 0.05, 5000, 2)
    assert choose_operating_point(g, i).threshold >= MIN_THRESHOLD


def test_the_band_is_wide_enough_to_hold_almost_all_of_the_owners_own_scores():
    g, i = gauss(0.75, 0.08, 300, 1), gauss(0.20, 0.10, 5000, 2)
    op = choose_operating_point(g, i, genuine_cover=0.99)
    assert (g >= op.floor).mean() >= 0.985
    assert op.floor <= op.threshold - 0.02


def test_overlapping_classes_are_reported_not_hidden():
    g, i = gauss(0.60, 0.10, 200, 1), gauss(0.45, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    assert not op.ready
    assert op.frr_at_threshold > 0.05
    assert any("false-reject" in n.lower() for n in op.notes)


def test_few_genuine_scores_make_the_result_provisional():
    g, i = gauss(0.85, 0.05, 12, 1), gauss(0.20, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    assert op.provisional and any("genuine" in n.lower() for n in op.notes)


def test_too_few_impostor_trials_to_resolve_the_target_are_flagged_and_padded():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.25, 0.08, 150, 2)
    op = choose_operating_point(g, i, far_target=0.001)
    assert op.provisional and any("impostor" in n.lower() for n in op.notes)
    assert op.threshold >= i.max()  # with 150 trials a 0.1% quantile is not resolvable: stay above the worst seen


def test_it_reports_intervals_not_just_rates():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.20, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    lo, hi = op.frr_interval
    assert 0.0 <= lo <= op.frr_at_threshold <= hi <= 1.0
    lo, hi = op.far_interval
    assert 0.0 <= lo <= op.far_at_threshold <= hi <= 1.0


def test_the_far_at_the_floor_is_reported_because_that_is_who_gets_a_second_chance():
    g, i = gauss(0.80, 0.06, 300, 1), gauss(0.30, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    assert op.far_at_floor >= op.far_at_threshold


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_scores_are_refused(bad):
    g, i = gauss(0.85, 0.05, 50, 1), gauss(0.20, 0.10, 500, 2)
    g[3] = bad
    with pytest.raises(ValueError):
        choose_operating_point(g, i)


def test_empty_inputs_are_refused():
    with pytest.raises(ValueError):
        choose_operating_point(np.array([]), gauss(0.2, 0.1, 100, 1))
    with pytest.raises(ValueError):
        choose_operating_point(gauss(0.8, 0.05, 100, 1), np.array([]))


def test_it_serialises_for_the_policy_file():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.20, 0.10, 5000, 2)
    d = choose_operating_point(g, i).to_dict()
    assert {"threshold", "floor", "grey_margin", "far_target", "n_genuine", "n_impostor", "provisional"} <= set(d)
