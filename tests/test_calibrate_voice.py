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


def test_well_separated_scores_put_the_threshold_between_the_edges_and_nearer_the_owner():
    """The owner can retry (a second sample, another attempt); a stranger is owed nothing. So the
    margin is not split evenly: the threshold sits 60% of the way up from the strangers' tail."""
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.20, 0.10, 5000, 2)
    op = choose_operating_point(g, i)
    worst_stranger = float(np.quantile(i, 0.999))
    owner_edge = float(np.quantile(g, 0.01))
    assert worst_stranger < op.threshold < owner_edge
    assert op.threshold == pytest.approx(worst_stranger + 0.6 * (owner_edge - worst_stranger), abs=1e-6)
    assert op.floor < op.threshold and op.floor > worst_stranger
    assert op.frr_at_threshold == 0.0 and op.far_at_threshold == 0.0
    assert op.ready


def test_an_easy_cohort_cannot_pull_the_threshold_below_what_hard_voices_reach():
    """Public voices are easier than a same-room judge. Against the corpus speakers the nearest
    stranger reached 0.60, so no cohort, however easy, may justify a threshold under the minimum."""
    g, i = gauss(0.90, 0.03, 200, 1), gauss(0.05, 0.05, 5000, 2)
    assert choose_operating_point(g, i).threshold >= MIN_THRESHOLD >= 0.55


def test_a_threshold_at_the_edge_of_the_strangers_would_leave_no_margin__this_one_does():
    g, i = gauss(0.85, 0.05, 200, 1), gauss(0.30, 0.08, 5000, 2)
    op = choose_operating_point(g, i, far_target=0.001)
    assert op.threshold - float(np.quantile(i, 0.999)) > 0.05


def test_when_the_classes_overlap_the_false_accept_rate_has_priority():
    g, i = gauss(0.60, 0.10, 200, 1), gauss(0.45, 0.10, 5000, 2)
    op = choose_operating_point(g, i, far_target=0.001)
    assert op.threshold == pytest.approx(np.quantile(i, 0.999), abs=1e-9)


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


class TestLeaveOneSessionOut:
    """Each session is scored against a template built from the *other* sessions only."""

    @staticmethod
    def owner(seed, n, base, noise=0.15):
        r = np.random.default_rng(seed)
        v = base + noise * r.standard_normal((n, base.size))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_every_clip_is_scored_once_as_genuine_and_the_cohort_once_per_fold(self):
        from kavach.calibrate_voice import loso_scores

        base = np.random.default_rng(0).standard_normal(64)
        sessions = {"S1": self.owner(1, 6, base), "S2": self.owner(2, 5, base), "S3": self.owner(3, 4, base)}
        cohort = np.random.default_rng(9).standard_normal((50, 64))
        cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
        out = loso_scores(sessions, cohort)
        assert len(out.genuine) == 15
        assert len(out.impostor) == 3 * 50
        assert set(out.by_session) == {"S1", "S2", "S3"}

    def test_a_held_out_session_never_contributes_to_its_own_template(self):
        from kavach.calibrate_voice import loso_scores

        r = np.random.default_rng(0)
        a, b = r.standard_normal(64), r.standard_normal(64)
        sessions = {"S1": self.owner(1, 6, a), "S2": self.owner(2, 6, a), "S3": self.owner(3, 6, b)}  # S3 is another voice
        cohort = r.standard_normal((20, 64))
        out = loso_scores(sessions, cohort)
        assert out.by_session["S3"].mean() < 0.4  # scored against S1+S2 only, so it looks like a stranger

    def test_one_session_cannot_be_cross_validated(self):
        from kavach.calibrate_voice import loso_scores

        with pytest.raises(ValueError, match="two"):
            loso_scores({"S1": np.ones((3, 8))}, np.ones((4, 8)))

    def test_non_finite_embeddings_are_refused(self):
        from kavach.calibrate_voice import loso_scores

        bad = np.ones((3, 8))
        bad[1, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            loso_scores({"S1": bad, "S2": np.ones((3, 8))}, np.ones((4, 8)))
