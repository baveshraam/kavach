import numpy as np
import pytest

from kavach.attacks.suite import wilson_interval
from kavach.eval.enrollee_stats import cluster_bootstrap_rate, d_prime


def test_the_bootstrap_is_deterministic_for_a_seed() -> None:
    flags = {"a": [True, False, False], "b": [False] * 3, "c": [True] * 2}
    assert cluster_bootstrap_rate(flags, seed=3) == cluster_bootstrap_rate(flags, seed=3)


def test_clustered_data_gets_a_wider_interval_than_wilson() -> None:
    """Four sittings, each all-or-nothing: 40 clips are not 40 independent trials."""
    flags = {f"s{i}": [i < 2] * 10 for i in range(4)}  # 2 sittings fully rejected
    k = sum(sum(v) for v in flags.values()); n = sum(len(v) for v in flags.values())
    w_lo, w_hi = wilson_interval(k, n)
    c_lo, c_hi = cluster_bootstrap_rate(flags, seed=1)
    assert (c_hi - c_lo) > (w_hi - w_lo)


def test_the_raw_bootstrap_is_degenerate_with_no_events_which_is_why_it_is_never_reported_alone() -> None:
    lo, hi = cluster_bootstrap_rate({"a": [False] * 5, "b": [False] * 5})
    assert lo == 0.0 and hi == 0.0


def test_one_cluster_is_reported_not_trusted() -> None:
    lo, hi = cluster_bootstrap_rate({"only": [True, False]})
    assert (lo, hi) == (0.0, 1.0)  # a single cluster cannot support an interval
    assert np.isfinite(lo) and np.isfinite(hi)


def test_d_prime_of_separated_groups_is_large_and_of_identical_is_zero() -> None:
    assert d_prime(np.array([0.9, 0.92, 0.88]), np.array([0.1, 0.12, 0.08])) > 5
    assert d_prime(np.array([0.5, 0.6]), np.array([0.5, 0.6])) == pytest.approx(0.0)


def test_the_reported_interval_is_never_narrower_than_wilson() -> None:
    """Two sittings with no rejections: the raw bootstrap says (0, 0); Wilson says (0, 4.6%).
    Reporting the first as 'the interval to trust' would be reviewing focus 3 in reverse."""
    from kavach.eval.enrollee_stats import reported_interval

    flags = {"S4": [False] * 40, "S5": [False] * 40}
    lo, hi, informative = reported_interval(0, 80, flags)
    w_lo, w_hi = wilson_interval(0, 80)
    assert lo <= w_lo and hi >= w_hi and hi > 0.0
    assert informative is False


def test_wilson_matches_a_hand_computed_value() -> None:
    # 0 of 40: upper bound z^2 / (n + z^2) = 3.8416 / 43.8416
    assert wilson_interval(0, 40)[1] == pytest.approx(0.0876, abs=2e-4)


def test_many_clusters_with_events_are_informative_and_contain_the_rate() -> None:
    from kavach.eval.enrollee_stats import reported_interval

    flags = {f"I{i}": [i % 3 == 0] * 5 for i in range(12)}
    k = sum(sum(v) for v in flags.values())
    lo, hi, informative = reported_interval(k, 60, flags)
    assert informative is True and lo <= k / 60 <= hi
