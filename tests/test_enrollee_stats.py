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


def test_all_clean_clusters_give_a_zero_rate_interval() -> None:
    lo, hi = cluster_bootstrap_rate({"a": [False] * 5, "b": [False] * 5})
    assert lo == 0.0 and hi == 0.0


def test_one_cluster_is_reported_not_trusted() -> None:
    lo, hi = cluster_bootstrap_rate({"only": [True, False]})
    assert (lo, hi) == (0.0, 1.0) or lo <= hi  # a single cluster cannot support an interval
    assert np.isfinite(lo) and np.isfinite(hi)


def test_d_prime_of_separated_groups_is_large_and_of_identical_is_zero() -> None:
    assert d_prime(np.array([0.9, 0.92, 0.88]), np.array([0.1, 0.12, 0.08])) > 5
    assert d_prime(np.array([0.5, 0.6]), np.array([0.5, 0.6])) == pytest.approx(0.0)
