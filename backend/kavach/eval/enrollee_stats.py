"""Interval and separation statistics for the single-enrollee evaluation."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap_rate(
    flags_by_cluster: dict[str, list[bool]], *, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """95% percentile interval of a pooled rate, resampling whole clusters.

    Clips from one sitting (or trials against one impostor) are not independent,
    so resampling clips understates the uncertainty. With one cluster the interval
    is not informative and is returned as such rather than as a tight number.
    """
    keys = list(flags_by_cluster)
    ks = np.array([sum(flags_by_cluster[k]) for k in keys], dtype=float)
    ns = np.array([len(flags_by_cluster[k]) for k in keys], dtype=float)
    if len(keys) < 2 or ns.sum() == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        total = ns[idx].sum()
        rates[b] = ks[idx].sum() / total if total else np.nan
    return float(np.nanpercentile(rates, 2.5)), float(np.nanpercentile(rates, 97.5))


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Standardised distance between two score distributions."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    pooled = np.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2.0) if len(g) > 1 and len(i) > 1 else 0.0
    return float((g.mean() - i.mean()) / pooled) if pooled > 1e-12 else 0.0
