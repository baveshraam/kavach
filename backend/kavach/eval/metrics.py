"""Verification metrics: EER, minDCF, DET curves.

These produce the numbers that go in the paper, so the implementations are
written to be checkable against the definitions rather than clever.

Conventions
-----------
* Scores are "higher = more likely genuine" everywhere in KAVACH.
* FAR (false accept rate) = P(accept | impostor). Also written FPR/FMR.
* FRR (false reject rate) = P(reject | genuine). Also written FNR/FNMR.
* EER is the operating point where FAR == FRR.

Small-sample warning
--------------------
With 25-30 speakers the number of genuine trials is small and EER confidence
intervals are wide. `bootstrap_eer_ci` is provided for exactly this reason:
**report the interval, not just the point estimate.** A bare "EER 8.3%" from
40 genuine trials invites a reviewer to ask what the error bar is, and the
honest answer is usually "several points wide".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DETPoint:
    """One operating point on a DET curve."""

    threshold: float
    far: float
    frr: float


@dataclass(frozen=True, slots=True)
class VerificationMetrics:
    """Full metric set for one system configuration."""

    eer: float
    """Equal error rate, in [0, 1]. Multiply by 100 for the paper."""

    eer_threshold: float
    """Score threshold at the EER operating point."""

    min_dcf: float
    """Minimum detection cost, under the parameters in `min_dcf_params`."""

    min_dcf_threshold: float

    far_at_frr_1pct: float
    """FAR when the threshold is set for 1% FRR -- a usability-first operating
    point. **NaN when no threshold reaches 1% FRR**, which is the normal state
    for a system with a veto: the veto's own false-reject cost is a floor.
    Render it as unattainable, never as 100%."""

    frr_at_far_1pct: float
    """FRR when the threshold is set for 1% FAR -- a security-first operating
    point. NaN when unattainable; see above."""

    auc: float
    """Area under the ROC curve."""

    n_genuine: int
    n_impostor: int

    det_curve: tuple[DETPoint, ...] = ()

    min_dcf_params: tuple[float, float, float] = (0.05, 1.0, 1.0)
    """(p_target, c_miss, c_fa) used for min_dcf."""

    @property
    def eer_percent(self) -> float:
        return self.eer * 100.0

    @property
    def is_reliable(self) -> bool:
        """Whether trial counts support a meaningful EER.

        Below ~30 trials per class the estimate is dominated by which
        particular speakers landed in the set.
        """
        return self.n_genuine >= 30 and self.n_impostor >= 30

    def summary(self) -> str:
        flag = "" if self.is_reliable else "  [!] small sample -- report CI"
        return (
            f"EER {self.eer_percent:6.2f}%  minDCF {self.min_dcf:.4f}  "
            f"FAR@1%FRR {format_rate(self.far_at_frr_1pct):>6}%  "
            f"FRR@1%FAR {format_rate(self.frr_at_far_1pct):>6}%  "
            f"(n={self.n_genuine}/{self.n_impostor}){flag}"
        )


def _rates(
    genuine: np.ndarray, impostor: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (thresholds, FAR, FRR) over all candidate thresholds.

    Uses every observed score as a candidate threshold, plus one below the
    lowest *finite* score, giving the exact empirical curve with no binning
    artefacts. The convention is: accept when score >= threshold.

    WHY "FINITE" IS DOING WORK THERE
    --------------------------------
    A score of -inf marks a trial that no threshold can accept -- a veto (see
    `eval.ablation`). Building the low anchor from `min(all) - 1.0` would make
    that anchor `-inf` too, and `-inf >= -inf` is True: every vetoed trial
    would be accepted at the bottom of the sweep, the curve would start at
    FRR=0, and "FAR at 1% FRR" would come back as an ordinary number for a
    system whose veto puts a hard floor under FRR.

    Anchoring below the lowest finite score instead makes -inf rejected
    everywhere, which is what a veto means, and the curve then correctly
    starts at the FRR the veto enforces.
    """
    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.unique(all_scores)

    finite = thresholds[np.isfinite(thresholds)]
    if len(finite) == 0:
        # Every trial was vetoed. One threshold, everything rejected.
        anchor = 0.0
        thresholds = np.array([anchor])
    else:
        thresholds = np.concatenate([[finite[0] - 1.0], finite])

    # far[i] = P(impostor >= t_i), frr[i] = P(genuine < t_i)
    far = np.array([np.mean(impostor >= t) for t in thresholds])
    frr = np.array([np.mean(genuine < t) for t in thresholds])
    return thresholds, far, frr


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold.

    Interpolates linearly between the two bracketing operating points where
    (FAR - FRR) changes sign, rather than returning the nearest sampled
    point. With few trials the sampled points are coarse and the nearest-point
    estimate is visibly biased.

    Returns:
        (eer, threshold).
    """
    thresholds, far, frr = _rates(genuine, impostor)
    diff = far - frr

    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        # No crossing (perfect or degenerate separation): fall back to the
        # point of closest approach.
        idx = int(np.argmin(np.abs(diff)))
        return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])

    i = int(sign_change[0])
    d0, d1 = diff[i], diff[i + 1]
    if d1 == d0:
        alpha = 0.0
    else:
        alpha = float(-d0 / (d1 - d0))

    eer = float(far[i] + alpha * (far[i + 1] - far[i]))
    threshold = float(thresholds[i] + alpha * (thresholds[i + 1] - thresholds[i]))
    return max(0.0, min(1.0, eer)), threshold


def compute_min_dcf(
    genuine: np.ndarray,
    impostor: np.ndarray,
    *,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    """Minimum normalised detection cost.

        DCF(t)  = c_miss * p_target * FRR(t) + c_fa * (1 - p_target) * FAR(t)
        minDCF  = min_t DCF(t) / min(c_miss * p_target, c_fa * (1 - p_target))

    `p_target=0.05` follows the ASVspoof/NIST-style convention of a low prior
    on the target. State whatever you use in the paper -- minDCF is not
    comparable across different parameter choices, and reviewers check.

    Returns:
        (min_dcf, threshold).
    """
    thresholds, far, frr = _rates(genuine, impostor)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    normaliser = min(c_miss * p_target, c_fa * (1.0 - p_target))
    dcf = dcf / normaliser
    idx = int(np.argmin(dcf))
    return float(dcf[idx]), float(thresholds[idx])


def _rate_at_fixed(
    genuine: np.ndarray, impostor: np.ndarray, *, target_frr: float | None = None,
    target_far: float | None = None
) -> float:
    """FAR at a fixed FRR, or FRR at a fixed FAR.

    Returns NaN when no threshold reaches the target -- **not 1.0.**

    That distinction is load-bearing once the system has a veto. A veto sets a
    floor on FRR that no threshold can go below, so "FAR at 1% FRR" simply does
    not exist for a system whose veto costs 1.25% FRR. Returning 1.0 there
    prints as "100.00" in a table next to a baseline's "46.70", and a reader
    compares the two as measured rates -- concluding that fusion is
    catastrophically worse at an operating point that fusion cannot occupy at
    all.

    NaN forces the caller to render it as unattainable, which is the true
    statement. `VerificationMetrics.summary` and `eval.ablation` both do.
    """
    thresholds, far, frr = _rates(genuine, impostor)
    if target_frr is not None:
        feasible = np.where(frr <= target_frr)[0]
        # Among thresholds meeting the FRR budget, take the one with lowest FAR.
        return float(far[feasible].min()) if len(feasible) else math.nan
    feasible = np.where(far <= target_far)[0]
    return float(frr[feasible].min()) if len(feasible) else math.nan


def format_rate(value: float, *, percent: bool = True) -> str:
    """Render a rate, or `n/a` when the operating point is unattainable."""
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def compute_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """ROC AUC, computed as the Mann-Whitney U statistic.

    Equivalent to P(random genuine scores above random impostor), with ties
    counted as half. Exact for small samples, unlike trapezoidal integration
    over a coarsely-sampled ROC.
    """
    n_g, n_i = len(genuine), len(impostor)
    if n_g == 0 or n_i == 0:
        return 0.5
    all_scores = np.concatenate([genuine, impostor])
    ranks = _rankdata(all_scores)
    rank_sum = float(ranks[:n_g].sum())
    u = rank_sum - n_g * (n_g + 1) / 2.0
    return u / (n_g * n_i)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Local implementation to avoid a scipy import."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def evaluate(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    *,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
    det_points: int = 200,
) -> VerificationMetrics:
    """Compute the full metric set for one configuration.

    Args:
        genuine_scores: Scores from same-speaker trials.
        impostor_scores: Scores from different-speaker (or attack) trials.
        p_target, c_miss, c_fa: minDCF parameters.
        det_points: Points to subsample for the DET curve. The stored curve is
            for plotting only; all scalar metrics use the full-resolution one.

    Raises:
        ValueError: If either score list is empty.
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)

    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError(
            f"Need both genuine and impostor scores to evaluate; got "
            f"{len(genuine)} genuine and {len(impostor)} impostor."
        )

    eer, eer_t = compute_eer(genuine, impostor)
    min_dcf, dcf_t = compute_min_dcf(
        genuine, impostor, p_target=p_target, c_miss=c_miss, c_fa=c_fa
    )

    thresholds, far, frr = _rates(genuine, impostor)
    if len(thresholds) > det_points:
        idx = np.linspace(0, len(thresholds) - 1, det_points).astype(int)
    else:
        idx = np.arange(len(thresholds))
    curve = tuple(
        DETPoint(threshold=float(thresholds[i]), far=float(far[i]), frr=float(frr[i]))
        for i in idx
    )

    return VerificationMetrics(
        eer=eer,
        eer_threshold=eer_t,
        min_dcf=min_dcf,
        min_dcf_threshold=dcf_t,
        far_at_frr_1pct=_rate_at_fixed(genuine, impostor, target_frr=0.01),
        frr_at_far_1pct=_rate_at_fixed(genuine, impostor, target_far=0.01),
        auc=compute_auc(genuine, impostor),
        n_genuine=len(genuine),
        n_impostor=len(impostor),
        det_curve=curve,
        min_dcf_params=(p_target, c_miss, c_fa),
    )


def bootstrap_eer_ci(
    genuine_scores: list[float] | np.ndarray,
    impostor_scores: list[float] | np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for EER. USE THIS IN THE PAPER.

    Resamples genuine and impostor score sets independently with replacement.
    With 25-30 speakers the interval is typically several percentage points
    wide, and reporting a bare point estimate would overstate precision.

    Args:
        n_resamples: Bootstrap draws. **0 means "point estimate only"** and
            returns NaN bounds -- callers that genuinely do not want an
            interval (ablation rows, which report EER deltas and no CI) would
            otherwise pay for hundreds of resamples per row. NaN rather than
            the point estimate repeated, because a zero-width interval is a
            claim of infinite precision, and `format_rate` already renders NaN
            as "n/a" everywhere it reaches a table.

    Returns:
        (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)

    point, _ = compute_eer(genuine, impostor)
    if n_resamples <= 0:
        return point, math.nan, math.nan

    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        g = rng.choice(genuine, size=len(genuine), replace=True)
        m = rng.choice(impostor, size=len(impostor), replace=True)
        samples[i], _ = compute_eer(g, m)

    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


__all__ = [
    "DETPoint",
    "VerificationMetrics",
    "format_rate",
    "evaluate",
    "compute_eer",
    "compute_min_dcf",
    "compute_auc",
    "bootstrap_eer_ci",
]
