"""Evaluation: verification metrics, ablations, fairness slices, attack tables.

Produces the numbers and figures that go into the paper. Pure numpy -- no ML
dependencies -- so evaluation can be re-run and figures regenerated without
loading any speech model.
"""

from .metrics import (
    DETPoint,
    VerificationMetrics,
    bootstrap_eer_ci,
    compute_auc,
    compute_eer,
    compute_min_dcf,
    evaluate,
)

__all__ = [
    "DETPoint",
    "VerificationMetrics",
    "evaluate",
    "compute_eer",
    "compute_min_dcf",
    "compute_auc",
    "bootstrap_eer_ci",
]
