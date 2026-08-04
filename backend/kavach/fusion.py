"""Score fusion: four branches into one accept/reject decision.

    S_spk    ECAPA-TDNN cosine        -- "does this sound like them?"
    S_csbg   CSBG likelihood ratio    -- "do they code-switch like them?"
    S_know   Cross-lingual answer match -- "do they know the answer?"
    S_live   Challenge freshness      -- "is this answer live?"

Two design points that matter more than the weights:

**Branches can be unavailable, and unavailable is not zero.** A short answer
yields too few tokens to score the CSBG; the semantic matcher may not be
installed. Scoring an unavailable branch as 0.0 would silently convert a
missing measurement into strong evidence of an impostor. Instead the branch
is dropped and the remaining weights renormalised, and the response records
which branches actually contributed.

**Liveness is a gate, not a weight.** An expired or reused challenge is not
"some evidence against" -- it is disqualifying, because it is exactly what a
replay attack looks like. Weighted fusion would let a very strong voice match
outvote it. `FusionPolicy.liveness_is_gate` keeps that from happening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BORDERLINE = "BORDERLINE"
    """Within `borderline_margin` of the threshold. An honest 'not sure' --
    escalate to a second factor rather than coin-flipping a security
    decision."""


class Branch(str, Enum):
    SPEAKER = "speaker_embedding"
    CSBG = "csbg"
    KNOWLEDGE = "knowledge"
    LIVENESS = "liveness"


@dataclass(slots=True)
class BranchScore:
    """One branch's contribution."""

    branch: Branch
    score: float
    threshold: float
    weight: float
    available: bool = True
    """False when the branch could not be measured. Excluded from fusion
    rather than counted as a zero."""

    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.available and self.score >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.branch.value,
            "score": round(self.score, 4),
            "threshold": round(self.threshold, 4),
            "weight": round(self.weight, 4),
            "passed": self.passed,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(slots=True)
class FusionPolicy:
    """How branches combine.

    Default weights encode how much each branch can be trusted, not measured
    optimality: the acoustic branch is the best-established, the CSBG is the
    research contribution but noisier on short probes, knowledge is binary-ish
    and strong when it fires. **Fit these on a dev split with `Calibrator`
    and report the fitted values** -- shipping the priors as if they were
    results would be dishonest.
    """

    weights: dict[Branch, float] = field(
        default_factory=lambda: {
            Branch.SPEAKER: 0.40,
            Branch.CSBG: 0.30,
            Branch.KNOWLEDGE: 0.30,
        }
    )
    threshold: float = 0.55
    borderline_margin: float = 0.05
    liveness_is_gate: bool = True
    """Treat liveness as pass/fail rather than a weighted term. Keep True --
    see the module docstring."""

    require_knowledge: bool = False
    """When True, a failed knowledge check rejects regardless of fusion.
    Off by default so the CSBG's contribution stays measurable in the attack
    evaluation: attack A4 assumes the attacker HAS the answer, and hard-gating
    knowledge would mask whether the CSBG caught them."""

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")


@dataclass(slots=True)
class FusionResult:
    """The final decision and everything behind it."""

    decision: Decision
    fused_score: float
    threshold: float
    branches: list[BranchScore]
    liveness_ok: bool = True
    explanation: list[str] = field(default_factory=list)
    contributing_branches: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "fused_score": round(self.fused_score, 4),
            "fused_threshold": round(self.threshold, 4),
            "liveness_ok": self.liveness_ok,
            "branches": [b.to_dict() for b in self.branches],
            "contributing_branches": self.contributing_branches,
            "explanation": self.explanation,
        }


def fuse(
    branches: list[BranchScore], policy: FusionPolicy | None = None
) -> FusionResult:
    """Combine branch scores into a decision.

    Args:
        branches: One BranchScore per measured branch. Unavailable branches
            may be included with `available=False`; they are excluded from the
            weighted sum and the remaining weights renormalised.
        policy: Weights and thresholds. Defaults to FusionPolicy().

    Returns:
        A FusionResult with a human-readable explanation.
    """
    policy = policy or FusionPolicy()
    by_branch = {b.branch: b for b in branches}

    # --- Liveness gate: disqualifying, not weighted. --------------------
    liveness = by_branch.get(Branch.LIVENESS)
    liveness_ok = True
    if policy.liveness_is_gate and liveness is not None:
        liveness_ok = liveness.passed
        if not liveness_ok:
            return FusionResult(
                decision=Decision.REJECT,
                fused_score=0.0,
                threshold=policy.threshold,
                branches=branches,
                liveness_ok=False,
                explanation=[
                    "Rejected: liveness check failed. "
                    + (liveness.detail or "The challenge was expired, reused, or unmatched."),
                    "This is the signature of a replay attack; no other branch can override it.",
                ],
                contributing_branches=[],
            )

    # --- Weighted fusion over available branches. -----------------------
    scored = [
        b for b in branches
        if b.branch in policy.weights and b.available
    ]
    if not scored:
        return FusionResult(
            decision=Decision.REJECT,
            fused_score=0.0,
            threshold=policy.threshold,
            branches=branches,
            liveness_ok=liveness_ok,
            explanation=[
                "Rejected: no verification branch could be measured. "
                "This is a system failure, not evidence about the speaker -- "
                "check ASR output and audio quality."
            ],
            contributing_branches=[],
        )

    total_weight = sum(policy.weights[b.branch] for b in scored)
    fused = sum(policy.weights[b.branch] * b.score for b in scored) / total_weight

    # --- Decision. ------------------------------------------------------
    if abs(fused - policy.threshold) <= policy.borderline_margin:
        decision = Decision.BORDERLINE
    elif fused >= policy.threshold:
        decision = Decision.ACCEPT
    else:
        decision = Decision.REJECT

    knowledge = by_branch.get(Branch.KNOWLEDGE)
    if policy.require_knowledge and knowledge is not None and not knowledge.passed:
        decision = Decision.REJECT

    return FusionResult(
        decision=decision,
        fused_score=fused,
        threshold=policy.threshold,
        branches=branches,
        liveness_ok=liveness_ok,
        explanation=_explain(decision, fused, policy, branches, scored),
        contributing_branches=[b.branch.value for b in scored],
    )


def _explain(
    decision: Decision,
    fused: float,
    policy: FusionPolicy,
    all_branches: list[BranchScore],
    scored: list[BranchScore],
) -> list[str]:
    """Plain-language reasons for the decision.

    This explainability is a genuine advantage of the graph formulation -- a
    512-dimensional embedding cannot say *which semantic class* caused a
    rejection. Worth a figure in the paper.
    """
    lines = [
        f"{decision.value}: fused score {fused:.3f} against threshold {policy.threshold:.3f}."
    ]

    for b in sorted(scored, key=lambda x: x.score - x.threshold):
        verdict = "passed" if b.passed else "failed"
        margin = b.score - b.threshold
        lines.append(
            f"  {b.branch.value}: {b.score:.3f} vs {b.threshold:.3f} "
            f"({verdict}, margin {margin:+.3f})"
            + (f" -- {b.detail}" if b.detail else "")
        )

    unavailable = [b for b in all_branches if not b.available]
    for b in unavailable:
        lines.append(
            f"  {b.branch.value}: not measured -- "
            + (b.detail or "insufficient evidence")
            + ". Excluded from fusion; remaining weights renormalised."
        )

    if decision is Decision.BORDERLINE:
        lines.append(
            "Score is within the borderline margin. Treat as inconclusive and "
            "escalate to a second factor rather than accepting."
        )
    return lines


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Calibrator:
    """Fits fusion weights on a labelled dev split.

    Uses logistic regression over branch scores, which yields weights that
    are directly interpretable as each branch's contribution -- and lets the
    paper report *fitted* weights rather than hand-chosen ones.

    **Fit on a dev split disjoint from the test speakers.** Fitting on the
    test set and reporting the resulting EER leaks, and a reviewer will ask
    which split the weights came from.
    """

    branches: tuple[Branch, ...] = (Branch.SPEAKER, Branch.CSBG, Branch.KNOWLEDGE)
    coefficients: dict[Branch, float] = field(default_factory=dict)
    intercept: float = 0.0
    fitted: bool = False

    def fit(
        self, samples: list[dict[Branch, float]], labels: list[int]
    ) -> dict[Branch, float]:
        """Fit weights.

        Args:
            samples: Per-trial {branch: score}. Missing branches are imputed
                with the column mean -- an unavailable branch must not be
                encoded as 0.0, which the model would read as strong evidence.
            labels: 1 for genuine, 0 for impostor.

        Returns:
            Normalised non-negative weights summing to 1.0, ready for
            FusionPolicy.
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ImportError(
                "scikit-learn is required for fusion calibration. "
                "Install with `pip install -r requirements-core.txt`."
            ) from exc

        if len(samples) != len(labels):
            raise ValueError(f"{len(samples)} samples but {len(labels)} labels.")
        if len(set(labels)) < 2:
            raise ValueError(
                "Calibration needs both genuine and impostor trials; got only one class."
            )

        matrix = np.array(
            [[s.get(b, np.nan) for b in self.branches] for s in samples], dtype=np.float64
        )
        col_means = np.nanmean(matrix, axis=0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(col_means, inds[1])

        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(matrix, labels)

        raw = dict(zip(self.branches, model.coef_[0]))
        self.intercept = float(model.intercept_[0])

        # Clip negatives: a negative weight would mean "sounding more like the
        # speaker is evidence against them", which is not a model we want to
        # ship even if a small dev split suggests it. A negative coefficient
        # here is a signal that branch is uninformative on your data --
        # investigate it rather than deploying it.
        clipped = {b: max(0.0, float(w)) for b, w in raw.items()}
        total = sum(clipped.values())
        if total <= 0:
            weights = {b: 1.0 / len(self.branches) for b in self.branches}
        else:
            weights = {b: w / total for b, w in clipped.items()}

        self.coefficients = weights
        self.fitted = True
        return weights

    def to_policy(self, **kw: Any) -> FusionPolicy:
        """Build a FusionPolicy from the fitted weights."""
        if not self.fitted:
            raise RuntimeError("Calibrator.fit() must be called before to_policy().")
        return FusionPolicy(weights=dict(self.coefficients), **kw)

    def report(self) -> str:
        if not self.fitted:
            return "Calibrator: not fitted"
        parts = [f"{b.value}={w:.3f}" for b, w in self.coefficients.items()]
        return f"Fitted fusion weights: {', '.join(parts)}"


def build_liveness_branch(
    *,
    challenge_valid: bool,
    matched_challenge: bool,
    response_latency_sec: float,
    max_latency_sec: float = 60.0,
    detail: str = "",
) -> BranchScore:
    """Construct the liveness branch from challenge-binding facts.

    Liveness here is *challenge freshness*, not acoustic spoof detection: a
    valid, unexpired, single-use challenge whose answer arrived in time. That
    scope boundary is deliberate and must be stated in the paper -- this
    defeats replay of an old recording, not a live acoustic attack.
    """
    ok = challenge_valid and matched_challenge and response_latency_sec <= max_latency_sec
    if not detail:
        if not challenge_valid:
            detail = "Challenge was expired, unknown, or already used."
        elif not matched_challenge:
            detail = "Response was not bound to the issued challenge."
        elif response_latency_sec > max_latency_sec:
            detail = f"Response arrived after {response_latency_sec:.0f}s."
        else:
            detail = f"Fresh response in {response_latency_sec:.1f}s."
    return BranchScore(
        branch=Branch.LIVENESS,
        score=1.0 if ok else 0.0,
        threshold=1.0,
        weight=0.0,
        available=True,
        detail=detail,
    )


__all__ = [
    "Decision",
    "Branch",
    "BranchScore",
    "FusionPolicy",
    "FusionResult",
    "fuse",
    "Calibrator",
    "build_liveness_branch",
]
