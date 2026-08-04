"""Score fusion: five branches into one accept/reject decision.

    S_spk    ECAPA-TDNN cosine        -- "does this sound like them?"
    S_csbg   CSBG likelihood ratio    -- "do they code-switch like them?"
    S_know   Cross-lingual answer match -- "do they know the answer?"
    S_live   Challenge freshness      -- "is this answer live?"
    S_int    Edit-artefact tests      -- "is this one honest capture?"

The first three are weighted; the last two are gates. That split is not a
convenience -- the first three all answer *who is this*, so averaging them is
coherent, while the last two answer *is this submission legitimate at all*,
which no amount of voice similarity can compensate for. See `kavach.integrity`
for the same argument made at length.

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

**A weighted average cannot express a veto, and the threat model needs one.**
Arithmetic: with weights (0.40, 0.30, 0.30) and a threshold of 0.55, a branch
weighted 0.30 can move the fused score by at most 0.30. An attacker who fools
the acoustic branch (0.85) and knows the answer (0.90) is already at 0.61 from
those two alone, so *no* CSBG score, not even zero, can pull the total below
threshold. That is precisely attack A4 -- the one the CSBG exists to stop --
and under plain weighted fusion the CSBG is structurally incapable of stopping
it. Raising its weight until the arithmetic works would be tuning to a single
adversary and would punish genuine speakers whose answers are short.

The right fix follows from what the CSBG score *is*. It is a log-likelihood
ratio against a background model, so a strongly negative value is not weak
support for identity -- it is affirmative evidence *against* it. Averaging
that away is a category error, the same one `liveness_is_gate` avoids.
`FusionPolicy.veto_thresholds` therefore lets a branch reject outright when
its score falls below a floor.

A veto is a false-reject risk and is deliberately set far below the fusion
threshold, so it fires only on strong contrary evidence rather than on a
merely unimpressive score. Two safeguards: an *unavailable* branch never
vetoes (so a probe too short to score cannot reject anyone), and the default
floor is a starting point to be fitted on the dev split. **Report the fitted
floor and the false-reject rate it costs.** A veto whose FRR cost is not
measured is not a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


#: Default CSBG veto floor, on the branch's [0, 1] score scale.
#:
#: WHERE THIS NUMBER COMES FROM -- and why an earlier guess was wrong
#: ------------------------------------------------------------------
#: The CSBG branch score is a length-normalised log-likelihood ratio squashed
#: by `csbg.scoring._squash` with scale 2.0, so a branch score `s` corresponds
#: to a raw LLR of `2 * ln(s / (1 - s))`. Three landmarks on that scale:
#:
#:     s = 0.50   LLR  0.00   speaker model and background explain the probe
#:                            equally well -- no evidence either way
#:     s = 0.35   LLR -1.24   background is ~3.5x more likely: strong evidence
#:                            against the claimed identity
#:     s = 0.15   LLR -3.47   background is ~32x more likely
#:
#: The floor was originally set to 0.15 by reasoning about the [0, 1] scale in
#: the abstract, without checking what the scorer emits. It does not emit that.
#: An impostor whose language choices contradict the claimed speaker in every
#: measured class lands around **0.29**, because the logistic compresses hard
#: once the LLR passes -2. A floor of 0.15 therefore sat *below* the impostor
#: mode: the veto was in the code, was covered by tests using hand-written
#: scores, and would never once have fired on a real probe.
#:
#: 0.35 is the point at which the background model is about 3.5x more likely
#: than the speaker's -- affirmative contrary evidence rather than an
#: unimpressive score, which is exactly what the veto is for. **Fit it on the
#: dev split and report the false-reject rate it costs**; a veto whose FRR cost
#: is not measured is not a result.
CSBG_VETO_FLOOR = 0.35


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
    INTEGRITY = "signal_integrity"
    """Edit- and duplicate-artefact evidence. A gate, never a weighted term --
    it answers "was this file assembled?", not "who is speaking?", and the two
    must not be averaged. See `kavach.integrity`."""


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

    integrity_is_gate: bool = True
    """Treat signal integrity as pass/fail. Keep True in any reported
    configuration; set False only for the ablation that measures what the gate
    is worth, and label that row as the ablation it is."""

    require_knowledge: bool = False
    """When True, a failed knowledge check rejects regardless of fusion.
    Off by default so the CSBG's contribution stays measurable in the attack
    evaluation: attack A4 assumes the attacker HAS the answer, and hard-gating
    knowledge would mask whether the CSBG caught them."""

    veto_thresholds: dict[Branch, float] = field(
        default_factory=lambda: {Branch.CSBG: CSBG_VETO_FLOOR}
    )
    """Per-branch floors below which the branch rejects outright.

    See the module docstring for why a weighted average alone cannot stop
    attack A4, and `CSBG_VETO_FLOOR` for where the default number comes from.
    It is a **starting point, not a fitted value** -- fit it on the dev split
    against the false-reject rate it costs, and report both.

    Set to `{}` to disable vetoes entirely, which is the correct setting for
    the ablation that measures what the veto is worth."""

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        for branch, floor in self.veto_thresholds.items():
            if floor >= self.threshold:
                raise ValueError(
                    f"Veto floor for {branch.value} is {floor}, at or above the fusion "
                    f"threshold {self.threshold}. A veto that fires on scores the fusion "
                    "would have rejected anyway does nothing; one that fires above the "
                    "threshold silently replaces fusion with a single-branch decision."
                )


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

    # --- Integrity gate: disqualifying, not weighted. --------------------
    # Second, after liveness, because liveness is decided on the ledger alone
    # and costs nothing, while this one needs the audio decoded and analysed.
    # An *unavailable* integrity branch skips the gate rather than failing it:
    # a check that did not run is missing evidence, not a finding.
    integrity = by_branch.get(Branch.INTEGRITY)
    integrity_ok = True
    if policy.integrity_is_gate and integrity is not None and integrity.available:
        integrity_ok = integrity.passed
        if not integrity_ok:
            return FusionResult(
                decision=Decision.REJECT,
                fused_score=0.0,
                threshold=policy.threshold,
                branches=branches,
                liveness_ok=liveness_ok,
                explanation=[
                    "Rejected: this recording shows signs of having been edited or "
                    "resubmitted. " + (integrity.detail or ""),
                    "A spliced file can carry a perfect voiceprint, because it is made "
                    "of the real speaker's voice -- so a strong acoustic match is not a "
                    "reason to accept it, and no branch overrides this.",
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

    # --- Vetoes: strong contrary evidence from one branch. ---------------
    # Applied after fusion so `fused_score` still reports what the weighted
    # sum actually produced -- a veto overrides the decision, it does not
    # rewrite the evidence. Only branches that were measured can veto.
    vetoes = [
        b
        for b in scored
        if b.branch in policy.veto_thresholds and b.score < policy.veto_thresholds[b.branch]
    ]
    if vetoes:
        decision = Decision.REJECT
        veto_lines = [
            f"Rejected on the {b.branch.value} branch alone: {b.score:.3f} is below the "
            f"veto floor of {policy.veto_thresholds[b.branch]:.3f}. This is not a weak "
            "score, it is evidence against identity, and no other branch overrides it."
            for b in vetoes
        ]
        return FusionResult(
            decision=decision,
            fused_score=fused,
            threshold=policy.threshold,
            branches=branches,
            liveness_ok=liveness_ok,
            explanation=veto_lines
            + _explain(decision, fused, policy, branches, scored),
            contributing_branches=[b.branch.value for b in scored],
        )

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
    "CSBG_VETO_FLOOR",
    "Decision",
    "Branch",
    "BranchScore",
    "FusionPolicy",
    "FusionResult",
    "fuse",
    "Calibrator",
    "build_liveness_branch",
]
