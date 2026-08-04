"""The offline evaluation: every number the paper reports comes from here.

`api/evaluation.py` scores whatever logins happened to be clicked through a
UI. This does the real thing: all-pairs trials, a dev/test split by *speaker*,
fusion weights and a veto floor fitted on dev only, and bootstrap intervals on
everything.

FOUR THINGS THAT ARE EASY TO GET WRONG, AND HOW EACH IS HANDLED
---------------------------------------------------------------

**1. Splitting by trial instead of by speaker.** A speaker's enrolled CSBG
appears in every trial that claims their identity. Put some of those trials in
dev and some in test and the model fitted on dev has already seen the test
speakers' graphs. `split_by_speaker` partitions *speakers*, and the split is
reported so a reviewer can check it.

**2. Scoring against a background model that contains the claimed speaker.**
The LLR asks "is this probe more likely under this speaker than under speakers
in general?". If "in general" includes them, every genuine score is pulled
toward zero. `_ubm_for` excludes the claimed speaker, every time.

**3. Fitting the veto floor on the metric you then report.** A veto is a
false-reject risk. `fit_veto_floor` searches the floor on **dev** against an
explicit FRR budget and returns the cost it found, so the reported floor comes
with the number it cost rather than as a bare architectural choice.

**4. Sweeping a threshold on a system that has a veto.** A veto is a decision,
not a score, so the naive move is to report a single operating point and no DET
curve. That is not necessary: a vetoed trial is one that no threshold can
accept, which is exactly a score of -infinity. `fused_scores` emits that, the
curve comes out correct, and it is visibly truncated at the FAR the veto
enforces -- which is the honest shape of the system's behaviour.

WHAT THIS MODULE DOES NOT DECIDE
--------------------------------
Where the branch scores come from. `build_trials` takes callables for the
acoustic and knowledge branches and marks them unavailable when not supplied,
so the same harness runs on simulated CSBG-only data today and on the recorded
corpus with ECAPA and the answer matcher later. Which branches were live is
recorded on the report; a table that silently dropped one would be comparing
different systems across rows.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..csbg.graph import CSBG
from ..csbg.scoring import (
    CohortNormaliser,
    ScoringWeights,
    build_background_model,
    score_llr,
)
from ..csbg.tokens import UtteranceTokens
from ..fusion import CSBG_VETO_FLOOR, Branch, Calibrator, FusionPolicy
from .metrics import (
    VerificationMetrics,
    bootstrap_eer_ci,
    evaluate,
    format_rate,
)

#: Score assigned to a vetoed trial. Any value below every real score works;
#: -inf is used because it is the one value no threshold can accept, which is
#: precisely what a veto means.
VETOED = -math.inf

#: Branch score for a trial where the branch could not be measured. Never 0.0:
#: on a [0, 1] scale that is maximal evidence against, and a missing
#: measurement is not evidence of anything.
UNAVAILABLE: float = math.nan


ScoreFn = Callable[[str, str, list[UtteranceTokens]], float]
"""(probe_speaker, claimed_speaker, probe) -> branch score in [0, 1]."""


# --------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Trial:
    """One (probe, claimed identity) pair with every branch scored.

    Branch scores are stored on their own scales and `nan` when unmeasured, so
    a stored trial log can be re-fused under a different policy without
    re-running any model. That matters more than it sounds: the ablation table
    is a dozen re-fusions of one expensive scoring pass, and re-scoring per row
    would make the rows incomparable if anything drifted between them.
    """

    probe_speaker: str
    claimed_speaker: str
    is_genuine: bool

    speaker_score: float = UNAVAILABLE
    csbg_score: float = UNAVAILABLE
    knowledge_score: float = UNAVAILABLE

    csbg_raw: float = 0.0
    """Pre-normalisation LLR, kept so cohort z-norm can be re-fitted."""

    n_scored_tokens: int = 0
    group: dict[str, str] = field(default_factory=dict)
    """Speaker attributes for fairness slicing: device, environment, gender."""

    def scores(self) -> dict[Branch, float]:
        return {
            Branch.SPEAKER: self.speaker_score,
            Branch.CSBG: self.csbg_score,
            Branch.KNOWLEDGE: self.knowledge_score,
        }

    def available(self, branch: Branch) -> bool:
        return not math.isnan(self.scores()[branch])


def build_trials(
    graphs: dict[str, CSBG],
    probes: dict[str, list[UtteranceTokens]],
    *,
    speaker_score_fn: ScoreFn | None = None,
    knowledge_score_fn: ScoreFn | None = None,
    min_scored_tokens: int = 5,
    scoring_weights: ScoringWeights | None = None,
    include_low_signal: bool = False,
    lid_confidence_floor: float = 0.0,
    groups: dict[str, dict[str, str]] | None = None,
) -> list[Trial]:
    """All-pairs scoring: every probe against every enrolled speaker.

    With N speakers and K probes each this is N*K genuine trials and N*(N-1)*K
    impostor trials. At N=25, K=10 that is 250 and 6000 -- enough for a stable
    EER, and the reason the harness scores rather than the routes.

    Args:
        graphs: Enrolled CSBG per speaker.
        probes: Held-out utterances per speaker. Must be disjoint from the
            speech the graphs were fitted on, or genuine trials are scored
            against their own training data.
        speaker_score_fn: Acoustic branch. Omit to run CSBG-only.
        knowledge_score_fn: Answer-match branch. Omit to run without it.
        min_scored_tokens: Below this the CSBG branch is marked unavailable.
        groups: speaker_id -> attributes, for `fairness_slices`.

    Returns:
        One Trial per (probe, claimed speaker) pair.
    """
    groups = groups or {}
    trials: list[Trial] = []

    for claimed, graph in graphs.items():
        ubm = _ubm_for(claimed, graphs)
        if ubm is None:
            continue
        for probe_speaker, utterances in probes.items():
            for utt in utterances:
                score = score_llr(
                    [utt],
                    graph,
                    ubm,
                    weights=scoring_weights,
                    include_low_signal=include_low_signal,
                    lid_confidence_floor=lid_confidence_floor,
                )
                reliable = score.n_scored_tokens >= min_scored_tokens
                trials.append(
                    Trial(
                        probe_speaker=probe_speaker,
                        claimed_speaker=claimed,
                        is_genuine=probe_speaker == claimed,
                        csbg_score=score.normalised_score if reliable else UNAVAILABLE,
                        csbg_raw=score.raw_score,
                        n_scored_tokens=score.n_scored_tokens,
                        speaker_score=(
                            speaker_score_fn(probe_speaker, claimed, [utt])
                            if speaker_score_fn
                            else UNAVAILABLE
                        ),
                        knowledge_score=(
                            knowledge_score_fn(probe_speaker, claimed, [utt])
                            if knowledge_score_fn
                            else UNAVAILABLE
                        ),
                        group=dict(groups.get(probe_speaker, {})),
                    )
                )
    return trials


def _ubm_for(claimed: str, graphs: dict[str, CSBG]) -> CSBG | None:
    """Leave-one-out background model, or None with too small a cohort."""
    others = [g for sid, g in graphs.items() if sid != claimed]
    if len(others) < 2:
        return None
    return build_background_model(others)


def apply_cohort_norm(trials: list[Trial], normaliser: CohortNormaliser) -> None:
    """Re-derive `csbg_score` from `csbg_raw` under a z-normaliser, in place.

    Raw LLRs are not comparable across claimed speakers: someone with unusual
    habits produces large-magnitude scores for everyone, and someone who
    behaves like the population produces near-zero scores even when genuine. A
    single global threshold over those is measuring the speakers' typicality,
    not the probes.

    Unavailable branches stay unavailable -- there is nothing to normalise.
    """
    for t in trials:
        if math.isnan(t.csbg_score):
            continue
        t.csbg_score = normaliser.apply_squashed(t.claimed_speaker, t.csbg_raw)


def fit_cohort_normaliser(dev: Sequence[Trial]) -> CohortNormaliser:
    """Fit per-speaker z-norm statistics from **dev impostors only**.

    Impostors only, because the statistics describe "what this speaker's model
    does to someone else's speech". Including genuine trials would drag the
    mean toward the target distribution and shrink exactly the separation the
    normaliser exists to expose.
    """
    normaliser = CohortNormaliser()
    by_speaker: dict[str, list[float]] = {}
    for t in dev:
        if not t.is_genuine:
            by_speaker.setdefault(t.claimed_speaker, []).append(t.csbg_raw)
    for speaker, scores in by_speaker.items():
        normaliser.fit_speaker(speaker, scores)
    return normaliser


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Split:
    """A dev/test partition, by speaker."""

    dev: list[Trial]
    test: list[Trial]
    dev_speakers: list[str]
    test_speakers: list[str]

    def summary(self) -> str:
        return (
            f"dev: {len(self.dev_speakers)} speakers / {len(self.dev)} trials | "
            f"test: {len(self.test_speakers)} speakers / {len(self.test)} trials"
        )


def split_by_speaker(
    trials: Sequence[Trial], *, dev_fraction: float = 0.4, seed: int = 0
) -> Split:
    """Partition by speaker so no speaker appears on both sides.

    A trial belongs to dev only when **both** its probe speaker and its claimed
    speaker are dev speakers. Cross-partition impostor trials are discarded
    rather than assigned to one side: a trial pairing a dev speaker's model
    with a test speaker's probe has one foot in each, and putting it in test
    means the threshold was fitted partly on that model.

    That discards roughly `2*f*(1-f)` of the impostor trials -- about half at
    `dev_fraction=0.4`. The alternative is a leak, and a leak is not cheaper
    than fewer trials, it is just harder to see.

    Raises:
        ValueError: If either side would have no speakers.
    """
    speakers = sorted({t.claimed_speaker for t in trials} | {t.probe_speaker for t in trials})
    if len(speakers) < 4:
        raise ValueError(
            f"Need at least 4 speakers to split; got {len(speakers)}. Below that, "
            "one side has too few speakers for a background model."
        )

    rng = random.Random(seed)
    shuffled = list(speakers)
    rng.shuffle(shuffled)
    n_dev = max(2, round(len(shuffled) * dev_fraction))
    n_dev = min(n_dev, len(shuffled) - 2)
    dev_speakers = set(shuffled[:n_dev])

    dev, test = [], []
    for t in trials:
        in_dev = t.probe_speaker in dev_speakers and t.claimed_speaker in dev_speakers
        in_test = t.probe_speaker not in dev_speakers and t.claimed_speaker not in dev_speakers
        if in_dev:
            dev.append(t)
        elif in_test:
            test.append(t)

    return Split(
        dev=dev,
        test=test,
        dev_speakers=sorted(dev_speakers),
        test_speakers=sorted(set(speakers) - dev_speakers),
    )


# --------------------------------------------------------------------------
# Fusion over trials
# --------------------------------------------------------------------------


def fused_scores(
    trials: Sequence[Trial], policy: FusionPolicy, branches: Sequence[Branch]
) -> np.ndarray:
    """Fused score per trial, with vetoes as -inf.

    Weights are renormalised over the branches that were actually measured on
    *each* trial, matching `fusion.fuse`. A trial with no measurable branch
    scores -inf: nothing was measured, so nothing supports acceptance.
    """
    active = [b for b in branches if b in policy.weights]
    out = np.empty(len(trials), dtype=np.float64)

    for i, t in enumerate(trials):
        scores = t.scores()
        usable = [b for b in active if not math.isnan(scores[b])]
        if not usable:
            out[i] = VETOED
            continue

        vetoed = any(
            b in policy.veto_thresholds and scores[b] < policy.veto_thresholds[b]
            for b in usable
        )
        if vetoed:
            out[i] = VETOED
            continue

        total = sum(policy.weights[b] for b in usable)
        out[i] = sum(policy.weights[b] * scores[b] for b in usable) / total
    return out


def _split_labels(
    trials: Sequence[Trial], scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    genuine = np.array([s for t, s in zip(trials, scores) if t.is_genuine])
    impostor = np.array([s for t, s in zip(trials, scores) if not t.is_genuine])
    return genuine, impostor


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FittedPolicy:
    """A policy fitted on dev, with the evidence behind each number."""

    policy: FusionPolicy
    weights_source: str
    """'logistic regression on dev' or why it fell back to the priors."""

    threshold_source: str
    veto_floor: float | None
    veto_frr_cost: float
    """FRR the veto costs on dev, over and above the threshold's own. **This is
    the number that must appear in the paper next to the floor.**"""

    veto_far_benefit: float
    """FAR the veto removes on dev. A floor whose benefit is zero is dead code
    with a false-reject cost, and should be dropped rather than reported."""

    def report(self) -> str:
        lines = [
            f"weights    : {', '.join(f'{b.value}={w:.3f}' for b, w in self.policy.weights.items())}",
            f"             ({self.weights_source})",
            f"threshold  : {self.policy.threshold:.4f} ({self.threshold_source})",
        ]
        if self.veto_floor is None:
            lines.append("veto       : none")
        else:
            lines.append(
                f"veto       : {self.veto_floor:.3f} on csbg -- "
                f"removes {self.veto_far_benefit:.2%} FAR, costs {self.veto_frr_cost:.2%} FRR"
            )
        return "\n".join(lines)


def fit_fusion_weights(
    dev: Sequence[Trial], branches: Sequence[Branch]
) -> tuple[dict[Branch, float], str]:
    """Fit branch weights by logistic regression on dev.

    Falls back to `FusionPolicy`'s priors -- restricted and renormalised to the
    branches in play -- when the fit is impossible or scikit-learn is absent.
    The fallback is reported rather than silent, because "fitted weights" and
    "the defaults" are different claims and a table must not conflate them.
    """
    usable = [b for b in branches if any(t.available(b) for t in dev)]
    if not usable:
        return {}, "no branch was measurable on dev"

    samples = [
        {b: t.scores()[b] for b in usable if t.available(b)} for t in dev
    ]
    labels = [int(t.is_genuine) for t in dev]

    calibrator = Calibrator(branches=tuple(usable))
    try:
        return calibrator.fit(samples, labels), "logistic regression on dev"
    except (ImportError, ValueError) as exc:
        priors = FusionPolicy().weights
        active = {b: priors.get(b, 1.0 / len(usable)) for b in usable}
        total = sum(active.values())
        return (
            {b: w / total for b, w in active.items()},
            f"PRIORS, NOT FITTED -- calibration failed: {exc}",
        )


def fit_threshold(
    dev: Sequence[Trial], policy: FusionPolicy, branches: Sequence[Branch]
) -> float:
    """Threshold at the dev EER operating point.

    EER rather than a cost-weighted point because the paper reports EER as its
    headline; if the deployment cared more about one error than the other,
    `compute_min_dcf`'s threshold is the one to use, and the choice must be
    stated either way.
    """
    scores = fused_scores(dev, policy, branches)
    genuine, impostor = _split_labels(dev, scores)
    if len(genuine) == 0 or len(impostor) == 0:
        return policy.threshold

    from .metrics import compute_eer

    finite_g = genuine[np.isfinite(genuine)]
    finite_i = impostor[np.isfinite(impostor)]
    if len(finite_g) == 0 or len(finite_i) == 0:
        return policy.threshold
    _, threshold = compute_eer(finite_g, finite_i)
    return float(threshold)


def fit_veto_floor(
    dev: Sequence[Trial],
    policy: FusionPolicy,
    branches: Sequence[Branch],
    *,
    branch: Branch = Branch.CSBG,
    max_frr_cost: float = 0.02,
    candidates: Sequence[float] | None = None,
) -> tuple[float | None, float, float]:
    """Highest veto floor whose false-reject cost stays inside a budget.

    The search is one-sided on purpose. A higher floor always catches more
    attackers and always rejects more genuine speakers, so there is no optimum
    to find -- only a price the deployment is willing to pay. `max_frr_cost`
    is that price, it is a *policy* decision rather than an empirical one, and
    it must be stated in the paper alongside the floor it produced.

    Returns:
        (floor, far_removed, frr_cost). `floor` is None when no candidate
        buys any FAR reduction, which is the correct outcome to report: a veto
        that catches nothing should be dropped, not tuned.
    """
    if candidates is None:
        candidates = [round(x, 3) for x in np.arange(0.05, policy.threshold, 0.025)]

    base = FusionPolicy(
        weights=dict(policy.weights),
        threshold=policy.threshold,
        borderline_margin=policy.borderline_margin,
        liveness_is_gate=policy.liveness_is_gate,
        require_knowledge=policy.require_knowledge,
        veto_thresholds={},
    )
    base_scores = fused_scores(dev, base, branches)
    base_g, base_i = _split_labels(dev, base_scores)
    if len(base_g) == 0 or len(base_i) == 0:
        return None, 0.0, 0.0

    base_far = float(np.mean(base_i >= policy.threshold))
    base_frr = float(np.mean(base_g < policy.threshold))

    best: tuple[float, float, float] | None = None
    for floor in candidates:
        if floor >= policy.threshold:
            continue
        trial_policy = FusionPolicy(
            weights=dict(policy.weights),
            threshold=policy.threshold,
            borderline_margin=policy.borderline_margin,
            liveness_is_gate=policy.liveness_is_gate,
            require_knowledge=policy.require_knowledge,
            veto_thresholds={branch: floor},
        )
        scores = fused_scores(dev, trial_policy, branches)
        g, i = _split_labels(dev, scores)
        far = float(np.mean(i >= policy.threshold))
        frr = float(np.mean(g < policy.threshold))

        frr_cost = frr - base_frr
        far_benefit = base_far - far
        if frr_cost <= max_frr_cost and far_benefit > 0:
            # Candidates ascend, so the last one inside budget is the highest.
            best = (floor, far_benefit, frr_cost)

    return best if best is not None else (None, 0.0, 0.0)


def fit_policy(
    dev: Sequence[Trial],
    branches: Sequence[Branch],
    *,
    max_veto_frr_cost: float = 0.02,
    borderline_margin: float = 0.05,
) -> FittedPolicy:
    """Fit weights, threshold and veto floor on dev, in that order.

    Order matters: the threshold is fitted under the fitted weights, and the
    veto floor is fitted under both. Fitting them independently would price the
    veto against a threshold the system does not use.
    """
    weights, weights_source = fit_fusion_weights(dev, branches)
    if not weights:
        return FittedPolicy(
            policy=FusionPolicy(veto_thresholds={}),
            weights_source=weights_source,
            threshold_source="default -- nothing was measurable",
            veto_floor=None,
            veto_frr_cost=0.0,
            veto_far_benefit=0.0,
        )

    provisional = FusionPolicy(
        weights=weights, borderline_margin=borderline_margin, veto_thresholds={}
    )
    threshold = fit_threshold(dev, provisional, branches)
    provisional.threshold = threshold

    floor, far_benefit, frr_cost = (None, 0.0, 0.0)
    if Branch.CSBG in weights:
        floor, far_benefit, frr_cost = fit_veto_floor(
            dev, provisional, branches, max_frr_cost=max_veto_frr_cost
        )

    return FittedPolicy(
        policy=FusionPolicy(
            weights=weights,
            threshold=threshold,
            borderline_margin=borderline_margin,
            veto_thresholds={Branch.CSBG: floor} if floor is not None else {},
        ),
        weights_source=weights_source,
        threshold_source="dev EER operating point",
        veto_floor=floor,
        veto_frr_cost=frr_cost,
        veto_far_benefit=far_benefit,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

#: The configurations the paper compares. Each adds exactly one branch to the
#: one before it, except `+ CSBG only`, which pairs the contribution with the
#: baseline and drops the knowledge branch -- the only column that isolates
#: what the CSBG does without a challenge-response protocol already helping.
CONFIGURATIONS: dict[str, tuple[Branch, ...]] = {
    "ECAPA alone": (Branch.SPEAKER,),
    "CSBG alone": (Branch.CSBG,),
    "+ Knowledge": (Branch.SPEAKER, Branch.KNOWLEDGE),
    "+ CSBG only": (Branch.SPEAKER, Branch.CSBG),
    "Full fusion": (Branch.SPEAKER, Branch.KNOWLEDGE, Branch.CSBG),
}


@dataclass(slots=True)
class ConfigurationResult:
    name: str
    branches: tuple[Branch, ...]
    metrics: VerificationMetrics
    eer_ci: tuple[float, float]
    n_vetoed: int = 0

    def row(self) -> str:
        lo, hi = self.eer_ci
        flag = "" if self.metrics.is_reliable else "  [!]"
        veto = f" ({self.n_vetoed} vetoed)" if self.n_vetoed else ""
        return (
            f"| {self.name} | {self.metrics.eer * 100:.2f} "
            f"[{lo * 100:.2f}-{hi * 100:.2f}] | {self.metrics.min_dcf:.4f} | "
            f"{format_rate(self.metrics.far_at_frr_1pct)} | "
            f"{format_rate(self.metrics.frr_at_far_1pct)} | "
            f"{self.metrics.n_genuine}/{self.metrics.n_impostor}{veto}{flag} |"
        )


def evaluate_configuration(
    name: str,
    trials: Sequence[Trial],
    policy: FusionPolicy,
    branches: Sequence[Branch],
    *,
    bootstrap: int = 500,
    seed: int = 0,
) -> ConfigurationResult | None:
    """EER, minDCF, DET and a bootstrap interval for one configuration.

    Returns None when the configuration has no measurable branch on this data
    -- an empty row is more honest than an EER of 0.5 that looks like a result.
    """
    active = [b for b in branches if b in policy.weights]
    if not active or not any(t.available(b) for t in trials for b in active):
        return None

    scores = fused_scores(trials, policy, branches)
    genuine, impostor = _split_labels(trials, scores)
    if len(genuine) == 0 or len(impostor) == 0:
        return None

    n_vetoed = int(np.sum(~np.isfinite(scores)))

    # -inf is passed through rather than substituted. Any finite stand-in,
    # however low, is acceptable at a low enough threshold -- which is exactly
    # what a veto is not. `metrics._rates` anchors its sweep below the lowest
    # *finite* score so -inf is rejected everywhere.
    metrics = evaluate(genuine, impostor)
    _, lo, hi = bootstrap_eer_ci(genuine, impostor, n_resamples=bootstrap, seed=seed)
    return ConfigurationResult(
        name=name,
        branches=tuple(active),
        metrics=metrics,
        eer_ci=(lo, hi),
        n_vetoed=n_vetoed,
    )


# --------------------------------------------------------------------------
# Ablations
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AblationRow:
    """One switch flipped, and what it cost."""

    name: str
    eer: float
    delta: float
    """Change in EER against the full system. Positive means the ablation is
    worse, i.e. the removed component was helping."""

    note: str = ""


def ablate_policy(
    test: Sequence[Trial], fitted: FittedPolicy, branches: Sequence[Branch]
) -> list[AblationRow]:
    """Ablations expressible by re-fusing existing trials.

    Cheap and exact: the branch scores do not change, so every row sees
    identical evidence and the differences are attributable to the switch
    alone. Ablations that change *scoring* (the class set, the transition
    stream, cohort normalisation) need a fresh `build_trials` and live in
    `run_ablation`.
    """
    base = evaluate_configuration("full", test, fitted.policy, branches)
    if base is None:
        return []
    rows: list[AblationRow] = []

    if fitted.policy.veto_thresholds:
        no_veto = FusionPolicy(
            weights=dict(fitted.policy.weights),
            threshold=fitted.policy.threshold,
            borderline_margin=fitted.policy.borderline_margin,
            veto_thresholds={},
        )
        result = evaluate_configuration("no veto", test, no_veto, branches)
        if result:
            rows.append(
                AblationRow(
                    name="vetoes disabled",
                    eer=result.metrics.eer,
                    delta=base.metrics.eer - result.metrics.eer,
                    note=(
                        "EER can *improve* without the veto while the attack table gets "
                        "worse: a veto trades false rejects for stopped attacks, and EER "
                        "over genuine-vs-impostor trials does not see the attack rows. "
                        "Read this against §5.1, not on its own."
                    ),
                )
            )

    equal = {b: 1.0 / len(fitted.policy.weights) for b in fitted.policy.weights}
    result = evaluate_configuration(
        "equal weights",
        test,
        FusionPolicy(
            weights=equal,
            threshold=fitted.policy.threshold,
            veto_thresholds=dict(fitted.policy.veto_thresholds),
        ),
        branches,
    )
    if result:
        rows.append(
            AblationRow(
                name="equal branch weights",
                eer=result.metrics.eer,
                delta=result.metrics.eer - base.metrics.eer,
                note="What the logistic-regression fit was worth.",
            )
        )

    return rows


# --------------------------------------------------------------------------
# Stability and fairness
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StabilityPoint:
    n_utterances: int
    approx_seconds: float
    eer: float
    ci_low: float
    ci_high: float


def stability_curve(
    enrolment: dict[str, list[UtteranceTokens]],
    probes: dict[str, list[UtteranceTokens]],
    *,
    budgets: Sequence[int] = (2, 5, 10, 20, 30),
    seconds_per_utterance: float = 6.0,
    bootstrap: int = 200,
    **build_kw: Any,
) -> list[StabilityPoint]:
    """EER against enrolment budget. **This curve answers two questions.**

    Read left to right it says how much speech a defender needs before a CSBG
    is usable. Read as an attacker's eavesdropping budget it says how much
    overheard speech is needed to steal one (§5.1.2) -- the same measurement,
    because the estimate an attacker forms and the graph a defender enrols
    converge at the same rate.

    Budgets are in utterances rather than seconds because that is what the
    corpus is indexed by; `seconds_per_utterance` converts for the axis label
    and should be replaced with measured durations when they exist.
    """
    points: list[StabilityPoint] = []
    for budget in budgets:
        graphs = {
            sid: CSBG.build(sid, utts[:budget])
            for sid, utts in enrolment.items()
            if len(utts) >= budget
        }
        if len(graphs) < 3:
            continue

        trials = build_trials(graphs, probes, **build_kw)
        genuine = np.array([t.csbg_score for t in trials if t.is_genuine and t.available(Branch.CSBG)])
        impostor = np.array(
            [t.csbg_score for t in trials if not t.is_genuine and t.available(Branch.CSBG)]
        )
        if len(genuine) < 5 or len(impostor) < 5:
            continue

        point, lo, hi = bootstrap_eer_ci(genuine, impostor, n_resamples=bootstrap)
        points.append(
            StabilityPoint(
                n_utterances=budget,
                approx_seconds=budget * seconds_per_utterance,
                eer=point,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return points


@dataclass(slots=True)
class FairnessSlice:
    condition: str
    group: str
    eer: float
    n_genuine: int
    n_impostor: int


def fairness_slices(
    trials: Sequence[Trial],
    policy: FusionPolicy,
    branches: Sequence[Branch],
    *,
    min_trials: int = 20,
) -> list[FairnessSlice]:
    """Per-group EER, sliced by speaker attribute.

    The audit that matters for this corpus is device and environment: an
    aggregate EER that is fine while doubling on a cheap phone in a noisy
    corridor is a system that fails the population it was built for. Groups
    below `min_trials` on either side are dropped, because a fairness table is
    read as a comparison and a bar built from four trials invites a conclusion
    it cannot support.
    """
    scores = fused_scores(trials, policy, branches)

    buckets: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for t, s in zip(trials, scores):
        value = s
        for condition, group in t.group.items():
            if not group:
                continue
            g, i = buckets.setdefault((condition, group), ([], []))
            (g if t.is_genuine else i).append(value)

    out: list[FairnessSlice] = []
    for (condition, group), (genuine, impostor) in sorted(buckets.items()):
        if len(genuine) < min_trials or len(impostor) < min_trials:
            continue
        metrics = evaluate(genuine, impostor)
        out.append(
            FairnessSlice(
                condition=condition,
                group=group,
                eer=metrics.eer,
                n_genuine=len(genuine),
                n_impostor=len(impostor),
            )
        )
    return out


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AblationReport:
    """Everything the evaluation section needs, in one object."""

    split: Split
    fitted: FittedPolicy
    configurations: list[ConfigurationResult] = field(default_factory=list)
    ablations: list[AblationRow] = field(default_factory=list)
    stability: list[StabilityPoint] = field(default_factory=list)
    fairness: list[FairnessSlice] = field(default_factory=list)
    measured_branches: tuple[Branch, ...] = ()
    caveats: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Verification results",
            "",
            f"Split: {self.split.summary()}",
            f"Branches measured: {', '.join(b.value for b in self.measured_branches) or 'none'}",
            "",
            "```",
            self.fitted.report(),
            "```",
            "",
            "| Configuration | EER % [95% CI] | minDCF | FAR@1%FRR | FRR@1%FAR | n gen/imp |",
            "|---|---|---|---|---|---|",
        ]
        lines.extend(c.row() for c in self.configurations)
        lines += [
            "",
            "`[!]` marks a configuration with fewer than 30 trials on a side; its "
            "interval is too wide to compare against another row.",
            "",
            f"minDCF parameters: p_target={self.configurations[0].metrics.min_dcf_params[0]}, "
            f"c_miss={self.configurations[0].metrics.min_dcf_params[1]}, "
            f"c_fa={self.configurations[0].metrics.min_dcf_params[2]}."
            if self.configurations
            else "",
        ]

        if self.ablations:
            lines += ["", "### Ablations", "", "| Removed | EER % | Δ EER | Note |", "|---|---|---|---|"]
            lines.extend(
                f"| {a.name} | {a.eer * 100:.2f} | {a.delta * 100:+.2f} | {a.note} |"
                for a in self.ablations
            )

        if self.stability:
            lines += [
                "",
                "### CSBG stability against enrolment budget",
                "",
                "Read left to right: how much speech a defender needs. Read as an "
                "attacker's eavesdropping budget: how much they need to steal it.",
                "",
                "| Utterances | ~seconds | EER % [95% CI] |",
                "|---|---|---|",
            ]
            lines.extend(
                f"| {p.n_utterances} | {p.approx_seconds:.0f} | "
                f"{p.eer * 100:.2f} [{p.ci_low * 100:.2f}-{p.ci_high * 100:.2f}] |"
                for p in self.stability
            )

        if self.fairness:
            lines += [
                "",
                "### Fairness slices",
                "",
                "| Condition | Group | EER % | n gen/imp |",
                "|---|---|---|---|",
            ]
            lines.extend(
                f"| {f.condition} | {f.group} | {f.eer * 100:.2f} | "
                f"{f.n_genuine}/{f.n_impostor} |"
                for f in self.fairness
            )

        if self.caveats:
            lines += ["", "### Caveats", ""]
            lines.extend(f"- {c}" for c in self.caveats)

        return "\n".join(lines)


def run_ablation(
    graphs: dict[str, CSBG],
    probes: dict[str, list[UtteranceTokens]],
    *,
    speaker_score_fn: ScoreFn | None = None,
    knowledge_score_fn: ScoreFn | None = None,
    groups: dict[str, dict[str, str]] | None = None,
    dev_fraction: float = 0.4,
    seed: int = 0,
    cohort_norm: bool = True,
    max_veto_frr_cost: float = 0.02,
    bootstrap: int = 500,
) -> AblationReport:
    """The whole offline evaluation, end to end.

    Order of operations, all of which matter:

    1. score every (probe, claimed speaker) pair once;
    2. split by **speaker**;
    3. fit cohort z-norm on **dev impostors**, apply to both sides;
    4. fit weights, then threshold, then veto floor -- on **dev**;
    5. evaluate every configuration on **test**, with bootstrap intervals;
    6. ablate by re-fusing the same test trials.

    Nothing fitted in 3-4 sees a test speaker. That is the property a reviewer
    checks first, and `AblationReport.split` records it so they can.
    """
    trials = build_trials(
        graphs,
        probes,
        speaker_score_fn=speaker_score_fn,
        knowledge_score_fn=knowledge_score_fn,
        groups=groups,
    )
    split = split_by_speaker(trials, dev_fraction=dev_fraction, seed=seed)

    caveats: list[str] = []
    if cohort_norm:
        normaliser = fit_cohort_normaliser(split.dev)
        if normaliser.means:
            apply_cohort_norm(split.dev, normaliser)
            apply_cohort_norm(split.test, normaliser)
            caveats.append(
                "CSBG scores are cohort z-normalised, with statistics fitted on dev "
                "impostors only. Report un-normalised results too -- z-norm usually "
                "helps materially, and hiding that is hiding a design decision."
            )
        else:
            caveats.append(
                "Cohort z-normalisation was requested but no dev impostor trials "
                "existed to fit it, so raw scores were used."
            )

    measured = tuple(
        b
        for b in (Branch.SPEAKER, Branch.CSBG, Branch.KNOWLEDGE)
        if any(t.available(b) for t in trials)
    )
    if len(measured) < 3:
        absent = [
            b.value for b in (Branch.SPEAKER, Branch.CSBG, Branch.KNOWLEDGE) if b not in measured
        ]
        caveats.append(
            f"Branches not measured on this run: {', '.join(absent)}. Rows that would "
            "have used them are omitted rather than scored, so this table compares "
            "fewer systems than the paper's -- it is not a partial version of it."
        )

    fitted = fit_policy(
        split.dev, measured, max_veto_frr_cost=max_veto_frr_cost
    )
    if fitted.veto_floor is None and Branch.CSBG in measured:
        caveats.append(
            "No veto floor bought any FAR reduction inside the "
            f"{max_veto_frr_cost:.0%} FRR budget on dev. Report that the veto was "
            "fitted and discarded -- that is a result about the branch, not an "
            "omission."
        )
    elif fitted.veto_floor is not None and abs(fitted.veto_floor - CSBG_VETO_FLOOR) > 0.1:
        caveats.append(
            f"The fitted veto floor ({fitted.veto_floor:.2f}) is far from the "
            f"reasoned default ({CSBG_VETO_FLOOR:.2f}). Use the fitted value and say "
            "so; the default was never a measurement."
        )

    configurations: list[ConfigurationResult] = []
    for name, branches in CONFIGURATIONS.items():
        if not set(branches) <= set(measured):
            continue
        policy = _restrict(fitted.policy, branches)
        result = evaluate_configuration(
            name, split.test, policy, branches, bootstrap=bootstrap, seed=seed
        )
        if result:
            configurations.append(result)

    return AblationReport(
        split=split,
        fitted=fitted,
        configurations=configurations,
        ablations=ablate_policy(split.test, fitted, measured),
        fairness=fairness_slices(split.test, fitted.policy, measured),
        measured_branches=measured,
        caveats=caveats,
    )


def _restrict(policy: FusionPolicy, branches: Sequence[Branch]) -> FusionPolicy:
    """Renormalise a policy onto a subset of branches.

    The columns must differ in which evidence is available, not in how the
    shared evidence is weighted -- otherwise a difference between two rows
    could be a weighting artefact rather than the branch that was added.
    """
    active = {b: policy.weights[b] for b in branches if b in policy.weights}
    total = sum(active.values())
    if total <= 0:
        active = {b: 1.0 / len(branches) for b in branches}
    else:
        active = {b: w / total for b, w in active.items()}
    return FusionPolicy(
        weights=active,
        threshold=policy.threshold,
        borderline_margin=policy.borderline_margin,
        veto_thresholds={
            b: t for b, t in policy.veto_thresholds.items() if b in branches
        },
    )


__all__ = [
    "CONFIGURATIONS",
    "UNAVAILABLE",
    "VETOED",
    "AblationReport",
    "AblationRow",
    "ConfigurationResult",
    "FairnessSlice",
    "FittedPolicy",
    "StabilityPoint",
    "Split",
    "Trial",
    "ablate_policy",
    "apply_cohort_norm",
    "build_trials",
    "evaluate_configuration",
    "fairness_slices",
    "fit_cohort_normaliser",
    "fit_fusion_weights",
    "fit_policy",
    "fit_threshold",
    "fit_veto_floor",
    "fused_scores",
    "run_ablation",
    "split_by_speaker",
    "stability_curve",
]
