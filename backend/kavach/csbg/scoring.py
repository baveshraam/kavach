"""CSBG similarity scoring.

Two scorers live here, used for different jobs.

1. Log-likelihood ratio (`score_llr`) -- THE VERIFICATION SCORER.
   Asks: are these tokens more probable under the claimed speaker's CSBG
   than under a background model of speakers in general?

       s = (1/N) * sum_t [ log P_spk(lang_t | class_t) - log P_ubm(lang_t | class_t) ]

   This is the standard target/background likelihood-ratio framework used for
   speaker verification, and it is the same formulation Doddington (2001)
   applied to word n-grams for idiolectal speaker recognition -- KAVACH
   substitutes (semantic class -> language choice) for his word n-grams. Using
   his framework is deliberate: it makes the relationship to the prior art
   explicit and legible to a reviewer.

   It is the right scorer for authentication because a login response is
   short (5-20 tokens). It evaluates the *observed tokens* under a
   well-estimated enrolment model, rather than fitting a second noisy model
   to 12 tokens and comparing two unreliable distributions.

2. Jensen-Shannon divergence (`score_jsd`) -- THE COMPARISON SCORER.
   Symmetric, bounded in [0, 1], and defined between two *graphs*. Used for
   speaker-vs-speaker comparison in the Graph Explorer, for the CSBG
   stability analysis (comparing a speaker's graph across sessions), and as
   an ablation baseline. It needs both sides well-estimated, so it is NOT
   used to score short login probes.

Why the background model matters
--------------------------------
Without it, a token sequence that is simply *common* scores high for
everyone. If every Tamil-English speaker says numbers in English, then
"speaks numbers in English" carries no identifying information -- and the
LLR correctly assigns it ~0 because P_spk and P_ubm agree. Only choices that
deviate from the population contribute. This is what makes the score a
biometric rather than a language-plausibility check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .graph import CSBG, LANG_INDEX, SmoothingConfig
from .ontology import (
    CHOICE_LANGUAGES,
    CLASS_INDEX,
    Language,
    SemanticClass,
    scoring_classes,
)
from .tokens import UtteranceTokens

#: Floor applied inside logs. Smoothing already keeps probabilities strictly
#: positive, so this only guards against a hand-built or corrupted graph
#: reaching the scorer with a zero.
_EPS = 1e-12

#: Token counts between which the graph-level metrics term ramps from zero
#: influence to its full weight. Below _METRICS_MIN_TOKENS the probe's CMI and
#: I-index are dominated by sampling noise (see CodeMixingMetrics.is_reliable);
#: by _METRICS_FULL_TOKENS they are worth their configured weight.
_METRICS_MIN_TOKENS = 20
_METRICS_FULL_TOKENS = 60


def _ramp(value: float, *, low: float, high: float) -> float:
    """Linear ramp from 0 at `low` to 1 at `high`, clamped outside.

    Used instead of a hard threshold anywhere a term's trustworthiness grows
    with evidence, so that scores stay continuous in probe length.
    """
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """Relative weight of each CSBG evidence stream.

    Defaults reflect how much each stream can be trusted on a short probe,
    not measured optimality:

    - Lexical choice is the best-estimated stream and carries most weight.
    - Transitions are ~4x sparser, so they are informative but noisier.
    - Graph-level metrics (CMI etc.) are near-meaningless on a 12-token
      response -- see CodeMixingMetrics.is_reliable -- so they get a small
      weight and are skipped entirely when the probe is too short.

    Fit these on a dev split with `kavach.fusion.calibration` and report the
    fitted values in the paper.
    """

    lexical: float = 0.60
    transition: float = 0.30
    metrics: float = 0.10

    def __post_init__(self) -> None:
        total = self.lexical + self.transition + self.metrics
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"ScoringWeights must sum to 1.0, got {total}")
        if min(self.lexical, self.transition, self.metrics) < 0:
            raise ValueError("ScoringWeights must be non-negative")


@dataclass(frozen=True, slots=True)
class ClassContribution:
    """Per-class breakdown of an LLR score, for the explainability panel.

    This is what lets KAVACH tell a user *why* they were rejected -- "you
    said place names in English, this speaker says them in Tamil" -- which a
    512-dimensional embedding cannot do. It is a genuine advantage of the
    graph formulation and worth a figure in the paper.
    """

    semantic_class: SemanticClass
    n_tokens: int
    llr: float
    """Summed log-likelihood ratio contributed by this class. Negative values
    are evidence *against* the claimed identity."""

    expected_language: Language
    expected_prob: float
    observed_language: Language
    observed_prob: float
    """P(observed_language | class) under the claimed speaker's model."""

    observed_counts: tuple[int, ...] = (0, 0)
    """Raw probe counts per language, indexed by LANG_INDEX (i.e. [TA, EN]).

    Kept alongside `observed_language` because the dominant language alone
    cannot distinguish "said it in English four times" from "said it in
    English three times and Tamil twice". The explainability panel needs the
    full empirical distribution to state a divergence rather than a winner."""

    @property
    def is_evidence_against(self) -> bool:
        return self.llr < 0

    @property
    def observed_distribution(self) -> tuple[float, ...]:
        """`observed_counts` normalised to a probability vector.

        Uniform when nothing was observed, so a divergence against it is 0
        rather than undefined.
        """
        total = sum(self.observed_counts)
        if total <= 0:
            n = len(self.observed_counts) or 1
            return tuple(1.0 / n for _ in range(n))
        return tuple(c / total for c in self.observed_counts)


@dataclass(slots=True)
class CSBGScore:
    """Result of scoring a probe against a claimed speaker's CSBG."""

    raw_score: float
    """Fused, length-normalised LLR. Unbounded; ~0 means uninformative."""

    normalised_score: float
    """`raw_score` after cohort z-normalisation, then squashed to [0, 1].
    This is what the fusion layer consumes."""

    lexical_llr: float
    transition_llr: float
    metrics_term: float

    n_scored_tokens: int
    """Language-choice tokens that contributed. Low values mean low confidence."""

    contributions: list[ClassContribution] = field(default_factory=list)
    """Sorted most-negative first: the strongest evidence against the claim."""

    z_score: float | None = None
    """Cohort z-score, if a normaliser was supplied."""

    @property
    def is_reliable(self) -> bool:
        """Whether enough tokens were scored for this to mean anything.

        Below 5 scored tokens the fusion layer should fall back to the other
        branches rather than trusting this score -- reported as
        `insufficient_evidence` in the API response.
        """
        return self.n_scored_tokens >= 5

    def top_evidence_against(self, k: int = 3) -> list[ClassContribution]:
        return [c for c in self.contributions if c.is_evidence_against][:k]


def build_background_model(
    graphs: list[CSBG],
    *,
    speaker_id: str = "__ubm__",
    smoothing: SmoothingConfig | None = None,
) -> CSBG:
    """Pool enrolled speakers into a universal background model.

    The UBM answers "what does a Tamil-English speaker *in general* do?", so
    the LLR can measure how far a probe deviates from the population rather
    than from uniform.

    Pooling is over raw counts, not over fitted probabilities. Averaging
    probabilities would weight a speaker with 40 tokens the same as one with
    900; pooling counts weights by evidence, which is what we want.

    Args:
        graphs: Enrolled speaker CSBGs. Should exclude the speaker being
            tested when computing an unbiased score -- see
            `CohortNormaliser.leave_one_out`.
        speaker_id: Identifier for the resulting pseudo-speaker.
        smoothing: Defaults to the smoothing of the first graph so the UBM and
            the speaker models share a parameterisation.

    Raises:
        ValueError: If `graphs` is empty. A UBM over nothing is uniform, and
            an LLR against uniform is a language-plausibility check silently
            masquerading as a biometric -- better to fail loudly.
    """
    if not graphs:
        raise ValueError(
            "Cannot build a background model from zero graphs. At least one "
            "enrolled speaker is required (and results are only meaningful "
            "with several)."
        )

    sm = smoothing or graphs[0].smoothing

    lexical_counts = np.sum([g.lexical_counts for g in graphs], axis=0)
    transition_counts = np.sum([g.transition_counts for g in graphs], axis=0)

    global_probs = CSBG._fit_global(lexical_counts, sm)
    superclass_probs = CSBG._fit_superclass(lexical_counts, global_probs, sm)
    lexical_probs = CSBG._fit_lexical(lexical_counts, superclass_probs, sm)
    transition_probs = CSBG._fit_transition(transition_counts, lexical_probs, sm)

    pooled_metrics = graphs[0].metrics
    return CSBG(
        speaker_id=speaker_id,
        lexical_probs=lexical_probs,
        lexical_counts=lexical_counts,
        transition_probs=transition_probs,
        transition_counts=transition_counts,
        global_probs=global_probs,
        metrics=pooled_metrics,
        n_utterances=sum(g.n_utterances for g in graphs),
        total_duration_sec=sum(g.total_duration_sec for g in graphs),
        smoothing=sm,
    )


def score_llr(
    probe: list[UtteranceTokens],
    reference: CSBG,
    ubm: CSBG,
    *,
    weights: ScoringWeights | None = None,
    include_low_signal: bool = False,
    lid_confidence_floor: float = 0.0,
    metrics_scale: float = 25.0,
) -> CSBGScore:
    """Score probe tokens against a claimed speaker's CSBG. THE MAIN SCORER.

    Args:
        probe: Annotated tokens from the login response.
        reference: Claimed speaker's enrolled CSBG.
        ubm: Background model, ideally excluding `reference`'s speaker.
        weights: Stream weights. Defaults to ScoringWeights().
        include_low_signal: Include LOW_SIGNAL_CLASSES. For ablation only.
        lid_confidence_floor: Ignore tokens below this LID confidence.
        metrics_scale: Divisor converting the CMI distance into the LLR's
            rough scale so the weights mean what they appear to mean. Only
            applied when the probe is long enough for metrics to be reliable.

    Returns:
        A CSBGScore. `normalised_score` is unset (equal to a squashed
        `raw_score`) until a CohortNormaliser is applied.
    """
    weights = weights or ScoringWeights()
    allowed = set(scoring_classes(include_low_signal))

    lexical_total = 0.0
    n_lexical = 0
    transition_total = 0.0
    n_transition = 0

    per_class_llr: dict[SemanticClass, float] = {}
    per_class_count: dict[SemanticClass, int] = {}
    per_class_observed: dict[SemanticClass, dict[Language, int]] = {}

    for utt in probe:
        tokens = (
            utt.confident_tokens(lid_confidence_floor)
            if lid_confidence_floor > 0
            else utt.choice_tokens
        )
        scored = [t for t in tokens if t.semantic_class in allowed]

        for tok in scored:
            ci = CLASS_INDEX[tok.semantic_class]
            li = LANG_INDEX[tok.language]
            llr = math.log(max(reference.lexical_probs[ci, li], _EPS)) - math.log(
                max(ubm.lexical_probs[ci, li], _EPS)
            )
            lexical_total += llr
            n_lexical += 1

            per_class_llr[tok.semantic_class] = per_class_llr.get(tok.semantic_class, 0.0) + llr
            per_class_count[tok.semantic_class] = per_class_count.get(tok.semantic_class, 0) + 1
            obs = per_class_observed.setdefault(tok.semantic_class, {})
            obs[tok.language] = obs.get(tok.language, 0) + 1

        # Transitions are taken over the *unfiltered* choice sequence so that
        # removing a low-signal token does not fabricate an adjacency between
        # its neighbours.
        for prev, cur in zip(tokens, tokens[1:]):
            if cur.semantic_class not in allowed:
                continue
            ci = CLASS_INDEX[cur.semantic_class]
            pi = LANG_INDEX[prev.language]
            li = LANG_INDEX[cur.language]
            transition_total += math.log(
                max(reference.transition_probs[ci, pi, li], _EPS)
            ) - math.log(max(ubm.transition_probs[ci, pi, li], _EPS))
            n_transition += 1

    # Length-normalise so a 6-token and a 60-token response are comparable.
    lexical_llr = lexical_total / n_lexical if n_lexical else 0.0
    transition_llr = transition_total / n_transition if n_transition else 0.0

    # Graph-level metrics (CMI, I-index) need a long probe to mean anything,
    # but they must not switch on abruptly: a hard cutoff at 20 tokens would
    # make a fixed decision threshold behave differently for a terse answer
    # than a chatty one, and response length is not under our control at
    # login. Instead the term ramps in with the evidence supporting it, and
    # the remaining weight is redistributed over the two LLR streams so the
    # effective weights always sum to 1.
    metrics_term = 0.0
    metrics_confidence = _ramp(n_lexical, low=_METRICS_MIN_TOKENS, high=_METRICS_FULL_TOKENS)
    if metrics_confidence > 0.0:
        from .metrics import compute_all_metrics

        probe_metrics = compute_all_metrics(probe)
        cmi_distance = abs(probe_metrics.cmi - reference.metrics.cmi) / 100.0
        i_distance = abs(probe_metrics.i_index - reference.metrics.i_index)
        # Negative: distance is evidence against, matching the LLR's sign.
        metrics_term = -((cmi_distance + i_distance) / 2.0) * metrics_scale

    w_metrics = weights.metrics * metrics_confidence
    llr_share = 1.0 - w_metrics
    llr_total = weights.lexical + weights.transition
    w_lexical = weights.lexical / llr_total * llr_share
    w_transition = weights.transition / llr_total * llr_share

    raw = w_lexical * lexical_llr + w_transition * transition_llr + w_metrics * metrics_term

    contributions = _build_contributions(
        per_class_llr, per_class_count, per_class_observed, reference
    )

    return CSBGScore(
        raw_score=raw,
        normalised_score=_squash(raw),
        lexical_llr=lexical_llr,
        transition_llr=transition_llr,
        metrics_term=metrics_term,
        n_scored_tokens=n_lexical,
        contributions=contributions,
    )


def _build_contributions(
    per_class_llr: dict[SemanticClass, float],
    per_class_count: dict[SemanticClass, int],
    per_class_observed: dict[SemanticClass, dict[Language, int]],
    reference: CSBG,
) -> list[ClassContribution]:
    """Assemble the per-class explainability breakdown, worst evidence first."""
    out: list[ClassContribution] = []
    for cls_, llr in per_class_llr.items():
        observed_counts = per_class_observed.get(cls_, {})
        if not observed_counts:
            continue
        observed_lang = max(observed_counts, key=lambda k: observed_counts[k])
        expected_lang, expected_p = reference.dominant_language(cls_)
        ref_probs = reference.p_lang_given_class(cls_)
        out.append(
            ClassContribution(
                semantic_class=cls_,
                n_tokens=per_class_count[cls_],
                llr=llr,
                expected_language=expected_lang,
                expected_prob=expected_p,
                observed_language=observed_lang,
                observed_prob=float(ref_probs[LANG_INDEX[observed_lang]]),
                observed_counts=tuple(
                    observed_counts.get(lang, 0) for lang in CHOICE_LANGUAGES
                ),
            )
        )
    out.sort(key=lambda c: c.llr)
    return out


def _squash(x: float, scale: float = 2.0) -> float:
    """Map an unbounded score to (0, 1) with a logistic.

    Presentation only -- the fusion layer and every EER computation use the
    raw or z-normalised score. Squashing before thresholding would be
    harmless but pointless; squashing before *averaging* would distort, so
    fusion deliberately does not consume this.
    """
    return 1.0 / (1.0 + math.exp(-x / scale))


# --------------------------------------------------------------------------
# Cohort normalisation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CohortNormaliser:
    """Z-normalisation of raw LLR scores against an impostor cohort.

    Raw LLRs are not comparable across speakers: a speaker with unusual
    habits gets large-magnitude scores for everyone, while a speaker who
    behaves like the population average gets near-zero scores even when
    genuine. Z-norm rescales per claimed speaker:

        z = (s - mu_spk) / sigma_spk

    where mu and sigma come from scoring *other speakers'* utterances against
    this speaker's model. This is standard practice in speaker verification
    and materially improves EER; report results both with and without it.
    """

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    default_std: float = 1.0

    def fit_speaker(self, speaker_id: str, impostor_scores: list[float]) -> None:
        """Fit normalisation statistics for one speaker.

        Args:
            speaker_id: Speaker whose model produced these scores.
            impostor_scores: Raw LLRs from other speakers' utterances scored
                against this speaker's CSBG.
        """
        if len(impostor_scores) < 2:
            # Not enough cohort data; fall back to identity normalisation
            # rather than a std of 0 that would produce infinities.
            self.means[speaker_id] = 0.0
            self.stds[speaker_id] = self.default_std
            return
        arr = np.asarray(impostor_scores, dtype=np.float64)
        self.means[speaker_id] = float(arr.mean())
        std = float(arr.std(ddof=1))
        self.stds[speaker_id] = std if std > 1e-6 else self.default_std

    def apply(self, speaker_id: str, raw_score: float) -> float:
        """Return the z-normalised score for `speaker_id`."""
        mean = self.means.get(speaker_id, 0.0)
        std = self.stds.get(speaker_id, self.default_std)
        return (raw_score - mean) / std

    def apply_squashed(self, speaker_id: str, raw_score: float) -> float:
        """Z-normalise and squash to (0, 1) -- the scale fusion consumes.

        The squash scale here is 1.5, not the 2.0 used on un-normalised
        scores: a z-score is already in units of the cohort's spread, so it
        occupies a much narrower range and needs less compression to land in
        a usable part of the logistic. Using the same constant for both would
        make z-normed scores cluster near 0.5 and quietly flatten the branch.
        """
        return _squash(self.apply(speaker_id, raw_score), scale=1.5)

    def normalise(self, speaker_id: str, score: CSBGScore) -> CSBGScore:
        """Apply z-norm to a CSBGScore in place and return it."""
        z = self.apply(speaker_id, score.raw_score)
        score.z_score = z
        score.normalised_score = _squash(z, scale=1.5)
        return score

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": dict(self.means), "stds": dict(self.stds)}

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, float]]) -> CohortNormaliser:
        out = cls()
        out.means = dict(data.get("means", {}))
        out.stds = dict(data.get("stds", {}))
        return out


# --------------------------------------------------------------------------
# Graph-to-graph comparison (explorer, stability analysis, ablation)
# --------------------------------------------------------------------------


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence in bits, in [0, 1] for base-2 logs."""
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return max(0.0, min(1.0, 0.5 * (kl_pm + kl_qm)))


@dataclass(frozen=True, slots=True)
class ClassDivergence:
    """Per-class JSD between two graphs, for the comparison view."""

    semantic_class: SemanticClass
    jsd: float
    a_ta_prob: float
    b_ta_prob: float
    a_count: float
    b_count: float


def score_jsd(
    a: CSBG,
    b: CSBG,
    *,
    include_low_signal: bool = False,
    min_count: float = 3.0,
) -> tuple[float, list[ClassDivergence]]:
    """Symmetric divergence between two well-estimated CSBGs, in [0, 1].

    Weighted by the geometric mean of both graphs' observation counts, so a
    class only influences the result when *both* speakers exercised it.

    NOT for scoring short login probes -- use `score_llr`. Intended for
    speaker-vs-speaker comparison, cross-session stability measurement, and
    as an ablation baseline against the LLR scorer.

    Args:
        min_count: Classes below this count in either graph are skipped.

    Returns:
        (weighted mean JSD, per-class divergences sorted descending).
    """
    classes = scoring_classes(include_low_signal)
    divergences: list[ClassDivergence] = []
    weights: list[float] = []

    for cls_ in classes:
        ca = a.observation_count(cls_)
        cb = b.observation_count(cls_)
        if ca < min_count or cb < min_count:
            continue
        pa = a.p_lang_given_class(cls_)
        pb = b.p_lang_given_class(cls_)
        jsd = _jensen_shannon(pa, pb)
        divergences.append(
            ClassDivergence(
                semantic_class=cls_,
                jsd=jsd,
                a_ta_prob=float(pa[LANG_INDEX[Language.TA]]),
                b_ta_prob=float(pb[LANG_INDEX[Language.TA]]),
                a_count=ca,
                b_count=cb,
            )
        )
        weights.append(math.sqrt(ca * cb))

    if not divergences:
        return 0.0, []

    total_w = sum(weights)
    mean_jsd = sum(d.jsd * w for d, w in zip(divergences, weights)) / total_w
    divergences.sort(key=lambda d: d.jsd, reverse=True)
    return mean_jsd, divergences


def discriminative_classes(
    speaker: CSBG, ubm: CSBG, *, min_count: float = 5.0, top_k: int = 5
) -> list[tuple[SemanticClass, float]]:
    """Classes where this speaker most deviates from the population.

    Drives adaptive challenge targeting: these are the domains where asking a
    question yields the most identifying information about this speaker.
    A speaker who says numbers in Tamil while everyone else uses English is
    most efficiently identified by being asked a question about numbers.

    Returns:
        Up to `top_k` (class, JSD-vs-UBM) pairs, most discriminative first.
    """
    scored: list[tuple[SemanticClass, float]] = []
    for cls_ in scoring_classes():
        if speaker.observation_count(cls_) < min_count:
            continue
        jsd = _jensen_shannon(speaker.p_lang_given_class(cls_), ubm.p_lang_given_class(cls_))
        scored.append((cls_, jsd))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
