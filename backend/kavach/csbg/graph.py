"""Code-Switch Behaviour Graph construction.

The CSBG is KAVACH's speaker template: a graph whose edges encode *which
language this speaker chooses for which semantic domain*, and *where they
switch*.

    Node sets
        V_C   21 semantic concept classes (ontology.SemanticClass)
        V_L   2 language nodes (TA, EN)

    Edge sets
        E_lex    class -> language, weight P(lang | class)
                 "this speaker says numbers in English 94% of the time"
        E_trans  (class, prev_lang) -> language, weight P(lang | class, prev_lang)
                 "...and switches INTO English for numbers even mid-Tamil"

    Graph attributes
        CMI, M-index, I-index, burstiness (metrics.CodeMixingMetrics)

The estimation problem
----------------------
Five minutes of speech gives ~600-900 tokens. Spread over 21 classes that is
~30 per class for E_lex -- workable. But E_trans has 21 x 2 x 2 = 84 cells,
many of which will be empty for any individual speaker. Maximum-likelihood
estimates there are useless: a class seen 3 times, all in English, would give
P(EN | class) = 1.0 and produce infinite divergence against a speaker who
said it in Tamil once.

The fix is hierarchical Dirichlet smoothing. Each distribution backs off
toward its parent:

    P(lang | class)             backs off to  P(lang | superclass)
    P(lang | superclass)        backs off to  P(lang)                [speaker marginal]
    P(lang | class, prev_lang)  backs off to  P(lang | class)

So a class with 3 observations sits close to the speaker's general tendency
and contributes little divergence, while a class with 80 observations
dominates its own estimate. Confidence scales with evidence automatically,
which is the behaviour we want for a biometric that must not fire on noise.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics import CodeMixingMetrics, compute_all_metrics
from .ontology import (
    CHOICE_LANGUAGES,
    CLASS_INDEX,
    CLASS_ORDER,
    N_CLASSES,
    ONTOLOGY_VERSION,
    SUPERCLASS_OF,
    Language,
    SemanticClass,
    SuperClass,
)
from .tokens import UtteranceTokens

#: Index of each language in every probability vector. Fixed as [TA, EN].
LANG_INDEX: dict[Language, int] = {lang: i for i, lang in enumerate(CHOICE_LANGUAGES)}
N_LANGS = len(CHOICE_LANGUAGES)


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    """Dirichlet concentration parameters for the backoff hierarchy.

    Each value is a pseudo-count: `class_alpha=4.0` means a class needs about
    4 real observations before its own evidence outweighs its parent's prior.

    Defaults are set from the sparsity arithmetic above, not tuned on data.
    Tune them on a dev split and report the values used -- and note that
    tuning them on the test speakers would leak.
    """

    class_alpha: float = 4.0
    """Pseudo-count pulling P(lang | class) toward its superclass prior."""

    superclass_alpha: float = 8.0
    """Pseudo-count pulling P(lang | superclass) toward the speaker marginal."""

    global_alpha: float = 2.0
    """Pseudo-count pulling the speaker marginal toward uniform. Small: with
    hundreds of tokens the marginal is well estimated and needs little help."""

    transition_alpha: float = 6.0
    """Pseudo-count pulling P(lang | class, prev_lang) toward P(lang | class).
    Higher than class_alpha because transition cells are ~4x sparser."""

    def __post_init__(self) -> None:
        for name in (
            "class_alpha",
            "superclass_alpha",
            "global_alpha",
            "transition_alpha",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")


@dataclass(slots=True)
class CSBG:
    """A speaker's Code-Switch Behaviour Graph.

    Construct with `CSBG.build(...)`; the fields are the fitted parameters.
    All probability arrays are indexed by `CLASS_ORDER` and `LANG_INDEX`.
    """

    speaker_id: str

    lexical_probs: np.ndarray
    """P(lang | class), shape (N_CLASSES, N_LANGS). Smoothed."""

    lexical_counts: np.ndarray
    """Raw observation counts, shape (N_CLASSES, N_LANGS). Unsmoothed."""

    transition_probs: np.ndarray
    """P(lang_j | class_j, lang_i), shape (N_CLASSES, N_LANGS, N_LANGS),
    indexed [class_j, prev_lang, next_lang]. Smoothed."""

    transition_counts: np.ndarray
    """Raw transition counts, same shape. Unsmoothed."""

    global_probs: np.ndarray
    """Speaker's marginal P(lang), shape (N_LANGS,). Smoothed."""

    metrics: CodeMixingMetrics
    """Graph-level code-mixing attributes."""

    n_utterances: int = 0
    total_duration_sec: float = 0.0
    ontology_version: str = ONTOLOGY_VERSION
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    # ---------------------------------------------------------------- build

    @classmethod
    def build(
        cls,
        speaker_id: str,
        utterances: list[UtteranceTokens],
        *,
        smoothing: SmoothingConfig | None = None,
        lid_confidence_floor: float = 0.0,
        total_duration_sec: float = 0.0,
    ) -> CSBG:
        """Fit a CSBG from a speaker's annotated utterances.

        Args:
            speaker_id: Identifier stored on the graph.
            utterances: Annotated utterances. May be empty -- the result is a
                valid graph sitting entirely at the smoothing prior, which
                scores near-chance against everything rather than crashing.
            smoothing: Dirichlet parameters. Defaults to SmoothingConfig().
            lid_confidence_floor: Drop language-choice tokens below this LID
                confidence. 0.0 keeps everything. Raising it trades coverage
                for annotation quality; see the LID-noise ablation.
            total_duration_sec: Stored for the enrolment-duration stability
                analysis. Not used in the maths.

        Returns:
            A fitted CSBG.
        """
        smoothing = smoothing or SmoothingConfig()

        lexical_counts = np.zeros((N_CLASSES, N_LANGS), dtype=np.float64)
        transition_counts = np.zeros((N_CLASSES, N_LANGS, N_LANGS), dtype=np.float64)

        for utt in utterances:
            choice = (
                utt.confident_tokens(lid_confidence_floor)
                if lid_confidence_floor > 0
                else utt.choice_tokens
            )

            for tok in choice:
                lexical_counts[CLASS_INDEX[tok.semantic_class], LANG_INDEX[tok.language]] += 1

            # Transitions stay inside the utterance -- see tokens.UtteranceTokens.
            for prev, cur in zip(choice, choice[1:]):
                transition_counts[
                    CLASS_INDEX[cur.semantic_class],
                    LANG_INDEX[prev.language],
                    LANG_INDEX[cur.language],
                ] += 1

        global_probs = cls._fit_global(lexical_counts, smoothing)
        superclass_probs = cls._fit_superclass(lexical_counts, global_probs, smoothing)
        lexical_probs = cls._fit_lexical(lexical_counts, superclass_probs, smoothing)
        transition_probs = cls._fit_transition(transition_counts, lexical_probs, smoothing)

        return cls(
            speaker_id=speaker_id,
            lexical_probs=lexical_probs,
            lexical_counts=lexical_counts,
            transition_probs=transition_probs,
            transition_counts=transition_counts,
            global_probs=global_probs,
            metrics=compute_all_metrics(utterances),
            n_utterances=len(utterances),
            total_duration_sec=total_duration_sec,
            ontology_version=ONTOLOGY_VERSION,
            smoothing=smoothing,
        )

    # ------------------------------------------------------- fitting stages

    @staticmethod
    def _fit_global(counts: np.ndarray, sm: SmoothingConfig) -> np.ndarray:
        """Speaker marginal P(lang), smoothed toward uniform."""
        totals = counts.sum(axis=0)
        uniform = np.full(N_LANGS, 1.0 / N_LANGS)
        return (totals + sm.global_alpha * uniform) / (totals.sum() + sm.global_alpha)

    @staticmethod
    def _fit_superclass(
        counts: np.ndarray, global_probs: np.ndarray, sm: SmoothingConfig
    ) -> dict[SuperClass, np.ndarray]:
        """P(lang | superclass), smoothed toward the speaker marginal."""
        pooled: dict[SuperClass, np.ndarray] = {
            sc: np.zeros(N_LANGS, dtype=np.float64) for sc in SuperClass
        }
        for cls_ in CLASS_ORDER:
            pooled[SUPERCLASS_OF[cls_]] += counts[CLASS_INDEX[cls_]]

        return {
            sc: (vec + sm.superclass_alpha * global_probs) / (vec.sum() + sm.superclass_alpha)
            for sc, vec in pooled.items()
        }

    @staticmethod
    def _fit_lexical(
        counts: np.ndarray,
        superclass_probs: dict[SuperClass, np.ndarray],
        sm: SmoothingConfig,
    ) -> np.ndarray:
        """P(lang | class), smoothed toward the superclass prior."""
        probs = np.zeros((N_CLASSES, N_LANGS), dtype=np.float64)
        for cls_ in CLASS_ORDER:
            i = CLASS_INDEX[cls_]
            prior = superclass_probs[SUPERCLASS_OF[cls_]]
            row = counts[i]
            probs[i] = (row + sm.class_alpha * prior) / (row.sum() + sm.class_alpha)
        return probs

    @staticmethod
    def _fit_transition(
        counts: np.ndarray, lexical_probs: np.ndarray, sm: SmoothingConfig
    ) -> np.ndarray:
        """P(lang_j | class_j, lang_i), smoothed toward P(lang_j | class_j)."""
        probs = np.zeros((N_CLASSES, N_LANGS, N_LANGS), dtype=np.float64)
        for ci in range(N_CLASSES):
            prior = lexical_probs[ci]
            for prev in range(N_LANGS):
                row = counts[ci, prev]
                probs[ci, prev] = (row + sm.transition_alpha * prior) / (
                    row.sum() + sm.transition_alpha
                )
        return probs

    # ------------------------------------------------------------ accessors

    def p_lang_given_class(self, cls_: SemanticClass) -> np.ndarray:
        """Smoothed P(lang | class) as a length-2 vector indexed [TA, EN]."""
        return self.lexical_probs[CLASS_INDEX[cls_]]

    def observation_count(self, cls_: SemanticClass) -> float:
        """Raw (unsmoothed) token count for a class -- the evidence behind it."""
        return float(self.lexical_counts[CLASS_INDEX[cls_]].sum())

    def observed_classes(self, min_count: float = 1.0) -> list[SemanticClass]:
        """Classes with at least `min_count` observations, most frequent first."""
        counted = [
            (cls_, self.observation_count(cls_))
            for cls_ in CLASS_ORDER
            if self.observation_count(cls_) >= min_count
        ]
        counted.sort(key=lambda pair: pair[1], reverse=True)
        return [cls_ for cls_, _ in counted]

    def sparse_classes(self, threshold: float = 5.0) -> list[SemanticClass]:
        """Classes with too few observations to be estimated independently.

        Surfaced in the enrolment UI as a warning and used by the challenge
        generator to decide what to probe for next.
        """
        return [c for c in CLASS_ORDER if self.observation_count(c) < threshold]

    @property
    def total_observations(self) -> float:
        return float(self.lexical_counts.sum())

    @property
    def density(self) -> float:
        """Fraction of classes with enough data to be estimated, in [0, 1].

        A rough completeness signal for the UI: a speaker at 0.3 has only
        exercised a third of the ontology and their CSBG will generalise
        poorly to challenges probing the rest.
        """
        return sum(1 for c in CLASS_ORDER if self.observation_count(c) >= 5.0) / N_CLASSES

    def dominant_language(self, cls_: SemanticClass) -> tuple[Language, float]:
        """Most probable language for a class, and its probability."""
        probs = self.p_lang_given_class(cls_)
        i = int(np.argmax(probs))
        return CHOICE_LANGUAGES[i], float(probs[i])

    # -------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation for persistence and the API."""
        return {
            "speaker_id": self.speaker_id,
            "ontology_version": self.ontology_version,
            "lexical_probs": self.lexical_probs.tolist(),
            "lexical_counts": self.lexical_counts.tolist(),
            "transition_probs": self.transition_probs.tolist(),
            "transition_counts": self.transition_counts.tolist(),
            "global_probs": self.global_probs.tolist(),
            "metrics": {
                "cmi": self.metrics.cmi,
                "m_index": self.metrics.m_index,
                "i_index": self.metrics.i_index,
                "burstiness": self.metrics.burstiness,
                "n_tokens": self.metrics.n_tokens,
                "n_choice_tokens": self.metrics.n_choice_tokens,
                "n_switches": self.metrics.n_switches,
                "ta_fraction": self.metrics.ta_fraction,
            },
            "n_utterances": self.n_utterances,
            "total_duration_sec": self.total_duration_sec,
            "smoothing": {
                "class_alpha": self.smoothing.class_alpha,
                "superclass_alpha": self.smoothing.superclass_alpha,
                "global_alpha": self.smoothing.global_alpha,
                "transition_alpha": self.smoothing.transition_alpha,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CSBG:
        """Inverse of `to_dict`.

        Raises:
            ValueError: If the stored graph was built under a different
                ontology version. Comparing graphs across ontology versions
                silently compares different class indices, which would
                produce plausible-looking nonsense.
        """
        stored = data.get("ontology_version")
        if stored != ONTOLOGY_VERSION:
            raise ValueError(
                f"CSBG for speaker {data.get('speaker_id')!r} was built under ontology "
                f"version {stored!r} but this build is {ONTOLOGY_VERSION!r}. "
                "Re-run enrolment to rebuild the graph."
            )

        m = data["metrics"]
        return cls(
            speaker_id=data["speaker_id"],
            lexical_probs=np.asarray(data["lexical_probs"], dtype=np.float64),
            lexical_counts=np.asarray(data["lexical_counts"], dtype=np.float64),
            transition_probs=np.asarray(data["transition_probs"], dtype=np.float64),
            transition_counts=np.asarray(data["transition_counts"], dtype=np.float64),
            global_probs=np.asarray(data["global_probs"], dtype=np.float64),
            metrics=CodeMixingMetrics(**m),
            n_utterances=data.get("n_utterances", 0),
            total_duration_sec=data.get("total_duration_sec", 0.0),
            ontology_version=stored,
            smoothing=SmoothingConfig(**data.get("smoothing", {})),
        )

    def __repr__(self) -> str:
        return (
            f"CSBG(speaker={self.speaker_id!r}, obs={self.total_observations:.0f}, "
            f"density={self.density:.2f}, CMI={self.metrics.cmi:.1f}, "
            f"I={self.metrics.i_index:.3f})"
        )


def class_frequency_weights(
    graph: CSBG, classes: tuple[SemanticClass, ...]
) -> dict[SemanticClass, float]:
    """Normalised observation-count weights over `classes`, summing to 1.

    Used to weight divergence terms so that a class the speaker actually
    exercised counts for more than one seen twice. Returns uniform weights
    when no class in the set was observed, so scoring degrades gracefully on
    an empty utterance instead of dividing by zero.
    """
    counts: Counter[SemanticClass] = Counter(
        {c: graph.observation_count(c) for c in classes}
    )
    total = sum(counts.values())
    if total <= 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: counts[c] / total for c in classes}
