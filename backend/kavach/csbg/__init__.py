"""Code-Switch Behaviour Graph -- KAVACH's research core.

The CSBG encodes a bilingual speaker's language-choice habits as a graph over
semantic concept classes, and scores probe utterances against it with a
likelihood ratio against a background model.

Typical use::

    from kavach.csbg import CSBG, build_background_model, score_llr

    graph = CSBG.build("speaker_07", enrolment_utterances)
    ubm = build_background_model([g for g in all_graphs if g.speaker_id != "speaker_07"])
    score = score_llr(login_utterance, graph, ubm)

This subpackage has no ML dependencies -- numpy and scipy only -- so it can be
tested and iterated on without loading torch or any speech model.
"""

from .graph import CSBG, SmoothingConfig, class_frequency_weights
from .metrics import (
    CodeMixingMetrics,
    compute_all_metrics,
    compute_burstiness,
    compute_cmi,
    compute_i_index,
    compute_m_index,
)
from .ontology import (
    CHOICE_LANGUAGES,
    CLASS_DESCRIPTIONS,
    CLASS_ORDER,
    ELICITABLE_CLASSES,
    LOW_SIGNAL_CLASSES,
    ONTOLOGY_VERSION,
    Language,
    SemanticClass,
    SuperClass,
    scoring_classes,
)
from .scoring import (
    ClassContribution,
    ClassDivergence,
    CohortNormaliser,
    CSBGScore,
    ScoringWeights,
    build_background_model,
    discriminative_classes,
    score_jsd,
    score_llr,
)
from .tokens import Token, UtteranceTokens, count_switch_points

__all__ = [
    # ontology
    "Language",
    "SemanticClass",
    "SuperClass",
    "CLASS_ORDER",
    "CLASS_DESCRIPTIONS",
    "CHOICE_LANGUAGES",
    "ELICITABLE_CLASSES",
    "LOW_SIGNAL_CLASSES",
    "ONTOLOGY_VERSION",
    "scoring_classes",
    # tokens
    "Token",
    "UtteranceTokens",
    "count_switch_points",
    # metrics
    "CodeMixingMetrics",
    "compute_all_metrics",
    "compute_cmi",
    "compute_m_index",
    "compute_i_index",
    "compute_burstiness",
    # graph
    "CSBG",
    "SmoothingConfig",
    "class_frequency_weights",
    # scoring
    "CSBGScore",
    "ScoringWeights",
    "ClassContribution",
    "ClassDivergence",
    "CohortNormaliser",
    "build_background_model",
    "score_llr",
    "score_jsd",
    "discriminative_classes",
]
