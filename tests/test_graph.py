"""Tests for CSBG construction, smoothing, and serialisation.

The smoothing hierarchy is the part most likely to be silently wrong: a bug
there produces graphs that look plausible and score badly, which is very hard
to diagnose from EER alone. These tests pin its behaviour precisely.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from kavach.csbg.graph import LANG_INDEX, CSBG, SmoothingConfig
from kavach.csbg.ontology import N_CLASSES, ONTOLOGY_VERSION, Language, SemanticClass
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.simulation import make_population, sample_session

TA, EN, NEU = Language.TA, Language.EN, Language.NEUTRAL


def make_utterance(pairs: list[tuple[Language, SemanticClass]], uid: str = "u") -> UtteranceTokens:
    return UtteranceTokens(
        utterance_id=uid,
        tokens=[
            Token(text=f"w{i}", language=lang, semantic_class=cls_)
            for i, (lang, cls_) in enumerate(pairs)
        ],
    )


class TestConstruction:
    def test_shapes(self):
        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)])])
        assert g.lexical_probs.shape == (N_CLASSES, 2)
        assert g.transition_probs.shape == (N_CLASSES, 2, 2)
        assert g.global_probs.shape == (2,)

    def test_all_distributions_normalised(self):
        """Every row must sum to 1 or the LLR is not a likelihood ratio."""
        profiles = make_population(1, seed=1)
        g = CSBG.build("s1", sample_session(profiles[0], random.Random(1)))

        assert g.global_probs.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(g.lexical_probs.sum(axis=1), 1.0, rtol=1e-9)
        np.testing.assert_allclose(g.transition_probs.sum(axis=2), 1.0, rtol=1e-9)

    def test_counts_are_unsmoothed(self):
        """Raw counts must stay integral -- smoothing applies only to probs."""
        g = CSBG.build(
            "s1",
            [make_utterance([(TA, SemanticClass.FOOD), (TA, SemanticClass.FOOD)])],
        )
        assert g.observation_count(SemanticClass.FOOD) == 2.0
        assert g.observation_count(SemanticClass.NUMBER) == 0.0

    def test_neutral_tokens_excluded_from_counts(self):
        g = CSBG.build(
            "s1",
            [
                make_utterance(
                    [
                        (TA, SemanticClass.FOOD),
                        (NEU, SemanticClass.FOOD),
                        (Language.NAMED_ENTITY, SemanticClass.FOOD),
                    ]
                )
            ],
        )
        assert g.observation_count(SemanticClass.FOOD) == 1.0

    def test_transitions_not_counted_across_utterances(self):
        """Two separate monolingual utterances must produce no cross-transition."""
        utterances = [
            make_utterance([(TA, SemanticClass.FOOD)], "u1"),
            make_utterance([(EN, SemanticClass.FOOD)], "u2"),
        ]
        g = CSBG.build("s1", utterances)
        assert g.transition_counts.sum() == 0.0

    def test_transitions_counted_within_utterance(self):
        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD), (EN, SemanticClass.NUMBER)])])
        idx = (
            list(SemanticClass).index(SemanticClass.NUMBER),
            LANG_INDEX[TA],
            LANG_INDEX[EN],
        )
        assert g.transition_counts[idx] == 1.0
        assert g.transition_counts.sum() == 1.0

    def test_empty_input_produces_valid_graph(self):
        """A speaker with no usable audio must not crash enrolment.

        The resulting graph sits entirely at the prior and will score near
        chance, which is the correct behaviour -- not an exception mid-enrolment.
        """
        g = CSBG.build("s1", [])
        assert g.total_observations == 0.0
        np.testing.assert_allclose(g.lexical_probs.sum(axis=1), 1.0, rtol=1e-9)
        np.testing.assert_allclose(g.global_probs, [0.5, 0.5])

    def test_lid_confidence_floor_filters(self):
        u = UtteranceTokens(
            utterance_id="u",
            tokens=[
                Token("a", TA, SemanticClass.FOOD, lid_confidence=0.95),
                Token("b", EN, SemanticClass.FOOD, lid_confidence=0.30),
            ],
        )
        assert CSBG.build("s", [u], lid_confidence_floor=0.5).observation_count(
            SemanticClass.FOOD
        ) == 1.0
        assert CSBG.build("s", [u]).observation_count(SemanticClass.FOOD) == 2.0


class TestSmoothing:
    def test_unobserved_class_falls_back_to_speaker_tendency(self):
        """A class never spoken must inherit the speaker's general bias.

        This is the core backoff property: an unseen class should not sit at
        50/50, it should reflect that this speaker mostly speaks Tamil.
        """
        utterances = [
            make_utterance([(TA, SemanticClass.FOOD)] * 40, "u1"),
            make_utterance([(TA, SemanticClass.KINSHIP)] * 40, "u2"),
        ]
        g = CSBG.build("s1", utterances)
        unseen = g.p_lang_given_class(SemanticClass.TECH_DIGITAL)
        assert unseen[LANG_INDEX[TA]] > 0.7, "unseen class ignored the speaker's Tamil bias"

    def test_sparse_class_pulled_toward_prior(self):
        """One observation must not produce a confident estimate."""
        utterances = [
            make_utterance([(TA, SemanticClass.FOOD)] * 50, "u1"),
            make_utterance([(EN, SemanticClass.NUMBER)], "u2"),  # single EN observation
        ]
        g = CSBG.build("s1", utterances)
        p_en = g.p_lang_given_class(SemanticClass.NUMBER)[LANG_INDEX[EN]]
        assert 0.2 < p_en < 0.75, f"single observation gave overconfident P(EN)={p_en:.3f}"

    def test_dense_class_dominates_its_own_estimate(self):
        """With plenty of evidence the estimate should approach the empirical rate."""
        utterances = [make_utterance([(EN, SemanticClass.NUMBER)] * 200, "u")]
        g = CSBG.build("s1", utterances)
        assert g.p_lang_given_class(SemanticClass.NUMBER)[LANG_INDEX[EN]] > 0.95

    def test_no_zero_probabilities(self):
        """Any zero would make the LLR infinite on a single surprising token."""
        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)] * 100)])
        assert (g.lexical_probs > 0).all()
        assert (g.transition_probs > 0).all()

    def test_superclass_backoff_shares_evidence(self):
        """Evidence should flow between classes in the same superclass.

        NUMBER and MONEY_COMMERCE are both QUANTITATIVE, so heavy English use
        in NUMBER should bias unseen MONEY_COMMERCE toward English more than
        it biases unrelated KINSHIP (SOCIAL).
        """
        utterances = [
            make_utterance([(EN, SemanticClass.NUMBER)] * 60, "u1"),
            make_utterance([(TA, SemanticClass.FOOD)] * 60, "u2"),
        ]
        g = CSBG.build("s1", utterances)
        money_en = g.p_lang_given_class(SemanticClass.MONEY_COMMERCE)[LANG_INDEX[EN]]
        kinship_en = g.p_lang_given_class(SemanticClass.KINSHIP)[LANG_INDEX[EN]]
        assert money_en > kinship_en, "superclass backoff did not share evidence"

    def test_invalid_smoothing_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            SmoothingConfig(class_alpha=0.0)


class TestAccessors:
    def test_dominant_language(self):
        g = CSBG.build("s1", [make_utterance([(EN, SemanticClass.NUMBER)] * 50)])
        lang, prob = g.dominant_language(SemanticClass.NUMBER)
        assert lang is EN
        assert prob > 0.9

    def test_sparse_classes_listed(self):
        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)] * 50)])
        sparse = g.sparse_classes(threshold=5.0)
        assert SemanticClass.FOOD not in sparse
        assert SemanticClass.TECH_DIGITAL in sparse

    def test_observed_classes_ordered_by_frequency(self):
        utterances = [
            make_utterance(
                [(TA, SemanticClass.FOOD)] * 10 + [(EN, SemanticClass.NUMBER)] * 30
            )
        ]
        g = CSBG.build("s1", utterances)
        assert g.observed_classes()[0] is SemanticClass.NUMBER

    def test_density_in_range(self):
        profiles = make_population(1, seed=3)
        g = CSBG.build("s1", sample_session(profiles[0], random.Random(3)))
        assert 0.0 <= g.density <= 1.0


class TestSerialisation:
    def test_roundtrip_preserves_everything(self):
        profiles = make_population(1, seed=7)
        original = CSBG.build("s1", sample_session(profiles[0], random.Random(7)))
        restored = CSBG.from_dict(original.to_dict())

        assert restored.speaker_id == original.speaker_id
        np.testing.assert_allclose(restored.lexical_probs, original.lexical_probs)
        np.testing.assert_allclose(restored.transition_probs, original.transition_probs)
        np.testing.assert_allclose(restored.lexical_counts, original.lexical_counts)
        assert restored.metrics.cmi == pytest.approx(original.metrics.cmi)
        assert restored.smoothing == original.smoothing

    def test_ontology_version_mismatch_rejected(self):
        """Loading a graph built under a different ontology must fail loudly.

        Class indices would silently shift, producing confident nonsense.
        """
        data = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)])]).to_dict()
        data["ontology_version"] = "0.9-old"
        with pytest.raises(ValueError, match="ontology version"):
            CSBG.from_dict(data)

    def test_current_version_accepted(self):
        data = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)])]).to_dict()
        assert data["ontology_version"] == ONTOLOGY_VERSION
        CSBG.from_dict(data)  # must not raise

    def test_json_serialisable(self):
        import json

        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)] * 5)])
        assert CSBG.from_dict(json.loads(json.dumps(g.to_dict()))).speaker_id == "s1"
