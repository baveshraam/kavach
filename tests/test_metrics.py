"""Tests for the published code-mixing metrics.

These formulas come from cited papers, so the tests check them against
hand-computed values rather than against the implementation's own output.
A regression here means the paper's reported CMI/I-index numbers are wrong.
"""

from __future__ import annotations

import math

import pytest

from kavach.csbg.metrics import (
    compute_all_metrics,
    compute_burstiness,
    compute_cmi,
    compute_i_index,
    compute_m_index,
)
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token, UtteranceTokens, count_switch_points


def tok(lang: Language, cls_: SemanticClass = SemanticClass.OTHER, text: str = "x") -> Token:
    return Token(text=text, language=lang, semantic_class=cls_)


def utt(languages: list[Language], uid: str = "u1") -> UtteranceTokens:
    return UtteranceTokens(utterance_id=uid, tokens=[tok(lang) for lang in languages])


TA, EN, NEU, NE = Language.TA, Language.EN, Language.NEUTRAL, Language.NAMED_ENTITY


class TestCMI:
    def test_monolingual_is_zero(self):
        """A speaker using one language has not mixed: CMI = 0."""
        assert compute_cmi([utt([TA] * 10)]) == 0.0
        assert compute_cmi([utt([EN] * 10)]) == 0.0

    def test_perfectly_balanced_is_fifty(self):
        """5 TA + 5 EN: 100 * (1 - 5/10) = 50."""
        assert compute_cmi([utt([TA] * 5 + [EN] * 5)]) == pytest.approx(50.0)

    def test_hand_computed_asymmetric(self):
        """8 TA + 2 EN: 100 * (1 - 8/10) = 20."""
        assert compute_cmi([utt([TA] * 8 + [EN] * 2)]) == pytest.approx(20.0)

    def test_language_independent_tokens_excluded(self):
        """Neutral tokens are the `u` term and must not change CMI.

        This is the formula's most commonly mis-implemented detail: named
        entities are excluded from the denominator, not counted as a third
        language.
        """
        without = compute_cmi([utt([TA] * 8 + [EN] * 2)])
        with_neutral = compute_cmi([utt([TA] * 8 + [EN] * 2 + [NEU] * 5 + [NE] * 3)])
        assert without == pytest.approx(with_neutral)

    def test_empty_input_is_zero(self):
        assert compute_cmi([]) == 0.0
        assert compute_cmi([utt([])]) == 0.0

    def test_all_neutral_is_zero(self):
        """No language choice was made, so no mixing occurred."""
        assert compute_cmi([utt([NEU, NE, NEU])]) == 0.0

    def test_pooled_not_averaged_across_utterances(self):
        """Aggregation pools counts; it does not average per-utterance CMIs.

        A 2-token utterance must not carry the same weight as a 20-token one.
        Pooled: 11 TA, 1 EN -> 100*(1 - 11/12) = 8.33.
        Naive per-utterance mean would give (0 + 50)/2 = 25.
        """
        utterances = [utt([TA] * 10, "long"), utt([TA, EN], "short")]
        assert compute_cmi(utterances) == pytest.approx(100 * (1 - 11 / 12), abs=1e-6)


class TestMIndex:
    def test_monolingual_is_zero(self):
        assert compute_m_index([utt([TA] * 10)]) == 0.0

    def test_balanced_is_one(self):
        """k=2, p=[0.5,0.5]: sum_sq=0.5, M=(1-0.5)/((2-1)*0.5)=1.0."""
        assert compute_m_index([utt([TA] * 5 + [EN] * 5)]) == pytest.approx(1.0)

    def test_hand_computed(self):
        """8 TA, 2 EN: sum_sq = 0.64+0.04 = 0.68, M = 0.32/0.68."""
        assert compute_m_index([utt([TA] * 8 + [EN] * 2)]) == pytest.approx(0.32 / 0.68)

    def test_empty_is_zero(self):
        assert compute_m_index([]) == 0.0


class TestIIndex:
    def test_no_switches(self):
        assert compute_i_index([utt([TA] * 5)]) == 0.0

    def test_every_boundary_switches(self):
        """Perfect alternation: 4 switches over 4 boundaries."""
        assert compute_i_index([utt([TA, EN, TA, EN, TA])]) == pytest.approx(1.0)

    def test_hand_computed(self):
        """TA TA EN EN TA -> switches at positions 2 and 4 = 2/4 = 0.5."""
        assert compute_i_index([utt([TA, TA, EN, EN, TA])]) == pytest.approx(0.5)

    def test_switches_not_counted_across_utterance_boundaries(self):
        """Two monolingual utterances in different languages have 0 switches.

        Concatenating them would fabricate a switch the speaker never made.
        This is the single most important correctness property in the module.
        """
        utterances = [utt([TA] * 5, "u1"), utt([EN] * 5, "u2")]
        assert compute_i_index(utterances) == 0.0

    def test_neutral_tokens_are_transparent(self):
        """'நான் Chennai போனேன்' (TA, NE, TA) contains no switch."""
        u = UtteranceTokens(
            utterance_id="u",
            tokens=[tok(TA), tok(NE), tok(TA)],
        )
        assert compute_i_index([u]) == 0.0

    def test_single_token_utterances_ignored(self):
        assert compute_i_index([utt([TA], "u1"), utt([EN], "u2")]) == 0.0


class TestBurstiness:
    def test_regular_alternation_is_negative(self):
        """Every span length 1 -> std 0 -> ratio 0 -> B = -1."""
        assert compute_burstiness([utt([TA, EN] * 6)]) == pytest.approx(-1.0)

    def test_bursty_is_higher_than_regular(self):
        """Long blocks then rapid alternation should exceed steady alternation."""
        regular = compute_burstiness([utt([TA, EN] * 8)])
        bursty = compute_burstiness([utt([TA] * 8 + [EN] * 8 + [TA, EN, TA, EN])])
        assert bursty > regular

    def test_too_few_spans_is_zero(self):
        assert compute_burstiness([utt([TA] * 5)]) == 0.0
        assert compute_burstiness([]) == 0.0


class TestSwitchPoints:
    def test_counts_language_changes(self):
        assert count_switch_points([tok(TA), tok(EN), tok(EN), tok(TA)]) == 2

    def test_empty_and_single(self):
        assert count_switch_points([]) == 0
        assert count_switch_points([tok(TA)]) == 0


class TestComputeAllMetrics:
    def test_consistency_with_individual_functions(self):
        utterances = [
            utt([TA, TA, EN, NEU, EN, TA], "u1"),
            utt([EN, TA, TA, NE, EN], "u2"),
        ]
        combined = compute_all_metrics(utterances)
        assert combined.cmi == pytest.approx(compute_cmi(utterances))
        assert combined.m_index == pytest.approx(compute_m_index(utterances))
        assert combined.i_index == pytest.approx(compute_i_index(utterances))
        assert combined.burstiness == pytest.approx(compute_burstiness(utterances))

    def test_token_counts(self):
        utterances = [utt([TA, TA, EN, NEU, NE], "u1")]
        m = compute_all_metrics(utterances)
        assert m.n_tokens == 5
        assert m.n_choice_tokens == 3
        assert m.ta_fraction == pytest.approx(2 / 3)

    def test_reliability_flag_tracks_token_count(self):
        assert not compute_all_metrics([utt([TA, EN] * 5)]).is_reliable  # 10 tokens
        assert compute_all_metrics([utt([TA, EN] * 15)]).is_reliable  # 30 tokens

    def test_empty_corpus_does_not_crash(self):
        """Degenerate input must return zeros, not NaN or an exception.

        Empty utterance lists reach this code whenever ASR returns nothing,
        which happens on silent or failed recordings.
        """
        m = compute_all_metrics([])
        assert m.cmi == 0.0
        assert m.n_tokens == 0
        assert not math.isnan(m.burstiness)
