"""The spoken phrase that binds a recording to one challenge.

A fixed sentence can be recorded once and played back; a phrase of words drawn at random
for each attempt cannot. The check is whether the *ASR transcript* contains the words that
were shown, in order, tolerating the small errors speech recognition makes on accented
English -- and nothing more: a transcript that does not contain them is the signature of a
recording made for some other attempt.
"""

from __future__ import annotations

import random
from difflib import SequenceMatcher
from itertools import combinations

import pytest

from kavach.phrase import FUZZY_RATIO, WORDS, match_phrase, new_phrase


class TestPool:
    def test_words_are_plain_lowercase_and_long_enough_to_be_heard_clearly(self):
        for w in WORDS:
            assert w.isascii() and w.isalpha() and w == w.lower(), w
            assert 4 <= len(w) <= 9, w

    def test_no_duplicates(self):
        assert len(set(WORDS)) == len(WORDS)

    def test_the_pool_is_large_enough_that_a_phrase_cannot_be_guessed_or_pre_recorded_cheaply(self):
        assert len(WORDS) >= 120

    def test_no_two_words_are_close_enough_for_the_fuzzy_match_to_confuse_them(self):
        """The matcher forgives 'tigers' for 'tiger'. It must not forgive 'marble'
        for 'maple', or a wrong word would count as a right one."""
        close = [(a, b) for a, b in combinations(WORDS, 2) if SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO]
        assert close == []


class TestNewPhrase:
    def test_has_the_requested_number_of_distinct_words_from_the_pool(self):
        p = new_phrase(6)
        assert len(p) == 6 and len(set(p)) == 6 and set(p) <= set(WORDS)

    def test_is_unpredictable_by_default(self):
        assert len({tuple(new_phrase(6)) for _ in range(20)}) == 20

    def test_a_seeded_rng_is_repeatable_for_tests(self):
        assert new_phrase(6, rng=random.Random(5)) == new_phrase(6, rng=random.Random(5))

    def test_refuses_more_words_than_the_pool_has(self):
        with pytest.raises(ValueError):
            new_phrase(len(WORDS) + 1)


class TestMatch:
    EXPECTED = ["tiger", "river", "mango", "window", "candle", "silver"]

    def test_an_exact_transcript_matches_completely(self):
        m = match_phrase(self.EXPECTED, "tiger river mango window candle silver")
        assert m.score == 1.0 and m.matched == 6 and m.total == 6

    def test_case_and_punctuation_do_not_matter(self):
        m = match_phrase(self.EXPECTED, "Tiger, river. MANGO! window; candle - silver.")
        assert m.score == 1.0

    def test_surrounding_chatter_does_not_matter(self):
        m = match_phrase(self.EXPECTED, "okay so tiger river uh mango window candle silver thank you")
        assert m.score == 1.0

    def test_a_plural_or_small_asr_slip_still_counts(self):
        m = match_phrase(self.EXPECTED, "tigers river mangos window candle silver")
        assert m.score == 1.0

    def test_one_dropped_word_still_scores_five_of_six(self):
        m = match_phrase(self.EXPECTED, "tiger river window candle silver")
        assert m.matched == 5 and m.score == pytest.approx(5 / 6)

    def test_a_different_phrase_scores_zero(self):
        assert match_phrase(self.EXPECTED, "bridge lantern pepper eagle forest ribbon").score == 0.0

    def test_the_right_words_in_the_wrong_order_do_not_pass(self):
        """A recording of these words in another order was not made for this challenge."""
        m = match_phrase(self.EXPECTED, "silver candle window mango river tiger")
        assert m.score < 0.5

    def test_silence_or_a_non_latin_transcript_scores_zero(self):
        assert match_phrase(self.EXPECTED, "").score == 0.0
        assert match_phrase(self.EXPECTED, "நான் இங்கே இருக்கிறேன்").score == 0.0

    def test_a_wrong_word_that_merely_resembles_a_right_one_is_not_forgiven(self):
        m = match_phrase(["marble"], "maple")
        assert m.matched == 0

    def test_it_reports_what_was_heard_for_the_explanation(self):
        m = match_phrase(self.EXPECTED, "tiger river window")
        assert "tiger" in m.detail and "mango" in m.detail  # names the words that were missing


class TestSpan:
    """Where in the recording the shown words were spoken: the voice is judged on that span."""

    @staticmethod
    def words(spec):
        from kavach.asr import Word
        return [Word(text=t, start_ms=s, end_ms=e) for t, s, e in spec]

    def test_the_span_runs_from_the_first_matched_word_to_the_last(self):
        w = self.words([("okay", 0, 300), ("Tiger,", 1000, 1400), ("river", 1500, 1900), ("mango.", 2000, 2500), ("thanks", 4000, 4400)])
        m = match_phrase(["tiger", "river", "mango"], "okay Tiger, river mango. thanks", words=w)
        assert m.span_ms == (1000, 2500)

    def test_speech_before_and_after_is_outside_the_span(self):
        w = self.words([("hey", 0, 400), ("tiger", 1000, 1400), ("river", 1500, 1900), ("let", 3000, 3200), ("me", 3200, 3300), ("try", 3300, 3500)])
        m = match_phrase(["tiger", "river"], "hey tiger river let me try", words=w)
        assert m.span_ms == (1000, 1900)

    def test_no_word_timings_means_no_span(self):
        assert match_phrase(["tiger"], "tiger").span_ms is None

    def test_no_match_means_no_span(self):
        w = self.words([("bridge", 0, 400)])
        assert match_phrase(["tiger", "river"], "bridge", words=w).span_ms is None

    def test_an_unordered_straggler_does_not_stretch_the_span(self):
        w = self.words([("river", 0, 400), ("tiger", 1000, 1400), ("river", 1500, 1900)])
        m = match_phrase(["tiger", "river"], "river tiger river", words=w)
        assert m.span_ms == (1000, 1900)
