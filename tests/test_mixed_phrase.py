"""The experimental mixed-language challenge: the Tamil word bank, the draw, and the matcher.

These pin logic with fake transcripts. They say nothing about whether a real recogniser hears real Tamil
words; that is measured on recordings by `kavach.asr_audit` and `kavach.mixed_eval`.
"""

from __future__ import annotations

import random

import pytest

from kavach.asr import Transcript, Word
from kavach.mixed_phrase import ORDER_SLACK_MS, match_mixed, new_mixed_phrase
from kavach.phrase import WORDS
from kavach.tamil_words import BANK, DEMO_WORDS, confusable_pairs, demo_pool, heard_as, is_tamil_token, normalise_english, normalise_tamil, phonetic_key, phonetic_same, tamil_same

STEP = 600  # ms per spoken word


def timeline(tokens, keep):
    """(english transcript, tamil transcript) for a recording of `tokens`, hearing only what `keep(i, token)` allows."""
    en, ta = [], []
    for i, t in enumerate(tokens):
        w = Word(text=t, start_ms=i * STEP, end_ms=i * STEP + 450, probability=0.9)
        (ta if is_tamil_token(t) else en).append((i, t, w))
    en_words = [w for i, t, w in en if keep(i, t)]
    ta_words = [w for i, t, w in ta if keep(i, t)]
    return (Transcript(text=" ".join(w.text for w in en_words), words=en_words),
            Transcript(text=" ".join(w.text for w in ta_words), words=ta_words))


def phrase(seed=1):
    return new_mixed_phrase(rng=random.Random(seed))


class TestTamilBank:
    def test_it_has_about_a_hundred_distinct_words(self):
        assert len(BANK) >= 100
        assert len({normalise_tamil(w.ta) for w in BANK}) == len(BANK)
        assert len({w.word_id for w in BANK}) == len(BANK)

    def test_every_entry_is_tamil_script_with_a_gloss_and_romanisation(self):
        for w in BANK:
            assert is_tamil_token(w.ta) and normalise_tamil(w.ta) == w.ta, w
            assert w.roman.isascii() and w.gloss, w

    def test_no_two_different_bank_words_are_called_the_same_word(self):
        assert confusable_pairs() == []

    def test_a_vowel_sign_or_pulli_changes_the_word(self):
        # Short words must match exactly: one combining mark is a different word.
        assert not tamil_same("மாடு", "மடு") and not tamil_same("நதி", "நாதி") and not tamil_same("கடை", "கடல்")

    def test_unicode_noise_is_not_a_difference(self):
        assert tamil_same("தண்ணீர்", "தண்ணீர்‌") and tamil_same("தண்ணீர்", " தண்ணீர், ")
        assert tamil_same("அம்மா", "அம்மா‍")

    def test_english_is_compared_case_and_punctuation_blind_but_not_fuzzy_here(self):
        assert normalise_english("Tiger,") == normalise_english("tiger") == "tiger"
        assert normalise_english("tiger") != normalise_english("tigers")


class TestPhoneticMatching:
    def test_the_demo_set_is_eight_words_from_the_bank(self):
        assert len(DEMO_WORDS) == 8 and len(demo_pool()) == 8

    def test_every_demo_word_is_accepted_in_tamil_script_and_in_its_romanisation(self):
        for w in demo_pool():
            assert heard_as(w.ta, w.ta) and heard_as(w.ta, w.roman), w

    @pytest.mark.parametrize("word,spelling", [("தண்ணீர்", "thanni"), ("தண்ணீர்", "tanir"), ("வீடு", "veedu"), ("வீடு", "vidu"), ("மலை", "malay"),
                                                  ("பூனை", "poonai"), ("கடல்", "kadal"), ("யானை", "yaanai"), ("புத்தகம்", "putagam"), ("அம்மா", "amma")])
    def test_ordinary_romanisations_of_each_word_are_heard_as_that_word(self, word, spelling):
        assert phonetic_same(word, spelling)

    def test_no_demo_word_is_heard_as_another_demo_word_in_either_script(self):
        for a in demo_pool():
            for b in demo_pool():
                if a is not b:
                    assert not heard_as(a.ta, b.ta) and not heard_as(a.ta, b.roman), (a.roman, b.roman)

    def test_the_english_meaning_of_a_demo_word_is_not_heard_as_it(self):
        for w in demo_pool():
            assert not heard_as(w.ta, w.gloss.split()[0]), w

    def test_the_draw_never_pairs_a_tamil_word_with_an_english_word_that_sounds_like_it(self):
        for seed in range(600):
            tokens, pos = new_mixed_phrase(rng=random.Random(seed), pool=demo_pool())
            eng = [t for t in tokens if not is_tamil_token(t)]
            assert not any(phonetic_same(tokens[i], e) for i in pos for e in eng)

    def test_keys_ignore_doubling_length_and_voicing_but_not_the_consonant_skeleton(self):
        assert phonetic_key("thanneer") == phonetic_key("தண்ணீர்") == phonetic_key("tannir")
        assert phonetic_key("malai") != phonetic_key("yanai")


class TestTamilNotEnforced:
    """`check_tamil=False` (KAVACH_MIXED_CHECK_TAMIL=false): the Tamil words are reported but do not decide the result."""

    def test_strict_is_the_default(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: i != pos[0])
        assert match_mixed(tokens, en, ta).failure_stage == "tamil_words"

    def test_an_unheard_tamil_word_no_longer_fails_but_is_still_reported(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: i != pos[0])
        m = match_mixed(tokens, en, ta, check_tamil=False)
        assert m.failure_stage == "" and m.score == 1.0 and not m.tamil_enforced
        assert m.tamil_matched == 1 and m.tamil_status[tokens[pos[0]]] == "not_heard"
        assert m.to_log()["tamil_enforced"] is False and "Tamil 1 of 2 heard" in m.detail

    def test_neither_tamil_word_heard_still_passes_on_the_english_words(self):
        tokens, _ = phrase()
        en, _ta = timeline(tokens, lambda i, t: not is_tamil_token(t))
        assert match_mixed(tokens, en, Transcript(text="", words=[]), check_tamil=False).failure_stage == ""

    def test_the_english_words_are_still_required(self):
        tokens, pos = phrase()
        drop = [i for i in range(12) if i not in pos][:5]
        en, ta = timeline(tokens, lambda i, t: i not in drop)
        m = match_mixed(tokens, en, ta, check_tamil=False)
        assert m.failure_stage == "english_words" and m.score == 0.0

    def test_a_recording_with_no_words_in_it_still_fails(self):
        tokens, _ = phrase()
        m = match_mixed(tokens, Transcript(text="", words=[]), Transcript(text="", words=[]), check_tamil=False)
        assert m.failure_stage == "english_words"


class TestDraw:
    def test_ten_english_and_two_tamil_words(self):
        tokens, pos = phrase()
        assert len(tokens) == 12 and len(pos) == 2
        assert [i for i, t in enumerate(tokens) if is_tamil_token(t)] == pos
        eng = [t for t in tokens if not is_tamil_token(t)]
        assert len(eng) == 10 and len(set(eng)) == 10 and set(eng) <= set(WORDS)

    def test_tamil_words_come_from_the_bank_and_differ(self):
        tokens, pos = phrase()
        ta = [tokens[i] for i in pos]
        assert ta[0] != ta[1] and {normalise_tamil(t) for t in ta} <= {w.ta for w in BANK}

    def test_positions_vary_and_never_touch_or_end_up_first_next_to_each_other(self):
        seen = set()
        for seed in range(300):
            _, pos = phrase(seed)
            assert pos[1] - pos[0] > 1, pos  # each Tamil word is flanked by English
            seen.add(tuple(pos))
        assert len(seen) > 30  # not a fixed slot

    def test_same_seed_same_challenge_and_different_seeds_differ(self):
        assert phrase(5) == phrase(5)
        assert len({tuple(phrase(s)[0]) for s in range(20)}) == 20

    def test_a_pool_restricts_the_tamil_words(self):
        pool = BANK[:5]
        for seed in range(40):
            tokens, pos = new_mixed_phrase(rng=random.Random(seed), pool=pool)
            assert {tokens[i] for i in pos} <= {w.ta for w in pool}

    def test_a_tamil_word_whose_translation_is_a_drawn_english_word_is_never_drawn(self):
        # The pool holds only the word for 'tiger'; whenever the English side holds 'tiger' that word must be refused.
        tiger = [w for w in BANK if w.gloss == "tiger"]
        other = [w for w in BANK if w.gloss != "tiger"][:3]
        for seed in range(400):
            tokens, pos = new_mixed_phrase(rng=random.Random(seed), pool=tiger + other)
            if "tiger" in tokens:
                assert tiger[0].ta not in tokens

    def test_too_small_a_pool_is_an_error_not_a_weaker_challenge(self):
        with pytest.raises(ValueError):
            new_mixed_phrase(rng=random.Random(0), pool=BANK[:1])


class TestMatch:
    def test_the_spoken_challenge_passes(self):
        tokens, _ = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        m = match_mixed(tokens, en, ta)
        assert (m.failure_stage, m.english_matched, m.tamil_matched, m.order_ok) == ("", 10, 2, True)
        assert m.score == 1.0 and m.missing_words == ()
        assert m.span_ms == (0, 11 * STEP + 450)

    def test_a_couple_of_missed_english_words_are_forgiven_like_the_english_login(self):
        tokens, pos = phrase()
        drop = [i for i in range(12) if i not in pos][:4]
        en, ta = timeline(tokens, lambda i, t: i not in drop)
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage == "" and m.english_matched == 6 and m.score == pytest.approx(0.6)

    def test_five_of_ten_english_words_is_not_enough(self):
        tokens, pos = phrase()
        drop = [i for i in range(12) if i not in pos][:5]
        en, ta = timeline(tokens, lambda i, t: i not in drop)
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage == "english_words" and m.score == 0.0

    @pytest.mark.parametrize("which", [0, 1])
    def test_one_unheard_tamil_word_fails_even_with_every_english_word_right(self, which):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: i != pos[which])
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage == "tamil_words" and m.score == 0.0 and m.tamil_matched == 1
        assert m.tamil_status[tokens[pos[which]]] == "not_heard"

    def test_a_different_tamil_word_is_not_the_asked_one(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        wrong = next(w.ta for w in BANK if w.ta not in tokens)
        ta.words[0] = Word(text=wrong, start_ms=ta.words[0].start_ms, end_ms=ta.words[0].end_ms, probability=0.9)
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage == "tamil_words" and m.tamil_matched == 1

    def test_the_english_only_reading_of_the_whole_phrase_fails_at_the_tamil_words(self):
        tokens, pos = phrase()
        en, _ = timeline(tokens, lambda i, t: True)
        english_only = Transcript(text=en.text, words=en.words)
        m = match_mixed(tokens, english_only, Transcript(text="", words=[]))
        assert m.failure_stage == "tamil_words" and m.tamil_matched == 0 and m.score == 0.0

    def test_a_tamil_word_the_english_decode_wrote_in_latin_letters_is_credited_when_it_sounds_right(self):
        tokens, pos = phrase()
        romans = {w.ta: w.roman for w in BANK}
        en_words = [Word(text=romans.get(t, t), start_ms=i * STEP, end_ms=i * STEP + 450, probability=0.9) for i, t in enumerate(tokens)]
        en = Transcript(text=" ".join(w.text for w in en_words), words=en_words)
        m = match_mixed(tokens, en, Transcript(text="", words=[]))
        assert m.failure_stage == "" and m.tamil_matched == 2 and m.english_matched == 10

    def test_the_romanisation_of_a_different_tamil_word_is_not_credited(self):
        tokens, pos = phrase()
        other = next(w for w in demo_pool() if w.ta not in tokens)
        romans = {w.ta: w.roman for w in BANK}
        en_words = [Word(text=other.roman if i == pos[0] else romans.get(t, t), start_ms=i * STEP, end_ms=i * STEP + 450, probability=0.9)
                    for i, t in enumerate(tokens)]
        en = Transcript(text=" ".join(w.text for w in en_words), words=en_words)
        m = match_mixed(tokens, en, Transcript(text="", words=[]))
        assert m.failure_stage == "tamil_words" and m.tamil_matched == 1

    def test_the_english_translation_of_a_tamil_word_is_not_a_tamil_word(self):
        tokens, pos = phrase()
        gloss = {w.ta: w.gloss.split()[0] for w in BANK}
        en_words = [Word(text=gloss.get(t, t), start_ms=i * STEP, end_ms=i * STEP + 450, probability=0.9) for i, t in enumerate(tokens)]
        en = Transcript(text=" ".join(w.text for w in en_words), words=en_words)
        assert match_mixed(tokens, en, Transcript(text="", words=[])).failure_stage == "tamil_words"

    def test_english_words_are_never_credited_from_the_tamil_decode(self):
        tokens, pos = phrase()
        _, ta = timeline(tokens, lambda i, t: True)
        english_in_tamil_decode = Transcript(text=" ".join(t for t in tokens), words=[
            Word(text=t, start_ms=i * STEP, end_ms=i * STEP + 450, probability=0.9) for i, t in enumerate(tokens)])
        m = match_mixed(tokens, Transcript(text="", words=[]), english_in_tamil_decode)
        assert m.english_matched == 0 and m.failure_stage == "english_words"

    def test_a_tamil_word_spoken_in_the_wrong_place_fails_the_order_check(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        for k, w in enumerate(ta.words):  # both Tamil words, in the right order, but spoken a minute after the English words
            ta.words[k] = Word(text=w.text, start_ms=w.start_ms + 60_000, end_ms=w.end_ms + 60_000, probability=0.9)
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage == "order" and m.score == 0.0 and not m.order_ok
        assert "wrong_place" in m.tamil_status.values()

    def test_jitter_between_the_two_decodes_within_the_slack_is_tolerated(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        w = ta.words[0]
        ta.words[0] = Word(text=w.text, start_ms=w.start_ms - ORDER_SLACK_MS + 50, end_ms=w.end_ms + ORDER_SLACK_MS - 50, probability=0.9)
        assert match_mixed(tokens, en, ta).failure_stage == ""

    def test_tamil_words_swapped_with_each_other_fail(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        a, b = ta.words
        ta.words[:] = [Word(text=b.text, start_ms=a.start_ms, end_ms=a.end_ms, probability=0.9),
                       Word(text=a.text, start_ms=b.start_ms, end_ms=b.end_ms, probability=0.9)]  # said in the other order
        m = match_mixed(tokens, en, ta)
        assert m.failure_stage in ("tamil_words", "order") and m.score == 0.0

    def test_a_decode_without_word_timings_cannot_pass(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: True)
        no_times = Transcript(text=ta.text, words=[])
        m = match_mixed(tokens, en, no_times)
        assert m.score == 0.0 and m.failure_stage in ("tamil_words", "no_timings")

    def test_the_log_record_keeps_english_tamil_and_order_apart(self):
        tokens, pos = phrase()
        en, ta = timeline(tokens, lambda i, t: i != pos[1])
        log = match_mixed(tokens, en, ta).to_log()
        assert log["english"] == "10/10" and log["tamil"] == "1/2" and log["failure_stage"] == "tamil_words"
