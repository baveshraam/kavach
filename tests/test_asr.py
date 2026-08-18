"""Tests for the ASR layer.

Almost all of this is about one setting. `suppress_numerals` claimed to make
Whisper write "six thirty" rather than "6.30", and for the life of the project
it passed `[-1]` -- which is faster-whisper's *own default*, a set of symbol
tokens with nothing to do with digits. So the flag did nothing, in either
position, and prompt 10 of the protocol -- the prompt whose entire job is to
elicit three numerals -- came back from the first real recordings as `2004`,
`10 Rs.` and `6.45`.

A digit is language-neutral. `500` records that a speaker said a number and
destroys which language they said it in, and that is the whole of the NUMBER
class. These tests pin the behaviour so it cannot silently revert.
"""

from __future__ import annotations

import numpy as np
import pytest

from kavach.asr import (
    REPETITION_FLOOR,
    Transcript,
    Word,
    WhisperASR,
    compare_transcripts,
)
from kavach.audio import Audio


class _StubTokenizer:
    """A tokenizer with a handful of digit pieces and a handful of word ones."""

    _VOCAB = ["hello", "0", "1", " 12", "world", "3.5", "999", "", "நான்"]

    def get_vocab_size(self) -> int:
        return len(self._VOCAB)

    def decode(self, ids: list[int]) -> str:
        return self._VOCAB[ids[0]]


class _StubModel:
    hf_tokenizer = _StubTokenizer()


class TestNumeralSuppression:
    @pytest.fixture
    def backend(self) -> WhisperASR:
        asr = WhisperASR(model_size="tiny")
        asr._model = _StubModel()
        return asr

    def test_collects_every_digit_only_token(self, backend):
        ids = backend._numeral_tokens()
        vocab = _StubTokenizer._VOCAB
        assert {vocab[i] for i in ids if i >= 0} == {"0", "1", " 12", "999"}

    def test_keeps_the_default_symbol_suppression(self, backend):
        """-1 is the library's own default set; suppressing numerals must add
        to it, not replace it."""
        assert -1 in backend._numeral_tokens()

    def test_leaves_words_and_mixed_pieces_alone(self, backend):
        """`3.5` is not digit-only, and suppressing it would break decimals in
        text that legitimately contains them."""
        ids = backend._numeral_tokens()
        vocab = _StubTokenizer._VOCAB
        for piece in ("hello", "world", "3.5", "நான்", ""):
            assert vocab.index(piece) not in ids

    def test_the_list_is_cached(self, backend):
        """It walks the whole vocabulary -- about 50k decodes on a real model,
        which is not something to repeat per utterance."""
        first = backend._numeral_tokens()
        assert backend._numeral_tokens() is first

    def test_a_library_without_the_tokenizer_degrades_to_the_default(self):
        """Rather than raising mid-transcription on a version bump."""
        asr = WhisperASR(model_size="tiny")
        asr._model = object()
        assert asr._numeral_tokens() == [-1]

    def test_the_flag_is_on_by_default(self):
        assert WhisperASR(model_size="tiny").suppress_numerals is True


class TestTranscript:
    def test_empty_transcript_is_detectable(self):
        assert Transcript(text="", words=[]).is_empty

    def test_word_texts_and_timings_line_up(self):
        t = Transcript(
            text="naan enga",
            words=[Word("naan", 0, 400), Word("enga", 500, 900)],
        )
        assert t.word_texts == ["naan", "enga"]
        assert t.timings == [(0, 400), (500, 900)]

    def test_gaps_between_words_are_measured(self):
        """The input to splice detection: concatenated words leave gaps that
        are unnaturally uniform or unnaturally long."""
        t = Transcript(text="a b", words=[Word("a", 0, 400), Word("b", 700, 900)])
        assert t.gaps_ms() == [300]

    def test_mean_confidence_of_no_words_is_zero_not_an_error(self):
        assert Transcript(text="", words=[]).mean_confidence == 0.0


class TestTranslationDetection:
    """Whisper translating instead of transcribing must not reach the CSBG.

    Worse than a loop, because a loop looks broken and this looks perfect. The
    output is fluent, correct English, so every check downstream passes and the
    utterance enters the graph as a speaker who chose English for all of it --
    a language choice nobody made, concentrated on whichever speaker the model
    found hardest.
    """

    #: The real failure, from the pilot: S04 prompt 10, whose reference is
    #: romanised Tamil ("Naan rendaayirathu naalu la piranthen ...").
    REAL_TRANSLATION = (
        "I was born in the year of two thousand and four. In my area, one plate "
        "of idli costs ten rupees and four idlis are served. Today, I woke up at "
        "six forty-five in the morning. I usually wake up at six thirty."
    )

    def test_the_real_corpus_translation_is_caught(self):
        reason = Transcript(
            text=self.REAL_TRANSLATION, language="ta", language_probability=0.98
        ).looks_translated()
        assert reason is not None
        assert "translated" in reason

    def test_genuine_code_mixed_output_is_not_flagged(self):
        """The same speaker, transcribed correctly, must pass."""
        t = Transcript(
            text=(
                "நான் two thousand and fourல பிறந்தேன். எங்க ஏரியால ஒரு plate idli "
                "ten rupees, morning six forty fiveக்கு எழுந்தேன்"
            ),
            language="ta",
        )
        assert t.looks_translated() is None

    def test_romanised_tamil_counts_as_tamil_surviving(self):
        """Latin script is not evidence of translation on its own.

        Whisper romanises Tamil constantly -- `lid.rules` exists for it -- so
        treating Latin script as English here would flag the single most common
        *correct* output this pipeline sees.
        """
        t = Transcript(text="naan college la irukken, appuram lab ku poren", language="ta")
        assert t.looks_translated() is None

    def test_an_english_recording_is_never_flagged(self):
        """Four of the seven pilot speakers read English-dominant scripts."""
        t = Transcript(
            text="I usually wake up at six thirty and take the bus to college.",
            language="en",
            language_probability=0.99,
        )
        assert t.looks_translated() is None

    def test_empty_transcript_is_not_a_translation(self):
        assert Transcript(text="", language="ta").looks_translated() is None

    def test_unknown_language_is_not_guessed_at(self):
        """No detected language means no claim -- silence, not a guess."""
        assert Transcript(text=self.REAL_TRANSLATION).looks_translated() is None

    def test_the_reason_names_the_evidence(self):
        """An operator has to be able to act on it without reading the source."""
        reason = Transcript(
            text=self.REAL_TRANSLATION, language="ta", language_probability=0.98
        ).looks_translated()
        assert "ta" in reason and "0.98" in reason


class TestHallucinatedScript:
    """Tamil-shaped text with other alphabets spliced into it.

    The third failure mode, and the one that follows from fixing the second:
    pushed to keep Tamil on audio it cannot decode, Whisper invents. Unlike a
    translation this looks wrong to a human -- but nothing was reading the
    manifest, and "a human would notice" is not a check.
    """

    def test_the_real_cases_are_caught(self):
        for text in (
            "எனக்கு கோயம்த்தூர் சரி விழிவு 초லோகளின் 거야 молодப்பயில்",
            "毛த சிறமுது குழ்வ இரண்டு ஆ கட்டத்தண்டு",
        ):
            assert Transcript(text=text).hallucinated_script() is not None

    def test_genuine_code_mixed_speech_is_clean(self):
        t = Transcript(
            text="recently நான் ஒரு movie பாத்தேன், story interesting இருந்தது"
        )
        assert t.hallucinated_script() is None

    def test_plain_english_is_clean(self):
        assert Transcript(text="My hometown is Coimbatore.").hallucinated_script() is None

    def test_it_reports_what_it_found(self):
        sample, count = Transcript(text="வணக்கம் 거야 молод").hallucinated_script()
        assert count > 0 and sample

    def test_empty_text_is_clean(self):
        assert Transcript(text="").hallucinated_script() is None


class TestRepetitionLoop:
    """Whisper's degenerate output must not reach the CSBG.

    A loop is worse than silence, and not by a little. It repeats a *plausible*
    word, so every copy is counted as a genuine language choice in one semantic
    class -- the speaker ends up with their most confident cell built from a
    word they never said. An empty transcript is handled everywhere; a
    degenerate one looks like data.
    """

    #: The real failure, trimmed: prompt 4 from speaker S06, where a price
    #: given as "three thousand five hundred" came back as `Rs.` sixty times.
    REAL_LOOP = (
        "last week was expensive, I bought a keyboard for "
        "Rs.Ls.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs which I had been putting "
        "of for months. metro card recharge, Rs.Ls.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs.Rs."
    )

    #: The Tamil loop this detector used to miss: speaker `aniruth`, prompt 1,
    #: where a name came back seventeen times. It measured 0.20 against a 0.25
    #: floor and passed, because `\w+` cut every Tamil word at its vowel signs.
    REAL_TAMIL_LOOP = "நார் வந்து என்னில் பாரவும்! " + "மனோஜ் அன்பர் " * 17

    def test_a_tamil_loop_is_caught(self):
        """Regression: the floor is only meaningful if the tokens are whole.

        Python's `\w` excludes Unicode combining marks, so it split "மனோஜ்"
        into "மன" + "ஜ" -- doubling the token count and halving every share.
        The detector was calibrated on an ASCII `Rs.` loop, where the two
        tokenisers agree, so a whole script's worth of loops went unseen.
        """
        found = Transcript(text=self.REAL_TAMIL_LOOP).repetition_loop()
        assert found is not None, (
            "a Tamil word repeated 17 times is a loop; if this fails the "
            "tokeniser is fragmenting Tamil again"
        )
        token, share = found
        assert token == "மனோஜ்", f"counted {token!r}, so the word was split"
        assert share > REPETITION_FLOOR

    def test_tamil_and_ascii_are_counted_in_the_same_units(self):
        """The share has to mean the same thing in both scripts.

        Otherwise one floor cannot serve both, and the calibration recorded on
        `REPETITION_FLOOR` silently applies to Latin text only.
        """
        tamil = Transcript(text="ஒரு நாள் காலையில " * 6).repetition_loop()
        latin = Transcript(text="one day morning " * 6).repetition_loop()
        assert (tamil is None) == (latin is None)
        if tamil and latin:
            assert abs(tamil[1] - latin[1]) < 0.01

    def test_normal_tamil_speech_is_not_a_loop(self):
        """No false alarm on real code-mixed answers."""
        text = (
            "என்னுடைய home town திருப்பூர், பிறகு வளர்ந்தது எல்லாமே திருப்பூர்ல தான், "
            "so திருப்பூர் ரொம்ப favourite place எனக்கு, personal connection இருக்கு"
        )
        assert Transcript(text=text).repetition_loop() is None

    def test_the_real_corpus_loop_is_caught(self):
        found = Transcript(text=self.REAL_LOOP).repetition_loop()
        assert found is not None
        token, share = found
        assert token == "rs"
        assert share > 0.25

    def test_ordinary_speech_is_not_flagged(self):
        text = (
            "last week was expensive I bought a keyboard for three thousand "
            "five hundred which I had been putting off for months metro card "
            "recharge five hundred I ate out four times"
        )
        assert Transcript(text=text).repetition_loop() is None

    def test_the_control_prompt_is_not_flagged(self):
        """"my name is ..." repeats `my` at 16.7%, the highest legitimate share
        measured on the real corpus. The floor sits above it deliberately."""
        assert Transcript(text="my name is Bavesh my friends").repetition_loop() is None

    def test_short_utterances_are_exempt(self):
        """In a four-token answer any repeated word is already 25%."""
        assert Transcript(text="yes yes yes yes").repetition_loop() is None

    def test_the_floor_is_adjustable_rather_than_hardcoded(self):
        assert Transcript(text=self.REAL_LOOP).repetition_loop(floor=0.99) is None

    def test_non_adjacent_repeats_still_count(self):
        """The real loop interleaves `Ls` and punctuation, so a run-length
        check would have missed it entirely."""
        text = " ".join(["rs x"] * 20)
        assert Transcript(text=text).repetition_loop() is not None


class TestLoopRetry:
    """A detected loop is re-decoded once with conditioning off."""

    class FakeModel:
        def __init__(self, texts: list[str]) -> None:
            self.texts = texts
            self.calls: list[dict] = []

        def transcribe(self, _samples, **kwargs):
            self.calls.append(kwargs)
            text = self.texts[min(len(self.calls) - 1, len(self.texts) - 1)]
            segment = type("S", (), {"text": text, "words": []})()
            info = type("I", (), {"language": "en", "language_probability": 0.9})()
            return [segment], info

    def _asr(self, texts: list[str]):
        """A real WhisperASR with a fake checkpoint behind its lazy loader.

        `_model` is set rather than the `model` property patched: assigning to
        `type(asr).model` mutates the class for the rest of the session, and
        the first version of this test did exactly that -- taking down an
        unrelated numeral-suppression test three classes away.
        """
        asr = WhisperASR(model_size="fake", suppress_numerals=False)
        model = self.FakeModel(texts)
        asr._model = model
        return asr, model

    def test_a_clean_first_decode_is_not_retried(self):
        asr, model = self._asr(["naan office ku poren today morning at nine"])
        asr.transcribe(Audio(np.zeros(16000, dtype=np.float32), 16000))
        assert len(model.calls) == 1, "a second decode is pure cost when the first is fine"

    def test_a_loop_triggers_one_retry_with_conditioning_off(self):
        clean = "I bought a keyboard for three thousand five hundred last week"
        asr, model = self._asr([TestRepetitionLoop.REAL_LOOP, clean])
        out = asr.transcribe(Audio(np.zeros(16000, dtype=np.float32), 16000))
        assert len(model.calls) == 2
        assert model.calls[0]["condition_on_previous_text"] is True
        assert model.calls[1]["condition_on_previous_text"] is False, (
            "the loop feeds on the model's own previous output; retrying without "
            "changing that is just rolling the dice again"
        )
        assert out.text == clean

    def test_two_loops_do_not_retry_forever(self):
        asr, model = self._asr([TestRepetitionLoop.REAL_LOOP])
        out = asr.transcribe(Audio(np.zeros(16000, dtype=np.float32), 16000))
        assert len(model.calls) == 2
        assert out.repetition_loop() is not None, (
            "a transcript that looped twice must still report it, or the "
            "annotation report will call the corpus clean"
        )


class TestTranscriptComparison:
    @staticmethod
    def _transcript(n_words: int, language: str = "ta") -> Transcript:
        return Transcript(
            text=" ".join(f"w{i}" for i in range(n_words)),
            words=[Word(f"w{i}", i * 100, i * 100 + 90) for i in range(n_words)],
            language=language,
        )

    def test_settings_that_agree_on_word_count_do_not_disagree(self):
        same = self._transcript(8)
        assert compare_transcripts({"auto": same, "ta": same, "en": same}).disagreement == 0.0

    def test_forcing_a_language_that_drops_half_the_words_is_visible(self):
        """This is the check behind leaving `whisper_language` on auto: forcing
        `ta` on code-mixed speech can swallow the English segments, and the
        symptom is a word count that collapses against the auto run."""
        comparison = compare_transcripts(
            {
                "auto": self._transcript(20),
                "ta": self._transcript(10),
                "en": self._transcript(18, language="en"),
            }
        )
        assert comparison.disagreement == pytest.approx(0.5)
        assert comparison.auto_language == "ta"

    def test_no_words_at_all_is_zero_rather_than_a_division_error(self):
        empty = Transcript(text="", words=[])
        assert compare_transcripts({"auto": empty, "ta": empty, "en": empty}).disagreement == 0.0


@pytest.mark.models
class TestAgainstARealCheckpoint:
    """Opt-in: needs faster-whisper and a downloaded checkpoint."""

    def test_suppression_changes_what_comes_out(self):
        pytest.importorskip("faster_whisper")
        sr = 16_000
        t = np.arange(sr * 2) / sr
        audio = Audio((0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

        plain = WhisperASR(model_size="tiny", suppress_numerals=False)
        suppressed = WhisperASR(model_size="tiny", suppress_numerals=True)
        assert suppressed._numeral_tokens() != [-1]
        assert len(suppressed._numeral_tokens()) > len([-1])
        # Both must still return a Transcript on nonsense input rather than raise.
        assert isinstance(plain.transcribe(audio), Transcript)
        assert isinstance(suppressed.transcribe(audio), Transcript)
