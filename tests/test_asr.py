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

from kavach.asr import Transcript, Word, WhisperASR, compare_transcripts
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
