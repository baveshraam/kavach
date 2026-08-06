"""Speech recognition for Tamil-English code-mixed audio.

Wraps `faster-whisper`, with word-level timestamps (needed for splice-attack
detection and for the token-level UI) and a numeral-suppression option that
matters more than it looks.

Known limitation, stated plainly because it bounds every downstream result
---------------------------------------------------------------------------
Whisper is not trained for code-switching. On Tamil-English mixed speech it
typically does one of three things, all of which harm the CSBG differently:

1.  Transcribes Tamil in Latin script ("naan enna panren"). Handled --
    `lid.rules` has a romanised-Tamil lexicon and the LLM adjudicates the
    rest.
2.  Drops or mangles the English segments when `language="ta"` is forced.
    This deletes real language choices, biasing CMI toward Tamil.
3.  Auto-detects one language per segment and applies it throughout,
    flattening genuine switches.

**Measure your WER on a hand-transcribed subset and report it.** Every CSBG
number inherits this error rate, and a reviewer will ask what it is. If
IndicWhisper or IndicConformer transcribe your audio better, use them --
`ASRBackend` exists so an alternative can be dropped in without touching the
pipeline.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

from .audio import Audio

#: Share of a transcript one token may occupy before it is called a loop.
#:
#: Measured against the real corpus: the highest legitimate value was 16.7%,
#: from `my` in the six-token "my name is ..." control prompt, and the one
#: genuine Whisper loop sat at 28.9%. 0.25 separates them with room on both
#: sides. Raise it if a prompt ever legitimately repeats a word -- a chant, a
#: counting task -- rather than deleting the check.
REPETITION_FLOOR = 0.25

#: Below this, share is meaningless: in a four-token answer any repeated word
#: is 25% on its own.
MIN_TOKENS_FOR_REPETITION = 12

#: Common Tamil-English function words. Passed to Whisper as an initial
#: prompt to bias it toward code-mixed output rather than snapping to a
#: single language. A hint, not a guarantee.
#: Biasing context prepended to every transcription. **Load-bearing. Do not
#: remove it to "let the model decide".**
#:
#: Measured on large-v3 against a real recording of prompt 10:
#:
#:   with:    "நான் two thousand and fourல பிறந்தேன். எங்க ஏரியால ஒரு
#:             plate idli ten rupees ... morning six forty fiveக்கு எழுந்தேன்"
#:   without: "I was born in the year of two thousand and four. One plate of
#:             idli in our area is ten rupees ... I woke up at six forty-five"
#:
#: Unprompted, Whisper *translates* the Tamil rather than transcribing it. The
#: output is fluent, plausible English and every code-switch in the utterance
#: is gone -- which is the entire measurement. A translated corpus would score
#: as a corpus of monolingual English speakers and the CSBG would separate
#: nobody, for a reason invisible in the manifest.
#:
#: The romanised Tamil in the prompt looks like it should push the model toward
#: romanised output, and it does not; it establishes "this audio is mixed, keep
#: both languages" and the model still writes Tamil in Tamil script. That split
#: is what `lid.rules` resolves for free.
CODE_MIX_PROMPT = (
    "This is a conversation in Tamil and English mixed together. "
    "naan enna panren, romba nalla irukku, college la, office ku, "
    "appuram, konjam, seri, amma appa anna akka."
)


@dataclass(frozen=True, slots=True)
class Word:
    """One recognised word with timing."""

    text: str
    start_ms: int
    end_ms: int
    probability: float = 1.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(slots=True)
class Transcript:
    """ASR output for one recording."""

    text: str
    words: list[Word] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration_sec: float = 0.0
    model: str = ""

    @property
    def word_texts(self) -> list[str]:
        return [w.text for w in self.words]

    @property
    def timings(self) -> list[tuple[int, int]]:
        """(start_ms, end_ms) per word, for `lid.to_tokens`."""
        return [(w.start_ms, w.end_ms) for w in self.words]

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.probability for w in self.words) / len(self.words)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def repetition_loop(self, *, floor: float = REPETITION_FLOOR) -> tuple[str, float] | None:
        """The token this transcript degenerated into, if it did.

        Whisper loops. Given audio it cannot decode -- and numerals with
        `suppress_numerals` on are a reliable trigger -- it emits one fragment
        over and over: `Rs.Ls.Rs.Rs.Rs.Rs.Rs.` for sixty tokens where the
        speaker said a price.

        This has to be caught before annotation, and not because it is untidy.
        A loop is a *plausible* token repeated, so it lands in one semantic
        class and one language, and the CSBG counts every copy. Sixty
        hallucinated `Rs` would make MONEY the most confident cell in that
        speaker's graph, built entirely from a word they never said. Silence
        would be safer than this, which is why an empty transcript is handled
        and a degenerate one must not be treated as ordinary text.

        Returns `(token, share)` for the offending token, or None. Uses a
        share of alphanumeric tokens rather than a run-length, because the
        loop interleaves punctuation and never repeats a token adjacently.
        """
        tokens = [t for t in re.findall(r"\w+", self.text.lower()) if t]
        if len(tokens) < MIN_TOKENS_FOR_REPETITION:
            return None
        top, count = Counter(tokens).most_common(1)[0]
        share = count / len(tokens)
        return (top, share) if share >= floor else None

    def gaps_ms(self) -> list[int]:
        """Silence between consecutive words.

        Splice attacks (A2) concatenate separately-recorded words, which
        leaves unnaturally uniform or unnaturally long inter-word gaps.
        Consumed by `attacks.detect_splice`.
        """
        return [b.start_ms - a.end_ms for a, b in zip(self.words, self.words[1:])]


class ASRBackend(Protocol):
    """Interface an ASR implementation must satisfy.

    Defined so IndicWhisper, IndicConformer, or a hosted API can replace
    faster-whisper without changes elsewhere. Swapping the backend and
    re-running the evaluation is a legitimate ablation.
    """

    def transcribe(self, audio: Audio, **kwargs: Any) -> Transcript: ...


class WhisperASR:
    """faster-whisper backend. Model loads lazily on first transcription."""

    def __init__(
        self,
        *,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        suppress_numerals: bool = True,
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.suppress_numerals = suppress_numerals
        self.download_root = download_root
        self._model: Any = None
        self._numeral_token_cache: list[int] | None = None

    def _numeral_tokens(self) -> list[int]:
        """Token ids that decode to digits, cached after the first call.

        `suppress_tokens=[-1]` is faster-whisper's own default and suppresses a
        set of *symbols*; it has nothing to do with numerals, so passing it was
        the same as passing nothing. The digits have to be enumerated from the
        tokenizer, which is what this does: every id whose decoded piece is
        made only of `0`-`9`, about 400 of them.

        Suppressing them is what forces Whisper to write "six thirty" instead
        of "6.30", and that distinction is the entire NUMBER class. A digit is
        language-neutral -- "500" records that the speaker said a number and
        destroys the evidence of *which language they said it in*, which is the
        only thing the CSBG reads. On real returns from this protocol, prompt
        10 exists to elicit exactly three numerals and came back as digits.

        `-1` is kept alongside so the default symbol suppression still applies.
        """
        if self._numeral_token_cache is not None:
            return self._numeral_token_cache

        tokenizer = getattr(self.model, "hf_tokenizer", None)
        if tokenizer is None:  # pragma: no cover - depends on library version
            self._numeral_token_cache = [-1]
            return self._numeral_token_cache

        ids = [-1]
        for token_id in range(tokenizer.get_vocab_size()):
            piece = tokenizer.decode([token_id]).strip()
            if piece and all(character in "0123456789" for character in piece):
                ids.append(token_id)
        self._numeral_token_cache = ids
        return ids

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ImportError(
                    "faster-whisper is required for transcription. "
                    "Install with `pip install -r requirements.txt`."
                ) from exc

            device = self.device
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )
        return self._model

    def transcribe(
        self,
        audio: Audio,
        *,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        **_: Any,
    ) -> Transcript:
        """Transcribe with word-level timestamps.

        Args:
            audio: 16 kHz mono.
            language: ISO code, or None to auto-detect. Overrides the
                constructor default for this call.
            initial_prompt: Biasing context. Defaults to CODE_MIX_PROMPT,
                which nudges Whisper toward mixed output.
            beam_size: Higher is slower and slightly more accurate.
            vad_filter: Drop non-speech regions before decoding. Reduces
                hallucinated text on silence -- a real Whisper failure mode
                that would otherwise inject phantom tokens into the CSBG.

        Returns:
            A Transcript. Empty audio yields an empty Transcript rather than
            raising, so a failed recording degrades to a rejected login
            instead of a 500.

            If the first decode degenerates into a repetition loop it is
            decoded once more with `condition_on_previous_text=False`; see
            `_decode`. The returned transcript is whichever attempt did not
            loop, and still carries `repetition_loop()` if both did.
        """
        first = self._decode(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=True,
        )
        if first.repetition_loop() is None:
            return first

        # Whisper loops when its own previous output conditions the next
        # window -- the repetition feeds itself. Turning that conditioning off
        # is the standard remedy and costs one extra decode on the rare
        # utterance that needs it. Observed on the real corpus: a price the
        # speaker gave as "three thousand five hundred" came back as `Rs.` for
        # sixty tokens, and the same audio decoded cleanly on the retry.
        second = self._decode(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
        )
        # Keep the retry only if it actually helped. A second loop is not
        # better than the first, and returning it would hide that the first
        # attempt looped too.
        return second if second.repetition_loop() is None else first

    def _decode(
        self,
        audio: Audio,
        *,
        language: str | None,
        initial_prompt: str | None,
        beam_size: int,
        vad_filter: bool,
        condition_on_previous_text: bool,
    ) -> Transcript:
        """One decode pass. See `transcribe` for the arguments."""
        lang = language if language is not None else self.language

        segments, info = self.model.transcribe(
            audio.samples,
            language=lang,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt if initial_prompt is not None else CODE_MIX_PROMPT,
            # Numerals as words, not digits: '5' is language-neutral and
            # loses the speaker's choice of English vs Tamil, and NUMBER is
            # one of the most discriminative CSBG classes. See _numeral_tokens.
            suppress_tokens=self._numeral_tokens() if self.suppress_numerals else [-1],
            condition_on_previous_text=condition_on_previous_text,
        )

        words: list[Word] = []
        parts: list[str] = []
        for segment in segments:
            parts.append(segment.text)
            for w in getattr(segment, "words", None) or []:
                text = w.word.strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start_ms=int(w.start * 1000),
                        end_ms=int(w.end * 1000),
                        probability=float(getattr(w, "probability", 1.0)),
                    )
                )

        return Transcript(
            text="".join(parts).strip(),
            words=words,
            language=getattr(info, "language", lang or ""),
            language_probability=float(getattr(info, "language_probability", 0.0)),
            duration_sec=audio.duration_sec,
            model=f"faster-whisper/{self.model_size}",
        )


def transcribe_both_languages(
    backend: WhisperASR, audio: Audio
) -> dict[str, Transcript]:
    """Transcribe forcing Tamil, forcing English, and auto-detecting.

    A diagnostic, not a production path. Code-mixed audio produces materially
    different transcripts under each setting, and comparing them on a sample
    of your corpus is the fastest way to decide which to standardise on --
    a decision the paper must state.

    Returns:
        {"auto": ..., "ta": ..., "en": ...}
    """
    return {
        "auto": backend.transcribe(audio, language=None),
        "ta": backend.transcribe(audio, language="ta"),
        "en": backend.transcribe(audio, language="en"),
    }


@dataclass(frozen=True, slots=True)
class TranscriptComparison:
    """Divergence between transcription settings on the same audio."""

    auto_language: str
    auto_words: int
    ta_words: int
    en_words: int
    auto_confidence: float

    @property
    def disagreement(self) -> float:
        """Relative spread in word count across settings, 0-1.

        High values mean the ASR is unstable on this audio and its transcript
        should not be trusted as ground truth without checking.
        """
        counts = [self.auto_words, self.ta_words, self.en_words]
        hi, lo = max(counts), min(counts)
        return (hi - lo) / hi if hi else 0.0


def compare_transcripts(results: dict[str, Transcript]) -> TranscriptComparison:
    """Summarise `transcribe_both_languages` output."""
    return TranscriptComparison(
        auto_language=results["auto"].language,
        auto_words=len(results["auto"].words),
        ta_words=len(results["ta"].words),
        en_words=len(results["en"].words),
        auto_confidence=results["auto"].mean_confidence,
    )


__all__ = [
    "Word",
    "Transcript",
    "ASRBackend",
    "WhisperASR",
    "CODE_MIX_PROMPT",
    "transcribe_both_languages",
    "compare_transcripts",
    "TranscriptComparison",
]
