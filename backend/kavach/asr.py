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

from dataclasses import dataclass, field
from typing import Any, Protocol

from .audio import Audio

#: Common Tamil-English function words. Passed to Whisper as an initial
#: prompt to bias it toward code-mixed output rather than snapping to a
#: single language. A hint, not a guarantee.
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
        """
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
            # one of the most discriminative CSBG classes.
            suppress_tokens=[-1] if self.suppress_numerals else None,
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
