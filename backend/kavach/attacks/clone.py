"""A3/A4/A5 -- voice cloning, and the quality check that makes it admissible.

THE METHODOLOGICAL TRAP
-----------------------
An attack that fails because the clone was bad is not evidence the defence
works. If the cloned Tamil comes out robotic and the acoustic branch rejects
it on its own, then the trial never tested the CSBG at all -- it tested the
TTS. Reporting that as "A4 rejected" would be straightforwardly wrong, and it
is the single easiest way to accidentally fake this paper's headline result.

So every clone is screened before it counts:

    a clone is an ADMISSIBLE A3/A4/A5 trial only if the acoustic branch
    accepts it.

`screen_clone()` enforces this and `CloneQualityReport.admissible` records it.
The suite reports **attack yield** -- the fraction of clones that got past
ECAPA -- next to every attack result. A row reading "A4: 0/40 accepted" means
nothing if the yield was 5%; it means everything if the yield was 90%.

Low yield is itself publishable, but as a different claim: "open TTS cannot
clone code-switched Tamil-English well enough to defeat a standard speaker
verifier" is the low-resource-ness-as-a-defensive-asset finding. It supports
the paper's framing. It just is not a result about the CSBG, and the two must
not be blurred together.

TAMIL SUPPORT IS NOT ASSUMED
----------------------------
XTTS-v2's advertised language list needs checking for `ta` before any of this
is committed to; `check_language_support()` asks the loaded model rather than
trusting a constant in this file. If Tamil is unsupported or poor, the
fallbacks are AI4Bharat IndicTTS and commercial multilingual TTS, and the
`CloneBackend` Protocol exists so swapping one in touches nothing else.

A code-mixed utterance is also a genuinely awkward case for any TTS: the text
is romanised Tamil interleaved with English, and a model conditioned on `ta`
may read the English words with Tamil grapheme-to-phoneme rules, or vice
versa. `SynthesisRequest.language` therefore records what the backend was
*told*, which is not always what the text *is*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..audio import Audio, AudioError, concatenate, prepare_for_embedding
from ..embedding import SpeakerEmbedding, SpeakerTemplate
from . import AttackType

DEFAULT_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

#: Minimum reference audio for a usable clone. Below this, XTTS-v2 produces
#: something recognisably wrong, which would suppress attack yield for a
#: reason unrelated to the defence.
MIN_REFERENCE_SEC = 6.0


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """One clone to generate."""

    text: str
    language: str = "ta"
    """The language code handed to the backend. Recorded because for
    code-mixed text it is a choice, not a fact -- see the module docstring."""

    speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Cannot synthesise empty text.")


@runtime_checkable
class CloneBackend(Protocol):
    """A voice cloning TTS backend.

    Kept minimal so XTTS-v2, IndicTTS, F5-TTS or a commercial API can each
    satisfy it. Backends must not mutate the reference clips.
    """

    def synthesise(self, request: SynthesisRequest, reference: list[Audio]) -> Audio:
        """Speak `request.text` in the voice of `reference`."""
        ...

    @property
    def name(self) -> str:
        """Backend identifier, recorded in trial provenance."""
        ...

    def supported_languages(self) -> list[str]:
        """Language codes the backend accepts."""
        ...


class XTTSCloner:
    """Coqui XTTS-v2 zero-shot voice cloning.

    The `TTS` package is imported lazily; the research core must stay
    installable without it.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_XTTS_MODEL,
        device: str = "cpu",
        sample_rate: int = 24000,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._tts: Any = None

    @property
    def name(self) -> str:
        return f"xtts::{self.model_name}"

    @property
    def tts(self) -> Any:
        """Lazily-loaded TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "The `TTS` package is required for voice cloning. Install it with "
                    "`pip install TTS`, or pass a different CloneBackend."
                ) from exc
            self._tts = TTS(self.model_name).to(self.device)
        return self._tts

    def supported_languages(self) -> list[str]:
        """Ask the loaded model, rather than trusting a hardcoded list."""
        langs = getattr(getattr(self.tts, "synthesizer", None), "tts_model", None)
        for attr in ("language_manager", "languages"):
            found = getattr(langs, attr, None) if langs is not None else None
            if found is not None:
                names = getattr(found, "language_names", found)
                if isinstance(names, (list, tuple)):
                    return list(names)
        return list(getattr(self.tts, "languages", []) or [])

    def synthesise(self, request: SynthesisRequest, reference: list[Audio]) -> Audio:
        """Clone the reference voice saying `request.text`.

        Raises:
            AudioError: If the reference audio is too short to clone from.
        """
        ref = _prepare_reference(reference)
        wav = self.tts.tts(
            text=request.text,
            speaker_wav=_as_float_list(ref),
            language=request.language,
            speed=request.speed,
        )
        samples = np.asarray(wav, dtype=np.float32)
        if not samples.size:
            raise AudioError("TTS backend returned empty audio.")
        return Audio(samples, self.sample_rate, f"clone::{request.language}")


class EchoCloner:
    """Test double: returns the reference audio, unchanged.

    Lets the whole attack pipeline run in CI without a TTS install. It is a
    *perfect* clone by construction, so any attack result it produces is
    meaningless as evidence -- which is the point. It makes the acoustic
    branch trivially fooled, so the CSBG's behaviour can be exercised in
    isolation. Every clip it returns is marked `source="echo_clone"` and the
    suite treats such trials as simulated.
    """

    @property
    def name(self) -> str:
        return "echo::test_double"

    def supported_languages(self) -> list[str]:
        return ["ta", "en"]

    def synthesise(self, request: SynthesisRequest, reference: list[Audio]) -> Audio:
        ref = _prepare_reference(reference)
        return Audio(ref.samples.copy(), ref.sample_rate, "echo_clone")


def _prepare_reference(reference: list[Audio]) -> Audio:
    """Join reference clips into one conditioning signal."""
    if not reference:
        raise AudioError("Voice cloning needs at least one reference clip.")
    ref = reference[0] if len(reference) == 1 else concatenate(reference, gap_ms=120)
    if ref.duration_sec < MIN_REFERENCE_SEC:
        raise AudioError(
            f"Only {ref.duration_sec:.1f}s of reference audio; cloning needs at least "
            f"{MIN_REFERENCE_SEC:.0f}s. A clone made from less will fail for reasons "
            "unrelated to the defence and would understate attacker power."
        )
    return ref


def _as_float_list(audio: Audio) -> list[float]:
    return audio.samples.astype(np.float32).tolist()


# --------------------------------------------------------------------------
# Admissibility screening
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CloneQualityReport:
    """Whether a clone is good enough to be a valid test of the CSBG."""

    similarity: float
    """Cosine similarity between the clone's embedding and the victim's
    template, on the same scale the acoustic branch scores on."""

    threshold: float
    admissible: bool
    """True when the acoustic branch accepts the clone. Only admissible
    clones count as A3/A4/A5 trials."""

    template_self_consistency: float = 0.0
    """The victim's own within-template similarity. Context for reading
    `similarity`: a clone at 0.65 against a template that only holds together
    at 0.70 is a much better clone than the raw number suggests."""

    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "similarity": round(self.similarity, 4),
            "threshold": round(self.threshold, 4),
            "admissible": self.admissible,
            "template_self_consistency": round(self.template_self_consistency, 4),
            "reason": self.reason,
        }


def screen_clone(
    clone: Audio,
    template: SpeakerTemplate,
    embedder: Any,
    *,
    threshold: float,
) -> CloneQualityReport:
    """Decide whether a clone is admissible as an attack trial.

    Args:
        clone: The synthesised audio.
        template: The victim's enrolled speaker template.
        embedder: Anything with `.embed(Audio) -> SpeakerEmbedding`.
        threshold: The acoustic branch's operating threshold. Pass the value
            the system actually uses -- screening at a laxer threshold than
            the deployed one would admit clones the real system would have
            stopped, inflating the CSBG's apparent workload.

    Returns:
        A CloneQualityReport.
    """
    embedding: SpeakerEmbedding = embedder.embed(prepare_for_embedding(clone))
    similarity = template.score(embedding)
    admissible = similarity >= threshold
    consistency = template.self_consistency

    if admissible:
        reason = (
            f"Clone scores {similarity:.3f} against the victim's template "
            f"(threshold {threshold:.3f}); the acoustic branch is fooled, so this "
            "trial tests the remaining branches."
        )
    else:
        reason = (
            f"Clone scores only {similarity:.3f} (threshold {threshold:.3f}); the "
            "acoustic branch stops it unaided. Not admissible as a test of the CSBG -- "
            "count it towards attack yield, not towards the attack's success rate."
        )
    return CloneQualityReport(
        similarity=similarity,
        threshold=threshold,
        admissible=admissible,
        template_self_consistency=consistency,
        reason=reason,
    )


@dataclass(slots=True)
class CloneBatchStats:
    """Yield statistics over a batch of generated clones.

    Reported alongside every clone-based attack row. Without it a low attack
    success rate is uninterpretable.
    """

    attempted: int = 0
    synthesised: int = 0
    admissible: int = 0
    similarities: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def synthesis_rate(self) -> float:
        """Fraction of requests the TTS produced audio for at all."""
        return self.synthesised / self.attempted if self.attempted else 0.0

    @property
    def yield_rate(self) -> float:
        """Fraction of attempts that produced an ECAPA-fooling clone.

        This is the number that decides whether the attack rows mean
        anything. Report it in the paper next to the attack table.
        """
        return self.admissible / self.attempted if self.attempted else 0.0

    @property
    def mean_similarity(self) -> float:
        return float(np.mean(self.similarities)) if self.similarities else 0.0

    def summary(self) -> str:
        return (
            f"{self.admissible}/{self.attempted} clones fooled the acoustic branch "
            f"(yield {self.yield_rate:.0%}, mean similarity {self.mean_similarity:.3f}); "
            f"{self.synthesised} of {self.attempted} synthesised successfully."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "synthesised": self.synthesised,
            "admissible": self.admissible,
            "synthesis_rate": round(self.synthesis_rate, 4),
            "yield_rate": round(self.yield_rate, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "n_failures": len(self.failures),
        }


def check_language_support(backend: CloneBackend, code: str = "ta") -> tuple[bool, str]:
    """Check a backend for a language before building a corpus against it.

    Returns:
        (supported, message). A backend that cannot enumerate its languages
        returns True with a caveat rather than blocking -- an unknown list is
        not evidence of absence -- but the caveat must reach the log.
    """
    try:
        langs = backend.supported_languages()
    except Exception as exc:  # noqa: BLE001 - backends vary wildly
        return True, f"Could not enumerate languages for {backend.name} ({exc}); assuming support."
    if not langs:
        return True, f"{backend.name} reported no language list; assuming support."
    if code in langs:
        return True, f"{backend.name} supports {code!r}."
    return False, (
        f"{backend.name} does not list {code!r} (has: {', '.join(sorted(langs)[:12])}). "
        "Use IndicTTS or a commercial backend, or report the absence as a finding."
    )


CLONE_ATTACKS: tuple[AttackType, ...] = (
    AttackType.A3_CLONE,
    AttackType.A4_CLONE_KNOWLEDGE,
    AttackType.A5_STYLE_ADAPTIVE,
)


__all__ = [
    "CLONE_ATTACKS",
    "CloneBackend",
    "CloneBatchStats",
    "CloneQualityReport",
    "EchoCloner",
    "MIN_REFERENCE_SEC",
    "SynthesisRequest",
    "XTTSCloner",
    "check_language_support",
    "screen_clone",
]
