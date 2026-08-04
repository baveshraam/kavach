"""A1 -- replay attacks: simulation, and detection as a liveness signal.

WHAT ACTUALLY STOPS A REPLAY HERE
---------------------------------
Not the detector in this file. A replay is a recording of the victim answering
some *earlier* challenge, and KAVACH picks a fresh challenge per session from a
single-use ledger with a TTL (`kavach.challenge.ChallengeLedger`). An old
recording answers the wrong question, so the knowledge branch rejects it and
the liveness gate rejects it. That is the primary defence and it is already
built.

This module adds a *supplementary* signal for the case the challenge-response
protocol cannot cover: the attacker recorded the victim answering this exact
challenge earlier, or coerced them into saying it. Then the answer is correct
and only the audio itself gives the attack away.

The cues below are honest but limited. Replay detection is an entire research
field with its own benchmark (ASVspoof's PA condition); this is not a
contribution and must not be presented as one. Cite ASVspoof, report these as
an engineering component, and be explicit that the thresholds are fitted to
one room and one pair of devices.

CALIBRATE BEFORE USE
--------------------
`SpectralCues` thresholds are device-dependent -- they encode what a *specific*
loudspeaker and microphone do to a signal. The defaults here are placeholders
chosen to be roughly sane for a phone speaker, not measurements. Run
`fit_spectral_thresholds()` on genuine-vs-replay recordings from the actual lab
hardware and report the fitted values, or leave this branch out of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..audio import Audio, AudioError

_EPS = 1e-10


# --------------------------------------------------------------------------
# Attack generation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayChannel:
    """A crude model of a loudspeaker-to-microphone playback chain.

    Three effects, in the order they physically happen: the loudspeaker and
    microphone band-limit the signal, the room adds reverberation, and the
    recording picks up ambient noise.

    This is a proxy for developing the detector, not a substitute for
    recording a real replay. See the package docstring.
    """

    low_cut_hz: float = 120.0
    """Below this the response rolls off. Small drivers cannot move enough air
    to reproduce sub-bass, so replayed speech loses the low end that a mouth
    close to a microphone produces."""

    high_cut_hz: float = 7000.0
    """Above this the response rolls off, from the speaker, the microphone,
    and any codec in between."""

    rolloff_octaves: float = 0.5
    """Width of the raised-cosine transition at each edge. A brick wall would
    be trivially detectable and would flatter the detector."""

    reverb_rt60_sec: float = 0.25
    """Reverberation time. 0 disables reverb."""

    snr_db: float = 30.0
    """Additive noise level. `math.inf` disables noise."""

    noise_colour: str = "pink"
    """'pink' (1/f, like room and electronics noise) or 'white'."""

    def describe(self) -> str:
        return (
            f"band {self.low_cut_hz:.0f}-{self.high_cut_hz:.0f} Hz, "
            f"RT60 {self.reverb_rt60_sec:.2f}s, SNR {self.snr_db:.0f} dB "
            f"({self.noise_colour})"
        )


def _band_response(freqs: np.ndarray, channel: ReplayChannel) -> np.ndarray:
    """Smooth band-limiting response, 1.0 in-band falling to 0.0 out of band."""
    resp = np.ones_like(freqs)
    width = max(channel.rolloff_octaves, 1e-3)

    # Work in octaves so the transition is symmetric on a log-frequency axis,
    # which is how the physical roll-offs actually behave.
    with np.errstate(divide="ignore", invalid="ignore"):
        lo_oct = np.log2(np.maximum(freqs, _EPS) / max(channel.low_cut_hz, _EPS))
        hi_oct = np.log2(np.maximum(freqs, _EPS) / max(channel.high_cut_hz, _EPS))

    # Raised cosine from 0 to 1 across `width` octaves either side.
    lo_ramp = np.clip(lo_oct / width, 0.0, 1.0)
    resp *= 0.5 - 0.5 * np.cos(np.pi * lo_ramp)

    hi_ramp = np.clip(-hi_oct / width, 0.0, 1.0)
    resp *= 0.5 - 0.5 * np.cos(np.pi * hi_ramp)

    resp[freqs <= 0] = 0.0
    return resp


def apply_band_limit(audio: Audio, channel: ReplayChannel) -> Audio:
    """Band-limit via FFT with a smooth response curve."""
    if not len(audio.samples):
        return audio
    spectrum = np.fft.rfft(audio.samples)
    freqs = np.fft.rfftfreq(len(audio.samples), 1.0 / audio.sample_rate)
    filtered = np.fft.irfft(spectrum * _band_response(freqs, channel), n=len(audio.samples))
    return Audio(filtered.astype(np.float32), audio.sample_rate, audio.source)


def synthetic_rir(sample_rate: int, rt60_sec: float, *, seed: int | None = None) -> np.ndarray:
    """Exponentially-decaying noise as a stand-in room impulse response.

    A real RIR has early reflections with structure; this has none. It
    reproduces the one property the detector cares about -- energy smeared
    across time, filling the gaps between words -- and nothing else.
    """
    rng = np.random.default_rng(seed)
    n = max(1, int(rt60_sec * sample_rate))
    decay = np.exp(-6.907 * np.arange(n) / n)  # -60 dB over rt60
    rir = rng.standard_normal(n) * decay
    rir[0] += 1.0  # direct path
    return (rir / np.linalg.norm(rir)).astype(np.float32)


def _coloured_noise(n: int, colour: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if colour == "white":
        return white
    if colour != "pink":
        raise ValueError(f"Unknown noise colour {colour!r}; use 'pink' or 'white'.")
    spectrum = np.fft.rfft(white)
    freqs = np.arange(len(spectrum))
    scale = 1.0 / np.sqrt(np.maximum(freqs, 1.0))
    return np.fft.irfft(spectrum * scale, n=n)


def add_noise(audio: Audio, snr_db: float, *, colour: str = "pink", seed: int | None = None) -> Audio:
    """Add noise at a target SNR measured against the signal's RMS."""
    if math.isinf(snr_db) or not len(audio.samples):
        return audio
    rng = np.random.default_rng(seed)
    noise = _coloured_noise(len(audio.samples), colour, rng)
    noise_rms = float(np.sqrt(np.mean(noise**2))) or 1.0
    target_rms = audio.rms / (10.0 ** (snr_db / 20.0))
    noisy = audio.samples + (noise * (target_rms / noise_rms)).astype(np.float32)
    return Audio(noisy.astype(np.float32), audio.sample_rate, audio.source)


def simulate_replay(
    audio: Audio,
    channel: ReplayChannel | None = None,
    *,
    seed: int | None = None,
) -> Audio:
    """Approximate what the victim's recording sounds like after playback.

    NOT a replay recording. Use this to develop and unit-test the detector;
    record real playback through real hardware for anything reported.

    Args:
        audio: The victim's genuine recording.
        channel: Playback chain model. Defaults to a phone-speaker-ish chain.
        seed: Seeds the RIR and the noise, so attacks are reproducible.

    Returns:
        Band-limited, reverberated, noisy audio at the same sample rate.
    """
    channel = channel or ReplayChannel()
    out = apply_band_limit(audio, channel)

    if channel.reverb_rt60_sec > 0 and len(out.samples):
        rir = synthetic_rir(out.sample_rate, channel.reverb_rt60_sec, seed=seed)
        wet = np.convolve(out.samples, rir)[: len(out.samples)]
        out = Audio(wet.astype(np.float32), out.sample_rate, out.source)

    out = add_noise(out, channel.snr_db, colour=channel.noise_colour, seed=seed)

    peak = out.peak
    if peak > 0.99:  # playback gain staging is rarely perfect
        out = Audio((out.samples / peak * 0.95).astype(np.float32), out.sample_rate, out.source)
    return Audio(out.samples, out.sample_rate, f"{audio.source}|simulated_replay")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def log_energy_envelope(audio: Audio, *, hop_ms: int = 10, win_ms: int = 25) -> np.ndarray:
    """Frame-wise log energy, mean-removed.

    The envelope is used instead of the raw waveform for near-duplicate
    matching because it survives the channel. Filtering, reverb and additive
    noise all destroy sample-level waveform correlation while leaving the
    rhythm of the utterance -- which syllables are loud, where the pauses fall
    -- almost intact. Two recordings of the *same performance* share that
    rhythm exactly; two recordings of a person saying the same words twice do
    not.
    """
    if not len(audio.samples):
        return np.zeros(0, dtype=np.float32)
    hop = max(1, int(hop_ms / 1000 * audio.sample_rate))
    win = max(hop, int(win_ms / 1000 * audio.sample_rate))
    n_frames = max(1, 1 + (len(audio.samples) - win) // hop) if len(audio.samples) >= win else 1

    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio.samples[i * hop : i * hop + win]
        energies[i] = np.log(float(np.mean(frame**2)) + _EPS)
    return energies - energies.mean()


def envelope_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Peak normalised cross-correlation of two envelopes, in [-1, 1].

    Cross-correlation rather than a plain dot product, so a replay that starts
    a few hundred milliseconds earlier or later still matches.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0

    n = 1 << (len(a) + len(b) - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n=n)
    return float(np.max(np.abs(corr)) / (na * nb))


@dataclass(frozen=True, slots=True)
class SpectralCues:
    """Band-energy ratios that a playback chain shifts.

    Thresholds are NOT measurements. Fit them with
    `fit_spectral_thresholds()` on genuine and replayed recordings from the
    hardware actually used, or drop this cue.
    """

    sub_bass_hz: float = 80.0
    high_band_hz: float = 6000.0

    min_sub_bass_ratio: float = 0.002
    """A replay through a small driver has near-zero energy below 80 Hz.
    Genuine close-mic speech has some, from breath and proximity effect."""

    min_high_ratio: float = 0.01
    """The playback chain rolls off the top end. Very little energy above
    6 kHz is suspicious -- but so is a bad microphone, hence 'suspicious',
    not 'rejected'."""

    sub_bass_evidence: float = 0.30
    high_band_evidence: float = 0.35
    """Evidence each cue contributes when it fires, combined by noisy-OR.

    With these defaults and a `spectral_threshold` of 0.5, **neither cue
    alone is enough** -- both must fire (0.30 and 0.35 combine to 0.545).
    That is deliberate. Each cue on its own also describes a cheap microphone
    or a muffled room, and a false replay accusation rejects a genuine user
    from their own account. A caller who has measured that one cue separates
    cleanly on their hardware can lower `ReplayDetector.spectral_threshold`
    to act on it alone; `cue_separation()` is how to find out."""


def band_energy_ratios(audio: Audio, cues: SpectralCues | None = None) -> tuple[float, float]:
    """Return (sub-bass fraction, high-band fraction) of total spectral energy."""
    cues = cues or SpectralCues()
    if not len(audio.samples):
        return 0.0, 0.0
    power = np.abs(np.fft.rfft(audio.samples)) ** 2
    freqs = np.fft.rfftfreq(len(audio.samples), 1.0 / audio.sample_rate)
    total = float(power.sum()) + _EPS
    return (
        float(power[freqs < cues.sub_bass_hz].sum()) / total,
        float(power[freqs > cues.high_band_hz].sum()) / total,
    )


@dataclass(slots=True)
class ReplayReport:
    """Why a recording did or did not look like a replay."""

    is_replay: bool
    score: float
    """0 = clean, 1 = certain replay. A max over the individual cues, not a
    calibrated probability -- do not put this on an axis without calibrating
    it first."""

    exact_hash_match: bool = False
    best_envelope_similarity: float = 0.0
    matched_source: str = ""
    sub_bass_ratio: float = 0.0
    high_band_ratio: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_replay": self.is_replay,
            "score": round(self.score, 4),
            "exact_hash_match": self.exact_hash_match,
            "best_envelope_similarity": round(self.best_envelope_similarity, 4),
            "matched_source": self.matched_source,
            "sub_bass_ratio": round(self.sub_bass_ratio, 6),
            "high_band_ratio": round(self.high_band_ratio, 6),
            "reasons": self.reasons,
        }


class ReplayDetector:
    """Checks a probe against previously seen recordings and for channel cues.

    Two independent mechanisms:

    1. **Duplicate detection.** Every accepted recording's hash and energy
       envelope are remembered. An exact hash match is conclusive. A high
       envelope correlation means the same *performance* was submitted twice,
       which no honest speaker does.

    2. **Spectral cues.** Band-energy ratios shifted by a playback chain.
       Weak, device-dependent, off by default.

    The envelope memory grows with every enrolment and session. For a
    deployment that would need bounding or indexing; at study scale a list is
    the right answer and pretending otherwise would be premature.
    """

    def __init__(
        self,
        *,
        envelope_threshold: float = 0.85,
        cues: SpectralCues | None = None,
        use_spectral_cues: bool = False,
        spectral_threshold: float = 0.5,
    ) -> None:
        self.envelope_threshold = envelope_threshold
        self.cues = cues or SpectralCues()
        self.use_spectral_cues = use_spectral_cues
        self.spectral_threshold = spectral_threshold
        self._hashes: dict[str, str] = {}
        self._envelopes: list[tuple[str, np.ndarray]] = []

    def remember(self, audio: Audio, *, label: str = "") -> None:
        """Record a recording as seen, so a later resubmission is caught."""
        label = label or audio.source or f"clip_{len(self._envelopes)}"
        self._hashes[audio.sha256()] = label
        self._envelopes.append((label, log_energy_envelope(audio)))

    def remember_many(self, clips: Iterable[Audio]) -> None:
        for clip in clips:
            self.remember(clip)

    def check(self, audio: Audio) -> ReplayReport:
        """Score a probe for replay evidence."""
        if not len(audio.samples):
            raise AudioError("Cannot check an empty recording for replay.")

        report = ReplayReport(is_replay=False, score=0.0)

        digest = audio.sha256()
        if digest in self._hashes:
            report.exact_hash_match = True
            report.matched_source = self._hashes[digest]
            report.best_envelope_similarity = 1.0
            report.score = 1.0
            report.is_replay = True
            report.reasons.append(
                f"Byte-identical to a previously submitted recording ({report.matched_source})."
            )
            return report

        probe_env = log_energy_envelope(audio)
        for label, stored in self._envelopes:
            sim = envelope_similarity(probe_env, stored)
            if sim > report.best_envelope_similarity:
                report.best_envelope_similarity = sim
                report.matched_source = label

        if report.best_envelope_similarity >= self.envelope_threshold:
            report.is_replay = True
            report.score = max(report.score, report.best_envelope_similarity)
            report.reasons.append(
                f"Energy envelope matches a stored recording ({report.matched_source}) at "
                f"{report.best_envelope_similarity:.2f}; the same performance was submitted twice."
            )

        report.sub_bass_ratio, report.high_band_ratio = band_energy_ratios(audio, self.cues)
        if self.use_spectral_cues:
            fired: list[float] = []
            if report.sub_bass_ratio < self.cues.min_sub_bass_ratio:
                fired.append(self.cues.sub_bass_evidence)
                report.reasons.append(
                    f"Almost no energy below {self.cues.sub_bass_hz:.0f} Hz "
                    f"({report.sub_bass_ratio:.4f}); consistent with a small loudspeaker."
                )
            if report.high_band_ratio < self.cues.min_high_ratio:
                fired.append(self.cues.high_band_evidence)
                report.reasons.append(
                    f"Little energy above {self.cues.high_band_hz:.0f} Hz "
                    f"({report.high_band_ratio:.4f}); consistent with a band-limited channel."
                )

            # Noisy-OR: independent weak cues accumulate without any one of
            # them being able to exceed its own evidence weight.
            spectral_score = 1.0
            for evidence in fired:
                spectral_score *= 1.0 - evidence
            spectral_score = 1.0 - spectral_score

            report.score = max(report.score, spectral_score)
            if spectral_score >= self.spectral_threshold:
                report.is_replay = True
            elif fired:
                report.reasons.append(
                    f"Spectral evidence {spectral_score:.2f} is below the "
                    f"{self.spectral_threshold:.2f} bar; suspicious, not conclusive."
                )

        if not report.reasons:
            report.reasons.append("No replay evidence found.")
        return report


def fit_spectral_thresholds(
    genuine: list[Audio], replayed: list[Audio], *, cues: SpectralCues | None = None
) -> SpectralCues:
    """Fit band-ratio thresholds on real recordings from the study hardware.

    Places each threshold midway between the class means in log space, which
    is the equal-error point when the two classes are roughly log-normal with
    similar spread. Crude on purpose -- if a threshold fitted this way does
    not separate the classes, the cue does not work on that hardware and
    belongs out of the paper rather than in it with a fancier fit.

    Raises:
        ValueError: If either class is empty.
    """
    if not genuine or not replayed:
        raise ValueError("Need both genuine and replayed recordings to fit thresholds.")
    cues = cues or SpectralCues()

    gen = np.array([band_energy_ratios(a, cues) for a in genuine])
    rep = np.array([band_energy_ratios(a, cues) for a in replayed])

    def midpoint(col: int) -> float:
        g = float(np.mean(np.log(gen[:, col] + _EPS)))
        r = float(np.mean(np.log(rep[:, col] + _EPS)))
        return float(np.exp((g + r) / 2.0))

    return SpectralCues(
        sub_bass_hz=cues.sub_bass_hz,
        high_band_hz=cues.high_band_hz,
        min_sub_bass_ratio=midpoint(0),
        min_high_ratio=midpoint(1),
        sub_bass_evidence=cues.sub_bass_evidence,
        high_band_evidence=cues.high_band_evidence,
    )


def cue_separation(
    genuine: list[Audio], replayed: list[Audio], *, cues: SpectralCues | None = None
) -> dict[str, float]:
    """How well each spectral cue separates the two classes, as d-prime.

    Run this **before** trusting a fitted threshold. `fit_spectral_thresholds`
    will happily place a threshold in the middle of two identical
    distributions, and the resulting detector then fires at chance on a cue
    that carries no information. A d-prime near zero means the cue does not
    work on this hardware and should be left out rather than tuned.

    Computed on log ratios, where the distributions are closer to normal.

    Returns:
        {"sub_bass": d', "high_band": d'}. Above roughly 2 is a usable cue;
        below 1 is not.
    """
    cues = cues or SpectralCues()
    if not genuine or not replayed:
        raise ValueError("Need both genuine and replayed recordings to measure separation.")

    gen = np.log(np.array([band_energy_ratios(a, cues) for a in genuine]) + _EPS)
    rep = np.log(np.array([band_energy_ratios(a, cues) for a in replayed]) + _EPS)

    def dprime(col: int) -> float:
        pooled = math.sqrt((float(np.var(gen[:, col])) + float(np.var(rep[:, col]))) / 2.0)
        if pooled < _EPS:
            return 0.0
        return abs(float(np.mean(gen[:, col])) - float(np.mean(rep[:, col]))) / pooled

    return {"sub_bass": dprime(0), "high_band": dprime(1)}


__all__ = [
    "ReplayChannel",
    "ReplayDetector",
    "ReplayReport",
    "SpectralCues",
    "add_noise",
    "apply_band_limit",
    "band_energy_ratios",
    "cue_separation",
    "envelope_similarity",
    "fit_spectral_thresholds",
    "log_energy_envelope",
    "simulate_replay",
    "synthetic_rir",
]
