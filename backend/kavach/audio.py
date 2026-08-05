"""Audio loading and preprocessing.

Shared by the ASR and speaker-embedding paths so both see byte-identical
input. If they preprocessed differently, a transcript and an embedding could
disagree about what was said -- and that disagreement would be invisible,
showing up only as unexplained score noise.

`soundfile`/`librosa` are imported lazily so the CSBG core stays importable
without them.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Everything downstream assumes this rate. ECAPA-TDNN and Whisper are both
#: trained at 16 kHz; resampling once here avoids each doing it differently.
TARGET_SAMPLE_RATE = 16_000


class AudioError(RuntimeError):
    """Audio could not be loaded or is unusable for verification."""


@dataclass(slots=True)
class Audio:
    """Mono float32 waveform at a known sample rate."""

    samples: np.ndarray
    sample_rate: int
    source: str = ""

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise AudioError(f"Expected mono audio, got shape {self.samples.shape}")
        if self.samples.dtype != np.float32:
            self.samples = self.samples.astype(np.float32)

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate if self.sample_rate else 0.0

    @property
    def rms(self) -> float:
        """Root-mean-square level. Near zero means silence or a dead mic."""
        return float(np.sqrt(np.mean(self.samples**2))) if len(self.samples) else 0.0

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if len(self.samples) else 0.0

    @property
    def clipping_ratio(self) -> float:
        """Fraction of samples at or near full scale.

        Phone recordings clip constantly, and clipping degrades ECAPA
        embeddings measurably. Surfaced in the enrolment UI so a bad take can
        be re-recorded rather than silently poisoning the speaker's template.
        """
        if not len(self.samples):
            return 0.0
        return float(np.mean(np.abs(self.samples) >= 0.99))

    def sha256(self) -> str:
        """Content hash. Used to detect exact replay of a stored recording."""
        return hashlib.sha256(self.samples.tobytes()).hexdigest()

    def slice_seconds(self, start: float, end: float) -> Audio:
        a = max(0, int(start * self.sample_rate))
        b = min(len(self.samples), int(end * self.sample_rate))
        return Audio(self.samples[a:b], self.sample_rate, f"{self.source}[{start}:{end}]")


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Pre-flight check on a recording."""

    duration_sec: float
    rms: float
    peak: float
    clipping_ratio: float
    is_silent: bool
    is_clipped: bool
    is_too_short: bool
    is_too_long: bool

    @property
    def is_usable(self) -> bool:
        return not (self.is_silent or self.is_too_short)

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.is_silent:
            out.append("Audio is silent or near-silent -- check the microphone.")
        if self.is_too_short:
            out.append(f"Only {self.duration_sec:.1f}s; speaker embeddings need at least 1s.")
        if self.is_too_long:
            out.append(f"{self.duration_sec:.1f}s is longer than expected; will be truncated.")
        if self.is_clipped:
            out.append(
                f"{self.clipping_ratio:.1%} of samples are clipped -- "
                "lower the input gain and re-record."
            )
        return out


def check_quality(
    audio: Audio, *, min_seconds: float = 1.0, max_seconds: float = 30.0
) -> QualityReport:
    """Assess whether a recording is fit for enrolment or verification."""
    return QualityReport(
        duration_sec=audio.duration_sec,
        rms=audio.rms,
        peak=audio.peak,
        clipping_ratio=audio.clipping_ratio,
        is_silent=audio.rms < 1e-4,
        is_clipped=audio.clipping_ratio > 0.01,
        is_too_short=audio.duration_sec < min_seconds,
        is_too_long=audio.duration_sec > max_seconds,
    )


def _load_wav_stdlib(path: Path) -> Audio | None:
    """Load a PCM WAV with the standard library, avoiding a soundfile import.

    Returns None for anything non-trivial (compressed, >16-bit) so the caller
    falls through to soundfile.
    """
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return None
            frames = w.readframes(w.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() > 1:
                data = data.reshape(-1, w.getnchannels()).mean(axis=1)
            return Audio(data, w.getframerate(), str(path))
    except (wave.Error, EOFError, ValueError):
        return None


def load_audio(
    path: str | Path, *, target_sr: int = TARGET_SAMPLE_RATE, mono: bool = True
) -> Audio:
    """Load an audio file, resampled to `target_sr` and mixed to mono.

    Tries the stdlib WAV reader first (no dependency), then soundfile, then
    librosa. Browser recordings arrive as WebM/Opus, which needs ffmpeg --
    see `decode_bytes`.

    Raises:
        AudioError: If the file cannot be read by any available backend.
    """
    p = Path(path)
    if not p.exists():
        raise AudioError(f"Audio file not found: {p}")

    audio: Audio | None = None
    if p.suffix.lower() == ".wav":
        audio = _load_wav_stdlib(p)

    if audio is None:
        try:
            import soundfile as sf

            data, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if data.ndim > 1 and mono:
                data = data.mean(axis=1)
            audio = Audio(np.asarray(data, dtype=np.float32), sr, str(p))
        except ImportError:
            pass
        except Exception as exc:
            raise AudioError(f"Could not read {p}: {exc}") from exc

    if audio is None:
        try:
            import librosa

            data, sr = librosa.load(str(p), sr=None, mono=mono)
            audio = Audio(np.asarray(data, dtype=np.float32), int(sr), str(p))
        except ImportError as exc:
            raise AudioError(
                f"Cannot read {p}: install `soundfile` or `librosa` "
                "(pip install -r requirements.txt)."
            ) from exc
        except Exception as exc:
            raise AudioError(f"Could not read {p}: {exc}") from exc

    if audio.sample_rate != target_sr:
        audio = resample(audio, target_sr)
    return audio


def resample(audio: Audio, target_sr: int) -> Audio:
    """Resample to `target_sr`.

    Uses librosa/scipy when available; otherwise falls back to linear
    interpolation, which is adequate for downsampling speech but introduces
    aliasing. The fallback warns rather than failing, because a working
    pipeline on a fresh machine beats a hard dependency error -- but real
    corpus processing should have librosa installed.
    """
    if audio.sample_rate == target_sr:
        return audio

    try:
        import librosa

        data = librosa.resample(
            audio.samples, orig_sr=audio.sample_rate, target_sr=target_sr
        )
        return Audio(np.asarray(data, dtype=np.float32), target_sr, audio.source)
    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(audio.sample_rate, target_sr)
        data = resample_poly(audio.samples, target_sr // g, audio.sample_rate // g)
        return Audio(np.asarray(data, dtype=np.float32), target_sr, audio.source)
    except ImportError:
        pass

    import warnings

    warnings.warn(
        "Resampling by linear interpolation (no librosa/scipy). This aliases; "
        "install librosa before processing corpus audio.",
        RuntimeWarning,
        stacklevel=2,
    )
    n_out = int(len(audio.samples) * target_sr / audio.sample_rate)
    x_old = np.linspace(0.0, 1.0, len(audio.samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return Audio(
        np.interp(x_new, x_old, audio.samples).astype(np.float32), target_sr, audio.source
    )


def decode_bytes(
    raw: bytes, *, suffix: str = ".webm", target_sr: int = TARGET_SAMPLE_RATE
) -> Audio:
    """Decode an in-memory audio blob, e.g. a browser MediaRecorder upload.

    WebM/Opus needs ffmpeg. Piping through ffmpeg avoids writing the upload to
    disk before it has been validated.

    Raises:
        AudioError: If decoding fails or ffmpeg is unavailable.
    """
    if suffix.lower() == ".wav":
        try:
            import soundfile as sf

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            audio = Audio(np.asarray(data, dtype=np.float32), int(sr), "upload")
            return resample(audio, target_sr) if sr != target_sr else audio
        except ImportError:
            pass
        except Exception:
            pass  # fall through to the stdlib reader, then ffmpeg

        # 16-bit PCM WAV needs no dependency at all, and it is what `save_wav`
        # writes -- so the round trip through the API works on a bare install
        # with neither soundfile nor ffmpeg present.
        try:
            with wave.open(io.BytesIO(raw), "rb") as w:
                if w.getsampwidth() == 2:
                    frames = w.readframes(w.getnframes())
                    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if w.getnchannels() > 1:
                        data = data.reshape(-1, w.getnchannels()).mean(axis=1)
                    audio = Audio(data.copy(), w.getframerate(), "upload")
                    return resample(audio, target_sr) if audio.sample_rate != target_sr else audio
        except (wave.Error, EOFError, ValueError):
            pass  # fall through to ffmpeg

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "f32le", "-acodec", "pcm_f32le",
                "-ac", "1", "-ar", str(target_sr),
                "pipe:1",
            ],
            input=raw,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise AudioError(
            "ffmpeg is required to decode browser audio (WebM/Opus). "
            "Install it and ensure it is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[:400]
        raise AudioError(f"ffmpeg failed to decode the upload: {detail}") from exc

    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if not len(data):
        raise AudioError("Decoded audio is empty -- the recording may have failed.")
    return Audio(data.copy(), target_sr, "upload")


def save_wav(audio: Audio, path: str | Path) -> Path:
    """Write a 16-bit PCM WAV. Used for archiving and for attack generation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm.tobytes())
    return p


def normalise_peak(audio: Audio, target_peak: float = 0.95) -> Audio:
    """Scale so the loudest sample sits at `target_peak`.

    Applied before embedding so recordings made at different input gains are
    comparable. Silent audio is returned unchanged rather than amplifying
    noise to full scale.
    """
    peak = audio.peak
    if peak < 1e-6:
        return audio
    return Audio(audio.samples * (target_peak / peak), audio.sample_rate, audio.source)


def trim_silence(audio: Audio, *, threshold_db: float = -40.0, pad_ms: int = 100) -> Audio:
    """Trim leading and trailing silence, keeping `pad_ms` either side.

    Leading silence dilutes an ECAPA embedding -- it is speaker-independent
    signal averaged into a speaker-specific representation. Returns the input
    unchanged if the whole clip is below threshold, so a silent recording
    stays detectably silent rather than becoming zero-length.
    """
    if not len(audio.samples):
        return audio

    frame = max(1, int(0.01 * audio.sample_rate))
    n_frames = len(audio.samples) // frame
    if n_frames < 2:
        return audio

    frames = audio.samples[: n_frames * frame].reshape(n_frames, frame)
    energy = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    db = 20 * np.log10(energy / (np.max(energy) + 1e-10))
    voiced = np.where(db > threshold_db)[0]
    if not len(voiced):
        return audio

    pad = int(pad_ms / 1000 * audio.sample_rate)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(audio.samples), (voiced[-1] + 1) * frame + pad)
    return Audio(audio.samples[start:end], audio.sample_rate, audio.source)


def leading_silence_sec(audio: Audio, *, threshold_db: float = -40.0) -> float:
    """Seconds before speech starts, by the same threshold `trim_silence` uses.

    The recording protocol asks participants to pause a beat after pressing
    record, and this is the check that they did. Two things depend on it.
    `integrity.py` estimates a noise floor from the head of the clip, and a
    file that opens mid-syllable gives it speech to measure instead. And a
    recorder that starts capturing late clips the first phoneme, which costs a
    token in a corpus where the tokens are the measurement.

    Returns the full duration if nothing crosses the threshold, so a silent
    file reads as "all silence" rather than "starts immediately".
    """
    if not len(audio.samples):
        return 0.0

    frame = max(1, int(0.01 * audio.sample_rate))
    n_frames = len(audio.samples) // frame
    if n_frames < 2:
        return 0.0

    frames = audio.samples[: n_frames * frame].reshape(n_frames, frame)
    energy = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    db = 20 * np.log10(energy / (np.max(energy) + 1e-10))
    voiced = np.where(db > threshold_db)[0]
    if not len(voiced):
        return audio.duration_sec
    return float(voiced[0] * frame / audio.sample_rate)


def estimate_snr_db(audio: Audio, *, window_sec: float = 0.3) -> float:
    """Crude speech-to-noise ratio: loud percentile minus quietest window.

    Deliberately not a VAD. The noise floor is the quietest `window_sec` of
    the file and the speech level is the 95th percentile of the same windows,
    which needs no model and no threshold tuning, and is stable on the one
    thing it is used for -- comparing recordings against each other and
    against a collection standard.

    It reads *low* on a file with no pauses at all, because then the quietest
    window still contains speech. That failure is in the safe direction: it
    flags a recording for a human to check rather than passing it silently.
    """
    sr = audio.sample_rate
    win = max(1, int(window_sec * sr))
    if len(audio.samples) < 2 * win:
        return 0.0

    step = max(1, win // 6)
    starts = range(0, len(audio.samples) - win, step)
    rms = np.array([float(np.sqrt(np.mean(audio.samples[s : s + win] ** 2))) for s in starts])
    noise = 20 * np.log10(rms.min() + 1e-12)
    speech = 20 * np.log10(float(np.percentile(rms, 95)) + 1e-12)
    return float(speech - noise)


def prepare_for_embedding(
    audio: Audio, *, max_seconds: float = 30.0, do_trim: bool = True
) -> Audio:
    """Standard preprocessing before ECAPA embedding extraction.

    Trim silence, peak-normalise, truncate. Applied identically at enrolment
    and at verification -- a mismatch would shift embeddings systematically
    and inflate the measured EER for reasons that have nothing to do with the
    speaker.
    """
    out = trim_silence(audio) if do_trim else audio
    out = normalise_peak(out)
    if out.duration_sec > max_seconds:
        out = out.slice_seconds(0.0, max_seconds)
    return out


def concatenate(clips: list[Audio], *, gap_ms: int = 0) -> Audio:
    """Join clips, optionally with silence between them.

    Used to build splice attacks (A2) from recorded word segments.

    Raises:
        AudioError: On empty input or mismatched sample rates.
    """
    if not clips:
        raise AudioError("Cannot concatenate an empty list of clips.")
    sr = clips[0].sample_rate
    if any(c.sample_rate != sr for c in clips):
        raise AudioError("All clips must share a sample rate before concatenation.")

    gap = np.zeros(int(gap_ms / 1000 * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, c in enumerate(clips):
        if i and len(gap):
            parts.append(gap)
        parts.append(c.samples)
    return Audio(np.concatenate(parts), sr, "concatenated")


__all__ = [
    "Audio",
    "AudioError",
    "QualityReport",
    "TARGET_SAMPLE_RATE",
    "load_audio",
    "decode_bytes",
    "save_wav",
    "resample",
    "normalise_peak",
    "trim_silence",
    "leading_silence_sec",
    "estimate_snr_db",
    "prepare_for_embedding",
    "check_quality",
    "concatenate",
]
