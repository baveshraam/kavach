"""A2 -- cut-and-paste splicing: construction and detection.

THE ATTACK
----------
The attacker holds recordings of the victim but not of the victim answering
*this* challenge. So they cut words and phrases out of what they have and
concatenate them into an answer. Every sample is the victim's real voice, so
the acoustic branch accepts and no synthesis detector can help: there is no
synthesis.

WHAT GIVES A SPLICE AWAY
------------------------
Two cues, aimed at two different attackers.

**The click.** A hard cut between two clips almost never lands on matching
waveform values, so the join is a step discontinuity -- audible as a click and
visible as a single-sample jump far outside the local signal statistics. This
catches the attacker who used `sox` and stopped there. A crossfade of even
10 ms removes it entirely, so it is only ever the first filter.

**The background.** The more durable cue. Segments cut from different
recordings carry different room tone: different noise floor, different
spectral tilt, different hum. Human speech has pauses, and in a genuine
recording the background is the same on both sides of every pause. In a splice
it steps. This survives crossfading, because crossfading blends 10 ms of a
mismatch that persists for seconds.

WHAT DEFEATS BOTH
-----------------
An attacker who records every source segment in one sitting, in one room, with
one microphone, and crossfades the joins. Then the background genuinely does
match and the signal-level cues are gone. That attacker is exactly why the
threat model puts A2 above A1: the defence that remains is that they can only
say words they have on tape, and stitching an answer out of a fixed inventory
tends to produce language choices that are not the speaker's own. Whether the
CSBG actually catches that is an experimental question, and the honest answer
may be "only sometimes" -- the threat-model table marks the knowledge branch
"partial" for A2 for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..audio import Audio, AudioError

_EPS = 1e-10


# --------------------------------------------------------------------------
# Attack generation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpliceConfig:
    """How careful the splicing attacker is.

    `naive()` and `careful()` bracket the range. Reporting both is what makes
    the A2 row meaningful -- a detector that only catches `naive()` has not
    shown much.
    """

    crossfade_ms: float = 0.0
    """Equal-power crossfade at each join. 0 = hard cut, which clicks."""

    gap_ms: float = 0.0
    """Pause inserted between segments, mimicking a natural one."""

    gap_fill: str = "silence"
    """'silence' writes exact zeros; 'room_tone' copies the quietest stretch
    of the preceding segment.

    Digital silence is its own giveaway -- no microphone ever records a run of
    exact zeros, so a pause with none is evidence of an edit regardless of
    anything else. A competent attacker pads with room tone lifted from one of
    their own recordings, which is what 'room_tone' models. It also makes the
    attack *harder* to catch in one way and easier in another: the pause stops
    being impossible, but the tone belongs to one segment and therefore
    mismatches the other's background at the join."""

    match_levels: bool = False
    """Scale each segment to a common RMS. An attacker who does not do this
    leaves obvious loudness steps."""

    @classmethod
    def naive(cls) -> SpliceConfig:
        """Hard cuts, no level matching. The `sox` attacker."""
        return cls(crossfade_ms=0.0, gap_ms=0.0, match_levels=False)

    @classmethod
    def careful(cls) -> SpliceConfig:
        """Crossfaded, level-matched, room-tone pauses."""
        return cls(crossfade_ms=15.0, gap_ms=80.0, gap_fill="room_tone", match_levels=True)


def _room_tone(clip: np.ndarray, n: int, sample_rate: int) -> np.ndarray:
    """Lift `n` samples of background from the quietest part of a clip.

    Finds the lowest-energy window of the required length and tiles it. Real
    room tone is what an attacker would pad with, since exact zeros are
    impossible in a recording.
    """
    if len(clip) < n or n <= 0:
        return np.zeros(max(n, 0), dtype=np.float32)

    hop = max(1, sample_rate // 100)
    best_start, best_energy = 0, np.inf
    for start in range(0, len(clip) - n + 1, hop):
        energy = float(np.mean(clip[start : start + n] ** 2))
        if energy < best_energy:
            best_start, best_energy = start, energy
    return clip[best_start : best_start + n].astype(np.float32).copy()


def _equal_power_fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power (constant-energy) fade curves.

    A linear crossfade dips in loudness at the midpoint because the two
    signals are uncorrelated; the sine/cosine pair keeps total power flat,
    which is what a careful attacker would use.
    """
    t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
    return np.cos(t), np.sin(t)


def splice_segments(
    segments: list[Audio], config: SpliceConfig | None = None
) -> Audio:
    """Build a spliced utterance from recorded segments.

    Args:
        segments: Clips cut from the victim's recordings, in the order they
            should be spoken. All must share a sample rate.
        config: Attacker sophistication. Defaults to `SpliceConfig.naive()`.

    Returns:
        The spliced audio, with `source` marked so it can never be mistaken
        for a genuine recording downstream.

    Raises:
        AudioError: On empty input or mismatched sample rates.
    """
    if not segments:
        raise AudioError("Cannot splice an empty list of segments.")
    sr = segments[0].sample_rate
    if any(s.sample_rate != sr for s in segments):
        raise AudioError("All segments must share a sample rate before splicing.")
    config = config or SpliceConfig.naive()

    clips = [s.samples.astype(np.float32) for s in segments]

    if config.match_levels:
        target = float(np.mean([float(np.sqrt(np.mean(c**2))) for c in clips if len(c)]) or 0.0)
        if target > _EPS:
            clips = [
                c * (target / max(float(np.sqrt(np.mean(c**2))), _EPS)) if len(c) else c
                for c in clips
            ]

    gap_n = int(config.gap_ms / 1000 * sr)
    fade_n = int(config.crossfade_ms / 1000 * sr)

    out = clips[0]
    for i, nxt in enumerate(clips[1:]):
        if gap_n:
            gap = (
                _room_tone(clips[i], gap_n, sr)
                if config.gap_fill == "room_tone"
                else np.zeros(gap_n, dtype=np.float32)
            )
            out = np.concatenate([out, gap])
        n = min(fade_n, len(out), len(nxt))
        if n > 1:
            down, up = _equal_power_fade(n)
            blended = out[-n:] * down + nxt[:n] * up
            out = np.concatenate([out[:-n], blended, nxt[n:]])
        else:
            out = np.concatenate([out, nxt])

    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = out / peak * 0.95
    return Audio(out.astype(np.float32), sr, "spliced")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _frames(samples: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Frame a signal into overlapping windows."""
    if len(samples) < win:
        padded = np.pad(samples, (0, win - len(samples)))
        return padded[None, :]
    view = np.lib.stride_tricks.sliding_window_view(samples, win)
    return view[::hop]


def _band_spectra(frames: np.ndarray, sample_rate: int, n_bands: int = 16) -> np.ndarray:
    """Log energy in log-spaced frequency bands, one row per frame.

    Log-spaced rather than linear so the low frequencies -- where room tone
    and hum live, and where recordings differ most -- are not averaged away
    into one wide bin.
    """
    window = np.hanning(frames.shape[1]).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sample_rate)

    edges = np.geomspace(50.0, max(sample_rate / 2.0, 100.0), n_bands + 1)
    out = np.empty((frames.shape[0], n_bands), dtype=np.float32)
    for b in range(n_bands):
        mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
        out[:, b] = (
            10.0 * np.log10(power[:, mask].mean(axis=1) + _EPS)
            if mask.any()
            else -100.0
        )
    return out


@dataclass(slots=True)
class SpliceBoundary:
    """One suspected join."""

    time_sec: float
    level_step_db: float
    """How far the background noise floor jumps across the pause."""

    spectral_distance_db: float
    """Mean absolute per-band difference in background spectrum."""

    click: bool = False
    """A step discontinuity in the waveform: a hard cut with no crossfade."""

    def to_dict(self) -> dict[str, object]:
        return {
            "time_sec": round(self.time_sec, 3),
            "level_step_db": round(self.level_step_db, 2),
            "spectral_distance_db": round(self.spectral_distance_db, 2),
            "click": self.click,
        }


@dataclass(slots=True)
class SpliceReport:
    """Splice evidence for one recording."""

    is_spliced: bool
    score: float
    """Max evidence across boundaries, roughly 0-1. Not calibrated."""

    boundaries: list[SpliceBoundary] = field(default_factory=list)
    n_clicks: int = 0
    digital_silence_runs: list[tuple[float, float]] = field(default_factory=list)
    """Runs of exact zeros, which a microphone cannot produce."""

    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_spliced": self.is_spliced,
            "score": round(self.score, 4),
            "n_clicks": self.n_clicks,
            "n_digital_silence_runs": len(self.digital_silence_runs),
            "boundaries": [b.to_dict() for b in self.boundaries],
            "reasons": self.reasons,
        }


def detect_clicks(
    audio: Audio, *, sigma: float = 12.0, min_jump_ratio: float = 0.5
) -> list[float]:
    """Find step discontinuities: times where the waveform jumps.

    Works on the *second* difference, not the first. Speech is band-limited,
    so it is locally close to a straight line: each sample sits near the
    average of its neighbours, and the second difference stays small. A hard
    cut breaks that -- the signal steps to an unrelated value and then
    continues smoothly, which is exactly what the second difference is
    largest at. The first difference is far less selective, because loud
    high-frequency speech produces large first differences all the time.

    Two conditions must both hold, and both are measured **locally**, in
    short blocks, rather than over the whole recording:

    - `sigma` times the block's median curvature. Catches a jump that is an
      outlier among its own neighbours.
    - `min_jump_ratio` times the block's RMS. Catches a jump that is large
      relative to the local signal.

    Locality is not a refinement, it is the whole thing. Speech is wildly
    non-stationary: curvature during a loud syllable dwarfs curvature during a
    quiet one, so a threshold set from the whole recording's statistics is far
    too low during loud passages and flags every one of them. Comparing each
    sample against its own 20 ms neighbourhood removes that entirely.

    Requiring both conditions makes the detector conservative. The local
    median collapses to zero on a constant stretch, where every rounding error
    becomes an infinite outlier; the amplitude floor alone fires on any
    broadband passage, since near-Nyquist noise has large curvature by
    construction. Together they miss quiet joins rather than inventing them,
    which is the right trade when a false positive accuses a genuine speaker
    of splicing.

    This is the weaker of the two splice cues regardless -- a 10 ms crossfade
    removes clicks entirely. `min_jump_ratio` is an engineering default and
    should be checked against genuine recordings from the study hardware.

    Precondition: the recording has a noise floor. A microphone always
    produces one, but a file that has been edited may contain runs of exact
    zeros, and signal emerging from digital silence looks like a
    discontinuity to any curvature test -- arguably correctly, since the
    silence was inserted. `detect_digital_silence` reports those runs
    directly, which is the better cue for that case.

    Returns:
        Times in seconds of suspected hard cuts.
    """
    if len(audio.samples) < 4:
        return []

    curvature = np.abs(np.diff(audio.samples, n=2))
    block = max(8, int(0.020 * audio.sample_rate))
    n_blocks = max(1, len(curvature) // block)

    # A quiet block gets a floor tied to the whole recording, not just to
    # itself. Without it, a near-silent stretch has a near-zero bar and every
    # rounding error there looks like a discontinuity. It is also the right
    # answer physically: a join between two passages that are both near
    # silence produces a step too small to hear, which is neither an
    # detectable artefact nor a useful one.
    global_rms = audio.rms
    quiet_floor = 0.1 * global_rms

    times: list[float] = []
    last = -np.inf
    for b in range(n_blocks):
        lo = b * block
        hi = len(curvature) if b == n_blocks - 1 else lo + block
        chunk = curvature[lo:hi]
        if not len(chunk):
            continue

        local_median = float(np.median(chunk))
        local_rms = max(float(np.sqrt(np.mean(audio.samples[lo : hi + 2] ** 2))), quiet_floor)
        threshold = max(sigma * local_median, min_jump_ratio * local_rms)

        for idx in np.flatnonzero(chunk > threshold):
            t = (lo + idx + 1) / audio.sample_rate  # +1: n=2 diff centres on idx+1
            if t - last > 0.02:  # collapse bursts into one reported join
                times.append(float(t))
                last = t
    return times


def detect_digital_silence(audio: Audio, *, min_ms: float = 10.0) -> list[tuple[float, float]]:
    """Find runs of exact zeros.

    A microphone never records digital silence. Even a soundproofed room in
    front of a muted preamp produces dither and thermal noise at the least
    significant bit. A run of exact zeros therefore did not come from a
    recording -- it was written by an editor, which is what
    `SpliceConfig(gap_fill="silence")` models.

    Strong evidence and nearly free to compute, but trivially avoided by an
    attacker who pads with room tone instead. It catches carelessness, not
    competence.

    Args:
        audio: The recording to check.
        min_ms: Shortest run to report. Isolated zero samples occur naturally
            at waveform crossings and mean nothing.

    Returns:
        (start, end) times in seconds of each run.
    """
    if not len(audio.samples):
        return []
    min_len = max(2, int(min_ms / 1000 * audio.sample_rate))
    runs = _runs(audio.samples == 0.0, min_len)
    return [(a / audio.sample_rate, b / audio.sample_rate) for a, b in runs]


def detect_splice(
    audio: Audio,
    *,
    hop_ms: int = 10,
    win_ms: int = 25,
    min_pause_ms: int = 60,
    context_ms: int = 150,
    level_step_db_threshold: float = 4.0,
    spectral_distance_threshold: float = 5.0,
    check_clicks: bool = True,
) -> SpliceReport:
    """Look for evidence that a recording was assembled from several takes.

    Finds pauses, then asks whether the background is the same on both sides
    of each one. In a genuine recording it is; across a splice it steps.

    Args:
        audio: The recording to check.
        hop_ms, win_ms: Analysis framing.
        min_pause_ms: Shortest pause treated as a candidate join. Below about
            60 ms a gap is more likely a stop consonant than an edit.
        context_ms: How much background either side to compare.
        level_step_db_threshold: Noise-floor step that counts as evidence.
        spectral_distance_threshold: Per-band background difference, in dB,
            that counts as evidence.
        check_clicks: Also run hard-cut detection.

    Returns:
        A SpliceReport. Thresholds are engineering defaults; fit them on
        genuine recordings from the study and report the operating point.

    Raises:
        AudioError: On an empty recording.
    """
    if not len(audio.samples):
        raise AudioError("Cannot check an empty recording for splicing.")

    report = SpliceReport(is_spliced=False, score=0.0)

    # Checked first: exact zeros mean the file was edited, and they also
    # invalidate the curvature test, whose scale estimates assume a noise
    # floor. Reporting the silence is both the stronger claim and the honest
    # one -- running the click detector over inserted silence would produce
    # detections at every boundary and attribute them to the wrong cause.
    silence_runs = detect_digital_silence(audio)
    report.digital_silence_runs = silence_runs
    if silence_runs:
        report.score = max(report.score, 0.9)
        report.reasons.append(
            f"{len(silence_runs)} run(s) of exact digital silence "
            f"(longest {max(b - a for a, b in silence_runs) * 1000:.0f} ms). "
            "No microphone records exact zeros; this was inserted by an editor."
        )

    click_times: list[float] = detect_clicks(audio) if check_clicks and not silence_runs else []
    report.n_clicks = len(click_times)

    hop = max(1, int(hop_ms / 1000 * audio.sample_rate))
    win = max(hop, int(win_ms / 1000 * audio.sample_rate))
    frames = _frames(audio.samples, win, hop)
    log_energy = 10.0 * np.log10(np.mean(frames**2, axis=1) + _EPS)
    spectra = _band_spectra(frames, audio.sample_rate)

    # Silence = the quiet tail of this recording's own energy distribution,
    # not an absolute dB value, so it holds for quiet and loud takes alike.
    floor = float(np.percentile(log_energy, 10))
    speech_peak = float(np.percentile(log_energy, 90))
    silence_cut = floor + 0.25 * max(speech_peak - floor, 1.0)
    is_silent = log_energy <= silence_cut

    min_pause_frames = max(1, int(min_pause_ms / hop_ms))
    context_frames = max(1, int(context_ms / hop_ms))

    # A pause needs real speech on both sides to be a join candidate. The
    # leading and trailing silence of a recording have nothing before or
    # after them, and estimating a background from the two or three frames
    # that happen to be there produces a confident comparison of noise.
    min_context = max(2, context_frames // 2)

    for start, end in _runs(is_silent, min_pause_frames):
        if start < min_context or end > len(log_energy) - min_context:
            continue

        # Compare the background on either SIDE of the pause, not the two
        # halves of its interior. The cut does not have to land in the middle
        # of the pause -- an attacker who pads with room tone from segment A
        # puts the join at the pause's trailing edge, and an interior
        # comparison would find A's tone on both halves and see nothing.
        half = max(1, (end - start) // 2)
        left = _background(
            log_energy, spectra, max(0, start - context_frames), start + half
        )
        right = _background(
            log_energy, spectra, end - half, min(len(log_energy), end + context_frames)
        )
        if left is None or right is None:
            continue  # pause at the very edge; no context to compare

        left_level, left_spectrum = left
        right_level, right_spectrum = right
        level_step = abs(left_level - right_level)
        spectral_dist = float(np.mean(np.abs(left_spectrum - right_spectrum)))

        t_mid = (start + end) / 2.0 * hop / audio.sample_rate
        near_click = any(abs(t - t_mid) < (end - start) * hop / audio.sample_rate for t in click_times)

        if (
            level_step >= level_step_db_threshold
            or spectral_dist >= spectral_distance_threshold
            or near_click
        ):
            report.boundaries.append(
                SpliceBoundary(
                    time_sec=t_mid,
                    level_step_db=level_step,
                    spectral_distance_db=spectral_dist,
                    click=near_click,
                )
            )
            evidence = max(
                level_step / max(level_step_db_threshold, _EPS),
                spectral_dist / max(spectral_distance_threshold, _EPS),
            )
            report.score = max(report.score, min(1.0, evidence / 2.0))

    # A click nowhere near a pause is still a join -- a cut made mid-word.
    accounted = {round(b.time_sec, 2) for b in report.boundaries}
    orphan_clicks = [t for t in click_times if round(t, 2) not in accounted]
    if orphan_clicks:
        report.score = max(report.score, 0.8)
        report.reasons.append(
            f"{len(orphan_clicks)} waveform discontinuit"
            f"{'y' if len(orphan_clicks) == 1 else 'ies'} not at a pause: "
            "a hard cut made mid-utterance."
        )

    if report.boundaries:
        worst = max(report.boundaries, key=lambda b: b.spectral_distance_db)
        report.reasons.append(
            f"{len(report.boundaries)} pause(s) with a background mismatch; worst at "
            f"{worst.time_sec:.2f}s ({worst.level_step_db:.1f} dB level step, "
            f"{worst.spectral_distance_db:.1f} dB spectral distance)."
        )

    report.is_spliced = report.score >= 0.5
    if not report.reasons:
        report.reasons.append("Background is consistent across all pauses; no splice evidence.")
    return report


def _background(
    log_energy: np.ndarray,
    spectra: np.ndarray,
    start: int,
    end: int,
    *,
    quiet_fraction: float = 0.3,
) -> tuple[float, np.ndarray] | None:
    """Estimate the room tone in a region: its quietest frames.

    Takes the lowest-energy `quiet_fraction` of frames rather than averaging
    the region, because the region deliberately spans speech as well as
    pause. Averaging would measure the speech, which differs between any two
    moments of a genuine recording and would make every pause look spliced.

    Returns:
        (mean log energy in dB, mean per-band spectrum), or None if the
        region is empty.
    """
    if end <= start:
        return None
    region = log_energy[start:end]
    k = max(1, int(len(region) * quiet_fraction))
    quietest = np.argsort(region)[:k] + start
    return float(np.mean(log_energy[quietest])), spectra[quietest].mean(axis=0)


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous True runs of at least `min_len`, as [start, end) indices."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask)))
    return runs


__all__ = [
    "SpliceBoundary",
    "SpliceConfig",
    "SpliceReport",
    "detect_clicks",
    "detect_digital_silence",
    "detect_splice",
    "splice_segments",
]
