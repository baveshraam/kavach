"""Decide whether the splice tests may be switched on.

    python -m kavach.calibrate_integrity \\
        --manifest data/corpus_v2/manifest.json --manifest data/corpus_v3/manifest.json

WHY THIS EXISTS
---------------
The splice detector was calibrated on synthetic audio and shipped with a floor
reasoned from it. On the first real corpus it rejected 167 of 168 genuine
clips. `integrity.calibrate_floor` is the function meant to fix a floor, but it
answers a narrower question than the one that matters: it picks the best floor
*given* that the scores separate the classes. It will happily report a floor of
0.0 as "feasible", because rejecting nothing meets any false-reject budget.
This command asks the prior question -- do the scores separate genuine speech
from edited speech at all -- and says so in words.

WHAT IT COMPARES, AND WHY THE COMPARISON IS DURATION-MATCHED
------------------------------------------------------------
Negatives are random windows cut from genuine clips. Positives are splices
built from windows of *different* clips of the same speaker, naive and
careful. Both classes have the same total duration, because the cues are
maxima over pauses and a 30 s recording has far more pauses than a 4.5 s one;
comparing a long genuine clip with a short splice measures length.

Windows keep `EDGE_MARGIN_SEC` clear of the clip's own ends. Real clips carry
decoder padding there -- runs of exact zeros -- and a window that reaches into
it hands the detector a "digital silence" finding that no editor wrote.

THE LIMIT OF THE POSITIVE CLASS
-------------------------------
Every speaker in the pilot corpus has one session, so every splice here is
same-sitting: one room, one microphone, one codec. That is the hardest case
for a background test and the one the module docstring of `integrity` already
concedes. A detector that fails here may still catch a cross-session splice;
this corpus cannot say, and the report does not claim it.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .attacks.splice import SpliceConfig, splice_segments
from .audio import Audio, AudioError, load_audio
from .integrity import (
    INTEGRITY_FLOOR,
    FloorCalibration,
    IntegrityChecker,
    IntegrityReport,
    calibrate_floor,
)

#: Share of tampered probes `calibrate_floor` must catch, at the 1% false-reject
#: budget, for the splice tests to count as useful. A DECISION, not a fitted
#: value: below one in two the gate is a coin flip an attacker can retry, and
#: it still costs genuine users their false rejects. `calibrate_floor` reports a
#: floor of 0.0 as feasible -- it meets the budget by rejecting nothing -- so
#: feasibility alone cannot be the test.
MIN_USEFUL_DETECTION = 0.5

#: False-reject ceiling for the gate, as a fraction. The gate runs before
#: fusion and nothing downstream can overturn it, so this is a budget.
MAX_FALSE_REJECT = 0.01

#: Keep windows this far from a clip's own start and end. See module docstring.
EDGE_MARGIN_SEC = 0.3

#: The cues, in report order. Every one reads "higher = more suspicious".
CUES = ("click_rate", "max_level", "max_spec", "n_bound", "interior_zero")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def auc(genuine: Sequence[float], spliced: Sequence[float]) -> float:
    """Probability a spliced probe scores higher than a genuine one.

    0.5 is no information, 1.0 is perfect separation. Ties count as half, so a
    cue that is constant -- as `interior_zero` is on any genuine window -- reads
    0.5 and not an accident of how the sort broke the tie.
    """
    g = np.asarray(genuine, dtype=float)
    s = np.asarray(spliced, dtype=float)
    if not len(g) or not len(s):
        return 0.5
    greater = (s[:, None] > g[None, :]).sum()
    equal = (s[:, None] == g[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(g) * len(s)))


def tpr_at_fpr(
    genuine: Sequence[float], spliced: Sequence[float], fpr: float
) -> tuple[float, float]:
    """Share of spliced probes above the threshold that flags `fpr` of genuine.

    Returns (true-positive rate, threshold). A probe is flagged when its cue is
    strictly above the threshold, matching the detectors' own `score >` tests.
    """
    g = np.asarray(genuine, dtype=float)
    s = np.asarray(spliced, dtype=float)
    if not len(g) or not len(s):
        return 0.0, float("nan")
    threshold = float(np.quantile(g, 1.0 - fpr))
    return float((s > threshold).mean()), threshold


def is_useful(cal: FloorCalibration) -> bool:
    """Whether a calibrated floor buys real detection, not just a low FRR."""
    return cal.feasible and cal.detection_rate >= MIN_USEFUL_DETECTION


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


def window(clip: Audio, seconds: float, rng: random.Random) -> Audio:
    """A random `seconds`-long window, clear of the clip's padded edges."""
    dur = len(clip.samples) / clip.sample_rate
    lo = EDGE_MARGIN_SEC
    hi = dur - seconds - EDGE_MARGIN_SEC
    if hi < lo:  # clip too short to keep the margin: centre what there is
        t0 = max(0.0, (dur - seconds) / 2.0)
    else:
        t0 = rng.uniform(lo, hi)
    return clip.slice_seconds(t0, min(t0 + seconds, dur))


def _cues(report: IntegrityReport, seconds: float) -> dict[str, float]:
    """The detector's own evidence, read off its report in one pass."""
    sp = report.splice
    if sp is None:
        return {name: 0.0 for name in CUES}
    interior = sum(
        1 for a, b in sp.digital_silence_runs if a > 0.05 and b < seconds - 0.05
    )
    return {
        "click_rate": sp.n_clicks / seconds,
        "max_level": max((b.level_step_db for b in sp.boundaries), default=0.0),
        "max_spec": max((b.spectral_distance_db for b in sp.boundaries), default=0.0),
        "n_bound": float(len(sp.boundaries)),
        "interior_zero": float(interior),
    }


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CueResult:
    name: str
    genuine_median: float
    genuine_p95: float
    auc: dict[str, float]
    tpr_at_1pct: dict[str, float]
    tpr_at_5pct: dict[str, float]


@dataclass(slots=True)
class Calibration:
    seconds: float
    n_genuine: int
    n_naive: int
    n_careful: int
    full_clip_rejected: int
    full_clip_total: int
    cues: list[CueResult] = field(default_factory=list)
    floor: FloorCalibration | None = None
    separates: bool = False


def measure(
    clips_by_speaker: dict[str, list[Audio]],
    *,
    seconds: float = 4.5,
    windows: int = 12,
    splices: int = 6,
    seed: int = 7,
) -> Calibration:
    """Run the detector over genuine windows and same-speaker splices.

    Args:
        clips_by_speaker: Genuine recordings, grouped by speaker. A speaker
            with fewer than three clips contributes genuine windows but no
            splices, because a splice needs three distinct sources.
        seconds: Probe length, the same for both classes.
        windows: Genuine windows per speaker.
        splices: Splices per speaker, per attacker type.
    """
    rng = random.Random(seed)
    checker = IntegrityChecker(check_replay=False, check_splice=True)

    all_clips = [c for clips in clips_by_speaker.values() for c in clips]
    full = [checker.check(c) for c in all_clips]

    genuine: list[IntegrityReport] = []
    naive: list[IntegrityReport] = []
    careful: list[IntegrityReport] = []
    for clips in clips_by_speaker.values():
        if not clips:
            continue
        for _ in range(windows):
            genuine.append(checker.check(window(rng.choice(clips), seconds, rng)))
        if len(clips) < 3:
            continue
        for config, bucket in (
            (SpliceConfig.naive(), naive),
            (SpliceConfig.careful(), careful),
        ):
            for _ in range(splices):
                segments = [window(c, seconds / 3.0, rng) for c in rng.sample(clips, 3)]
                bucket.append(checker.check(splice_segments(segments, config)))

    result = Calibration(
        seconds=seconds,
        n_genuine=len(genuine),
        n_naive=len(naive),
        n_careful=len(careful),
        full_clip_rejected=sum(1 for r in full if not r.clean),
        full_clip_total=len(full),
    )

    g_cues = [_cues(r, seconds) for r in genuine]
    for name in CUES:
        g = [c[name] for c in g_cues]
        row = CueResult(
            name=name,
            genuine_median=float(np.median(g)) if g else 0.0,
            genuine_p95=float(np.quantile(g, 0.95)) if g else 0.0,
            auc={},
            tpr_at_1pct={},
            tpr_at_5pct={},
        )
        for label, bucket in (("naive", naive), ("careful", careful)):
            s = [_cues(r, seconds)[name] for r in bucket]
            row.auc[label] = auc(g, s)
            row.tpr_at_1pct[label] = tpr_at_fpr(g, s, 0.01)[0]
            row.tpr_at_5pct[label] = tpr_at_fpr(g, s, 0.05)[0]
        result.cues.append(row)

    if genuine and (naive or careful):
        result.floor = calibrate_floor(
            genuine, naive + careful, max_false_reject=MAX_FALSE_REJECT
        )
        result.separates = is_useful(result.floor)
    return result


def render(c: Calibration) -> str:
    """The report, ending in an instruction rather than a table."""
    rejected = c.full_clip_rejected / c.full_clip_total if c.full_clip_total else 0.0
    lines = [
        "Splice-test calibration on genuine recordings",
        "",
        f"whole genuine clips rejected at the live floor ({INTEGRITY_FLOOR}): "
        f"{c.full_clip_rejected} / {c.full_clip_total} ({rejected:.1%})",
        f"probes of {c.seconds:.1f} s, duration-matched: {c.n_genuine} genuine windows, "
        f"{c.n_naive} naive splices, {c.n_careful} careful splices",
        "splices are cut from different clips of one speaker -- the same sitting, "
        "the only kind a one-session corpus can build",
        "",
        f"{'cue':14s} {'genuine med':>11s} {'p95':>7s} | "
        f"{'naive AUC':>9s} {'TPR@1%':>7s} {'TPR@5%':>7s} | "
        f"{'careful AUC':>11s} {'TPR@1%':>7s} {'TPR@5%':>7s}",
    ]
    for cue in c.cues:
        lines.append(
            f"{cue.name:14s} {cue.genuine_median:11.2f} {cue.genuine_p95:7.2f} | "
            f"{cue.auc.get('naive', 0.5):9.2f} {cue.tpr_at_1pct.get('naive', 0.0):7.2f} "
            f"{cue.tpr_at_5pct.get('naive', 0.0):7.2f} | "
            f"{cue.auc.get('careful', 0.5):11.2f} {cue.tpr_at_1pct.get('careful', 0.0):7.2f} "
            f"{cue.tpr_at_5pct.get('careful', 0.0):7.2f}"
        )

    lines.append("")
    if c.floor is not None:
        lines.append(f"calibrate_floor (false-reject <= {MAX_FALSE_REJECT:.0%}): {c.floor.summary()}")
    lines.append("")
    if c.separates and c.floor is not None:
        lines.append(
            f"VERDICT: the edit tests catch {c.floor.detection_rate:.0%} of splices at a "
            f"{c.floor.false_reject_rate:.1%} false-reject cost (bar: "
            f"{MIN_USEFUL_DETECTION:.0%}). They may be enabled with "
            f"KAVACH_INTEGRITY_CHECK_SPLICE=true; set INTEGRITY_FLOOR to "
            f"{c.floor.floor:.3f} only after re-checking the margin around it."
        )
    else:
        got = f"{c.floor.detection_rate:.0%}" if c.floor is not None else "n/a"
        lines.append(
            f"VERDICT: do not enable. Detection at the {MAX_FALSE_REJECT:.0%} "
            f"false-reject budget is {got}, under the {MIN_USEFUL_DETECTION:.0%} bar. "
            "Leave `integrity_check_splice` off; the duplicate (replay) test is "
            "unaffected. Lowering the floor cannot help -- the cues do not separate "
            "the classes."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_clips(manifests: Sequence[Path]) -> dict[str, list[Audio]]:
    """Every readable recording in the manifests, grouped by speaker id.

    Excluded utterances are included: they were excluded from the *graph*
    because Whisper translated them, and the audio is still a genuine capture.
    """
    from .corpus import load_manifest

    out: dict[str, list[Audio]] = {}
    for path in manifests:
        corpus = load_manifest(path)
        root = corpus.root or Path()
        for u in corpus.utterances:
            if not u.audio_path:
                continue
            try:
                out.setdefault(u.speaker_id, []).append(load_audio(root / u.audio_path))
            except AudioError:
                continue
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.calibrate_integrity",
        description="Do the splice tests separate edited speech from genuine speech?",
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="Corpus manifest; repeat for several corpora.")
    parser.add_argument("--seconds", type=float, default=4.5, help="Probe length.")
    parser.add_argument("--windows", type=int, default=12, help="Genuine windows per speaker.")
    parser.add_argument("--splices", type=int, default=6, help="Splices per speaker, per type.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, help="Write the report to a file as well.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clips = load_clips(args.manifest)
    if not clips:
        print(
            "error: no readable audio in those manifests. The corpus audio is not "
            "part of the repository (see README, 'What is not in this repository').",
            file=sys.stderr,
        )
        return 2

    out = render(
        measure(
            clips,
            seconds=args.seconds,
            windows=args.windows,
            splices=args.splices,
            seed=args.seed,
        )
    )
    print(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
