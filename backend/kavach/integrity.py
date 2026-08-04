"""Signal integrity: is this one continuous capture, or an assembled file?

WHY THIS IS A GATE AND NOT A BRANCH WEIGHT
-------------------------------------------
The three fusion branches all answer the same question -- *is this the
enrolled speaker?* -- from different evidence, so averaging them is coherent.
Signal integrity answers a different question entirely: *is this file a
recording of somebody speaking, once, just now?* A spliced file can carry a
perfect voiceprint, because it is made of the victim's real voice; the
acoustic branch will happily score it 0.95 and the weighted average will carry
it over the line. That is not a tuning problem. Averaging a "who" score with a
"was this edited" score means a good enough voice buys tolerance for an edit,
which is exactly backwards -- the better the voice match on a file we can
prove was assembled, the *more* alarming it is.

So integrity is disqualifying, like liveness, and for the same reason. See
`kavach.fusion` for the parallel argument about the CSBG veto.

WHAT THIS COSTS, STATED UP FRONT
---------------------------------
A gate that fires on a genuine recording locks out a real user, and no other
branch can rescue them. That makes the floor a false-reject budget, not a
detection target, and it is why `INTEGRITY_FLOOR` is set from a measured
distribution rather than chosen because it sounded strict. `calibrate_floor`
is the function that sets it and it optimises detection *subject to* an FRR
ceiling, never the other way round.

WHAT IT CATCHES AND WHAT IT DOES NOT
-------------------------------------
Catches, per `attacks.splice`: exact digital silence (no microphone produces
it), waveform discontinuities, and background-noise steps across pauses.
Catches, per `attacks.replay`: byte-identical resubmission, and the same
performance submitted twice under a different encoding.

Does NOT catch: a re-recording of a loudspeaker in the same room as enrolment
(the spectral cues for that are device-dependent and off by default), or a
splice made from segments recorded seconds apart in one sitting, where there
is no background step to find because the background never changed. The second
is the honest limit of this approach and belongs in the paper's limitations
section, not in a footnote -- an attacker who records all their raw material
in one session defeats the background test by construction.

Neither detector is a neural anti-spoofing model. This is not an
ASVspoof-style countermeasure and must not be described as one; it is a set of
edit-artefact tests, which is a strictly narrower and much more checkable
claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .attacks.replay import ReplayDetector, ReplayReport
from .attacks.splice import SpliceReport, detect_splice
from .audio import Audio
from .fusion import Branch, BranchScore

#: Floor on the integrity score, below which a probe is rejected outright.
#:
#: MEASURED, NOT CHOSEN -- and the difference cost 5% FRR
#: ------------------------------------------------------
#: This constant was first set to 0.55 by reasoning about where the detectors'
#: outputs "ought to" fall. Running `calibrate_floor` over 80 clean and 80
#: spliced synthetic recordings showed what that would have done:
#:
#:     floor 0.25    FRR 0.00%    detection 82.5%
#:     floor 0.55    FRR 5.00%    detection 82.5%   <- the reasoned guess
#:
#: Identical detection, and one genuine speaker in twenty locked out of their
#: account by a gate no other branch can overturn. The evidence is bimodal --
#: the detectors emit ~1.0 or a specific low value, almost nothing between --
#: so raising the floor across the empty middle buys nothing at all and only
#: reaches further into the clean tail. This is the second time in this project
#: that a threshold reasoned about in one module was wrong about what another
#: module actually emits; see `fusion.CSBG_VETO_FLOOR` for the first.
#:
#: WHAT 0.25 CATCHES, BY ATTACKER
#: -------------------------------
#: All at 0.0% false-reject on the clean set:
#:
#:     naive splice, cross-session    82.5%    d' 2.60
#:     naive splice, same session     63.7%    d' 1.66   <- report this one
#:     careful splice, cross-session  88.8%    d' 4.97
#:
#: The middle row is the operating point to quote: an attacker who records all
#: their raw material in one sitting leaves no background step, so only the
#: click test can fire, and a third of their attempts get through.
#:
#: WHY NOT 0.20, WHICH A GRID SEARCH PREFERS
#: -----------------------------------------
#: `calibrate_floor` sometimes returns 0.200 on this data, with detection
#: indistinguishable from 0.25 -- and 0.200 is a trap. The detectors emit fixed
#: evidence weights for their categorical findings, 0.9 for inserted digital
#: silence and 0.8 for a hard cut away from any pause, so those probes score
#: exactly 0.10 and 0.20. The comparison is `score < floor`, so a floor of
#: exactly 0.20 catches none of the orphan-click cases while looking optimal to
#: a grid search. The floor has to sit strictly inside the gap between the
#: highest categorical level it must catch (0.20) and the lowest clean score
#: (0.274). `tests/test_integrity.py::test_the_floor_does_not_sit_on_a_
#: discrete_evidence_level` asserts that gap still exists and still holds it.
#:
#: The careful attacker being *easier* to catch is not an error. `careful()`
#: inserts a room-tone pause to make the join sound natural -- and a pause at
#: the join is precisely what the background test examines. The naive hard cut
#: has no pause, so nothing but the click detector ever looks at it. Making a
#: splice sound better makes it easier to detect here, which means the
#: attacker's best move against this particular defence is the crude one.
#:
#: **These are synthetic recordings.** Real room tone from one sitting will
#: narrow the gap further. Re-run the calibration on genuine recordings from
#: the study hardware before reporting any number that depends on this gate.
INTEGRITY_FLOOR = 0.25


@dataclass(slots=True)
class IntegrityReport:
    """Edit- and replay-artefact evidence for one probe.

    Attributes:
        score: 1.0 = no artefact found, 0.0 = conclusive tampering. This is
            `1 - max(evidence)` over the detectors, so it inherits their
            calibration -- which is to say it has none. It is comparable
            against `INTEGRITY_FLOOR` and against itself, and nowhere else.
        splice: The splice detector's report, or None if it did not run.
        replay: The replay detector's report, or None if it did not run.
        reasons: Human-readable, aggregated from both detectors.
    """

    score: float
    splice: SpliceReport | None = None
    replay: ReplayReport | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.score >= INTEGRITY_FLOOR

    def to_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 4),
            "clean": self.clean,
            "splice": self.splice.to_dict() if self.splice else None,
            "replay": self.replay.to_dict() if self.replay else None,
            "reasons": self.reasons,
        }


class IntegrityChecker:
    """Runs the edit-artefact and duplicate tests over a probe.

    Holds the replay detector's memory of previously seen recordings, which is
    the only stateful part. That memory is what makes duplicate detection work
    at all, and it is why the pipeline persists envelopes across restarts: a
    replay detector that forgets everything when the server restarts catches
    replays only within a single process lifetime, which is not a security
    property, it is an accident of uptime.

    Args:
        check_splice: Run the edit-artefact tests.
        check_replay: Run the duplicate tests.
        floor: Score below which `build_integrity_branch` rejects.
    """

    def __init__(
        self,
        *,
        check_splice: bool = True,
        check_replay: bool = True,
        floor: float = INTEGRITY_FLOOR,
        replay_detector: ReplayDetector | None = None,
    ) -> None:
        self.check_splice = check_splice
        self.check_replay = check_replay
        self.floor = floor
        self.replay = replay_detector or ReplayDetector()

    def remember(self, audio: Audio, *, label: str = "") -> None:
        """Add a recording to the duplicate memory."""
        self.replay.remember(audio, label=label)

    def check(self, audio: Audio) -> IntegrityReport:
        """Score a probe for edit and duplicate artefacts.

        Never raises on a short or odd recording: an integrity check that
        crashes takes the whole verification down, and a probe too short to
        analyse is a *quality* problem, which `check_quality` already reports.
        A detector that could not run contributes no evidence rather than
        contributing an accusation.
        """
        evidence = 0.0
        reasons: list[str] = []
        splice_report: SpliceReport | None = None
        replay_report: ReplayReport | None = None

        if self.check_splice:
            try:
                splice_report = detect_splice(audio)
            except Exception:  # noqa: BLE001 -- see docstring
                splice_report = None
            if splice_report is not None and splice_report.score > 0.0:
                evidence = max(evidence, splice_report.score)
                reasons.extend(splice_report.reasons)

        if self.check_replay:
            try:
                replay_report = self.replay.check(audio)
            except Exception:  # noqa: BLE001
                replay_report = None
            if replay_report is not None and replay_report.score > 0.0:
                evidence = max(evidence, replay_report.score)
                reasons.extend(replay_report.reasons)

        if not reasons:
            reasons.append("No edit or duplicate artefacts found.")

        return IntegrityReport(
            score=1.0 - evidence,
            splice=splice_report,
            replay=replay_report,
            reasons=reasons,
        )


def build_integrity_branch(
    report: IntegrityReport | None, *, floor: float = INTEGRITY_FLOOR
) -> BranchScore:
    """Turn an integrity report into the gate branch.

    A None report means the check did not run -- the branch reports
    `available=False` and `fuse` skips the gate entirely. That is deliberate:
    an integrity test that could not be performed is missing evidence, and
    missing evidence must not read as either a pass or a failure. The
    alternative, defaulting to "clean", would silently disable the gate the
    moment the detector errored, which is the worst of the three options
    because nothing in the output would say so.
    """
    if report is None:
        return BranchScore(
            branch=Branch.INTEGRITY,
            score=0.0,
            threshold=floor,
            weight=0.0,
            available=False,
            detail="Signal-integrity checks did not run.",
        )
    detail = report.reasons[0] if report.reasons else ""
    return BranchScore(
        branch=Branch.INTEGRITY,
        score=report.score,
        threshold=floor,
        weight=0.0,
        available=True,
        detail=detail,
    )


@dataclass(slots=True)
class FloorCalibration:
    """What a candidate integrity floor costs and buys.

    Attributes:
        floor: The chosen score floor.
        false_reject_rate: Share of *genuine* recordings it would lock out.
        detection_rate: Share of *tampered* recordings it would catch.
        n_genuine, n_tampered: Sample sizes behind those two rates.
        feasible: False when no floor meets the FRR budget, in which case
            `floor` is the most permissive one tried and the gate should be
            left off rather than shipped at a rate nobody chose.
    """

    floor: float
    false_reject_rate: float
    detection_rate: float
    n_genuine: int
    n_tampered: int
    feasible: bool = True

    def summary(self) -> str:
        if not self.feasible:
            return (
                f"No integrity floor meets the false-reject budget "
                f"({self.n_genuine} genuine, {self.n_tampered} tampered). "
                "The gate should stay off."
            )
        return (
            f"floor {self.floor:.3f}: catches {self.detection_rate:6.1%} of tampered "
            f"probes at a {self.false_reject_rate:.1%} false-reject cost "
            f"({self.n_genuine} genuine, {self.n_tampered} tampered)"
        )


def calibrate_floor(
    genuine: list[IntegrityReport],
    tampered: list[IntegrityReport],
    *,
    max_false_reject: float = 0.01,
    grid: int = 101,
) -> FloorCalibration:
    """Pick the strictest floor that stays inside the false-reject budget.

    The asymmetry is the whole design. A missed splice costs one fraudulent
    acceptance, which the other three branches still get a chance to catch. A
    false integrity rejection is final -- the gate runs before fusion and
    nothing downstream can overturn it -- so a locked-out genuine user has no
    recourse but to try again and be rejected again, because the artefact is a
    property of their hardware, not of that attempt. Detection is therefore
    maximised *subject to* the FRR ceiling and never traded against it.

    Args:
        genuine: Reports from recordings known to be single honest captures.
        tampered: Reports from recordings known to be edited or resubmitted.
        max_false_reject: FRR ceiling, as a fraction.
        grid: Number of candidate floors swept over [0, 1].

    Returns:
        A FloorCalibration. Check `feasible` before using `floor`: when even
        the most permissive floor exceeds the budget, the detectors are firing
        on genuine audio and the answer is to fix them, not to ship the gate.
    """
    if not genuine:
        raise ValueError(
            "Cannot calibrate an integrity floor without genuine recordings. "
            "The false-reject rate is the binding constraint and there is "
            "nothing to measure it on."
        )

    gen = [r.score for r in genuine]
    tam = [r.score for r in tampered]

    best: FloorCalibration | None = None
    for i in range(grid):
        floor = i / (grid - 1)
        frr = sum(1 for s in gen if s < floor) / len(gen)
        if frr > max_false_reject:
            continue
        det = (sum(1 for s in tam if s < floor) / len(tam)) if tam else 0.0
        candidate = FloorCalibration(
            floor=floor,
            false_reject_rate=frr,
            detection_rate=det,
            n_genuine=len(gen),
            n_tampered=len(tam),
        )
        # Strictly greater, so ties keep the *lowest* floor that achieves the
        # detection rate -- the cheapest way to buy it, and the one with the
        # most headroom when real audio turns out noisier than this sample.
        if best is None or candidate.detection_rate > best.detection_rate:
            best = candidate

    if best is None:
        return FloorCalibration(
            floor=0.0,
            false_reject_rate=sum(1 for s in gen if s < 0.0) / len(gen),
            detection_rate=0.0,
            n_genuine=len(gen),
            n_tampered=len(tam),
            feasible=False,
        )
    return best


def separation(genuine: list[IntegrityReport], tampered: list[IntegrityReport]) -> float:
    """Standardised distance between the two score distributions.

    Reported alongside the floor because the floor alone hides the thing that
    actually matters: whether there is a gap to put it in. A floor with d' of
    0.3 is a coin flip dressed as a threshold.
    """
    if not genuine or not tampered:
        return 0.0
    gen = [r.score for r in genuine]
    tam = [r.score for r in tampered]
    mu_g = sum(gen) / len(gen)
    mu_t = sum(tam) / len(tam)
    var_g = sum((s - mu_g) ** 2 for s in gen) / max(1, len(gen) - 1)
    var_t = sum((s - mu_t) ** 2 for s in tam) / max(1, len(tam) - 1)
    pooled = math.sqrt(max((var_g + var_t) / 2.0, 1e-12))
    return (mu_g - mu_t) / pooled


__all__ = [
    "INTEGRITY_FLOOR",
    "IntegrityReport",
    "IntegrityChecker",
    "build_integrity_branch",
    "FloorCalibration",
    "calibrate_floor",
    "separation",
]
