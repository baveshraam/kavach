"""Tests for the signal-integrity gate.

THE POINT OF THIS FILE
----------------------
`INTEGRITY_FLOOR` is a number in one module compared against scores produced
by two others. That is exactly the shape of the bug this project has now hit
twice: `fusion.CSBG_VETO_FLOOR` was set to a value the CSBG scorer could never
emit, and it went unnoticed because the fusion tests used hand-written branch
scores and the scorer tests never looked at fusion. Both modules were correct;
the constant passed between them was not, and nothing tested the seam.

So `TestFloorCalibration` does not assert that the floor equals 0.25. It
generates audio, runs the *real* detectors, and asserts that the floor still
separates the classes the way the constant's docstring claims. If someone
tightens `detect_clicks` or changes the evidence weights, this fails with the
measured numbers in the message rather than silently locking out genuine
speakers.

WHY THE GENERATOR FADES ITS PAUSES
-----------------------------------
`speechlike()` ramps in and out of every pause over 8 ms. An earlier version
zeroed the envelope instantly, which is a waveform discontinuity -- the exact
artefact `detect_clicks` exists to find. Every "clean" recording then tested
positive and the two distributions were indistinguishable (d' = 0.48). The
control has to be free of the artefact it is a control for. No vocal tract
stops in one sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from kavach.attacks.replay import ReplayDetector
from kavach.attacks.splice import SpliceConfig, splice_segments
from kavach.audio import Audio
from kavach.fusion import Branch, BranchScore, Decision, FusionPolicy, fuse
from kavach.integrity import (
    INTEGRITY_FLOOR,
    IntegrityChecker,
    IntegrityReport,
    build_integrity_branch,
    calibrate_floor,
    separation,
)

SR = 16_000


def speechlike(
    *,
    seconds: float = 4.0,
    noise_db: float = -45.0,
    f0: float = 140.0,
    seed: int = 0,
    pauses: int = 3,
) -> Audio:
    """A recording with harmonics, syllables, faded pauses and a noise floor.

    Not speech, and not claimed to be. It has the four properties the
    detectors actually read -- a harmonic stack, a syllable-rate envelope,
    pauses with smooth edges, and a stationary background -- which is what
    makes it a usable control. Anything more would be modelling a vocal tract
    to test a click detector.
    """
    r = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR

    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 9):
        sig += (0.6**k) * np.sin(2 * np.pi * f0 * k * t + r.uniform(0, 2 * np.pi))

    env = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    if pauses and seconds > 1.0:
        fade = int(0.008 * SR)
        ramp = np.hanning(2 * fade)[:fade]
        for start in r.uniform(0.3, max(0.4, seconds - 0.5), size=pauses):
            a, b = int(start * SR), min(int((start + 0.20) * SR), n)
            if b - a <= 2 * fade:
                continue
            env[a : a + fade] *= ramp[::-1]
            env[a + fade : b - fade] = 0.0
            env[b - fade : b] *= ramp

    sig *= env
    sig /= max(float(np.max(np.abs(sig))), 1e-9)
    sig = 0.4 * sig + (10 ** (noise_db / 20)) * r.standard_normal(n)
    return Audio(sig.astype(np.float32), SR, f"u{seed}")


def spliced(
    *, seed: int, config: SpliceConfig, same_session: bool = False, rng: np.random.Generator
) -> Audio:
    """Three segments joined. `same_session` shares one noise floor."""
    noise = rng.uniform(-55, -35)
    f0 = rng.uniform(90, 220)
    segments = [
        speechlike(
            seconds=1.6,
            seed=seed + k,
            noise_db=noise if same_session else rng.uniform(-55, -35),
            f0=f0 if same_session else rng.uniform(90, 220),
            pauses=1,
        )
        for k in range(3)
    ]
    return splice_segments(segments, config)


@pytest.fixture(scope="module")
def measured() -> dict[str, list[IntegrityReport]]:
    """Real detector output over clean and tampered audio.

    Module-scoped: 320 recordings analysed once, not once per test. The
    detectors are deterministic given the seeds, so sharing is safe.
    """
    checker = IntegrityChecker(check_replay=False)
    rng = np.random.default_rng(7)

    clean = [
        checker.check(
            speechlike(seed=i, noise_db=rng.uniform(-55, -35), f0=rng.uniform(90, 220))
        )
        for i in range(80)
    ]
    naive = [
        checker.check(spliced(seed=1000 + i * 3, config=SpliceConfig.naive(), rng=rng))
        for i in range(80)
    ]
    one_session = [
        checker.check(
            spliced(
                seed=5000 + i * 3,
                config=SpliceConfig.naive(),
                same_session=True,
                rng=rng,
            )
        )
        for i in range(80)
    ]
    careful = [
        checker.check(spliced(seed=9000 + i * 3, config=SpliceConfig.careful(), rng=rng))
        for i in range(80)
    ]
    return {
        "clean": clean,
        "naive": naive,
        "one_session": one_session,
        "careful": careful,
    }


def rates(reports: list[IntegrityReport], floor: float) -> float:
    return sum(1 for r in reports if r.score < floor) / len(reports)


class TestFloorCalibration:
    """The seam between the detectors and the constant that thresholds them."""

    def test_the_shipped_floor_rejects_no_genuine_recording(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """The binding constraint, asserted first because it is the one that
        harms a real person. A missed splice costs one fraudulent acceptance
        that three other branches still get to catch; a false integrity
        rejection is final and repeats every time that speaker tries."""
        frr = rates(measured["clean"], INTEGRITY_FLOOR)
        assert frr == 0.0, (
            f"The integrity gate would lock out {frr:.1%} of genuine recordings. "
            f"Clean scores: min={min(r.score for r in measured['clean']):.3f}, "
            f"floor={INTEGRITY_FLOOR}."
        )

    def test_the_shipped_floor_is_as_good_as_calibration_can_get(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """Re-derive the constant's *performance* rather than its exact value.

        Asserting `cal.floor == INTEGRITY_FLOOR` was the first version of this
        test and it was wrong. `calibrate_floor` breaks ties toward the lowest
        floor, and the tampered scores are piled on a few discrete evidence
        levels, so the argmax lands wherever the lowest tampered score happens
        to sit -- 0.200 on one sample, 0.250 on another, with identical
        detection. That is sampling noise in a tie-break, not a finding about
        the detector, and a test that fails on it teaches nothing.

        What matters is that the shipped floor is not leaving detection on the
        table at zero false-reject cost.
        """
        cal = calibrate_floor(
            measured["clean"], measured["naive"], max_false_reject=0.0
        )
        assert cal.feasible
        shipped = rates(measured["naive"], INTEGRITY_FLOOR)
        assert shipped >= cal.detection_rate - 1e-9, (
            f"The shipped floor {INTEGRITY_FLOOR} detects {shipped:.1%}, but "
            f"{cal.floor:.3f} would detect {cal.detection_rate:.1%} at the same "
            "zero false-reject cost. Move the constant."
        )

    def test_the_floor_does_not_sit_on_a_discrete_evidence_level(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """The knife-edge that the naive calibration walks straight into.

        The detectors emit fixed evidence weights for their categorical
        findings -- 0.9 for inserted digital silence, 0.8 for a hard cut away
        from any pause -- so those probes score exactly 0.10 and 0.20. A floor
        of exactly 0.20 fails to catch the 0.20 cases, because the comparison
        is `score < floor`. It looks like the optimum to a grid search and is
        one rounding error from catching nothing in that category.

        The floor therefore has to sit strictly inside the gap: above every
        categorical evidence level it is meant to catch, below the lowest clean
        score. This asserts that gap still exists and still contains it.
        """
        clean_min = min(r.score for r in measured["clean"])
        categorical = [0.10, 0.20]  # digital silence, orphan click

        assert INTEGRITY_FLOOR > max(categorical) + 1e-6, (
            f"Floor {INTEGRITY_FLOOR} does not strictly exceed the orphan-click "
            f"score of {max(categorical)}, so hard cuts away from a pause are "
            "not caught."
        )
        assert INTEGRITY_FLOOR < clean_min, (
            f"Floor {INTEGRITY_FLOOR} is at or above the lowest clean score "
            f"{clean_min:.3f}; genuine recordings would be rejected."
        )

    def test_raising_the_floor_buys_nothing_and_costs_speakers(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """The specific mistake this constant used to embody.

        0.55 was the reasoned guess. It has identical detection and a real
        false-reject cost, because the evidence is bimodal and the space
        between the modes is empty. Encoded as a test so nobody re-reasons
        their way back to it.
        """
        detection_low = rates(measured["naive"], INTEGRITY_FLOOR)
        detection_high = rates(measured["naive"], 0.55)
        frr_high = rates(measured["clean"], 0.55)

        assert detection_high <= detection_low + 1e-9, (
            "A higher floor now detects more, so the bimodality argument in "
            "INTEGRITY_FLOOR's docstring no longer holds -- recalibrate."
        )
        assert frr_high > 0.0, (
            "0.55 no longer costs any false rejects. The clean distribution has "
            "moved; re-read the docstring's numbers before trusting them."
        )

    def test_the_classes_are_actually_separated(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """A floor is only meaningful if there is a gap to put it in.

        d' below about 1.0 would mean the gate is a coin flip wearing a
        threshold, and the honest response would be to turn it off rather than
        to tune it.
        """
        d = separation(measured["clean"], measured["naive"])
        assert d > 1.5, f"Integrity d' has fallen to {d:.2f}; the gate is not separating."

    def test_same_session_splices_are_the_hard_case(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """The limitation named in the module docstring, asserted so it stays
        named. An attacker who records everything in one sitting leaves no
        background step, and a meaningful fraction gets through."""
        caught = rates(measured["one_session"], INTEGRITY_FLOOR)
        cross = rates(measured["naive"], INTEGRITY_FLOOR)
        assert caught < cross, (
            "Same-session splices are no longer harder to catch than "
            "cross-session ones. That would be good news, and it would mean the "
            "detector found a cue that does not depend on the background -- "
            "check what changed before relaxing the documented limitation."
        )
        assert 0.3 < caught < 0.9, (
            f"Same-session detection is {caught:.1%}; the docstring quotes ~64%."
        )

    def test_a_careful_attacker_is_not_harder_to_catch(
        self, measured: dict[str, list[IntegrityReport]]
    ) -> None:
        """The counterintuitive finding, pinned.

        `careful()` adds a room-tone pause so the join sounds natural -- and a
        pause at the join is exactly what the background test reads. Polish
        makes this attack *more* detectable, so the attacker's best move
        against this defence is the crude one. If this ever flips, the claim in
        INTEGRITY_FLOOR's docstring is stale.
        """
        assert rates(measured["careful"], INTEGRITY_FLOOR) >= rates(
            measured["naive"], INTEGRITY_FLOOR
        )


class TestCalibrationContract:
    def test_calibration_needs_genuine_recordings(self) -> None:
        with pytest.raises(ValueError, match="genuine"):
            calibrate_floor([], [IntegrityReport(score=0.1)])

    def test_infeasible_when_clean_audio_trips_every_floor(self) -> None:
        """When even floor 0 exceeds the budget the answer is to fix the
        detector, not to ship the gate at a rate nobody chose."""
        cal = calibrate_floor(
            [IntegrityReport(score=-1.0)] * 10,
            [IntegrityReport(score=-2.0)] * 10,
            max_false_reject=0.0,
        )
        assert not cal.feasible
        assert "should stay off" in cal.summary()

    def test_ties_prefer_the_lower_floor(self) -> None:
        """Two floors with equal detection: take the cheaper one, which has
        more headroom when real audio turns out noisier than the sample."""
        clean = [IntegrityReport(score=1.0)] * 20
        tampered = [IntegrityReport(score=0.0)] * 20
        cal = calibrate_floor(clean, tampered, max_false_reject=0.0)
        assert cal.detection_rate == 1.0
        assert cal.floor < 0.5

    def test_separation_is_zero_without_both_classes(self) -> None:
        assert separation([], [IntegrityReport(score=0.0)]) == 0.0
        assert separation([IntegrityReport(score=1.0)], []) == 0.0


class TestBranchConstruction:
    def test_a_check_that_did_not_run_is_unavailable_not_clean(self) -> None:
        """The failure mode that would silently disable the gate.

        Defaulting a missing check to "clean" is the worst of the three
        options, because nothing in the output would say the gate stopped
        working.
        """
        branch = build_integrity_branch(None)
        assert branch.available is False
        assert branch.passed is False

    def test_an_unavailable_gate_does_not_reject(self) -> None:
        result = fuse(
            [
                build_integrity_branch(None),
                BranchScore(Branch.SPEAKER, 0.9, 0.62, 1.0),
            ],
            FusionPolicy(weights={Branch.SPEAKER: 1.0}, veto_thresholds={}),
        )
        assert result.decision is Decision.ACCEPT

    def test_a_tampered_probe_rejects_over_a_perfect_voice(self) -> None:
        """The whole argument for making this a gate.

        A spliced file is made of the victim's real voice, so the acoustic
        branch scores it 0.99 -- and under weighted fusion that would carry it.
        """
        result = fuse(
            [
                build_integrity_branch(IntegrityReport(score=0.0, reasons=["hard cut"])),
                BranchScore(Branch.SPEAKER, 0.99, 0.62, 1.0),
            ],
            FusionPolicy(weights={Branch.SPEAKER: 1.0}, veto_thresholds={}),
        )
        assert result.decision is Decision.REJECT
        assert any("edited or resubmitted" in line for line in result.explanation)

    def test_the_rejection_says_why_a_good_voice_did_not_save_it(self) -> None:
        """A user told only "rejected" after a perfect voice match learns
        nothing. The explanation has to name the actual reason."""
        result = fuse(
            [
                build_integrity_branch(IntegrityReport(score=0.0, reasons=["hard cut"])),
                BranchScore(Branch.SPEAKER, 0.99, 0.62, 1.0),
            ],
            FusionPolicy(weights={Branch.SPEAKER: 1.0}, veto_thresholds={}),
        )
        assert any("voiceprint" in line for line in result.explanation)

    def test_the_gate_can_be_switched_off_for_its_own_ablation(self) -> None:
        result = fuse(
            [
                build_integrity_branch(IntegrityReport(score=0.0)),
                BranchScore(Branch.SPEAKER, 0.99, 0.62, 1.0),
            ],
            FusionPolicy(
                weights={Branch.SPEAKER: 1.0},
                veto_thresholds={},
                integrity_is_gate=False,
            ),
        )
        assert result.decision is Decision.ACCEPT


class TestDuplicateDetection:
    def test_the_same_file_twice_is_caught(self) -> None:
        clip = speechlike(seed=1)
        checker = IntegrityChecker()
        checker.remember(clip, label="enrol_1")
        assert checker.check(clip).score < INTEGRITY_FLOOR

    def test_a_different_recording_is_not(self) -> None:
        checker = IntegrityChecker()
        checker.remember(speechlike(seed=1), label="enrol_1")
        assert checker.check(speechlike(seed=2, f0=210.0)).score >= INTEGRITY_FLOOR

    def test_memory_survives_being_rebuilt(self) -> None:
        """A replay detector that forgets on restart catches replays only
        within one process lifetime, which is uptime rather than a security
        property. `Pipeline._reload_integrity_memory` is what makes this hold
        across a restart; this asserts the mechanism it relies on."""
        clip = speechlike(seed=3)
        detector = ReplayDetector()
        detector.remember(clip, label="utt_abc")
        rebuilt = IntegrityChecker(replay_detector=detector)
        report = rebuilt.check(clip)
        assert report.score < INTEGRITY_FLOOR
        assert report.replay is not None
        assert report.replay.matched_source == "utt_abc"


class TestRobustness:
    def test_a_detector_that_errors_does_not_take_down_verification(self) -> None:
        """An integrity check that raises would reject every login. Contributing
        no evidence is the correct degradation."""

        class Exploding(ReplayDetector):
            def check(self, audio: Audio):  # type: ignore[override]
                raise RuntimeError("boom")

        checker = IntegrityChecker(check_splice=False, replay_detector=Exploding())
        report = checker.check(speechlike(seed=4))
        assert report.score == 1.0
        assert report.replay is None

    def test_a_very_short_clip_is_a_quality_problem_not_an_accusation(self) -> None:
        tiny = Audio(np.zeros(64, dtype=np.float32) + 0.01, SR, "tiny")
        report = IntegrityChecker(check_replay=False).check(tiny)
        assert report.score >= INTEGRITY_FLOOR
