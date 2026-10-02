"""Tests for `kavach.calibrate_integrity`, the command that decides whether the
splice tests may be switched on.

The module exists because the splice detector was calibrated on synthetic audio
and rejected 167 of 168 genuine corpus clips. These tests check the *machinery*
on synthetic audio -- that it can tell separable from inseparable, and that it
refuses to call a detector useful when it is not. What the real corpus says is
the command's output, not a test assertion: a test pinned to today's corpus
would fail the day someone adds a speaker, and would pass vacuously on a
checkout that has no audio.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from kavach.audio import Audio
from kavach.calibrate_integrity import (
    MIN_USEFUL_DETECTION,
    Calibration,
    auc,
    is_useful,
    measure,
    render,
    tpr_at_fpr,
    window,
)
from kavach.integrity import FloorCalibration

from test_integrity import speechlike


class TestStatistics:
    def test_auc_of_perfectly_separated_scores_is_one(self) -> None:
        assert auc([1, 2, 3], [10, 11, 12]) == 1.0

    def test_auc_of_identical_distributions_is_one_half(self) -> None:
        assert auc([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(0.5)

    def test_auc_counts_ties_as_half(self) -> None:
        """Every score equal is no information, not perfect separation."""
        assert auc([5, 5, 5], [5, 5, 5]) == pytest.approx(0.5)

    def test_tpr_at_fpr_reads_the_threshold_off_the_genuine_scores(self) -> None:
        genuine = list(range(100))  # 0..99
        spliced = [1000.0] * 10
        tpr, threshold = tpr_at_fpr(genuine, spliced, 0.05)
        assert tpr == 1.0
        assert 90 <= threshold <= 99, "5% FPR on 0..99 puts the threshold near 94"

    def test_tpr_at_fpr_is_zero_when_the_classes_overlap_completely(self) -> None:
        genuine = [float(i) for i in range(100)]
        tpr, _ = tpr_at_fpr(genuine, genuine, 0.01)
        assert tpr <= 0.02


class TestWindow:
    def test_a_window_is_the_requested_length(self) -> None:
        clip = speechlike(seconds=8.0, seed=3)
        w = window(clip, 2.5, random.Random(0))
        assert len(w.samples) / w.sample_rate == pytest.approx(2.5, abs=0.01)

    def test_a_window_stays_clear_of_the_clips_own_edges(self) -> None:
        """Decoder padding sits at the edges of real clips as runs of exact
        zeros. A window that includes it turns padding into an 'interior'
        digital-silence finding -- 78 of the 168 genuine rejections."""
        rng = random.Random(1)
        edge = int(0.3 * 16_000)
        samples = np.full(16_000 * 8, 0.1, dtype=np.float32)
        samples[:edge] = 0.0
        samples[-edge:] = 0.0
        clip = Audio(samples, 16_000, "padded")
        for _ in range(50):
            w = window(clip, 2.0, rng)
            assert not np.any(w.samples == 0.0), "a window reached into the padded edge"


class TestMeasure:
    @staticmethod
    def _clips(*, same_room: bool) -> dict[str, list]:
        """Three speakers x six 8 s clips. `same_room` shares one noise floor
        within a speaker, as one sitting does; otherwise every clip differs."""
        out: dict[str, list] = {}
        r = np.random.default_rng(5)
        for spk in ("S1", "S2", "S3"):
            noise = float(r.uniform(-55, -35))
            f0 = float(r.uniform(90, 220))
            out[spk] = [
                speechlike(
                    seconds=8.0,
                    seed=100 * len(out) + k,
                    noise_db=noise if same_room else float(r.uniform(-55, -35)),
                    f0=f0 if same_room else float(r.uniform(90, 220)),
                    pauses=3,
                )
                for k in range(6)
            ]
        return out

    def test_it_counts_what_it_measured(self) -> None:
        c = measure(self._clips(same_room=False), seconds=3.0, windows=4, splices=3, seed=1)
        assert isinstance(c, Calibration)
        assert c.n_genuine == 3 * 4
        assert c.n_naive == 3 * 3 and c.n_careful == 3 * 3
        assert c.full_clip_total == 18
        assert {cue.name for cue in c.cues} >= {"click_rate", "max_level", "max_spec"}

    def test_it_separates_splices_whose_room_tone_differs(self) -> None:
        """Positive control. If the machinery could not say 'yes' on audio built
        to be separable, its 'no' on the real corpus would mean nothing."""
        c = measure(self._clips(same_room=False), seconds=3.0, windows=12, splices=8, seed=2)
        best = max(max(cue.auc.values()) for cue in c.cues)
        assert best > 0.7, f"no cue separated clearly different room tones (best AUC {best:.2f})"


class TestVerdict:
    def test_a_floor_that_catches_nothing_is_not_useful(self) -> None:
        """`calibrate_floor` reports a 0.0 floor as *feasible* -- it meets the
        FRR budget by rejecting nothing. Feasible is not useful."""
        cal = FloorCalibration(
            floor=0.0, false_reject_rate=0.0, detection_rate=0.0,
            n_genuine=100, n_tampered=100, feasible=True,
        )
        assert is_useful(cal) is False

    def test_a_floor_that_catches_most_splices_is_useful(self) -> None:
        cal = FloorCalibration(
            floor=0.25, false_reject_rate=0.005, detection_rate=0.9,
            n_genuine=100, n_tampered=100, feasible=True,
        )
        assert is_useful(cal) is True

    def test_the_bar_is_stated_and_not_trivial(self) -> None:
        assert 0.5 <= MIN_USEFUL_DETECTION < 1.0

    def test_the_report_says_to_leave_it_off_when_nothing_separates(self) -> None:
        c = measure(
            {"S1": [speechlike(seconds=8.0, seed=i, noise_db=-45.0, f0=140.0) for i in range(6)]},
            seconds=3.0, windows=4, splices=3, seed=3,
        )
        c.separates = False
        text = render(c)
        assert "leave" in text.lower() and "off" in text.lower()
        assert "AUC" in text

    def test_the_report_says_it_may_be_enabled_when_something_separates(self) -> None:
        c = measure(
            {"S1": [speechlike(seconds=8.0, seed=i, noise_db=-45.0, f0=140.0) for i in range(6)]},
            seconds=3.0, windows=4, splices=3, seed=3,
        )
        c.separates = True
        assert "leave" not in render(c).lower()
