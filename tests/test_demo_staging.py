"""Staged demo clips must not be mistaken for replays.

Found by running the preflight against the real backend: the 'Impostor voice'
demo clip, a 20 s cut of a 24 s stored recording, was rejected by the integrity
gate before the voiceprint ever ran. The replay detector was right -- envelope
similarity is the peak cross-correlation normalised by the full energies, which
for a cut is about sqrt(cut / clip), so 20 s of 24 s scores ~0.91 against a 0.85
threshold. The staging was wrong. These tests pin the staging rule to the
detector, because a constant in one module that must agree with another module
is exactly where this project's bugs hide.
"""

from __future__ import annotations

import pytest

from kavach.attacks.replay import ReplayDetector
from kavach.demo_check import (
    STAGING_MAX_CUT_SEC,
    STAGING_MAX_SHARE,
    STAGING_MIN_SOURCE_SEC,
    cut_plan,
)
from test_integrity import speechlike


def similarity(clip, cut) -> float:
    detector = ReplayDetector()
    detector.remember(clip, label="stored")
    return detector.check(cut).best_envelope_similarity


def test_a_cut_that_is_most_of_a_stored_clip_reads_as_a_replay() -> None:
    clip = speechlike(seconds=24.0, seed=1, pauses=10)
    detector = ReplayDetector()
    assert similarity(clip, clip.slice_seconds(2.0, 22.0)) >= detector.envelope_threshold


@pytest.mark.parametrize("seconds", [24.0, 30.0, 37.0])
def test_a_planned_cut_stays_well_clear_of_the_replay_threshold(seconds) -> None:
    clip = speechlike(seconds=seconds, seed=2, pauses=12)
    start, length = cut_plan(seconds)
    sim = similarity(clip, clip.slice_seconds(start, start + length))
    assert sim < ReplayDetector().envelope_threshold - 0.1, (
        f"a {length:.1f}s cut of a {seconds:.0f}s clip scores {sim:.2f}"
    )


def test_a_planned_cut_is_a_bounded_share_of_the_source() -> None:
    for seconds in (20.0, 24.0, 40.0, 90.0):
        start, length = cut_plan(seconds)
        assert length <= STAGING_MAX_SHARE * seconds + 1e-9
        assert length <= STAGING_MAX_CUT_SEC
        assert start + length <= seconds


def test_a_cut_is_long_enough_to_score() -> None:
    _, length = cut_plan(STAGING_MIN_SOURCE_SEC)
    assert length >= 8.0, "under about eight seconds the CSBG and the matcher have little to read"


def test_a_clip_too_short_to_cut_safely_is_not_staged() -> None:
    assert cut_plan(STAGING_MIN_SOURCE_SEC - 0.1) is None
