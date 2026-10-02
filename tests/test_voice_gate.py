"""The voiceprint is a gate in the live login, not one vote in a weighted average.

Why this exists: with weights (speaker 0.40, CSBG 0.30, knowledge 0.30) and a threshold of
0.55, someone whose voice scores 0.45 -- well below the 0.62 voiceprint threshold -- but who
knows the answer (a hometown on a slide, a fact the panel was told) is accepted:
0.40*0.45 + 0.30*0.8 + 0.30*1.0 = 0.72. A judge does not need to sound like the presenter.
"""

from __future__ import annotations

import math

from kavach.fusion import Branch, BranchScore, Decision, FusionPolicy, fuse

VOICE_THRESHOLD = 0.62


def b(branch: Branch, score: float, threshold: float = 0.5, available: bool = True) -> BranchScore:
    return BranchScore(branch=branch, score=score, threshold=threshold, weight=0.0, available=available)


def voice(score: float, available: bool = True) -> BranchScore:
    return b(Branch.SPEAKER, score, VOICE_THRESHOLD, available)


def others(csbg: float = 0.8, knowledge: float = 1.0) -> list[BranchScore]:
    return [b(Branch.CSBG, csbg), b(Branch.KNOWLEDGE, knowledge, 0.7)]


GATED = FusionPolicy(voice_gate=True, voice_grey_margin=0.08)


def test_the_weighted_average_alone_lets_a_wrong_voice_in__the_hole_this_closes():
    ungated = fuse([voice(0.45), *others()], FusionPolicy())
    assert ungated.decision is Decision.ACCEPT


def test_a_known_answer_cannot_outvote_a_wrong_voice():
    result = fuse([voice(0.45), *others()], GATED)
    assert result.decision is Decision.REJECT
    text = " ".join(result.explanation).lower()
    assert "voice" in text and "no other branch" in text, result.explanation


def test_a_voice_far_below_the_threshold_is_rejected_even_when_everything_else_is_perfect():
    result = fuse([voice(0.21), *others(csbg=1.0, knowledge=1.0)], GATED)
    assert result.decision is Decision.REJECT


def test_a_voice_just_under_the_threshold_asks_for_a_second_sample_instead_of_accepting():
    """Inside the grey zone (threshold - margin <= score < threshold) the voice is
    inconclusive: BORDERLINE means 'step up', and it must never be ACCEPT."""
    result = fuse([voice(0.58), *others()], GATED)
    assert result.decision is Decision.BORDERLINE
    assert any("second" in line.lower() or "step" in line.lower() for line in result.explanation)


def test_the_grey_zone_does_not_rescue_a_login_the_other_branches_reject():
    result = fuse([voice(0.58), *others(csbg=0.05, knowledge=0.0)], GATED)
    assert result.decision is Decision.REJECT


def test_the_floor_is_the_threshold_minus_the_margin():
    just_below_floor = fuse([voice(VOICE_THRESHOLD - 0.08 - 0.001), *others()], GATED)
    at_floor = fuse([voice(VOICE_THRESHOLD - 0.08), *others()], GATED)
    assert just_below_floor.decision is Decision.REJECT
    assert at_floor.decision is Decision.BORDERLINE


def test_a_voice_that_passes_still_needs_the_rest_to_agree():
    assert fuse([voice(0.93), *others()], GATED).decision is Decision.ACCEPT
    assert fuse([voice(0.93), *others(csbg=0.1, knowledge=0.0)], GATED).decision is Decision.REJECT


def test_a_voice_that_could_not_be_measured_fails_closed():
    """Unavailable is not zero for a weighted term, but for the identity factor
    'we could not tell who this is' must not become 'accept on the other two'."""
    result = fuse([voice(0.0, available=False), *others()], GATED)
    assert result.decision is Decision.REJECT
    assert "could not be measured" in " ".join(result.explanation)


def test_a_missing_voice_branch_fails_closed():
    assert fuse(others(), GATED).decision is Decision.REJECT


def test_a_nan_voice_score_is_rejected():
    result = fuse([voice(math.nan), *others()], GATED)
    assert result.decision is Decision.REJECT


def test_the_gate_leaves_the_research_default_alone():
    """Ablations and the paper's fusion tables use FusionPolicy(); the gate is a
    property of the live system's policy, so it is opt-in at this layer."""
    assert FusionPolicy().voice_gate is False
