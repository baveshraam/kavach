import numpy as np
import pytest

from kavach.embedding import SpeakerEmbedding
from kavach.eval.enrollee import LIMITS, TestClip, evaluate_enrollee, render

D = 16


def vec(centre: int, noise: float, rng) -> SpeakerEmbedding:
    v = np.zeros(D); v[centre] = 1.0
    return SpeakerEmbedding(v + noise * rng.standard_normal(D))


def world(*, genuine_noise=0.05, impostor_noise=0.05, n_imp=6, seed=0):
    rng = np.random.default_rng(seed)
    enrol = {s: [vec(0, genuine_noise, rng) for _ in range(10)] for s in ("S1", "S2", "S3")}
    test = {"S4": [TestClip(vec(0, genuine_noise, rng), "DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") for _ in range(20)]}
    imp = {f"I{k}": [vec(1 + k, impostor_noise, rng) for _ in range(10)] for k in range(n_imp)}
    return enrol, test, imp


def run(**kw):
    enrol, test, imp = world(**{k: v for k, v in kw.items() if k in ("genuine_noise", "impostor_noise", "n_imp")})
    return evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, seed=5)


def test_a_perfectly_separated_world_has_no_errors_and_a_positive_gap() -> None:
    r = run()
    assert r.frr_at_system == 0.0 and r.far_at_system == 0.0 and r.gap > 0


def test_an_overlapping_world_has_errors_and_no_gap() -> None:
    r = run(genuine_noise=0.9, impostor_noise=0.9)
    assert (r.frr_at_system > 0 or r.far_at_system > 0) and r.gap < 0.2


def test_the_held_out_session_is_never_in_the_template() -> None:
    enrol, test, imp = world()
    rng = np.random.default_rng(1)
    # poison the held-out session: if it leaked into enrolment the template would move
    test["S4"] = [TestClip(vec(7, 0.01, rng), "DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") for _ in range(10)]
    r = evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, seed=5)
    assert r.frr_at_system == 1.0  # a different "voice" is rejected; it did not enrol itself


def test_the_dev_test_split_shares_no_impostor_speaker() -> None:
    r = run()
    assert r.dev_impostors and r.test_impostors and not set(r.dev_impostors) & set(r.test_impostors)


def test_a_single_enrolment_session_skips_the_fitted_threshold_and_says_so() -> None:
    enrol, test, imp = world()
    r = evaluate_enrollee(enrol={"S1": enrol["S1"]}, test=test, impostors=imp, system_threshold=0.62)
    assert r.fitted_threshold is None and any("one enrolment session" in n for n in r.notes)


def test_small_condition_groups_are_dropped_not_printed_as_rates() -> None:
    enrol, test, imp = world()
    test["S4"].append(TestClip(test["S4"][0].embedding, "PHONE", "OFFICE", "free"))
    r = evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, min_group=5)
    assert ("DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") in r.slices
    assert ("PHONE", "OFFICE", "free") not in r.slices


def test_both_intervals_are_reported_and_the_limits_lead_the_report() -> None:
    text = render(run())
    assert text.startswith(LIMITS.splitlines()[0]) or LIMITS in text[:2000]
    assert "Wilson" in text and "cluster" in text.lower()
    assert "one enrolled speaker" in text and "same-sentence" in text


def test_impostors_are_listed_by_pseudonym_with_their_highest_score() -> None:
    r = run()
    assert set(r.by_impostor) == {f"I{k}" for k in range(6)}
    assert all("max" in v for v in r.by_impostor.values())


def test_the_cli_refuses_a_session_that_is_both_enrolled_and_held_out(tmp_path) -> None:
    from kavach.eval.enrollee import main

    assert main(["--studio", str(tmp_path / "S04"), "--enrol-sessions", "S1,S2", "--test-sessions", "S2",
                 "--impostors", str(tmp_path / "m.json")]) == 2


def test_the_cli_refuses_when_the_named_sessions_have_no_clips(tmp_path) -> None:
    from kavach.eval.enrollee import main

    class NoEmbedder:
        def embed(self, audio):
            raise AssertionError("nothing should be embedded")

    assert main(["--studio", str(tmp_path / "S04"), "--enrol-sessions", "S1", "--test-sessions", "S4",
                 "--impostors", str(tmp_path / "m.json")], embedder=NoEmbedder()) == 2


def test_a_single_held_out_session_says_its_session_interval_is_uninformative() -> None:
    """One held-out sitting is one cluster: its bootstrap interval is (0, 100%) and
    must not be read as a result. The report has to say so and say what to do."""
    r = run()  # world() has exactly one test session
    assert any("one held-out session" in n for n in r.notes), r.notes
    assert "second held-out session" in render(r)
