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


def two_sittings():
    enrol, test, imp = world()
    rng = np.random.default_rng(9)
    test["S5"] = [TestClip(vec(0, 0.05, rng), "HEADSET", "QUIET_ROOM", "read") for _ in range(20)]
    return enrol, test, imp


def test_two_clean_sittings_never_print_a_zero_width_interval() -> None:
    enrol, test, imp = two_sittings()
    r = evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, seed=5)
    assert r.frr_at_system == 0.0
    assert r.frr_reported[1] > 0.0, r.frr_reported
    assert r.frr_informative is False
    assert "not informative" in render(r)


def test_the_report_names_its_sessions_clip_counts_and_template() -> None:
    text = render(run())
    for needle in ("S1", "S2", "S3", "S4", "template", "re-enrol"):
        assert needle in text, needle


def test_the_template_is_exactly_the_enrolment_clips() -> None:
    """A leaked held-out session would change this count; clean clips could hide it in a score."""
    assert run().n_enrol_clips == 30


def test_no_genuine_or_no_impostor_trials_is_a_clear_error() -> None:
    enrol, test, imp = world()
    with pytest.raises(ValueError, match="genuine"):
        evaluate_enrollee(enrol=enrol, test={}, impostors=imp, system_threshold=0.62)
    with pytest.raises(ValueError, match="impostor"):
        evaluate_enrollee(enrol=enrol, test=test, impostors={}, system_threshold=0.62)


def test_a_non_finite_embedding_is_refused_not_counted_as_a_good_score() -> None:
    enrol, test, imp = world()
    bad = SpeakerEmbedding(np.full(D, np.nan))
    test["S4"][0] = TestClip(bad, "DEMO_LAPTOP_MIC", "QUIET_ROOM", "read")
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62)


def test_a_perfectly_separated_dev_set_is_fitted_at_the_midpoint_of_the_gap() -> None:
    """compute_eer returns the LOWEST dev genuine score when the classes do not overlap, which
    produced a fitted FRR that came from the tie-break, not the voice."""
    r = run()
    assert r.fitted_threshold is not None and 0.3 < r.fitted_threshold < 0.7, r.fitted_threshold


def test_the_fitted_rates_carry_n_and_an_interval() -> None:
    r = run()
    assert r.fitted_n_genuine == 20 and r.fitted_n_impostor > 0
    assert r.fitted_frr_interval[0] <= r.fitted_frr <= r.fitted_frr_interval[1]
    assert "dev template" in render(r)


def test_results_json_carries_the_limits() -> None:
    d = run().to_dict()
    assert d["limits"] == LIMITS


def test_the_per_trial_scores_are_kept_for_the_audit_trail() -> None:
    r = run()
    assert len(r.trials) == r.n_genuine + r.n_impostor and {"label", "group", "score"} <= set(r.trials[0])
    assert "trials" not in r.to_dict()  # per-trial scores stay out of the tracked JSON


def test_collect_impostors_never_includes_the_enrollee() -> None:
    from types import SimpleNamespace

    from kavach.eval.enrollee import collect_impostors

    corpus = SimpleNamespace(root=None, utterances=[
        SimpleNamespace(speaker_id="S04", audio_path="a.wav"),
        SimpleNamespace(speaker_id="S08", audio_path="b.wav"),
        SimpleNamespace(speaker_id="S09", audio_path=""),
    ])
    got = collect_impostors(corpus, exclude={"S04"}, embed=lambda path: str(path))
    assert set(got) == {"S08"}


class _FakeEmbedder:
    def embed(self, audio):
        v = np.zeros(D)
        v[int(abs(audio.samples[:200].sum() * 1000)) % D] = 1.0
        return SpeakerEmbedding(v + 0.01)


def _studio(tmp_path, sessions):
    from clone_helpers import tone
    from kavach.studio.store import StudioStore

    store = StudioStore(tmp_path / "S04", "S04")
    for sid in sessions:
        for i in range(3):
            store.add_clip(audio=tone(seconds=2.0, seed=i), session_id=sid, kind="read", prompt_id="demo1",
                           device="DEMO_LAPTOP_MIC", environment="QUIET_ROOM", orig_bytes=b"x", orig_ext=".wav")
    return tmp_path / "S04"


def test_the_cli_refuses_when_any_named_session_has_no_clips(tmp_path, capsys) -> None:
    from kavach.eval.enrollee import main

    studio = _studio(tmp_path, ["S1", "S4"])  # S2 is named below but was never recorded
    code = main(["--studio", str(studio), "--enrol-sessions", "S1,S2", "--test-sessions", "S4",
                 "--impostors", str(tmp_path / "m.json")], embedder=_FakeEmbedder())
    assert code == 2 and "S2" in capsys.readouterr().err


def test_the_cli_tolerates_spaces_in_the_session_lists(tmp_path, capsys) -> None:
    """'S4, S5' used to become a session called ' S5' with no clips, silently."""
    from kavach.eval.enrollee import main

    studio = _studio(tmp_path, ["S1", "S4"])
    code = main(["--studio", str(studio), "--enrol-sessions", "S1", "--test-sessions", "S4, S5",
                 "--impostors", str(tmp_path / "m.json")], embedder=_FakeEmbedder())
    err = capsys.readouterr().err
    assert code == 2 and "S5" in err and " S5" not in err.replace("S5", "", 1)
