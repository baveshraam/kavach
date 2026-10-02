"""Calibrating the live voice policy from the presenter's Studio sessions and the cached cohorts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clone_helpers import tone
from kavach.calibrate_voice import calibrate_from_studio, load_voice_policy
from kavach.embedding import SpeakerEmbedding
from kavach.studio.store import StudioStore

KW = dict(kind="read", prompt_id="demo1", environment="QUIET_ROOM", orig_bytes=b"x", orig_ext=".webm")
BASE = np.random.default_rng(0).standard_normal(192)


class OwnerEmbedder:
    """Every clip embeds near the owner; counts how many probes it was asked for."""

    def __init__(self) -> None:
        self.n = 0
        self.rng = np.random.default_rng(1)

    def embed(self, audio):
        self.n += 1
        return SpeakerEmbedding(BASE + 0.35 * self.rng.standard_normal(192))


@pytest.fixture()
def studio(tmp_path):
    s = StudioStore(tmp_path / "studio" / "S04", "S04")
    for session, device in (("S1", "DEMO_LAPTOP_MIC"), ("S2", "HEADSET"), ("S3", "DEMO_LAPTOP_MIC")):
        for k in range(4):
            s.add_clip(session_id=session, device=device, audio=tone(seconds=12.0, seed=hash((session, k)) % 997), **KW)
    return s


@pytest.fixture()
def cohort(tmp_path):
    r = np.random.default_rng(5)
    path = tmp_path / "cohort.npz"
    np.savez(path, vec=r.standard_normal((600, 192)), speaker=np.array([f"X{i % 40}" for i in range(600)]),
             session=np.array(["one"] * 600), dur=np.full(600, 5.0), path=np.array([""] * 600))
    return path


def test_probes_are_cut_to_the_length_of_a_phrase(studio, cohort, tmp_path):
    emb = OwnerEmbedder()
    result = calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=emb, cohort_files=[cohort], probe_seconds=5.0)
    # 12 clips of 12 s: two full 5 s probes each, the 2 s tail is dropped -> 24 held-out probes (and the enrolment pass embeds whole clips)
    assert len(result.loso.genuine) == 24


def test_it_scores_each_session_against_the_others_and_reports_each(studio, cohort):
    result = calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)
    assert set(result.loso.by_session) == {"S1", "S2", "S3"}
    assert "S2" in result.report and "HEADSET" in result.report


def test_the_report_leads_with_what_it_cannot_claim(studio, cohort):
    result = calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)
    assert result.report.startswith("LIMITS")
    assert "optimistic" in result.report.lower()


def test_the_enrollee_is_never_their_own_impostor(studio, tmp_path):
    r = np.random.default_rng(5)
    path = tmp_path / "with_self.npz"
    np.savez(path, vec=BASE + 0.1 * r.standard_normal((50, 192)), speaker=np.array(["S04"] * 50),
             session=np.array(["one"] * 50), dur=np.full(50, 5.0), path=np.array([""] * 50))
    with pytest.raises(ValueError, match="impostor"):
        calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[path], probe_seconds=5.0, exclude_speakers={"S04"})


def test_a_policy_file_is_written_when_asked(studio, cohort, tmp_path):
    out = tmp_path / "voice_policy.json"
    calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0, out=out)
    p = load_voice_policy(out)
    assert p is not None and p.sessions == ["S1", "S2", "S3"]
    assert json.loads(out.read_text(encoding="utf-8"))["n_genuine"] == 24


def test_nothing_is_written_when_not_asked(studio, cohort, tmp_path):
    calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)
    assert not list(tmp_path.glob("*.json"))


def test_a_session_with_no_clips_is_refused(studio, cohort):
    with pytest.raises(ValueError, match="S9"):
        calibrate_from_studio(studio, ["S1", "S9"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)


def test_a_clip_changed_after_recording_is_refused(studio, cohort):
    victim = studio.clips()[0]
    studio.wav_path(victim).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)


def test_the_command_writes_the_policy_and_a_report(studio, cohort, tmp_path, capsys):
    from kavach.calibrate_voice import main

    out = tmp_path / "policy.json"
    rc = main(["--studio", str(studio.root), "--sessions", "S1,S2,S3", "--cohort", str(cohort),
               "--out", str(out), "--report", str(tmp_path / "report.md")], embedder=OwnerEmbedder())
    assert rc == 0 and out.exists()
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("LIMITS")
    assert "threshold" in capsys.readouterr().out.lower()


def test_the_command_refuses_a_missing_session_with_exit_code_2(studio, cohort, tmp_path, capsys):
    from kavach.calibrate_voice import main

    rc = main(["--studio", str(studio.root), "--sessions", "S1,S9", "--cohort", str(cohort),
               "--out", str(tmp_path / "p.json")], embedder=OwnerEmbedder())
    assert rc == 2 and "S9" in capsys.readouterr().err
    assert not (tmp_path / "p.json").exists()


def test_the_command_does_not_write_a_policy_with_dry_run(studio, cohort, tmp_path):
    from kavach.calibrate_voice import main

    out = tmp_path / "p.json"
    rc = main(["--studio", str(studio.root), "--sessions", "S1,S2,S3", "--cohort", str(cohort),
               "--out", str(out), "--dry-run"], embedder=OwnerEmbedder())
    assert rc == 0 and not out.exists()


# ---- probes of the login's own task -------------------------------------------------------


@pytest.fixture()
def studio_with_words(tmp_path):
    s = StudioStore(tmp_path / "studio2" / "S04", "S04")
    for session in ("S1", "S2", "S3"):
        for k in range(3):
            s.add_clip(session_id=session, device="DEMO_LAPTOP_MIC", audio=tone(seconds=12.0, seed=hash((session, k)) % 997), **KW)
        for k in range(4):
            kw = dict(KW, kind="words", prompt_id=f"words_{session}_{k:02d}")
            s.add_clip(session_id=session, device="DEMO_LAPTOP_MIC", audio=tone(seconds=6.0, seed=hash((session, "w", k)) % 997), **kw)
    return s


def test_probes_of_kind_words_are_used_whole_and_the_template_still_uses_everything(studio_with_words, cohort):
    emb = OwnerEmbedder()
    result = calibrate_from_studio(studio_with_words, ["S1", "S2", "S3"], embedder=emb, cohort_files=[cohort], probe_kind="words")
    assert len(result.loso.genuine) == 12                      # 3 sessions x 4 words clips, not chunked
    assert result.report.count("words") >= 1


def test_without_a_probe_kind_every_clip_is_cut_into_phrase_length_probes(studio_with_words, cohort):
    result = calibrate_from_studio(studio_with_words, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_seconds=5.0)
    assert len(result.loso.genuine) > 12


def test_a_session_with_no_words_clips_cannot_be_asked_for_words_probes(studio, cohort):
    with pytest.raises(ValueError, match="words"):
        calibrate_from_studio(studio, ["S1", "S2", "S3"], embedder=OwnerEmbedder(), cohort_files=[cohort], probe_kind="words")


def test_the_command_prefers_words_probes_when_every_session_has_them(studio_with_words, cohort, tmp_path, capsys):
    from kavach.calibrate_voice import main

    rc = main(["--studio", str(studio_with_words.root), "--sessions", "S1,S2,S3", "--cohort", str(cohort), "--dry-run"], embedder=OwnerEmbedder())
    assert rc == 0
    assert "words" in capsys.readouterr().out.lower()


def test_the_command_falls_back_to_chunking_when_there_are_no_words_clips(studio, cohort, tmp_path, capsys):
    from kavach.calibrate_voice import main

    rc = main(["--studio", str(studio.root), "--sessions", "S1,S2,S3", "--cohort", str(cohort), "--dry-run"], embedder=OwnerEmbedder())
    assert rc == 0
    assert "chunk" in capsys.readouterr().out.lower()
