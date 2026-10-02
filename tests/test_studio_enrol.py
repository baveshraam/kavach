"""Enrol the demo's voiceprint from the Studio sessions, through the same audio chain as the login.

The demo's S04 template was built from phone recordings; the login arrives through the laptop
microphone, Opus-coded in the browser. Enrolling from Studio sessions (recorded through that same
path) is what makes the measured numbers describe the demo. The tool must never touch the template
when anything is wrong, and must say what the template was built from.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from clone_helpers import tone
from kavach.api.store import Store
from kavach.embedding import SpeakerEmbedding
from kavach.studio.enrol import EnrolError, enrol_from_studio
from kavach.studio.store import StudioStore

KW = dict(kind="read", prompt_id="demo1", environment="QUIET_ROOM", orig_bytes=b"x", orig_ext=".webm")


class CountingEmbedder:
    """Embeds each clip to a different vector so the template says which clips went in."""

    def __init__(self) -> None:
        self.n = 0

    def embed(self, audio):
        self.n += 1
        v = np.zeros(192)
        v[self.n % 192] = 1.0
        return SpeakerEmbedding(v)


@pytest.fixture()
def world(tmp_path):
    db = Store(tmp_path / "kavach.db", tmp_path / "raw")
    sid = db.create_speaker({"display_name": "S04 · Presenter", "consent_given": True})["id"]
    studio = StudioStore(tmp_path / "studio" / "S04", "S04")
    for session, device, n in (("S1", "DEMO_LAPTOP_MIC", 3), ("S2", "HEADSET", 2), ("S3", "DEMO_LAPTOP_MIC", 2)):
        for k in range(n):
            studio.add_clip(session_id=session, device=device, audio=tone(seconds=2.0, seed=hash((session, k)) % 1000), **KW)
    yield db, sid, studio, tmp_path
    db.close()


def test_the_template_is_built_from_the_named_sessions_only(world):
    db, sid, studio, _ = world
    emb = CountingEmbedder()
    summary = enrol_from_studio(db, studio, "S04", ["S1", "S2"], embedder=emb)
    stored = db.load_template(sid)
    assert len(stored["embeddings"]) == 5 and emb.n == 5
    assert summary["n_clips"] == 5 and summary["sessions"] == ["S1", "S2"]


def test_the_template_records_what_it_was_built_from(world):
    db, sid, studio, _ = world
    enrol_from_studio(db, studio, "S04", ["S1", "S3"], embedder=CountingEmbedder())
    prov = db.load_template(sid)["provenance"]
    assert prov["source"] == "studio" and prov["sessions"] == ["S1", "S3"]
    assert prov["devices"] == ["DEMO_LAPTOP_MIC"] and prov["n_clips"] == 5


def test_a_session_with_no_clips_is_refused_and_the_template_is_untouched(world):
    db, sid, studio, _ = world
    enrol_from_studio(db, studio, "S04", ["S1"], embedder=CountingEmbedder())
    before = db.load_template(sid)
    with pytest.raises(EnrolError, match="S9"):
        enrol_from_studio(db, studio, "S04", ["S2", "S9"], embedder=CountingEmbedder())
    assert db.load_template(sid) == before


def test_an_unknown_speaker_is_refused(world):
    db, _, studio, _ = world
    with pytest.raises(EnrolError, match="S77"):
        enrol_from_studio(db, studio, "S77", ["S1"], embedder=CountingEmbedder())


def test_a_clip_changed_after_recording_is_refused_and_nothing_is_written(world):
    db, sid, studio, _ = world
    victim = [c for c in studio.clips() if c.session_id == "S2"][0]
    studio.wav_path(victim).write_bytes(b"tampered")
    with pytest.raises(EnrolError, match="hash"):
        enrol_from_studio(db, studio, "S04", ["S2"], embedder=CountingEmbedder())
    assert db.load_template(sid) is None


def test_the_database_is_backed_up_before_the_template_is_replaced(world):
    db, sid, studio, tmp_path = world
    backups = tmp_path / "backups"
    summary = enrol_from_studio(db, studio, "S04", ["S1"], embedder=CountingEmbedder(), backup_dir=backups)
    assert list(backups.glob("kavach.db.pre-studio-enrol-*"))
    assert summary["backup"]


def test_no_embedder_means_no_enrolment(world):
    db, _, studio, _ = world
    with pytest.raises(EnrolError, match="embedder"):
        enrol_from_studio(db, studio, "S04", ["S1"], embedder=None)
