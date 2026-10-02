from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from clone_helpers import tone
from kavach.audio import Audio
from kavach.studio.store import DEVICES, ENVIRONMENTS, KINDS, StudioError, StudioStore

KW = dict(
    session_id="S1", kind="read", prompt_id="demo1", device="DEMO_LAPTOP_MIC",
    environment="QUIET_ROOM", orig_bytes=b"RIFFxxxx", orig_ext=".webm",
)


def store(tmp_path) -> StudioStore:
    return StudioStore(tmp_path / "studio" / "S04", "S04")


def test_a_clip_is_written_and_indexed_last(tmp_path) -> None:
    s = store(tmp_path)
    rec = s.add_clip(audio=tone(seconds=3.0), text_hint="நாளைக்கு meeting", **KW)
    assert s.wav_path(rec).exists()
    assert (s.root / "S1" / f"{rec.clip_id}.orig").read_bytes() == b"RIFFxxxx"
    assert [c.clip_id for c in s.clips()] == [rec.clip_id]
    assert rec.duration_sec == pytest.approx(3.0, abs=0.01) and len(rec.sha256_wav) == 64


def test_the_index_is_append_only(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=2.0), **KW)
    first = s.index_path.read_bytes()
    s.add_clip(audio=tone(seconds=2.0, seed=1), **KW)
    assert s.index_path.read_bytes().startswith(first)
    assert len(s.index_path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_failed_write_leaves_no_index_entry(tmp_path, monkeypatch) -> None:
    import kavach.studio.store as mod

    s = store(tmp_path)
    monkeypatch.setattr(mod, "save_wav", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        s.add_clip(audio=tone(seconds=2.0), **KW)
    assert s.clips() == []


@pytest.mark.parametrize("audio,reason", [
    (Audio(np.zeros(16000 * 3, dtype=np.float32), 16000, "s"), "no speech"),
    (tone(seconds=0.4), "at least"),
])
def test_unusable_audio_is_refused_and_nothing_is_written(tmp_path, audio, reason) -> None:
    s = store(tmp_path)
    with pytest.raises(StudioError, match=reason):
        s.add_clip(audio=audio, **KW)
    assert s.clips() == []
    assert not list(s.root.rglob("*.wav")) and not list(s.root.rglob("*.orig"))


@pytest.mark.parametrize("field,value", [
    ("kind", "shout"), ("device", "TOASTER"), ("environment", "MOON"),
    ("session_id", "../evil"), ("session_id", ""), ("prompt_id", ""),
])
def test_bad_labels_are_refused(tmp_path, field, value) -> None:
    s = store(tmp_path)
    with pytest.raises(StudioError):
        s.add_clip(audio=tone(seconds=2.0), **{**KW, field: value})
    assert s.clips() == []


def test_a_hostile_extension_is_neutralised(tmp_path) -> None:
    s = store(tmp_path)
    rec = s.add_clip(audio=tone(seconds=2.0), **{**KW, "orig_ext": "/../x"})
    assert rec.orig_ext == ".bin"
    assert not any(p.name == "x" for p in tmp_path.rglob("*"))


def test_next_session_id_skips_used_ones(tmp_path) -> None:
    s = store(tmp_path)
    assert s.next_session_id() == "S1"
    s.add_clip(audio=tone(seconds=2.0), **KW)
    assert s.next_session_id() == "S2"


def test_summary_counts_minutes_and_slices(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=6.0), **KW)
    s.add_clip(audio=tone(seconds=6.0, seed=1), **{**KW, "kind": "free", "device": "PHONE", "session_id": "S2"})
    summ = s.summary()
    assert summ["clips"] == 2 and summ["minutes"] == pytest.approx(0.2, abs=0.01)
    assert summ["by_kind"]["read"]["clips"] == 1 and summ["by_device"]["PHONE"]["clips"] == 1
    assert set(summ["by_session"]) == {"S1", "S2"}


def test_a_corrupt_index_line_names_its_line(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=2.0), **KW)
    with s.index_path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(StudioError, match="line 2"):
        s.clips()


def test_the_closed_sets_are_what_the_spec_says() -> None:
    assert KINDS == ("read", "free", "fact")
    assert DEVICES == ("DEMO_LAPTOP_MIC", "PHONE", "HEADSET", "OTHER")
    assert "QUIET_ROOM" in ENVIRONMENTS
