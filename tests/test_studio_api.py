from __future__ import annotations

import shutil
import subprocess

import pytest

from clone_helpers import tone
from kavach.audio import Audio, save_wav
from test_clone_bank_api import build_client

import numpy as np

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def studio(tmp_path, **kw):
    return build_client(tmp_path, studio_enabled=True, studio_speakers=["S04"], **kw)


def wav_bytes(tmp_path, seconds=3.0, seed=0, silent=False) -> bytes:
    p = tmp_path / f"c{seed}.wav"
    audio = Audio(np.zeros(int(16000 * seconds), dtype=np.float32), 16000, "s") if silent else tone(seconds=seconds, seed=seed)
    save_wav(audio, p)
    return p.read_bytes()


FORM = {"speaker": "S04", "session_id": "S1", "kind": "read", "prompt_id": "demo1",
        "device": "DEMO_LAPTOP_MIC", "environment": "QUIET_ROOM"}


def post(client, data: bytes, name="c.wav", **over):
    return client.post("/api/studio/clips", files={"audio": (name, data, "audio/wav")}, data={**FORM, **over})


def test_everything_is_a_404_while_the_studio_is_off(tmp_path) -> None:
    client, *_ = build_client(tmp_path)
    assert client.get("/api/studio/plan", params={"speaker": "S04"}).status_code == 404
    assert client.get("/api/studio/summary", params={"speaker": "S04"}).status_code == 404
    assert post(client, wav_bytes(tmp_path)).status_code == 404


def test_a_speaker_not_on_the_allowlist_is_a_403(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert client.get("/api/studio/plan", params={"speaker": "S09"}).status_code == 403
    assert post(client, wav_bytes(tmp_path), speaker="S09").status_code == 403


def test_the_plan_lists_devices_environments_and_items(tmp_path) -> None:
    client, store, _, settings = studio(tmp_path)
    sp = store.create_speaker({"display_name": "S04 · Tester"})
    client.put(f"/api/speakers/{sp['id']}/skg", json=[{"subject": "x", "predicate": "hometown", "object": "T"}])
    body = client.get("/api/studio/plan", params={"speaker": "S04"}).json()
    assert body["sessionId"] == "S1" and body["hasFacts"] is True
    assert "DEMO_LAPTOP_MIC" in body["devices"] and "QUIET_ROOM" in body["environments"]
    kinds = {i["kind"] for i in body["items"]}
    assert kinds == {"read", "free", "fact"} and body["estimatedMinutes"] > 0


def test_a_clip_is_stored_and_summarised(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    r = post(client, wav_bytes(tmp_path))
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["clips"] == 1
    index = settings.data_dir / "studio" / "S04" / "index.jsonl"
    assert len(index.read_text(encoding="utf-8").splitlines()) == 1
    s = client.get("/api/studio/summary", params={"speaker": "S04"}).json()
    assert s["by_kind"]["read"]["clips"] == 1


@pytest.mark.parametrize("over,status", [({"kind": "shout"}, 422), ({"device": "TOASTER"}, 422), ({"session_id": "../x"}, 422)])
def test_bad_labels_are_a_422_and_write_nothing(tmp_path, over, status) -> None:
    client, _, _, settings = studio(tmp_path)
    assert post(client, wav_bytes(tmp_path), **over).status_code == status
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_silent_and_tiny_clips_are_refused_with_a_reason(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    silent = post(client, wav_bytes(tmp_path, silent=True))
    tiny = post(client, wav_bytes(tmp_path, seconds=0.4, seed=1))
    assert silent.status_code == 422 and "no speech" in silent.json()["detail"]
    assert tiny.status_code == 422 and "at least" in tiny.json()["detail"]
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_empty_and_unreadable_uploads_are_a_400(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert post(client, b"").status_code == 400


@needs_ffmpeg
@pytest.mark.parametrize("ext,codec", [("webm", ["-c:a", "libopus"]), ("m4a", ["-c:a", "aac"])])
def test_a_real_browser_or_phone_encoding_is_stored_as_a_16k_wav(tmp_path, ext, codec) -> None:
    client, _, _, settings = studio(tmp_path)
    (tmp_path / "in.wav").write_bytes(wav_bytes(tmp_path, seconds=6.0))
    out = tmp_path / f"o.{ext}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "in.wav"), *codec, str(out)], check=False)
    if not out.exists():
        pytest.skip("this ffmpeg cannot encode that")
    r = client.post("/api/studio/clips", files={"audio": (f"c.{ext}", out.read_bytes(), "audio/webm")}, data=FORM)
    assert r.status_code == 200, r.text
    clip = r.json()["clip"]
    assert clip["duration_sec"] == pytest.approx(6.0, abs=0.6) and clip["orig_ext"] == f".{ext}"


def test_an_upload_over_the_size_cap_is_a_413(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert post(client, b"x" * (31 * 1024 * 1024)).status_code == 413


def test_audio_over_the_length_cap_is_a_422(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    r = post(client, wav_bytes(tmp_path, seconds=125.0))
    assert r.status_code == 422 and "longer than" in r.json()["detail"]
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


@needs_ffmpeg
def test_undecodable_bytes_are_a_400_and_write_nothing(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    r = post(client, b"this is not audio at all" * 40)
    assert r.status_code == 400
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_a_second_device_in_the_same_session_is_a_422(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert post(client, wav_bytes(tmp_path)).status_code == 200
    r = post(client, wav_bytes(tmp_path, seed=1), device="PHONE")
    assert r.status_code == 422 and "one sitting" in r.json()["detail"]
