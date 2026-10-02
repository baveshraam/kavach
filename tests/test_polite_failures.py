"""The live login must fail politely.

Probed on 2026-10-02 with hostile and odd inputs: no server errors, but three
things an examiner would see. A garbage upload showed raw decoder output; the
first non-16 kHz upload stalled for seconds while librosa loaded; silence went
on to Whisper, which invents text for it, and the invention was scored.
"""

from __future__ import annotations

import io
import shutil
import threading
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app, get_pipeline, get_settings, get_store
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.audio import AudioError, decode_bytes, warm_resample
from kavach.config import Settings

SR = 16_000


def wav(seconds: float, kind: str = "tone") -> bytes:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    x = np.zeros(n) if kind == "silence" else 0.3 * np.sin(2 * np.pi * 140 * t)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
    )


@pytest.fixture()
def store(settings) -> Store:
    s = Store(settings.db_path, settings.audio_dir)
    yield s
    s.close()


@pytest.fixture()
def pipeline(store, settings) -> Pipeline:
    return Pipeline(store, settings)


@pytest.fixture()
def client(store, pipeline, settings) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def challenge_id(client: TestClient) -> str:
    speaker = client.post(
        "/api/speakers", json={"displayName": "Probe", "consentGiven": True}
    ).json()
    client.put(
        f"/api/speakers/{speaker['id']}/skg",
        json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
    )
    return client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()["id"]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class TestUnreadableUploads:
    @needs_ffmpeg
    def test_garbage_does_not_show_decoder_output(self) -> None:
        with pytest.raises(AudioError) as exc:
            decode_bytes(b"not audio at all" * 50, suffix=".wav")
        message = str(exc.value)
        assert "ffmpeg" not in message.lower() and "[in#" not in message, message
        assert "WAV" in message, "say which formats would have worked"

    @needs_ffmpeg
    def test_the_route_returns_a_400_with_that_message(self, client: TestClient) -> None:
        response = client.post(
            "/api/authenticate",
            files={"audio": ("x.wav", b"not audio at all" * 50, "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        assert response.status_code == 400
        assert "ffmpeg" not in response.json()["detail"].lower()


class TestWarmUp:
    def test_warm_resample_runs(self) -> None:
        assert warm_resample() is None

    def test_start_up_pays_the_resampling_cost_in_the_background(
        self, store, pipeline, settings, monkeypatch
    ) -> None:
        """The first non-16 kHz upload took 3-12 s because librosa loads lazily.
        That must be paid at start-up, in the warm-up thread, not on stage."""
        import kavach.api.app as app_module

        done = threading.Event()
        monkeypatch.setattr(app_module, "_store", store)
        monkeypatch.setattr(app_module, "_pipeline", pipeline)
        monkeypatch.setattr(app_module, "warm_resample", done.set)
        cfg = settings.model_copy(update={"warm_models_on_start": True})
        with TestClient(app_module.create_app(cfg)):
            assert done.wait(15), "start-up never called warm_resample"


class TestUnusableAudio:
    def test_silence_is_rejected_without_running_a_model(
        self, client: TestClient, pipeline: Pipeline, monkeypatch
    ) -> None:
        def boom(*a, **k):
            raise AssertionError("a model ran on audio with nothing in it")

        monkeypatch.setattr(pipeline, "annotate", boom)
        response = client.post(
            "/api/authenticate",
            files={"audio": ("s.wav", wav(3.0, "silence"), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        assert response.status_code == 200
        body = response.json()
        text = " ".join(body["explanation"]).lower()
        assert body["decision"] == "REJECT"
        assert "no speech" in text and "system failure" not in text, body["explanation"]

    def test_a_clip_that_is_too_short_says_how_short(
        self, client: TestClient, pipeline: Pipeline, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            pipeline, "annotate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))
        )
        response = client.post(
            "/api/authenticate",
            files={"audio": ("s.wav", wav(0.3), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        text = " ".join(response.json()["explanation"])
        assert response.json()["decision"] == "REJECT"
        assert "0.3" in text and "at least" in text, text

    def test_ordinary_audio_is_not_caught_by_the_early_reject(self, client: TestClient) -> None:
        response = client.post(
            "/api/authenticate",
            files={"audio": ("t.wav", wav(3.0), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        text = " ".join(response.json()["explanation"]).lower()
        assert "no speech" not in text and "seconds long" not in text, text


class TestRealWorldFormats:
    """Phone voice memos and browser recordings arrive in containers ffmpeg
    cannot always read from a pipe. An MP4/M4A written without +faststart keeps
    its `moov` atom at the END of the file; read from a non-seekable pipe,
    ffmpeg says "partial file", exits 0 and emits nothing, and the upload was
    reported as "Decoded audio is empty". Found by rehearsing the demo against
    the real backend with a real .m4a."""

    @pytest.mark.parametrize(
        "ext,codec",
        [
            ("m4a", ["-c:a", "aac"]),
            ("m4a", ["-c:a", "aac", "-ar", "44100"]),
            ("webm", ["-c:a", "libopus"]),
            ("ogg", ["-c:a", "libvorbis"]),
            ("mp3", ["-c:a", "libmp3lame"]),
        ],
    )
    @needs_ffmpeg
    def test_a_real_encoded_upload_decodes_in_full(self, tmp_path, ext, codec) -> None:
        import subprocess

        (tmp_path / "in.wav").write_bytes(wav(10.0))
        out = tmp_path / f"o.{ext}"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "in.wav"), *codec, str(out)],
            check=False,
        )
        if not out.exists():
            pytest.skip(f"this ffmpeg build cannot encode {ext} with {codec}")
        audio = decode_bytes(out.read_bytes(), suffix=f".{ext}")
        assert audio.duration_sec == pytest.approx(10.0, abs=0.6), (ext, audio.duration_sec)
