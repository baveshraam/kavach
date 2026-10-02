"""The 'read these words' login: voice, freshness and nothing else.

A phrase challenge needs no knowledge-graph facts, no language tagger and no network: it is
the fast, offline-capable path. It is accepted only when every gate passes -- a live,
unused challenge, an unedited recording, the words that were shown, and the enrolled voice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app, get_pipeline, get_settings, get_store
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.asr import Transcript
from kavach.audio import Audio, save_wav
from kavach.config import Settings
from kavach.embedding import EMBEDDING_DIM, SpeakerEmbedding, SpeakerTemplate

_COUNTER = iter(range(10_000))


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM)
    return v / np.linalg.norm(v)


TEMPLATE_VEC = unit(1)


def probe_with_cosine(c: float) -> np.ndarray:
    """A unit vector whose cosine with the template is exactly `c`."""
    other = unit(2) - (unit(2) @ TEMPLATE_VEC) * TEMPLATE_VEC
    other /= np.linalg.norm(other)
    return c * TEMPLATE_VEC + np.sqrt(1 - c * c) * other


class FakeEmbedder:
    def __init__(self, vector) -> None:
        self.vector = np.asarray(vector, float)

    def embed(self, audio):
        return SpeakerEmbedding(self.vector)


class FakeASR:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return Transcript(text=self.text, words=[])


class ExplodingLID:
    """A phrase login must never reach the language tagger (it needs the network)."""

    last_llm_error = None
    last_fallback_coverage = None

    def tag_utterance(self, *a, **k):
        raise AssertionError("the tagger must not run for a phrase challenge")


def wav_bytes(tmp_path: Path, seconds: float = 4.0) -> bytes:
    n = next(_COUNTER)
    rng = np.random.default_rng(n)
    t = np.linspace(0, seconds, int(16_000 * seconds), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * (140 + n) * t) + 0.02 * rng.standard_normal(len(t))
    path = tmp_path / f"clip_{n}.wav"
    save_wav(Audio(wave.astype(np.float32), 16_000), path)
    return path.read_bytes()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path, audio_dir=tmp_path / "raw", attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db", offline=False,
    )


@pytest.fixture()
def store(settings: Settings):
    s = Store(settings.db_path, settings.audio_dir)
    yield s
    s.close()


@pytest.fixture()
def pipeline(store: Store, settings: Settings) -> Pipeline:
    p = Pipeline(store, settings)
    p._lid = ExplodingLID()
    return p


@pytest.fixture()
def client(store, pipeline, settings) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@pytest.fixture()
def speaker(client: TestClient, store: Store) -> str:
    r = client.post("/api/speakers", json={"displayName": "Presenter", "consentGiven": True})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    store.save_template(sid, SpeakerTemplate.from_embeddings(sid, [SpeakerEmbedding(TEMPLATE_VEC)]).to_dict())
    return sid


def issue(client: TestClient, sid: str, **extra) -> dict:
    r = client.post("/api/challenge", json={"speakerId": sid, "kind": "phrase", **extra})
    assert r.status_code == 200, r.text
    return r.json()


def login(client, pipeline, tmp_path, challenge, *, heard: str, cosine: float):
    pipeline._embedder = FakeEmbedder(probe_with_cosine(cosine))
    pipeline._asr = FakeASR(heard)
    r = client.post(
        "/api/authenticate",
        files={"audio": ("a.wav", wav_bytes(tmp_path), "audio/wav")},
        data={"challengeId": challenge["id"]},
    )
    assert r.status_code == 200, r.text
    return r.json(), pipeline._asr


def branch(result: dict, name: str) -> dict | None:
    return next((b for b in result["branches"] if b["name"] == name), None)


class TestIssuing:
    def test_a_speaker_with_no_facts_can_still_be_challenged_for_a_phrase(self, client, speaker):
        c = issue(client, speaker)
        assert c["kind"] == "phrase" and len(c["phrase"]) == 6
        assert all(w in c["questionText"] for w in c["phrase"])

    def test_every_challenge_is_a_different_phrase(self, client, speaker):
        assert len({tuple(issue(client, speaker)["phrase"]) for _ in range(10)}) == 10

    def test_a_question_challenge_is_still_the_default(self, client, speaker):
        r = client.post("/api/challenge", json={"speakerId": speaker})
        assert r.status_code == 409  # no facts: the question path is unchanged

    def test_an_unknown_kind_is_a_client_error(self, client, speaker):
        r = client.post("/api/challenge", json={"speakerId": speaker, "kind": "magic"})
        assert r.status_code == 422


class TestVerifying:
    def test_the_right_voice_saying_the_shown_words_is_accepted(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        result, asr = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.85)
        assert result["decision"] == "ACCEPT", result["explanation"]
        assert branch(result, "phrase")["passed"] and branch(result, "speaker_embedding")["passed"]
        assert branch(result, "csbg") is None and branch(result, "knowledge") is None
        assert result["transcript"] == " ".join(c["phrase"])

    def test_the_words_are_transcribed_as_english_without_the_code_mixing_bias(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        _, asr = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.85)
        assert asr.calls[0].get("language") == "en"
        assert asr.calls[0].get("initial_prompt") == ""

    def test_a_different_voice_is_rejected_even_when_it_says_the_right_words(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.25)
        assert result["decision"] == "REJECT"
        assert "voice does not match" in " ".join(result["explanation"])

    def test_the_right_voice_saying_other_words_is_rejected__that_is_what_a_replay_looks_like(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        result, _ = login(client, pipeline, tmp_path, c, heard="bridge lantern pepper eagle forest ribbon", cosine=0.95)
        assert result["decision"] == "REJECT"
        assert not branch(result, "phrase")["passed"]
        text = " ".join(result["explanation"]).lower()
        assert "words" in text and "replay" in text, result["explanation"]

    def test_a_voice_just_under_the_threshold_is_borderline_never_accepted(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.58)
        assert result["decision"] == "BORDERLINE"

    def test_one_misheard_word_is_forgiven(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        heard = " ".join(c["phrase"][:-1])
        result, _ = login(client, pipeline, tmp_path, c, heard=heard, cosine=0.85)
        assert result["decision"] == "ACCEPT", result["explanation"]

    def test_no_speech_recognition_means_no_login__fail_closed(self, client, pipeline, speaker, tmp_path, settings):
        c = issue(client, speaker)
        pipeline._embedder = FakeEmbedder(probe_with_cosine(0.95))
        pipeline._asr = None
        pipeline._failed["asr"] = "not installed"
        r = client.post(
            "/api/authenticate",
            files={"audio": ("a.wav", wav_bytes(tmp_path), "audio/wav")},
            data={"challengeId": c["id"]},
        ).json()
        assert r["decision"] == "REJECT"
        assert "could not" in " ".join(r["explanation"]).lower()

    def test_a_phrase_challenge_is_single_use(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        first, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.85)
        second, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.85)
        assert first["decision"] == "ACCEPT"
        assert second["decision"] == "REJECT"
        assert any("already been used" in line for line in second["explanation"])
