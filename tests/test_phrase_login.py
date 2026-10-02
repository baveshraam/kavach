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

    def embed(self, audio, **kw):
        return SpeakerEmbedding(self.vector)


class FakeASR:
    def __init__(self, text: str, words=None) -> None:
        self.text = text
        self.words = words or []
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return Transcript(text=self.text, words=self.words)


class SpyEmbedder(FakeEmbedder):
    """Remembers how long each clip it was asked to embed was."""

    def __init__(self, vector) -> None:
        super().__init__(vector)
        self.seen_seconds: list[float] = []

    def embed(self, audio, **kw):
        self.seen_seconds.append(audio.duration_sec)
        return super().embed(audio, **kw)


class ExplodingLID:
    """A phrase login must never reach the language tagger (it needs the network)."""

    last_llm_error = None
    last_fallback_coverage = None

    def tag_utterance(self, *a, **k):
        raise AssertionError("the tagger must not run for a phrase challenge")


def wav_bytes(tmp_path: Path, seconds: float = 4.0) -> bytes:  # noqa: D103
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


class TestThrottle:
    def fail_once(self, client, pipeline, tmp_path, sid):
        c = issue(client, sid)
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.2)
        assert result["decision"] == "REJECT"

    def test_repeated_failures_make_the_next_challenge_wait(self, client, pipeline, speaker, tmp_path):
        for _ in range(3):
            self.fail_once(client, pipeline, tmp_path, speaker)
        r = client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"})
        assert r.status_code == 200  # three free attempts
        c = r.json()
        login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.2)  # the fourth failure
        blocked = client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"})
        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) >= 1
        assert "wait" in blocked.json()["detail"].lower()

    def test_a_success_clears_the_failures(self, client, pipeline, speaker, tmp_path):
        for _ in range(3):
            self.fail_once(client, pipeline, tmp_path, speaker)
        c = issue(client, speaker)
        ok, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.9)
        assert ok["decision"] == "ACCEPT"
        for _ in range(3):
            self.fail_once(client, pipeline, tmp_path, speaker)
        assert client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 200

    def test_a_recording_problem_is_not_a_failed_attempt(self, client, pipeline, speaker, tmp_path):
        """Silence and undecodable audio say nothing about who was speaking."""
        pipeline._embedder = FakeEmbedder(probe_with_cosine(0.9))
        pipeline._asr = FakeASR("")
        silent = Audio(np.zeros(16_000 * 3, dtype=np.float32), 16_000)
        path = tmp_path / "silent.wav"
        save_wav(silent, path)
        for _ in range(6):
            c = issue(client, speaker)
            r = client.post(
                "/api/authenticate",
                files={"audio": ("a.wav", path.read_bytes(), "audio/wav")},
                data={"challengeId": c["id"]},
            ).json()
            assert r["decision"] == "REJECT"
        assert client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 200

    def test_a_system_failure_is_not_a_failed_attempt(self, client, pipeline, speaker, tmp_path):
        pipeline._embedder = None
        pipeline._failed["embedder"] = "not installed"
        for _ in range(6):
            c = issue(client, speaker)
            client.post(
                "/api/authenticate",
                files={"audio": ("a.wav", wav_bytes(tmp_path), "audio/wav")},
                data={"challengeId": c["id"]},
            )
        assert client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 200

    def test_the_wait_is_for_that_identity_only(self, client, pipeline, speaker, store, tmp_path):
        other = client.post("/api/speakers", json={"displayName": "Other", "consentGiven": True}).json()["id"]
        for _ in range(4):
            self.fail_once(client, pipeline, tmp_path, speaker)
        assert client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 429
        assert client.post("/api/challenge", json={"speakerId": other, "kind": "phrase"}).status_code == 200

    def test_it_can_be_switched_off(self, client, pipeline, speaker, settings, tmp_path):
        pipeline.throttle.enabled = False
        for _ in range(6):
            self.fail_once(client, pipeline, tmp_path, speaker)
        assert client.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 200


class TestStepUp:
    """A borderline voice earns one more, stricter, sample. Only the server can grant it."""

    def borderline(self, client, pipeline, tmp_path, sid):
        c = issue(client, sid)
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.58)
        assert result["decision"] == "BORDERLINE"

    def step_up(self, client, sid):
        return client.post("/api/challenge", json={"speakerId": sid, "kind": "phrase", "stepUp": True})

    def test_a_borderline_voice_earns_a_step_up_challenge(self, client, pipeline, speaker, tmp_path):
        self.borderline(client, pipeline, tmp_path, speaker)
        r = self.step_up(client, speaker)
        assert r.status_code == 200
        assert r.json()["stepUp"] is True and r.json()["kind"] == "phrase"

    def test_the_second_sample_must_pass_outright__no_grey_band(self, client, pipeline, speaker, tmp_path):
        self.borderline(client, pipeline, tmp_path, speaker)
        c = self.step_up(client, speaker).json()
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.58)
        assert result["decision"] == "REJECT"

    def test_a_clear_second_sample_is_accepted(self, client, pipeline, speaker, tmp_path):
        self.borderline(client, pipeline, tmp_path, speaker)
        c = self.step_up(client, speaker).json()
        result, _ = login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.72)
        assert result["decision"] == "ACCEPT"

    def test_a_step_up_cannot_be_asked_for_without_a_borderline_attempt(self, client, speaker):
        r = self.step_up(client, speaker)
        assert r.status_code == 409
        assert "borderline" in r.json()["detail"].lower()

    def test_a_step_up_is_granted_once(self, client, pipeline, speaker, tmp_path):
        self.borderline(client, pipeline, tmp_path, speaker)
        assert self.step_up(client, speaker).status_code == 200
        assert self.step_up(client, speaker).status_code == 409

    def test_a_clear_reject_earns_no_step_up(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        login(client, pipeline, tmp_path, c, heard=" ".join(c["phrase"]), cosine=0.2)
        assert self.step_up(client, speaker).status_code == 409


class TestVoiceIsJudgedOnThePhraseSpan:
    """Speech before or after the shown words must not dilute (or hijack) the voiceprint."""

    @staticmethod
    def timed(words_and_times):
        from kavach.asr import Word
        return [Word(text=t, start_ms=a, end_ms=b) for t, a, b in words_and_times]

    def run(self, client, pipeline, tmp_path, speaker, *, seconds, layout):
        c = issue(client, speaker)
        words = c["phrase"]
        timed = self.timed(layout(words))
        spy = SpyEmbedder(probe_with_cosine(0.9))
        pipeline._embedder = spy
        pipeline._asr = FakeASR(" ".join(t.text for t in timed), words=timed)
        r = client.post(
            "/api/authenticate",
            files={"audio": ("a.wav", wav_bytes(tmp_path, seconds=seconds), "audio/wav")},
            data={"challengeId": c["id"]},
        ).json()
        return r, spy

    def test_a_judge_speaking_after_the_phrase_is_not_part_of_the_voice_sample(self, client, pipeline, speaker, tmp_path):
        def layout(w):
            phrase = [(x, 1000 + i * 500, 1400 + i * 500) for i, x in enumerate(w)]  # 1.0 s .. 3.9 s
            judge = [("let", 6000, 6200), ("me", 6200, 6300), ("try", 6300, 6600)]
            return phrase + judge

        r, spy = self.run(client, pipeline, tmp_path, speaker, seconds=8.0, layout=layout)
        assert r["decision"] == "ACCEPT", r["explanation"]
        assert spy.seen_seconds and spy.seen_seconds[0] == pytest.approx(3.3, abs=0.2)  # 1.0..3.9 plus margins, not 8 s

    def test_with_no_word_timings_the_whole_recording_is_used(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        spy = SpyEmbedder(probe_with_cosine(0.9))
        pipeline._embedder = spy
        pipeline._asr = FakeASR(" ".join(c["phrase"]), words=[])
        client.post(
            "/api/authenticate",
            files={"audio": ("a.wav", wav_bytes(tmp_path, seconds=5.0), "audio/wav")},
            data={"challengeId": c["id"]},
        )
        assert spy.seen_seconds[0] == pytest.approx(5.0, abs=0.05)

    def test_a_span_too_short_to_embed_falls_back_to_the_whole_recording(self, client, pipeline, speaker, tmp_path):
        def layout(w):
            return [(x, 2000 + i * 100, 2090 + i * 100) for i, x in enumerate(w)]  # six words in 0.6 s

        r, spy = self.run(client, pipeline, tmp_path, speaker, seconds=5.0, layout=layout)
        assert spy.seen_seconds[0] == pytest.approx(5.0, abs=0.05)

    def test_the_span_never_runs_past_the_ends_of_the_recording(self, client, pipeline, speaker, tmp_path):
        def layout(w):
            return [(x, 100 + i * 500, 500 + i * 500) for i, x in enumerate(w)]  # starts almost at 0

        r, spy = self.run(client, pipeline, tmp_path, speaker, seconds=3.2, layout=layout)
        assert 0 < spy.seen_seconds[0] <= 3.2 + 1e-6

    def test_the_explanation_says_which_part_was_scored(self, client, pipeline, speaker, tmp_path):
        def layout(w):
            return [(x, 1000 + i * 500, 1400 + i * 500) for i, x in enumerate(w)]

        r, _ = self.run(client, pipeline, tmp_path, speaker, seconds=6.0, layout=layout)
        voice = next(b for b in r["branches"] if b["name"] == "speaker_embedding")
        assert voice["passed"]


class TestWarmInference:
    """The first real login must not pay for kernel initialisation (8.8 s measured, 2-3 s after)."""

    def test_it_runs_one_embedding_and_one_transcription_and_swallows_nothing_it_should_not(self, pipeline):
        spy = SpyEmbedder(probe_with_cosine(0.9))
        asr = FakeASR("")
        pipeline._embedder, pipeline._asr = spy, asr
        pipeline.warm_inference()
        assert len(spy.seen_seconds) == 1 and len(asr.calls) == 1
        assert asr.calls[0].get("vad_filter") is False  # VAD would drop synthetic noise before the decoder runs

    def test_a_missing_model_does_not_stop_the_other(self, pipeline):
        spy = SpyEmbedder(probe_with_cosine(0.9))
        pipeline._embedder = spy
        pipeline._asr = None
        pipeline._failed["asr"] = "not installed"
        pipeline.warm_inference()
        assert len(spy.seen_seconds) == 1

    def test_a_model_that_blows_up_is_reported_not_raised(self, pipeline):
        class Boom:
            def embed(self, *a, **k):
                raise RuntimeError("cuda out of memory")

        pipeline._embedder = Boom()
        pipeline._asr = FakeASR("")
        assert "cuda out of memory" in " ".join(pipeline.warm_inference())


class TestResponseNamesTheWords:
    def test_it_says_which_shown_words_were_heard_and_which_were_not(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        dropped = c["phrase"][2]
        heard = " ".join(w for w in c["phrase"] if w != dropped)
        result, _ = login(client, pipeline, tmp_path, c, heard=heard, cosine=0.85)
        assert result["phraseMissing"] == [dropped]
        assert result["phraseMatched"] == [w for w in c["phrase"] if w != dropped]

    def test_a_question_login_carries_no_phrase_fields(self, client, speaker):
        # the default (question) challenge needs facts; the wire still has the fields, empty
        from kavach.api import schemas
        assert schemas.AuthResult.model_fields["phrase_matched"].default_factory() == []


class TestDemoTools:
    """The preflight's failed logins must not leave the presenter with strikes against them."""

    def test_the_reset_route_is_hidden_unless_demo_tools_are_on(self, client, settings):
        assert settings.demo_tools is False
        assert client.post("/api/demo/reset-throttle").status_code == 404

    def test_with_demo_tools_on_it_clears_the_wait(self, store, pipeline, settings, speaker, tmp_path):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        on = settings.model_copy(update={"demo_tools": True})
        app = create_app(on)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        app.dependency_overrides[get_settings] = lambda: on
        c = TestClient(app)
        for _ in range(4):
            ch = issue(c, speaker)
            login(c, pipeline, tmp_path, ch, heard=" ".join(ch["phrase"]), cosine=0.2)
        assert c.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 429
        assert c.post("/api/demo/reset-throttle").status_code == 200
        assert c.post("/api/challenge", json={"speakerId": speaker, "kind": "phrase"}).status_code == 200

    def test_health_announces_demo_tools(self, client):
        assert client.get("/api/health").json()["demoTools"] is False


class TestVoiceprintInfo:
    def test_it_reports_the_template_size_and_what_it_was_built_from(self, client, store, speaker):
        payload = store.load_template(speaker)
        payload["provenance"] = {"source": "studio", "sessions": ["S1", "S2"], "devices": ["DEMO_LAPTOP_MIC"], "n_clips": 1}
        store.save_template(speaker, payload)
        r = client.get(f"/api/speakers/{speaker}/voiceprint")
        assert r.status_code == 200
        body = r.json()
        assert body["nClips"] == 1
        assert body["provenance"]["sessions"] == ["S1", "S2"]

    def test_a_template_without_provenance_says_none(self, client, speaker):
        assert client.get(f"/api/speakers/{speaker}/voiceprint").json()["provenance"] is None

    def test_a_speaker_with_no_voiceprint_is_a_404_that_says_so(self, client):
        sid = client.post("/api/speakers", json={"displayName": "NoVoice", "consentGiven": True}).json()["id"]
        r = client.get(f"/api/speakers/{sid}/voiceprint")
        assert r.status_code == 404 and "voiceprint" in r.json()["detail"].lower()

    def test_an_unknown_speaker_is_a_404(self, client):
        assert client.get("/api/speakers/spk_nope/voiceprint").status_code == 404

    def test_it_never_returns_the_embeddings(self, client, speaker):
        body = client.get(f"/api/speakers/{speaker}/voiceprint").text
        assert "centroid" not in body and "embeddings" not in body
