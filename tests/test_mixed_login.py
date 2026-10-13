"""The experimental Tamil-English login through the real API and pipeline, with a fake recogniser and voice.

What this proves: the gates are wired, fail closed, name the stage they failed at, and leave the English login
alone. What it cannot prove: that a real recogniser hears real Tamil words, or that a real voice clears the
gate. Those are measured on recordings (`kavach.mixed_eval`) and reported there, not here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.asr import Transcript
from kavach.audio import decode_bytes
from kavach.config import Settings
from kavach.tamil_words import BANK, is_tamil_token

# Fixtures and fakes are the English login's own: the mixed login must pass through the same machinery.
from test_mixed_phrase import timeline
from test_phrase_login import (  # noqa: F401  (fixtures are used by name)
    FakeASR,
    FakeEmbedder,
    branch,
    client,
    issue,
    pipeline,
    probe_with_cosine,
    speaker,
    store,
    wav_bytes,
)


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path, audio_dir=tmp_path / "raw", attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db", offline=False, mixed_enabled=True,
    )


class LangASR:
    """A recogniser that answers each language's decode with its own transcript, and remembers how it was asked."""

    def __init__(self, english: Transcript, tamil: Transcript) -> None:
        self.by_language = {"en": english, "ta": tamil}
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return self.by_language[kwargs["language"]]


def issue_mixed(client: TestClient, sid: str, **extra) -> dict:
    r = client.post("/api/challenge", json={"speakerId": sid, "kind": "mixed", **extra})
    assert r.status_code == 200, r.text
    return r.json()


def say(client, pipeline, tmp_path, challenge, *, keep=lambda i, t: True, cosine=0.85, audio_bytes=None, asr=None):
    en, ta = timeline(challenge["phrase"], keep)
    pipeline._embedder = FakeEmbedder(probe_with_cosine(cosine))
    pipeline._asr = asr or LangASR(en, ta)
    r = client.post(
        "/api/authenticate",
        files={"audio": ("a.wav", audio_bytes or wav_bytes(tmp_path), "audio/wav")},
        data={"challengeId": challenge["id"]},
    )
    assert r.status_code == 200, r.text
    return r.json(), pipeline._asr


class TestTheExperimentIsOptIn:
    def test_the_shipped_default_is_off(self):
        assert Settings.model_fields["mixed_enabled"].default is False

    def test_when_off_a_mixed_challenge_is_refused_and_says_why(self, client, pipeline, speaker, settings):
        pipeline.settings = settings.model_copy(update={"mixed_enabled": False})
        r = client.post("/api/challenge", json={"speakerId": speaker, "kind": "mixed"})
        assert r.status_code == 409 and "switched off" in r.json()["detail"]

    def test_the_english_login_is_unaffected_by_the_flag(self, client, speaker):
        c = issue(client, speaker)
        assert c["kind"] == "phrase" and len(c["phrase"]) == 10 and not any(is_tamil_token(w) for w in c["phrase"])


class TestIssuingAMixedChallenge:
    def test_ten_english_words_and_two_tamil_words_in_one_phrase(self, client, speaker):
        c = issue_mixed(client, speaker)
        assert c["kind"] == "mixed" and len(c["phrase"]) == 12
        assert sum(is_tamil_token(w) for w in c["phrase"]) == 2
        assert all(w in c["questionText"] for w in c["phrase"])

    def test_the_tamil_words_move_around_and_every_challenge_is_new(self, client, speaker):
        cs = [issue_mixed(client, speaker) for _ in range(25)]
        assert len({tuple(c["phrase"]) for c in cs}) == 25
        assert len({tuple(i for i, w in enumerate(c["phrase"]) if is_tamil_token(w)) for c in cs}) > 8

    def test_an_anonymous_mixed_login_is_allowed_like_the_english_one(self, client, speaker):
        assert issue_mixed(client, "anonymous")["kind"] == "mixed"

    def test_a_step_up_is_not_available_without_a_borderline_attempt(self, client, speaker):
        r = client.post("/api/challenge", json={"speakerId": speaker, "kind": "mixed", "stepUp": True})
        assert r.status_code == 409


class TestVerifyingAMixedChallenge:
    def test_the_right_voice_saying_the_whole_challenge_is_accepted(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        result, asr = say(client, pipeline, tmp_path, c)
        assert result["decision"] == "ACCEPT", result["explanation"]
        assert branch(result, "phrase")["passed"] and branch(result, "speaker_embedding")["passed"]
        assert [k["language"] for k in asr.calls] == ["en", "ta"]  # two decodes, English then Tamil

    def test_the_english_login_still_makes_exactly_one_english_decode(self, client, pipeline, speaker, tmp_path):
        c = issue(client, speaker)
        pipeline._embedder = FakeEmbedder(probe_with_cosine(0.85))
        pipeline._asr = FakeASR(" ".join(c["phrase"]))
        r = client.post("/api/authenticate", files={"audio": ("a.wav", wav_bytes(tmp_path), "audio/wav")}, data={"challengeId": c["id"]})
        assert r.json()["decision"] == "ACCEPT"
        assert [k.get("language") for k in pipeline._asr.calls] == ["en"]

    @pytest.mark.parametrize("which", [0, 1])
    def test_a_perfect_voice_cannot_make_up_for_an_unheard_tamil_word(self, client, pipeline, speaker, tmp_path, which):
        c = issue_mixed(client, speaker)
        tamil_at = [i for i, w in enumerate(c["phrase"]) if is_tamil_token(w)]
        result, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: i != tamil_at[which], cosine=0.99)
        assert result["decision"] == "REJECT"
        assert not branch(result, "phrase")["passed"] and branch(result, "speaker_embedding")["passed"]
        assert "tamil words" in " ".join(result["explanation"]).lower()

    def test_reading_only_the_english_words_is_rejected(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: not is_tamil_token(t), cosine=0.99)
        assert result["decision"] == "REJECT" and not branch(result, "phrase")["passed"]

    def test_a_different_voice_is_rejected_even_when_every_word_is_right(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, cosine=0.25)
        assert result["decision"] == "REJECT"
        assert branch(result, "phrase")["passed"] and not branch(result, "speaker_embedding")["passed"]
        assert "voice does not match" in " ".join(result["explanation"])

    def test_a_voice_just_under_the_threshold_is_borderline_never_accepted(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, cosine=0.58)
        assert result["decision"] == "BORDERLINE"

    def test_a_tamil_word_read_in_the_wrong_place_is_rejected(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        en, ta = timeline(c["phrase"], lambda i, t: True)
        from kavach.asr import Word

        w = ta.words[0]
        ta.words[0] = Word(text=w.text, start_ms=60_000, end_ms=60_400, probability=0.9)  # a minute after the English
        result, _ = say(client, pipeline, tmp_path, c, cosine=0.99, asr=LangASR(en, ta))
        assert result["decision"] == "REJECT" and "order" in " ".join(result["explanation"]).lower()

    def test_no_speech_recognition_means_no_mixed_login__fail_closed(self, client, pipeline, speaker, tmp_path, settings):
        c = issue_mixed(client, speaker)
        pipeline._embedder = FakeEmbedder(probe_with_cosine(0.99))
        pipeline._asr = None
        pipeline._failed["asr"] = "model missing"
        pipeline.settings = settings.model_copy(update={"offline": True})
        r = client.post("/api/authenticate", files={"audio": ("a.wav", wav_bytes(tmp_path), "audio/wav")}, data={"challengeId": c["id"]})
        assert r.json()["decision"] == "REJECT"

    def test_the_voice_score_and_the_words_result_are_separate_branches(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, cosine=0.85, keep=lambda i, t: i != 0)
        phrase, voice = branch(result, "phrase"), branch(result, "speaker_embedding")
        assert phrase is not None and voice is not None and phrase is not voice
        assert voice["score"] == pytest.approx(0.85, abs=1e-3)  # the voice was scored even though the words failed


class TestTamilNotEnforcedLogin:
    @pytest.fixture()
    def relaxed(self, pipeline, settings):
        pipeline.settings = settings.model_copy(update={"mixed_check_tamil": False})

    def test_the_shipped_default_is_strict(self):
        assert Settings.model_fields["mixed_check_tamil"].default is True

    def test_with_it_off_the_right_voice_and_the_english_words_are_accepted_and_the_tamil_words_are_reported(self, client, pipeline, speaker, tmp_path, relaxed):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: not is_tamil_token(t), cosine=0.85)
        assert result["decision"] == "ACCEPT", result["explanation"]
        assert "Tamil words: 0 of 2 heard" in " ".join(result["explanation"])  # the Tamil words are reported, heard or not

    def test_with_it_off_a_wrong_voice_is_still_rejected(self, client, pipeline, speaker, tmp_path, relaxed):
        c = issue_mixed(client, speaker)
        result, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: not is_tamil_token(t), cosine=0.25)
        assert result["decision"] == "REJECT" and not branch(result, "speaker_embedding")["passed"]

    def test_with_it_off_other_english_words_are_still_rejected(self, client, pipeline, speaker, tmp_path, relaxed):
        old, new = issue_mixed(client, speaker), issue_mixed(client, speaker)
        en, ta = timeline(old["phrase"], lambda i, t: True)
        result, _ = say(client, pipeline, tmp_path, new, cosine=0.99, asr=LangASR(en, ta))
        assert result["decision"] == "REJECT" and not branch(result, "phrase")["passed"]

    def test_with_it_off_replay_and_freshness_still_hold(self, client, pipeline, speaker, tmp_path, relaxed):
        c = issue_mixed(client, speaker)
        first, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: not is_tamil_token(t))
        again, _ = say(client, pipeline, tmp_path, c, keep=lambda i, t: not is_tamil_token(t))
        assert first["decision"] == "ACCEPT" and again["decision"] == "REJECT"
        stored = wav_bytes(tmp_path)
        pipeline.remember_recording(decode_bytes(stored, suffix=".wav"), label="enrolment")
        out, asr = say(client, pipeline, tmp_path, issue_mixed(client, speaker), audio_bytes=stored, keep=lambda i, t: not is_tamil_token(t))
        assert out["decision"] == "REJECT" and asr.calls == []


class TestReplayAndFreshness:
    def test_a_challenge_can_be_answered_once(self, client, pipeline, speaker, tmp_path):
        c = issue_mixed(client, speaker)
        first, _ = say(client, pipeline, tmp_path, c)
        again, _ = say(client, pipeline, tmp_path, c)
        assert first["decision"] == "ACCEPT" and again["decision"] == "REJECT"
        assert "already been used" in " ".join(again["explanation"]) or "replay" in " ".join(again["explanation"]).lower()

    def test_a_recording_of_an_earlier_challenge_does_not_answer_a_new_one(self, client, pipeline, speaker, tmp_path):
        old, new = issue_mixed(client, speaker), issue_mixed(client, speaker)
        en, ta = timeline(old["phrase"], lambda i, t: True)  # the recording says the OLD phrase
        result, _ = say(client, pipeline, tmp_path, new, cosine=0.99, asr=LangASR(en, ta))
        assert result["decision"] == "REJECT" and not branch(result, "phrase")["passed"]

    def test_a_stored_enrolment_recording_replayed_is_stopped_before_any_recognition(self, client, pipeline, speaker, tmp_path):
        stored = wav_bytes(tmp_path)
        pipeline.remember_recording(decode_bytes(stored, suffix=".wav"), label="enrolment")
        c = issue_mixed(client, speaker)
        result, asr = say(client, pipeline, tmp_path, c, audio_bytes=stored, cosine=0.99)
        assert result["decision"] == "REJECT"
        assert asr.calls == []  # the integrity gate ran first: no recogniser, no voiceprint spent on it

    def test_failed_mixed_attempts_slow_the_account_like_any_other(self, client, pipeline, speaker, tmp_path):
        for _ in range(4):
            c = issue_mixed(client, speaker)
            say(client, pipeline, tmp_path, c, cosine=0.2)
        r = client.post("/api/challenge", json={"speakerId": speaker, "kind": "mixed"})
        assert r.status_code == 429
