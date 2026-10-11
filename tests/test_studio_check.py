"""The Studio's 'did you say it?' check: ticks the shown words as Whisper hears them, saves nothing.

It exists so a recording session gives the presenter a visible confirmation after every clip, and so a clip
where the words were not said can be redone before it is kept.
"""

from __future__ import annotations

from kavach.asr import Transcript
from test_studio_api import FORM, studio, wav_bytes

WORDS = "bottle,dragon,peacock,window,village,mirror,mango,jewel,festival,eagle"


class FakeASR:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return Transcript(text=self.text)


def online(tmp_path, asr):
    """A Studio client whose speech recognition is `asr` (the suite is offline unless a test says otherwise)."""
    client, _, pipeline, settings = studio(tmp_path, offline=False)
    pipeline._asr = asr
    return client, settings


def check(client, data: bytes, **over):
    form = {"speaker": FORM["speaker"], "words": WORDS, **over}
    return client.post("/api/studio/check", files={"audio": ("c.wav", data, "audio/wav")}, data=form)


def test_everything_is_a_404_while_the_studio_is_off(tmp_path) -> None:
    from test_clone_bank_api import build_client

    client, *_ = build_client(tmp_path)
    assert check(client, wav_bytes(tmp_path)).status_code == 404


def test_a_speaker_not_on_the_allowlist_is_a_403(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert check(client, wav_bytes(tmp_path), speaker="S09").status_code == 403


def test_all_the_words_heard_is_ok_and_every_word_is_ticked(tmp_path) -> None:
    client, _ = online(tmp_path, FakeASR("Bottle, dragon, peacock, window, village, mirror, mango, jewel, festival, eagle."))
    body = check(client, wav_bytes(tmp_path)).json()
    assert body["available"] is True and body["ok"] is True
    assert body["matched"] == 10 and body["total"] == 10
    assert body["matchedWords"] == WORDS.split(",") and body["missingWords"] == []


def test_a_clip_with_too_few_words_is_not_ok_and_names_the_missing_ones(tmp_path) -> None:
    client, _ = online(tmp_path, FakeASR("bottle dragon peacock window"))
    body = check(client, wav_bytes(tmp_path)).json()
    assert body["ok"] is False and body["matched"] == 4
    assert body["missingWords"] == ["village", "mirror", "mango", "jewel", "festival", "eagle"]
    assert body["heard"] == "bottle dragon peacock window"


def test_it_uses_the_same_bar_as_the_login(tmp_path) -> None:
    """Six of ten in order clears it: the same share the Unlock login needs."""
    client, settings = online(tmp_path, FakeASR("bottle dragon peacock window village mirror"))
    assert check(client, wav_bytes(tmp_path)).json()["ok"] is (6 / 10 >= settings.phrase_min_match)


def test_it_transcribes_like_the_login_does(tmp_path) -> None:
    """English, no code-mix prompt: a different setting would measure a different thing."""
    asr = FakeASR("bottle")
    client, _ = online(tmp_path, asr)
    check(client, wav_bytes(tmp_path))
    assert asr.calls[0]["language"] == "en" and asr.calls[0]["initial_prompt"] == ""


def test_checking_saves_nothing(tmp_path) -> None:
    client, settings = online(tmp_path, FakeASR("bottle"))
    check(client, wav_bytes(tmp_path))
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_without_speech_recognition_it_says_so_instead_of_failing(tmp_path) -> None:
    """Recording must never be blocked by a missing model: the UI then just saves."""
    client, *_ = studio(tmp_path, offline=True)
    r = check(client, wav_bytes(tmp_path))
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False and body["ok"] is None and body["reason"]


def test_empty_upload_is_a_400_and_no_words_is_a_422(tmp_path) -> None:
    client, _ = online(tmp_path, FakeASR("bottle"))
    assert check(client, b"").status_code == 400
    assert check(client, wav_bytes(tmp_path), words=" , ").status_code == 422
