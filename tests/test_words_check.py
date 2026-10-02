"""Does speech recognition hear the presenter's six words? The login depends on it as much as on the voice."""

from __future__ import annotations

import pytest

from clone_helpers import tone
from kavach.asr import Transcript
from kavach.studio.store import StudioStore
from kavach.studio.words_check import check_words

KW = dict(device="DEMO_LAPTOP_MIC", environment="QUIET_ROOM", orig_bytes=b"x", orig_ext=".webm")


class ScriptedASR:
    """Hears whatever it is told, per call, in order."""

    def __init__(self, heard):
        self.heard = list(heard)
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return Transcript(text=self.heard.pop(0), words=[])


@pytest.fixture()
def studio(tmp_path):
    s = StudioStore(tmp_path / "studio" / "S04", "S04")
    for session in ("S1", "S2"):
        for k in range(3):
            s.add_clip(session_id=session, kind="words", prompt_id=f"words_{session}_{k:02d}",
                       text_hint="Say: tiger, river, mango, window, candle, silver",
                       audio=tone(seconds=6.0, seed=k + (0 if session == "S1" else 50)), **KW)
        s.add_clip(session_id=session, kind="read", prompt_id="demo1", audio=tone(seconds=6.0, seed=99), **KW)
    return s


def test_only_the_six_words_clips_are_checked(studio):
    asr = ScriptedASR(["tiger river mango window candle silver"] * 6)
    r = check_words(studio, asr=asr, threshold=0.67)
    assert r.n == 6 and len(asr.calls) == 6


def test_it_transcribes_the_way_the_login_does(studio):
    asr = ScriptedASR(["tiger river mango window candle silver"] * 6)
    check_words(studio, asr=asr, threshold=0.67)
    assert asr.calls[0].get("language") == "en" and asr.calls[0].get("initial_prompt") == ""


def test_the_pass_rate_is_the_share_that_would_have_cleared_the_gate(studio):
    heard = ["tiger river mango window candle silver"] * 4 + ["tiger river"] + ["something else entirely"]
    r = check_words(studio, asr=ScriptedASR(heard), threshold=0.67)
    assert r.n == 6 and r.passed == 4
    assert r.pass_rate == pytest.approx(4 / 6)


def test_failures_are_listed_with_what_was_expected_and_heard(studio):
    heard = ["tiger river mango window candle silver"] * 5 + ["tiger river"]
    r = check_words(studio, asr=ScriptedASR(heard), threshold=0.67)
    assert len(r.failures) == 1
    f = r.failures[0]
    assert "tiger" in f["expected"] and f["heard"] == "tiger river" and "mango" in f["missing"]


def test_words_that_are_missed_are_counted_across_clips(studio):
    heard = ["tiger river window candle silver"] * 6        # 'mango' is never heard
    r = check_words(studio, asr=ScriptedASR(heard), threshold=0.67)
    assert r.word_misses["mango"] == 6 and "tiger" not in r.word_misses


def test_the_report_says_what_to_do_when_the_rate_is_low(studio):
    r = check_words(studio, asr=ScriptedASR(["nothing"] * 6), threshold=0.67)
    assert r.pass_rate == 0.0
    assert "larger" in r.report().lower() or "fewer words" in r.report().lower()


def test_no_words_clips_is_an_error(tmp_path):
    s = StudioStore(tmp_path / "studio" / "S04", "S04")
    s.add_clip(session_id="S1", kind="read", prompt_id="demo1", audio=tone(seconds=6.0), **KW)
    with pytest.raises(ValueError, match="words"):
        check_words(s, asr=ScriptedASR([]), threshold=0.67)


def test_a_clip_changed_after_recording_is_refused(studio):
    victim = next(c for c in studio.clips() if c.kind == "words")
    studio.wav_path(victim).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        check_words(studio, asr=ScriptedASR(["x"] * 6), threshold=0.67)
