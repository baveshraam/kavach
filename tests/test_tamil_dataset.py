"""The local Tamil-word dataset: text, language, transcription and an audio reference, and no audio in the index."""

from __future__ import annotations

import json

import pytest

from clone_helpers import tone
from kavach.asr import Transcript
from kavach.studio.plan import build_plan, estimate_minutes, mixed_items, tamil_word_items
from kavach.studio.store import KINDS, StudioStore
from kavach.tamil_dataset import build_dataset, heard_exactly
from kavach.tamil_words import BANK, by_id, load_recorded, words_index_path

KW = dict(session_id="S1", kind="tamil", device="DEMO_LAPTOP_MIC", environment="QUIET_ROOM", orig_bytes=b"RIFFxxxx", orig_ext=".webm")


class AnswerByOrder:
    """A recogniser that returns the next scripted transcription, whatever it is asked."""

    def __init__(self, *texts: str) -> None:
        self.texts, self.calls = list(texts), []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return Transcript(text=self.texts.pop(0), words=[])


def studio(tmp_path) -> StudioStore:
    return StudioStore(tmp_path / "studio" / "S04", "S04")


def record(s: StudioStore, word_id: str, seed: int):
    return s.add_clip(audio=tone(seconds=2.0, seed=seed), prompt_id=word_id, text_hint=by_id()[word_id].ta, **KW)


class TestHeardExactly:
    def test_the_word_alone_is_heard(self):
        assert heard_exactly("தண்ணீர்", "தண்ணீர்") and heard_exactly("தண்ணீர்", " தண்ணீர். ") and heard_exactly("அம்மா", "அம்மா அம்மா")

    def test_another_word_or_extra_tamil_is_not(self):
        assert not heard_exactly("அம்மா", "அப்பா") and not heard_exactly("அம்மா", "அம்மா வந்தாள்") and not heard_exactly("அம்மா", "")

    def test_a_latin_transcription_is_not_hearing_the_tamil_word(self):
        assert not heard_exactly("அம்மா", "amma") and not heard_exactly("அம்மா", "mother")


class TestBuild:
    def test_rows_hold_text_language_transcription_and_an_audio_reference(self, tmp_path):
        s = studio(tmp_path)
        a, b = BANK[0], BANK[1]
        ca, cb = record(s, a.word_id, 1), record(s, b.word_id, 2)
        summary = build_dataset(s, AnswerByOrder(a.ta, "வேறு"), tmp_path, model_name="fake")
        rows = load_recorded(tmp_path)
        assert [r["word_id"] for r in rows] == [a.word_id, b.word_id]
        r = rows[0]
        assert (r["text"], r["language"], r["romanised"], r["transcription"]) == (a.ta, "ta", a.roman, a.ta)
        assert r["heard"] is True and rows[1]["heard"] is False
        assert r["audio_ref"] == f"studio/S04/S1/{ca.clip_id}.wav" and (tmp_path / r["audio_ref"]).exists()
        assert r["sha256_wav"] == ca.sha256_wav and r["asr_model"] == "fake"
        assert summary["words_heard"] == 1 and summary["words_not_heard"] == [b.word_id]

    def test_the_index_holds_references_not_audio(self, tmp_path):
        s = studio(tmp_path)
        record(s, BANK[0].word_id, 1)
        build_dataset(s, AnswerByOrder(BANK[0].ta), tmp_path)
        raw = words_index_path(tmp_path).read_bytes()
        assert b"RIFF" not in raw and len(raw) < 2000
        assert all(isinstance(json.loads(l), dict) for l in raw.decode("utf-8").splitlines())

    def test_the_tamil_decode_is_what_transcribes_each_word(self, tmp_path):
        s = studio(tmp_path)
        record(s, BANK[0].word_id, 1)
        asr = AnswerByOrder(BANK[0].ta)
        build_dataset(s, asr, tmp_path)
        assert asr.calls[0]["language"] == "ta"

    def test_nothing_recorded_is_a_refusal_not_an_empty_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="kind 'tamil'"):
            build_dataset(studio(tmp_path), AnswerByOrder(), tmp_path)
        assert load_recorded(tmp_path) == []

    def test_a_clip_for_a_word_not_in_the_bank_is_reported_and_skipped(self, tmp_path):
        s = studio(tmp_path)
        s.add_clip(audio=tone(seconds=2.0), prompt_id="ta999", text_hint="x", **KW)
        record(s, BANK[0].word_id, 3)
        summary = build_dataset(s, AnswerByOrder(BANK[0].ta), tmp_path)
        assert summary["unknown_prompt_ids"] == ["ta999"] and summary["clips"] == 1

    def test_a_clip_whose_file_changed_since_it_was_indexed_stops_the_build(self, tmp_path):
        from kavach.studio.store import StudioError

        s = studio(tmp_path)
        c = record(s, BANK[0].word_id, 1)
        s.wav_path(c).write_bytes(s.wav_path(c).read_bytes()[:-100])
        with pytest.raises(StudioError):
            build_dataset(s, AnswerByOrder(BANK[0].ta), tmp_path)


class TestStudioPlans:
    def test_the_new_kinds_are_known_to_the_studio(self):
        assert "tamil" in KINDS and "mixed" in KINDS and KINDS[:4] == ("read", "free", "fact", "words")

    def test_the_tamil_plan_is_every_bank_word_once_in_a_session_fixed_order(self):
        items = tamil_word_items("S7")
        assert [i.kind for i in items] == ["tamil"] * len(BANK)
        assert sorted(i.prompt_id for i in items) == sorted(w.word_id for w in BANK)
        assert [i.prompt_id for i in tamil_word_items("S7")] == [i.prompt_id for i in items]
        assert [i.prompt_id for i in tamil_word_items("S8")] != [i.prompt_id for i in items]

    def test_the_mixed_plan_shows_twelve_words_with_two_in_tamil_and_is_repeatable(self):
        items = mixed_items("S7", 6)
        assert all(i.kind == "mixed" and i.text_ta.startswith("Say: ") for i in items)
        assert [i.text_ta for i in mixed_items("S7", 6)] == [i.text_ta for i in items]
        assert len({i.text_ta for i in items}) == 6
        assert all(len(i.text_ta[5:].split(", ")) == 12 for i in items)

    def test_the_default_plans_are_unchanged_by_the_new_modes(self):
        assert {i.kind for i in build_plan("S1", [])} == {"read", "free", "words"}
        assert {i.kind for i in build_plan("S1", [], voice_only=True)} == {"words"}
        assert {i.kind for i in build_plan("S1", [], mode="tamil")} == {"tamil"}
        assert {i.kind for i in build_plan("S1", [], mode="mixed")} == {"mixed"}
        assert estimate_minutes(tamil_word_items("S1")) > 0
