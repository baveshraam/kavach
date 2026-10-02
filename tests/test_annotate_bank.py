from __future__ import annotations

import numpy as np
import pytest

from clone_helpers import add_clip, new_bank
from kavach.asr import Transcript, Word
from kavach.attacks.annotate_bank import annotate_bank
from kavach.attacks.bank import CloneBank
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.embedding import SpeakerEmbedding, SpeakerTemplate
from kavach.skg import SpeakerKG


class FakeASR:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio, fast=False):
        self.calls += 1
        words = [Word(w, i * 300, i * 300 + 250) for i, w in enumerate(self.text.split())]
        return Transcript(text=self.text, words=words)


class FakeLID:
    last_llm_error = None

    def tag_utterance(self, text, *, utterance_id, speaker_id=None, timings=None):
        return UtteranceTokens(
            utterance_id=utterance_id,
            tokens=[Token(w, Language.EN, SemanticClass.OTHER, 0.9) for w in text.split()],
            speaker_id=speaker_id,
            transcript=text,
        )


class FakeEmbedder:
    """Every clip embeds to `vector`; the template decides admissibility."""

    def __init__(self, vector) -> None:
        self.vector = np.asarray(vector, dtype=float)

    def embed(self, audio):
        return SpeakerEmbedding(self.vector)


class FakeMatcher:
    def match(self, answer, expected):
        class R:
            score = 1.0 if expected.lower() in answer.lower() else 0.1

        return R()


def template(vector):
    return SpeakerTemplate.from_embeddings("spk_victim", [SpeakerEmbedding(np.asarray(vector, float))])


def skg_with(**facts) -> SpeakerKG:
    kg = SpeakerKG("spk_victim")
    for pred, value in facts.items():
        kg.add_fact(pred, value)
    return kg


def run(bank, *, text="my hometown is Thanjavur", embed=(1, 0, 0), tmpl=(1, 0, 0), asr=None, **kw):
    return annotate_bank(
        bank,
        speaker_id="spk_victim",
        asr=asr or FakeASR(text),
        lid=FakeLID(),
        embedder=FakeEmbedder(embed),
        template=template(tmpl),
        matcher=FakeMatcher(),
        skg=skg_with(hometown="Thanjavur"),
        threshold=0.62,
        **kw,
    )


def test_a_clip_gets_measured_scores(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    report = run(bank)
    clip = CloneBank.load(bank.root, allowed_victims=["S04"]).clips[0]
    assert report.annotated == 1
    assert clip.ecapa_similarity == pytest.approx(1.0) and clip.admissible is True
    assert clip.answer_score == 1.0 and clip.transcript == "my hometown is Thanjavur"
    assert clip.tokens and clip.tokens[0]["language"] == Language.EN.value


def test_a_clone_the_voiceprint_stops_is_inadmissible_but_still_recorded(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, embed=(0, 1, 0))  # orthogonal to the template
    clip = bank.clips[0]
    assert clip.admissible is False and clip.ecapa_similarity == pytest.approx(0.0)
    assert bank.yield_summary().yield_rate == 0.0


def test_a_wrong_answer_scores_low(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, text="no idea honestly")
    assert bank.clips[0].answer_score == pytest.approx(0.1)


def test_a_translated_transcript_is_flagged_and_excluded(tmp_path) -> None:
    """Whisper writing fluent English for Tamil speech is a fabricated choice."""
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)

    # Transcript has slots, so override the method on a subclass.
    class T(Transcript):
        __slots__ = ()

        def looks_translated(self):
            return "no Tamil share"

    class Translated(FakeASR):
        def transcribe(self, audio, fast=False):
            base = super().transcribe(audio, fast)
            return T(text=base.text, words=base.words)

    report = run(bank, asr=Translated("my hometown is Thanjavur"))
    assert report.flagged == [bank.clips[0].clip_id]
    assert bank.clips[0].flags and bank.measured() == []


def test_foreign_script_tokens_are_dropped(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, text="my 안녕 hometown")
    assert all("안녕" != t["text"] for t in bank.clips[0].tokens)


def test_already_annotated_clips_are_skipped_unless_redone(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=True)
    asr = FakeASR("x")
    assert run(bank, asr=asr).skipped == 1 and asr.calls == 0
    assert run(bank, asr=asr, redo=True).annotated == 1 and asr.calls == 1


def test_progress_is_saved_after_every_clip(tmp_path) -> None:
    """A crash on clip 3 must not lose clips 1 and 2."""
    bank = new_bank(tmp_path)
    for _ in range(3):
        add_clip(bank, annotated=False)

    class DiesOnThird(FakeASR):
        def transcribe(self, audio, fast=False):
            if self.calls == 2:
                raise RuntimeError("boom")
            return super().transcribe(audio, fast)

    with pytest.raises(RuntimeError):
        run(bank, asr=DiesOnThird("my hometown is Thanjavur"))
    reloaded = CloneBank.load(bank.root, allowed_victims=["S04"])
    assert sum(c.annotated for c in reloaded.clips) == 2
