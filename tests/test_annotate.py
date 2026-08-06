"""Tests for the annotation pass.

Most of this file is about one negative result. The read-speech scripts give
every utterance a known reference, which looks like it should make ASR
scoreable -- and it does not, because the scripts romanise Tamil and Whisper
does not. Two attempts at a WER against them produced 281% and then 96%, and
both looked like numbers. The tests pin the refusal so a later change cannot
quietly reintroduce either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kavach import annotate as A
from kavach import corpus as C
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token


class TestNormalisation:
    def test_case_and_punctuation_are_dropped(self):
        assert A.normalise_for_wer("Naan, enga VEETLA!") == ["naan", "enga", "veetla"]

    def test_combining_and_composed_tamil_compare_equal(self):
        """Otherwise identical-looking words differ and the WER is inflated by
        an encoding detail rather than a transcription error."""
        composed = "நீ"
        decomposed = "ந" + "ீ"
        assert A.normalise_for_wer(composed) == A.normalise_for_wer(decomposed)

    def test_empty_text_is_no_words_not_one_empty_word(self):
        assert A.normalise_for_wer("   ") == []


class TestIsLatin:
    @pytest.mark.parametrize("word", ["family", "six-forty-five", "whatsapp'la", "Idli"])
    def test_latin_words(self, word):
        assert A.is_latin(word)

    @pytest.mark.parametrize("word", ["நான்", "வீட்ல", "அம்மா"])
    def test_tamil_script_words(self, word):
        assert not A.is_latin(word)

    def test_a_bare_numeral_has_no_letters_and_is_not_latin(self):
        """Which is the point: a digit carries no language evidence at all."""
        assert not A.is_latin("2004")


class TestWordErrorRate:
    def test_identical_text_scores_zero(self):
        assert A.word_error_rate("naan enga poren", "naan enga poren") == 0.0

    def test_one_substitution_in_three(self):
        assert A.word_error_rate("naan enga poren", "naan enga varen") == pytest.approx(1 / 3)

    def test_insertions_can_exceed_one(self):
        """A hallucinating ASR is not capped at 100%, and clamping it would
        hide exactly the failure worth seeing."""
        assert A.word_error_rate("naan", "naan naan naan naan") > 1.0

    def test_empty_reference_and_empty_hypothesis_is_zero(self):
        assert A.word_error_rate("", "") == 0.0

    def test_empty_reference_with_output_is_one_not_a_zero_division(self):
        assert A.word_error_rate("", "something") == 1.0

    def test_latin_only_filters_both_sides(self):
        assert A.word_error_rate("நான் enga poren", "வீட்ல enga poren", latin_only=True) == 0.0


class TestComparability:
    #: What a participant was asked to read: romanised Tamil with English nouns.
    REFERENCE = "Naan enga family kooda dhaan iruken. Enga veetla amma, appa irukkanga."
    #: What Whisper returns: Tamil in Tamil script, English left in Latin.
    HYPOTHESIS = "நான் எங்க family கூட தான் இருக்கேன். எங்க வீட்ல அம்மா, அப்பா இருக்காங்க."

    def test_a_romanised_reference_is_not_comparable_to_tamil_script_output(self):
        assert not A.transcripts_are_comparable(self.REFERENCE, self.HYPOTHESIS)

    def test_and_the_wer_it_would_have_produced_is_absurd(self):
        """Kept as a test so the magnitude is on record: this is what was
        nearly reported."""
        assert A.word_error_rate(self.REFERENCE, self.HYPOTHESIS) > 0.9

    def test_restricting_to_latin_does_not_rescue_it(self):
        """The obvious repair, and it is wrong: romanised Tamil *is* Latin, so
        the filter strips nothing from the reference and all the Tamil from the
        hypothesis."""
        assert A.word_error_rate(self.REFERENCE, self.HYPOTHESIS, latin_only=True) > 0.5

    def test_two_transcripts_in_the_same_orthography_are_comparable(self):
        assert A.transcripts_are_comparable(
            "I live with my parents", "I live with my parent"
        )
        assert A.transcripts_are_comparable(self.HYPOTHESIS, self.HYPOTHESIS)

    def test_latin_fraction_of_a_romanised_script_is_one(self):
        assert A.latin_fraction(self.REFERENCE) == 1.0

    def test_latin_fraction_of_mixed_output_is_between(self):
        assert 0.0 < A.latin_fraction(self.HYPOTHESIS) < 1.0

    def test_no_words_is_zero_rather_than_an_error(self):
        assert A.latin_fraction("") == 0.0


def _corpus_with_references(tmp_path: Path) -> C.Corpus:
    corpus = C.Corpus(name="t", provenance=C.Provenance.SCRIPTED, root=tmp_path)
    corpus.speakers.append(C.SpeakerRecord(speaker_id="S01", consent_ref="c", script_id="A"))
    corpus.sessions.append(C.SessionRecord(session_id="S01_s1", speaker_id="S01"))
    for prompt in C.PROTOCOL_V1[:3]:
        corpus.utterances.append(
            C.UtteranceRecord(
                utterance_id=f"S01_s1_{prompt.prompt_id}",
                session_id="S01_s1",
                speaker_id="S01",
                prompt_id=prompt.prompt_id,
                reference_transcript="Naan enga veetla iruken",
                transcript="நான் எங்க வீட்ல இருக்கேன்",
            )
        )
    return corpus


class TestTagging:
    def test_rules_only_tagging_marks_the_source_and_counts_guesses(self, tmp_path):
        corpus = _corpus_with_references(tmp_path)
        report = A.tag_corpus(corpus, pipeline=A.LIDPipeline())

        assert report.tagged == 3
        assert not report.used_llm
        for u in corpus.utterances:
            assert u.tokens is not None
            assert u.annotation_source is C.AnnotationSource.RULES

    def test_rules_only_output_is_never_corpus_grade(self, tmp_path):
        """Every token lands in SemanticClass.OTHER, so the CSBG has one class
        containing everything and every speaker's graph is identical."""
        corpus = _corpus_with_references(tmp_path)
        report = A.tag_corpus(corpus, pipeline=A.LIDPipeline())
        assert not report.is_corpus_grade

        classes = {t.semantic_class for u in corpus.utterances for t in u.tokens or []}
        assert classes == {SemanticClass.OTHER}

    def test_the_report_says_so_in_words(self, tmp_path):
        corpus = _corpus_with_references(tmp_path)
        report = A.tag_corpus(corpus, pipeline=A.LIDPipeline())
        assert "Not corpus-grade" in report.to_markdown()
        assert "ANTHROPIC_API_KEY" in report.to_markdown()

    def test_guessed_tokens_are_attributed_per_utterance_not_accumulated(self, tmp_path):
        """`PipelineStats.fallback_guesses` is a running total across the whole
        corpus, so writing it straight onto each record would give the last
        utterance everyone else's guesses. `Corpus.reportability` sums the
        per-utterance counts, and that sum has to equal the pipeline's total."""
        corpus = _corpus_with_references(tmp_path)
        for u in corpus.utterances:
            u.transcript = "naan enga veetla iruken"  # all Latin, so all guessed

        pipeline = A.LIDPipeline()
        A.tag_corpus(corpus, pipeline=pipeline)

        per_utterance = [u.n_guessed_tokens for u in corpus.utterances]
        assert all(n == per_utterance[0] > 0 for n in per_utterance)
        assert sum(per_utterance) == pipeline.stats.fallback_guesses

    def test_already_tagged_utterances_are_left_alone(self, tmp_path):
        corpus = _corpus_with_references(tmp_path)
        sentinel = [Token(text="x", language=Language.EN, semantic_class=SemanticClass.FOOD)]
        corpus.utterances[0].tokens = sentinel

        A.tag_corpus(corpus, pipeline=A.LIDPipeline())
        assert corpus.utterances[0].tokens == sentinel

    def test_force_retags_them(self, tmp_path):
        corpus = _corpus_with_references(tmp_path)
        corpus.utterances[0].tokens = []
        A.tag_corpus(corpus, force=True, pipeline=A.LIDPipeline())
        assert corpus.utterances[0].tokens

    def test_an_utterance_without_a_transcript_is_not_tagged(self, tmp_path):
        """`tokens is None` and `tokens == []` mean different things; an
        untranscribed utterance must stay in the first state."""
        corpus = _corpus_with_references(tmp_path)
        corpus.utterances[0].transcript = ""
        A.tag_corpus(corpus, pipeline=A.LIDPipeline())
        assert corpus.utterances[0].tokens is None


class TestTranslationReporting:
    """A translated transcript has to be loud in the report.

    It is the one defect here that looks like success: correct English, clean
    WER if no reference exists, and a graph that says the speaker chose English
    throughout. If the report does not name it, nothing else will.
    """

    def test_translated_utterances_appear_in_the_report(self):
        report = A.AnnotationReport(transcribed=2, tagged=0)
        report.translated.append(("S04_s1_p10_numbers", "detected language 'ta' but ..."))
        md = report.to_markdown()
        assert "S04_s1_p10_numbers" in md
        assert "translated" in md.lower()

    def test_the_report_says_not_to_annotate_them(self):
        """The instruction, not just the observation."""
        report = A.AnnotationReport(transcribed=1, tagged=0)
        report.translated.append(("S04_s1_p08_travel", "reason"))
        md = report.to_markdown()
        assert "do not annotate" in md.lower()
        assert "re-transcribe" in md.lower()

    def test_a_clean_run_says_nothing_about_translation(self):
        """No false alarm on the ordinary case."""
        report = A.AnnotationReport(transcribed=5, tagged=5)
        assert "translated" not in report.to_markdown().lower()

    def test_translation_is_tracked_separately_from_loops(self):
        """Different defects, different fixes; one list would hide that."""
        report = A.AnnotationReport(transcribed=2, tagged=0)
        report.degenerate.append(("u1", "Rs", 0.4))
        report.translated.append(("u2", "reason"))
        md = report.to_markdown()
        assert "u1" in md and "u2" in md
        assert md.count("do not annotate") == 2


class TestReportRoundTrip:
    def test_reference_and_asr_transcripts_survive_serialisation(self, tmp_path):
        """They are different fields on purpose: conflating them would make
        the reference the annotation input and any WER zero by construction."""
        corpus = _corpus_with_references(tmp_path)
        corpus.utterances[0].asr_wer = 0.25
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        reloaded = C.load_manifest(path)

        u = reloaded.utterances[0]
        assert u.reference_transcript == "Naan enga veetla iruken"
        assert u.transcript == "நான் எங்க வீட்ல இருக்கேன்"
        assert u.asr_wer == 0.25

    def test_absent_wer_stays_none_rather_than_becoming_zero(self, tmp_path):
        """None means 'not measurable'; 0.0 would mean 'perfect'."""
        corpus = _corpus_with_references(tmp_path)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        assert C.load_manifest(path).utterances[0].asr_wer is None


# --------------------------------------------------------------------------
# Partial runs
# --------------------------------------------------------------------------


def _taggable_corpus(n: int = 25) -> C.Corpus:
    corpus = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
    corpus.speakers.append(C.SpeakerRecord(speaker_id="S00", consent_ref="c/S00"))
    corpus.sessions.append(C.SessionRecord(session_id="S00_s1", speaker_id="S00"))
    for i in range(n):
        corpus.utterances.append(
            C.UtteranceRecord(
                utterance_id=f"u{i:02d}", session_id="S00_s1", speaker_id="S00",
                transcript="நான் morning six மணிக்கு",
            )
        )
    return corpus


class _CountingTagger:
    """Tags plausibly, and optionally dies on the Nth call."""

    supports_batch = False

    def __init__(self, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.retries = 0

    def tag(self, tokens, *, context=None):
        from kavach.lid.llm import TaggedToken

        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("429 rate limit")
        return [
            TaggedToken(
                text=t,
                language=Language.EN if t.isascii() else Language.TA,
                semantic_class=SemanticClass.OTHER,
                confidence=0.9,
            )
            for t in tokens
        ]


def _pipeline(tagger):
    from kavach.lid.pipeline import LIDPipeline

    return LIDPipeline(llm_tagger=tagger)


class TestPartialTaggingRuns:
    """A corpus pass against a free tier can die on the fiftieth utterance.

    Losing the first forty-nine turns a rate limit into an hour of re-tagging
    against the same rate limit, so what is already tagged has to survive the
    failure and there has to be a way to pick up from it.
    """

    def test_a_failure_saves_what_was_already_tagged(self, tmp_path):
        corpus = _taggable_corpus(25)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        tagger = _CountingTagger(fail_at=15)

        with pytest.raises(RuntimeError):
            A.tag_corpus(corpus, pipeline=_pipeline(tagger), manifest_path=path)

        reloaded = C.load_manifest(path)
        tagged = [u for u in reloaded.utterances if u.tokens]
        assert len(tagged) == 14

    def test_the_failure_says_how_far_it_got_and_how_to_resume(self, tmp_path):
        corpus = _taggable_corpus(25)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        with pytest.raises(RuntimeError, match=r"14 of 25"):
            A.tag_corpus(
                corpus, pipeline=_pipeline(_CountingTagger(fail_at=15)),
                manifest_path=path,
            )

    def test_the_failure_names_the_resume_flag(self, tmp_path):
        corpus = _taggable_corpus(25)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        with pytest.raises(RuntimeError, match=r"--resume"):
            A.tag_corpus(
                corpus, pipeline=_pipeline(_CountingTagger(fail_at=15)),
                manifest_path=path,
            )

    def test_resume_only_retags_what_is_not_llm_tagged(self, tmp_path):
        """--force would re-send the ones that succeeded, against the same rate
        limit that stopped the run."""
        corpus = _taggable_corpus(25)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        with pytest.raises(RuntimeError):
            A.tag_corpus(
                corpus, pipeline=_pipeline(_CountingTagger(fail_at=15)),
                manifest_path=path,
            )

        resumed = C.load_manifest(path)
        second = _CountingTagger()
        report = A.tag_corpus(
            resumed, resume=True, pipeline=_pipeline(second), manifest_path=path
        )
        assert report.tagged == 11
        assert second.calls == 11

    def test_resume_leaves_nothing_untagged(self, tmp_path):
        corpus = _taggable_corpus(25)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        with pytest.raises(RuntimeError):
            A.tag_corpus(
                corpus, pipeline=_pipeline(_CountingTagger(fail_at=15)),
                manifest_path=path,
            )
        resumed = C.load_manifest(path)
        A.tag_corpus(resumed, resume=True, pipeline=_pipeline(_CountingTagger()),
                     manifest_path=path)
        final = C.load_manifest(path)
        assert all(u.tokens for u in final.utterances)
        assert all(
            u.annotation_source is C.AnnotationSource.LLM for u in final.utterances
        )

    def test_resume_replaces_rules_only_tokens_from_a_smoke_test(self, tmp_path):
        """Plain `--stage tag` skips anything with tokens, and rules-only
        tokens from a no-key smoke test are exactly what needs replacing."""
        corpus = _taggable_corpus(3)
        for u in corpus.utterances:
            u.tokens = [Token(text="x", language=Language.TA,
                              semantic_class=SemanticClass.OTHER)]
            u.annotation_source = C.AnnotationSource.RULES

        skipped = _CountingTagger()
        A.tag_corpus(corpus, pipeline=_pipeline(skipped))
        assert skipped.calls == 0

        retagged = _CountingTagger()
        A.tag_corpus(corpus, resume=True, pipeline=_pipeline(retagged))
        assert retagged.calls == 3

    def test_force_still_retags_everything(self, tmp_path):
        corpus = _taggable_corpus(5)
        A.tag_corpus(corpus, pipeline=_pipeline(_CountingTagger()))
        again = _CountingTagger()
        A.tag_corpus(corpus, force=True, pipeline=_pipeline(again))
        assert again.calls == 5

    def test_a_clean_run_still_saves_once_at_the_end(self, tmp_path):
        corpus = _taggable_corpus(3)
        path = C.save_manifest(corpus, tmp_path / "manifest.json")
        A.tag_corpus(corpus, pipeline=_pipeline(_CountingTagger()), manifest_path=path)
        assert all(u.tokens for u in C.load_manifest(path).utterances)

    def test_no_manifest_path_says_nothing_was_saved(self, tmp_path):
        """Silently losing the work would be worse than saying so."""
        corpus = _taggable_corpus(25)
        with pytest.raises(RuntimeError, match="Nothing was saved"):
            A.tag_corpus(corpus, pipeline=_pipeline(_CountingTagger(fail_at=15)))

    def test_the_report_carries_the_retry_count(self, tmp_path):
        corpus = _taggable_corpus(3)
        tagger = _CountingTagger()
        tagger.retries = 7
        report = A.tag_corpus(corpus, pipeline=_pipeline(tagger))
        assert report.retries == 7
        assert "fighting a rate limit" in report.to_markdown()

    def test_the_report_carries_the_transliteration_count(self, tmp_path):
        """It is the size of a silent one-directional bias; it has to reach the
        report, not just the pipeline's stats object."""
        corpus = _taggable_corpus(2)
        pipeline = _pipeline(_CountingTagger())
        report = A.tag_corpus(corpus, pipeline=pipeline)
        assert report.transliteration_recovered == pipeline.stats.transliteration_recovered
