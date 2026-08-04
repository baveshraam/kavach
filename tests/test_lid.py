"""Tests for word-level language ID.

LID accuracy is the accuracy ceiling for everything downstream: a token tagged
EN that was really Tamil is a fabricated switch point, which corrupts both the
CSBG and the published I-index. These tests pin the rule cascade; the LLM
stage is validated separately against the hand-annotated set (see
tests/test_lid_validation.py once the corpus exists).
"""

from __future__ import annotations

import pytest

from kavach.csbg.ontology import Language, SemanticClass
from kavach.lid import rules
from kavach.lid.llm import (
    TaggedToken,
    build_system_prompt,
    estimate_cost,
    to_tokens,
)
from kavach.lid.pipeline import LIDPipeline

TA, EN, NEU, NE = (
    Language.TA,
    Language.EN,
    Language.NEUTRAL,
    Language.NAMED_ENTITY,
)


class TestScriptDetection:
    def test_tamil_script(self):
        assert rules.script_of("நான்") == "tamil"
        assert rules.script_of("வணக்கம்") == "tamil"

    def test_latin_script(self):
        assert rules.script_of("college") == "latin"
        assert rules.script_of("Hello") == "latin"

    def test_no_letters(self):
        assert rules.script_of("123") == "none"
        assert rules.script_of("!?.") == "none"

    def test_mixed_resolves_by_majority(self):
        """'college-ல' is mostly Latin, so the stem dominates."""
        assert rules.script_of("collegeல") == "latin"

    def test_other_indic_detected(self):
        assert rules.script_of("नमस्ते") == "other_indic"


class TestTokenTagging:
    def test_tamil_script_is_certain(self):
        result = rules.tag_token("நான்")
        assert result.language is TA
        assert result.confidence == 1.0
        assert result.reason == "tamil_script"

    def test_digits_are_neutral(self):
        assert rules.tag_token("42").language is NEU
        assert rules.tag_token("3.14").language is NEU

    def test_punctuation_is_neutral(self):
        assert rules.tag_token("?").language is NEU
        assert rules.tag_token("...").language is NEU

    def test_romanised_tamil_from_lexicon(self):
        """The case a naive script check gets wrong."""
        for word in ["naan", "romba", "panren", "appuram", "irukku"]:
            result = rules.tag_token(word)
            assert result.language is TA, f"{word!r} should be Tamil, got {result}"

    def test_case_insensitive_lexicon(self):
        assert rules.tag_token("Naan").language is TA
        assert rules.tag_token("ROMBA").language is TA

    def test_lexicon_strips_trailing_punctuation(self):
        assert rules.tag_token("naan,").language is TA
        assert rules.tag_token("romba.").language is TA

    def test_unknown_latin_defers_to_llm(self):
        """Guessing here would bias every romanised token toward English."""
        result = rules.tag_token("serendipity")
        assert result.language is None
        assert result.reason == "latin_needs_llm"

    def test_homographs_defer_to_llm(self):
        """'enna' is Tamil for 'what' but also a proper name."""
        assert rules.tag_token("enna").language is None
        assert rules.tag_token("anna").language is None


class TestIntraWordCodeMixing:
    def test_english_stem_tamil_suffix_tags_as_english(self):
        """'college-la' -> EN.

        The stem carries the concept; the Tamil suffix supplies grammar. See
        the TAMIL_SUFFIXES docstring -- this is a stated linguistic choice
        that must be defended in the paper, not an accident.
        """
        result = rules.tag_token("college-la")
        assert result.language is EN
        assert result.reason == "english_stem_tamil_suffix"

    def test_tamil_stem_tamil_suffix_stays_tamil(self):
        assert rules.tag_token("veedu-la").language is TA

    def test_suffix_splitting(self):
        assert rules.strip_tamil_suffix("college-la") == ("college", "la")
        assert rules.strip_tamil_suffix("bus-ku") == ("bus", "ku")

    def test_non_suffix_hyphen_untouched(self):
        """A real hyphenated English word must not be split."""
        assert rules.strip_tamil_suffix("well-known") == ("well-known", None)

    def test_unhyphenated_agglutination_defers(self):
        """Splitting 'collegela' heuristically would mangle English words."""
        assert rules.tag_token("collegela").language is None


class TestTokenisation:
    def test_splits_on_whitespace(self):
        assert rules.simple_tokenise("naan college poren") == ["naan", "college", "poren"]

    def test_keeps_intra_word_hyphen(self):
        assert "college-la" in rules.simple_tokenise("naan college-la poren")

    def test_separates_punctuation(self):
        assert rules.simple_tokenise("naan poren?") == ["naan", "poren", "?"]

    def test_handles_tamil_script(self):
        assert rules.simple_tokenise("நான் போறேன்") == ["நான்", "போறேன்"]

    def test_mixed_script(self):
        assert rules.simple_tokenise("நான் college போறேன்") == ["நான்", "college", "போறேன்"]

    def test_empty(self):
        assert rules.simple_tokenise("") == []
        assert rules.simple_tokenise("   ") == []


class TestResolutionRate:
    def test_reports_share_resolved(self):
        results = rules.tag_tokens(["நான்", "42", "serendipity", "romba"])
        # Tamil script, digit, and lexicon hit resolve; the unknown does not.
        assert rules.resolution_rate(results) == pytest.approx(0.75)

    def test_empty_is_zero(self):
        assert rules.resolution_rate([]) == 0.0

    def test_realistic_sentence_mostly_resolves(self):
        """Cost sanity check -- the rules stage must carry real load."""
        sentence = "naan நேத்து college-la irundhen appuram 5 மணிக்கு veedu போனேன்"
        results = rules.tag_tokens(rules.simple_tokenise(sentence))
        assert rules.resolution_rate(results) > 0.8


class TestSystemPrompt:
    def test_deterministic(self):
        """Any variation silently breaks prompt caching."""
        assert build_system_prompt() == build_system_prompt()

    def test_contains_every_class(self):
        prompt = build_system_prompt()
        for cls_ in SemanticClass:
            assert cls_.value in prompt, f"{cls_.value} missing from tagging prompt"

    def test_contains_every_language_tag(self):
        prompt = build_system_prompt()
        for lang in Language:
            assert lang.value in prompt

    def test_long_enough_to_cache(self):
        """Below the model's minimum cacheable prefix, caching silently no-ops.

        Claude Opus 5's minimum is 512 tokens; ~4 chars/token puts the floor
        around 2048 characters. Verify the real hit rate at runtime via
        TaggingStats.cache_hit_rate -- this only catches gross regressions.
        """
        assert len(build_system_prompt()) > 2048

    def test_states_the_romanised_tamil_rule(self):
        """The single most important instruction in the prompt."""
        prompt = build_system_prompt().lower()
        assert "romanis" in prompt and "naan" in prompt


class TestTokenConversion:
    def test_converts_and_preserves_fields(self):
        tagged = [
            TaggedToken(text="naan", language=TA, semantic_class=SemanticClass.FUNCTION_WORD, confidence=0.95),
            TaggedToken(text="college", language=EN, semantic_class=SemanticClass.EDU_WORK, confidence=1.0),
        ]
        tokens = to_tokens(tagged, timings=[(0, 300), (300, 800)])
        assert [t.text for t in tokens] == ["naan", "college"]
        assert tokens[0].language is TA
        assert tokens[1].semantic_class is SemanticClass.EDU_WORK
        assert tokens[1].start_ms == 300

    def test_missing_timings_default_to_zero(self):
        tagged = [TaggedToken(text="x", language=EN, semantic_class=SemanticClass.OTHER)]
        assert to_tokens(tagged)[0].start_ms == 0

    def test_confidence_clamped(self):
        """Structured-output schemas cannot express numeric bounds."""
        assert TaggedToken(text="x", language=EN, semantic_class=SemanticClass.OTHER, confidence=1.7).confidence == 1.0
        assert TaggedToken(text="x", language=EN, semantic_class=SemanticClass.OTHER, confidence=-0.5).confidence == 0.0


class TestPipelineWithoutLLM:
    """Rules-only mode: for plumbing tests, never for corpus annotation."""

    def test_tags_without_api_access(self):
        pipeline = LIDPipeline(llm_tagger=None)
        utt = pipeline.tag_utterance("naan நேத்து college poren", utterance_id="u1")
        assert len(utt.tokens) == 4
        assert utt.tokens[0].language is TA  # naan
        assert utt.tokens[1].language is TA  # Tamil script

    def test_flags_guessed_tokens(self):
        """Fallback guesses must be visible, not silent."""
        pipeline = LIDPipeline(llm_tagger=None)
        pipeline.tag_utterance("serendipity ephemeral", utterance_id="u1")
        assert pipeline.stats.fallback_guesses == 2
        assert not pipeline.stats.is_corpus_grade

    def test_clean_input_is_corpus_grade(self):
        pipeline = LIDPipeline(llm_tagger=None)
        pipeline.tag_utterance("நான் romba நல்லா irukken", utterance_id="u1")
        assert pipeline.stats.is_corpus_grade

    def test_guessed_tokens_have_low_confidence(self):
        """So a confidence floor can exclude them from the CSBG."""
        pipeline = LIDPipeline(llm_tagger=None)
        utt = pipeline.tag_utterance("serendipity", utterance_id="u1")
        assert utt.tokens[0].lid_confidence < 0.5

    def test_empty_transcript(self):
        pipeline = LIDPipeline(llm_tagger=None)
        utt = pipeline.tag_utterance("", utterance_id="u1")
        assert utt.tokens == []
        assert utt.utterance_id == "u1"

    def test_preserves_metadata(self):
        pipeline = LIDPipeline(llm_tagger=None)
        utt = pipeline.tag_utterance("naan poren", utterance_id="u7", speaker_id="spk_3")
        assert utt.speaker_id == "spk_3"
        assert utt.transcript == "naan poren"

    def test_stats_accumulate(self):
        pipeline = LIDPipeline(llm_tagger=None)
        pipeline.tag_utterance("naan poren", utterance_id="u1")
        pipeline.tag_utterance("நான் போறேன்", utterance_id="u2")
        assert pipeline.stats.total_tokens == 4


class TestCostEstimate:
    def test_caching_is_cheaper(self):
        assert (
            estimate_cost(25_000, cached=True)["total_usd"]
            < estimate_cost(25_000, cached=False)["total_usd"]
        )

    def test_batch_is_half(self):
        est = estimate_cost(25_000)
        assert est["total_usd_batch"] == pytest.approx(est["total_usd"] * 0.5, rel=0.01)

    def test_corpus_scale_is_affordable(self):
        """~30 speakers x 5 min is well under $100 -- the design premise."""
        assert estimate_cost(25_000)["total_usd_batch"] < 100.0
