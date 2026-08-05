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
from kavach.lid import llm as llm_mod
from kavach.lid.llm import (
    PROVIDERS,
    PROVIDERS_BY_NAME,
    TaggedToken,
    available_providers,
    build_system_prompt,
    estimate_cost,
    make_tagger,
    to_tokens,
)
from kavach.lid.pipeline import LIDPipeline

ALL_KEY_VARS = sorted(
    {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}
    | {k for p in PROVIDERS for k in p.env_keys}
)


@pytest.fixture
def no_keys(monkeypatch):
    """A machine with no provider key anywhere, and no `.env` to find.

    Without the `_env_file_loaded` reset these tests would pass or fail
    depending on whether some earlier test had already triggered the one-shot
    load -- and without stubbing the loader they would read the developer's
    real `.env` and see their real key.
    """
    for var in ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm_mod, "_env_file_loaded", True)
    return monkeypatch

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


class _StubTagger:
    """Returns a fixed tag list, so merge behaviour can be tested without a key."""

    supports_batch = False

    def __init__(self, tags: list[TaggedToken]) -> None:
        self._tags = tags

    def tag(self, tokens, *, context=None):
        return self._tags


def _tag(text, language, confidence, semantic_class=SemanticClass.OTHER):
    return TaggedToken(
        text=text, language=language, semantic_class=semantic_class, confidence=confidence
    )


class TestTransliterationRecovery:
    """Whisper sometimes writes English in Tamil script.

    Observed on real returns: "morning six thirty" came back as "மானிங்க்
    சிக்ஸ் தெட்டி". Script rules then call it Tamil at confidence 1.0, and the
    rules-beat-LLM override used to make that final -- recording an English
    choice the speaker did make as a Tamil one, concentrated in NUMBER and
    TIME_DATE, which are among the most discriminative CSBG classes.
    """

    def test_confident_llm_english_overrules_tamil_script(self):
        text = "மானிங்க் சிக்ஸ்"
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([
                _tag("மானிங்க்", EN, 0.95, SemanticClass.TIME_DATE),
                _tag("சிக்ஸ்", EN, 0.95, SemanticClass.NUMBER),
            ])
        )
        tokens = pipeline.tag_utterance(text, utterance_id="u1").tokens

        assert [t.language for t in tokens] == [EN, EN]
        assert pipeline.stats.transliteration_recovered == 2

    def test_unconfident_llm_does_not_overrule_tamil_script(self):
        """Script stays the default. Only a confident model call displaces it."""
        text = "மானிங்க் சிக்ஸ்"
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([
                _tag("மானிங்க்", EN, 0.60),
                _tag("சிக்ஸ்", EN, 0.84),
            ])
        )
        tokens = pipeline.tag_utterance(text, utterance_id="u1").tokens

        assert [t.language for t in tokens] == [TA, TA]
        assert pipeline.stats.transliteration_recovered == 0

    def test_semantic_class_survives_an_override_either_way(self):
        """Rules never determine class, so the model's must be kept in both
        branches -- dropping it would empty the class the token belongs to."""
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([_tag("சிக்ஸ்", EN, 0.60, SemanticClass.NUMBER)])
        )
        tokens = pipeline.tag_utterance("சிக்ஸ்", utterance_id="u1").tokens

        assert tokens[0].language is TA  # script won
        assert tokens[0].semantic_class is SemanticClass.NUMBER  # class still the model's

    def test_agreement_is_not_counted_as_a_recovery(self):
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([_tag("வணக்கம்", TA, 1.0)])
        )
        pipeline.tag_utterance("வணக்கம்", utterance_id="u1")
        assert pipeline.stats.transliteration_recovered == 0

    def test_trust_rules_off_disables_the_override_entirely(self):
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([_tag("மானிங்க்", EN, 0.60)]),
            trust_rules_over_llm=False,
        )
        tokens = pipeline.tag_utterance("மானிங்க்", utterance_id="u1").tokens
        assert tokens[0].language is EN
        assert pipeline.stats.transliteration_recovered == 0

    def test_count_appears_in_the_summary(self):
        """It has to be visible; a silent one-directional bias is the problem."""
        pipeline = LIDPipeline(
            llm_tagger=_StubTagger([_tag("சிக்ஸ்", EN, 0.99)])
        )
        pipeline.tag_utterance("சிக்ஸ்", utterance_id="u1")
        assert "translit 1" in pipeline.stats.summary()


class TestProviderResolution:
    """Which tagger the environment produces.

    The failure this guards against is silent: no key resolves, the pipeline
    degrades to rules-only, rules never assign semantic class, every token
    becomes OTHER, and every speaker's graph comes out identical -- a corpus
    that annotates cleanly and separates nobody.
    """

    def test_no_key_anywhere_returns_none(self, no_keys):
        assert available_providers() == []
        assert make_tagger() is None

    def test_gemini_key_selects_gemini(self, no_keys):
        no_keys.setenv("GEMINI_API_KEY", "test-key")
        assert [p.name for p in available_providers()] == ["gemini"]
        assert make_tagger().provider.name == "gemini"

    def test_google_api_key_is_accepted_for_gemini(self, no_keys):
        """AI Studio hands out the same key under two names."""
        no_keys.setenv("GOOGLE_API_KEY", "test-key")
        assert [p.name for p in available_providers()] == ["gemini"]

    def test_anthropic_wins_when_present(self, no_keys):
        no_keys.setenv("ANTHROPIC_API_KEY", "test-key")
        no_keys.setenv("GEMINI_API_KEY", "test-key")
        assert type(make_tagger()).__name__ == "LLMTagger"

    def test_gemini_beats_groq_when_both_present(self, no_keys):
        """Ordering is a Tamil-quality judgement, not an accident. See PROVIDERS."""
        no_keys.setenv("GROQ_API_KEY", "test-key")
        no_keys.setenv("GEMINI_API_KEY", "test-key")
        assert make_tagger().provider.name == "gemini"

    def test_explicit_provider_overrides_preference(self, no_keys):
        no_keys.setenv("GEMINI_API_KEY", "test-key")
        no_keys.setenv("GROQ_API_KEY", "test-key")
        assert make_tagger("groq").provider.name == "groq"

    def test_unknown_provider_names_the_valid_ones(self, no_keys):
        with pytest.raises(ValueError, match="gemini"):
            make_tagger("gemeni")

    def test_openai_compatible_taggers_declare_no_batch_api(self):
        """`pipeline.tag_corpus` branches on this; a wrong default would make
        it submit a batch job to an endpoint that has none."""
        assert make_tagger("gemini", model="m").supports_batch is False


class TestDotenvLoading:
    """`.env` has to reach `os.environ`, which it did not used to.

    `config.Settings` reads the same file but only maps `KAVACH_`-prefixed
    names onto its own fields and exports nothing, so an unprefixed
    `GEMINI_API_KEY=...` on disk left `make_tagger()` returning None with the
    key sitting two directories up.
    """

    def test_env_file_key_becomes_visible(self, tmp_path, monkeypatch):
        for var in ALL_KEY_VARS:
            monkeypatch.delenv(var, raising=False)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(llm_mod, "_env_file_loaded", False)

        assert [p.name for p in available_providers()] == ["gemini"]
        assert PROVIDERS_BY_NAME["gemini"].api_key() == "from-dotenv"

    def test_real_environment_wins_over_the_file(self, tmp_path, monkeypatch):
        """An exported key is a deliberate act; a checked-out `.env` may be stale."""
        for var in ALL_KEY_VARS:
            monkeypatch.delenv(var, raising=False)
        (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("GEMINI_API_KEY", "from-environ")
        monkeypatch.setattr(llm_mod, "_env_file_loaded", False)

        llm_mod.load_api_keys()
        assert PROVIDERS_BY_NAME["gemini"].api_key() == "from-environ"

    def test_reads_the_file_once_not_per_call(self, monkeypatch):
        """`api_key()` runs on every provider on every lookup; without the
        one-shot guard each of those would re-read the filesystem."""
        import dotenv

        reads = []
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: reads.append(1))
        monkeypatch.setattr(llm_mod, "_env_file_loaded", False)

        llm_mod.load_api_keys()
        llm_mod.load_api_keys()
        available_providers()

        assert len(reads) == 1

    def test_missing_env_file_is_not_an_error(self, tmp_path, monkeypatch):
        """A fresh checkout has no `.env`; that is the normal state."""
        for var in ALL_KEY_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(llm_mod, "_env_file_loaded", False)

        llm_mod.load_api_keys()
        assert make_tagger() is None


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


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


class _Status(Exception):
    """An SDK-shaped API error. Both anthropic and openai set `status_code`."""

    def __init__(self, status: int, headers: dict | None = None) -> None:
        super().__init__(f"status {status}")
        self.status_code = status
        if headers is not None:
            self.response = type("R", (), {"headers": headers})()


class TestIsTransient:
    """Which failures are worth spending another request on.

    Getting this wrong in either direction costs: retrying a deterministic
    failure burns a free tier's quota to receive the same answer five more
    times, and not retrying a 429 aborts a corpus pass partway.
    """

    def test_rate_limit_is_transient(self):
        assert llm_mod.is_transient(_Status(429))

    def test_server_errors_are_transient(self):
        assert all(llm_mod.is_transient(_Status(s)) for s in (500, 502, 503, 504, 529))

    def test_client_errors_are_not(self):
        """A 401 will still be a 401 in eight seconds."""
        assert not llm_mod.is_transient(_Status(401))
        assert not llm_mod.is_transient(_Status(404))
        assert not llm_mod.is_transient(_Status(400))

    def test_a_timeout_is_not_retried_as_a_status(self):
        """408 is deliberately absent: the request may have been served, and
        re-sending a tagging call that succeeded costs quota for nothing."""
        assert 408 not in llm_mod.RETRYABLE_STATUS
        assert not llm_mod.is_transient(_Status(408))

    def test_alignment_errors_are_not_transient(self):
        """A token-count mismatch is deterministic. Retrying it five times gets
        the same mismatch five times."""
        assert not llm_mod.is_transient(ValueError("returned 4 tokens for 5 inputs"))

    def test_sdk_exception_names_are_recognised_without_importing_them(self):
        """The two SDKs raise different classes for the same condition, and
        importing either here would undo the lazy import."""
        for name in ("RateLimitError", "APITimeoutError", "APIConnectionError"):
            exc = type(name, (Exception,), {})()
            assert llm_mod.is_transient(exc), name


class TestWithRetries:
    def test_returns_on_first_success(self):
        assert llm_mod.with_retries(lambda: "ok") == "ok"

    def test_retries_until_it_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise _Status(429)
            return "ok"

        assert llm_mod.with_retries(flaky, sleep=lambda _: None) == "ok"
        assert len(calls) == 3

    def test_gives_up_and_raises_the_last_error(self):
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(_Status(429)),
                attempts=3, sleep=lambda _: None,
            )

    def test_makes_exactly_the_allowed_number_of_attempts(self):
        calls = []

        def always_fails():
            calls.append(1)
            raise _Status(503)

        with pytest.raises(_Status):
            llm_mod.with_retries(always_fails, attempts=4, sleep=lambda _: None)
        assert len(calls) == 4

    def test_does_not_retry_a_deterministic_failure(self):
        calls = []

        def bad_alignment():
            calls.append(1)
            raise ValueError("token alignment")

        with pytest.raises(ValueError):
            llm_mod.with_retries(bad_alignment, sleep=lambda _: None)
        assert len(calls) == 1

    def test_backoff_grows(self):
        slept: list[float] = []
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(_Status(429)),
                attempts=5, sleep=slept.append,
            )
        assert len(slept) == 4
        # Jittered, so compare the envelope rather than exact values.
        assert slept[-1] > slept[0]

    def test_jitter_makes_two_runs_differ(self):
        """Without jitter, requests rate-limited together retry together, hit
        the limit together, and the backoff accomplishes nothing."""
        def collect() -> list[float]:
            slept: list[float] = []
            with pytest.raises(_Status):
                llm_mod.with_retries(
                    lambda: (_ for _ in ()).throw(_Status(429)),
                    attempts=5, sleep=slept.append,
                )
            return slept

        assert collect() != collect()

    def test_honours_retry_after(self):
        """The provider knows when its window resets; guessing shorter just
        burns another attempt."""
        slept: list[float] = []
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(_Status(429, {"retry-after": "7"})),
                attempts=2, sleep=slept.append,
            )
        assert slept == [7.0]

    def test_caps_an_absurd_retry_after(self):
        """A provider asking for an hour should not hang the run for an hour."""
        slept: list[float] = []
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(_Status(429, {"retry-after": "3600"})),
                attempts=2, sleep=slept.append,
            )
        assert slept == [120.0]

    def test_an_unparseable_retry_after_falls_back_to_the_schedule(self):
        """HTTP-date form is legal and not worth parsing; the schedule is."""
        slept: list[float] = []
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(
                    _Status(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
                ),
                attempts=2, sleep=slept.append,
            )
        assert len(slept) == 1 and 0 < slept[0] <= llm_mod.RETRY_BASE_SECONDS

    def test_on_retry_is_told_what_happened(self):
        seen: list[tuple[int, float, str]] = []
        with pytest.raises(_Status):
            llm_mod.with_retries(
                lambda: (_ for _ in ()).throw(_Status(429)),
                attempts=3,
                on_retry=lambda a, d, e: seen.append((a, d, type(e).__name__)),
                sleep=lambda _: None,
            )
        assert [s[0] for s in seen] == [1, 2]


class TestOutputBudget:
    """A tagging response that runs out of output tokens is valid text and
    invalid JSON. The parser then reports "EOF while parsing a string at line
    106", which names neither the cause nor anything to do about it -- and this
    killed a real 56-utterance corpus pass on the second utterance."""

    def test_short_utterances_get_the_floor(self):
        assert llm_mod.output_budget(1) == llm_mod.DEFAULT_MAX_TOKENS
        assert llm_mod.output_budget(10) == llm_mod.DEFAULT_MAX_TOKENS

    def test_it_scales_with_the_utterance(self):
        assert llm_mod.output_budget(100) > llm_mod.output_budget(50)

    def test_the_real_utterance_that_broke_it_now_fits(self):
        """S01_s1_p02_food is 64 words and consumed the whole 4096 budget."""
        assert llm_mod.output_budget(64) > 4096

    def test_the_longest_utterance_in_the_corpus_fits(self):
        """S04_s1_p03_commute is 98 words. Tamil JSON-escapes to six characters
        per character, so the per-word cost is far above what ASCII suggests."""
        assert llm_mod.output_budget(98) >= 96 * 98

    def test_it_never_returns_less_than_the_floor(self):
        assert all(llm_mod.output_budget(n) >= llm_mod.DEFAULT_MAX_TOKENS
                   for n in range(0, 200, 7))
