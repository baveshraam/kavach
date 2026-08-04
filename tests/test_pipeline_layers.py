"""Tests for audio, SKG, challenge generation, matching, and fusion.

Nothing here loads an ML model or calls an API -- ECAPA and Whisper are
wrapped behind lazy imports, and the LLM paths have offline fallbacks. The
whole file runs in a couple of seconds.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kavach.audio import (
    Audio,
    AudioError,
    check_quality,
    concatenate,
    normalise_peak,
    trim_silence,
)
from kavach.challenge import (
    Challenge,
    ChallengeError,
    ChallengeGenerator,
    ChallengeLedger,
    select_target,
)
from kavach.csbg.graph import CSBG
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.scoring import build_background_model
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.embedding import SpeakerEmbedding, SpeakerTemplate
from kavach.fusion import (
    Branch,
    BranchScore,
    Calibrator,
    Decision,
    FusionPolicy,
    build_liveness_branch,
    fuse,
)
from kavach.matcher import (
    AnswerMatcher,
    entity_match,
    exact_match,
    levenshtein,
    normalise,
    phonetic_key,
    phonetic_match,
)
from kavach.skg import FACT_TYPES, SKGStore, SpeakerKG, slugify

TA, EN = Language.TA, Language.EN


def sine(seconds: float, sr: int = 16000, freq: float = 220.0, amp: float = 0.3) -> Audio:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return Audio((amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)


# ==========================================================================
# Audio
# ==========================================================================


class TestAudio:
    def test_duration(self):
        assert sine(2.0).duration_sec == pytest.approx(2.0)

    def test_rejects_stereo(self):
        with pytest.raises(AudioError, match="mono"):
            Audio(np.zeros((100, 2), dtype=np.float32), 16000)

    def test_silence_detected(self):
        silent = Audio(np.zeros(16000, dtype=np.float32), 16000)
        assert check_quality(silent).is_silent
        assert not check_quality(silent).is_usable

    def test_clipping_detected(self):
        clipped = Audio(np.ones(16000, dtype=np.float32), 16000)
        report = check_quality(clipped)
        assert report.is_clipped
        assert any("clipped" in w for w in report.warnings)

    def test_short_audio_unusable(self):
        assert not check_quality(sine(0.3)).is_usable

    def test_peak_normalisation(self):
        assert normalise_peak(sine(1.0, amp=0.1)).peak == pytest.approx(0.95, abs=0.01)

    def test_normalising_silence_does_not_amplify_noise(self):
        silent = Audio(np.zeros(1000, dtype=np.float32), 16000)
        assert normalise_peak(silent).peak == 0.0

    def test_trim_silence(self):
        sr = 16000
        padded = np.concatenate(
            [np.zeros(sr, dtype=np.float32), sine(1.0).samples, np.zeros(sr, dtype=np.float32)]
        )
        trimmed = trim_silence(Audio(padded, sr))
        assert trimmed.duration_sec < 3.0
        assert trimmed.duration_sec > 0.9

    def test_trim_leaves_all_silent_audio_intact(self):
        """A silent clip must stay detectably silent, not become zero-length."""
        silent = Audio(np.zeros(16000, dtype=np.float32), 16000)
        assert len(trim_silence(silent).samples) == 16000

    def test_concatenate_for_splice_attack(self):
        joined = concatenate([sine(1.0), sine(1.0)], gap_ms=100)
        assert joined.duration_sec == pytest.approx(2.1, abs=0.01)

    def test_concatenate_rejects_mismatched_rates(self):
        with pytest.raises(AudioError, match="sample rate"):
            concatenate([sine(1.0, sr=16000), sine(1.0, sr=8000)])

    def test_content_hash_detects_identical_replay(self):
        a, b = sine(1.0), sine(1.0)
        assert a.sha256() == b.sha256()
        assert a.sha256() != sine(1.0, freq=440).sha256()


# ==========================================================================
# Speaker embeddings (no model load)
# ==========================================================================


class TestSpeakerEmbedding:
    def test_normalised_on_construction(self):
        emb = SpeakerEmbedding(np.array([3.0, 4.0]))
        assert np.linalg.norm(emb.vector) == pytest.approx(1.0)

    def test_zero_vector_rejected(self):
        """A zero embedding means silent audio, not an identity."""
        with pytest.raises(ValueError, match="Degenerate"):
            SpeakerEmbedding(np.zeros(192))

    def test_identical_vectors_score_one(self):
        v = np.random.default_rng(0).normal(size=192)
        assert SpeakerEmbedding(v).similarity(SpeakerEmbedding(v)) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = SpeakerEmbedding(np.array([1.0, 0.0]))
        b = SpeakerEmbedding(np.array([0.0, 1.0]))
        assert a.similarity(b) == pytest.approx(0.0)

    def test_template_centroid(self):
        rng = np.random.default_rng(1)
        base = rng.normal(size=192)
        embs = [SpeakerEmbedding(base + 0.05 * rng.normal(size=192)) for _ in range(5)]
        template = SpeakerTemplate.from_embeddings("s1", embs)
        assert template.score(embs[0]) > 0.9

    def test_template_rejects_empty(self):
        with pytest.raises(ValueError, match="no embeddings"):
            SpeakerTemplate.from_embeddings("s1", [])

    def test_self_consistency_flags_mixed_speakers(self):
        """Low consistency means the enrolment clips disagree on who this is."""
        rng = np.random.default_rng(2)
        consistent = SpeakerTemplate.from_embeddings(
            "s1", [SpeakerEmbedding(np.ones(192) + 0.01 * rng.normal(size=192)) for _ in range(4)]
        )
        mixed = SpeakerTemplate.from_embeddings(
            "s2", [SpeakerEmbedding(rng.normal(size=192)) for _ in range(4)]
        )
        assert consistent.self_consistency > mixed.self_consistency

    def test_template_roundtrip(self):
        rng = np.random.default_rng(3)
        t = SpeakerTemplate.from_embeddings(
            "s1", [SpeakerEmbedding(rng.normal(size=192)) for _ in range(3)]
        )
        restored = SpeakerTemplate.from_dict(t.to_dict())
        assert restored.speaker_id == "s1"
        np.testing.assert_allclose(restored.centroid.vector, t.centroid.vector)


# ==========================================================================
# Speaker Knowledge Graph
# ==========================================================================


class TestSpeakerKG:
    def test_add_and_get(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        assert kg.get("hometown").value == "Thanjavur"

    def test_facts_by_semantic_class(self):
        """The query that drives adaptive challenge targeting."""
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        kg.add_fact("favouriteFood", "kothu parotta")
        food = kg.facts_for_class(SemanticClass.FOOD)
        assert len(food) == 1
        assert food[0].value == "kothu parotta"

    def test_replacing_a_fact(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        kg.add_fact("hometown", "Madurai")
        assert len(kg) == 1
        assert kg.get("hometown").value == "Madurai"

    def test_remove_fact(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        assert kg.remove_fact("hometown")
        assert not kg.remove_fact("hometown")

    def test_completeness(self):
        kg = SpeakerKG("s1")
        assert kg.completeness == 0.0
        for ft in FACT_TYPES:
            kg.add_fact(ft.predicate, "x")
        assert kg.completeness == 1.0

    def test_slugify_preserves_tamil(self):
        """A Tamil-script answer must not collapse to an empty slug."""
        assert slugify("தஞ்சாவூர்") != "unknown"
        assert slugify("kothu parotta") == "kothu_parotta"
        assert slugify("!!!") == "unknown"

    def test_rdf_graph_builds(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        assert len(kg.graph) > 0

    def test_sparql_query_by_class(self):
        """Proves the SKG is queried as a graph, not used as a dict."""
        kg = SpeakerKG("s1")
        kg.add_fact("favouriteFood", "kothu parotta")
        kg.add_fact("hometown", "Thanjavur")
        rows = kg.query_by_class(SemanticClass.FOOD)
        assert ("favouriteFood", "kothu parotta") in rows
        assert not any(p == "hometown" for p, _ in rows)

    def test_graph_rebuilds_after_change(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        before = len(kg.graph)
        kg.add_fact("favouriteFood", "idli")
        assert len(kg.graph) > before

    def test_roundtrip(self):
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur", raw_answer="naan Thanjavur", verified=True)
        restored = SpeakerKG.from_dict(kg.to_dict())
        assert restored.get("hometown").value == "Thanjavur"
        assert restored.get("hometown").verified

    def test_anonymised_view_leaks_nothing(self):
        """This is what may ship with a released corpus."""
        kg = SpeakerKG("s1")
        kg.add_fact("hometown", "Thanjavur")
        kg.add_fact("siblingName", "Priya")
        anon = kg.anonymised()
        blob = str(anon)
        assert "Thanjavur" not in blob
        assert "Priya" not in blob
        assert anon["n_facts"] == 2
        assert "hometown" in anon["predicates"]


class TestSKGStore:
    def test_get_or_create(self):
        store = SKGStore()
        assert store.get_or_create("s1") is store.get_or_create("s1")

    def test_delete_supports_erasure_requests(self):
        store = SKGStore()
        store.get_or_create("s1").add_fact("hometown", "Thanjavur")
        assert store.delete("s1")
        assert store.get("s1") is None
        assert not store.delete("s1")


# ==========================================================================
# Challenge generation
# ==========================================================================


def make_skg(speaker_id: str = "s1") -> SpeakerKG:
    kg = SpeakerKG(speaker_id)
    kg.add_fact("hometown", "Thanjavur")
    kg.add_fact("favouriteFood", "kothu parotta")
    kg.add_fact("hostelRoom", "214")
    return kg


def utt(pairs, uid="u"):
    return UtteranceTokens(
        utterance_id=uid,
        tokens=[Token(f"w{i}", lang, cls_) for i, (lang, cls_) in enumerate(pairs)],
    )


class TestChallengeTargeting:
    def test_picks_most_discriminative_class(self):
        """The adaptive-targeting claim, tested directly.

        The population says numbers in English; this speaker says them in
        Tamil. NUMBER should be chosen, so the question is aimed at their tell.
        """
        population = [
            CSBG.build(f"p{i}", [utt([(EN, SemanticClass.NUMBER)] * 80 + [(TA, SemanticClass.FOOD)] * 80)])
            for i in range(5)
        ]
        odd = CSBG.build(
            "s1", [utt([(TA, SemanticClass.NUMBER)] * 80 + [(TA, SemanticClass.FOOD)] * 80)]
        )
        ubm = build_background_model(population)

        fact, target, jsd = select_target(odd, ubm, make_skg())
        assert target is SemanticClass.NUMBER
        assert fact.predicate == "hostelRoom"
        assert jsd > 0.2

    def test_falls_back_without_csbg(self):
        """First login after enrolment: targeting is an optimisation only."""
        fact, target, jsd = select_target(None, None, make_skg())
        assert fact is not None
        assert jsd == 0.0

    def test_respects_exclusions(self):
        _, target, _ = select_target(
            None, None, make_skg(), exclude_classes={SemanticClass.FOOD, SemanticClass.NUMBER}
        )
        assert target is SemanticClass.PLACE_LOCAL

    def test_empty_skg_raises(self):
        with pytest.raises(ChallengeError, match="no knowledge-graph facts"):
            select_target(None, None, SpeakerKG("empty"))


class TestChallengeLedger:
    def test_single_use(self):
        """Reuse would let an observed answer be replayed."""
        ledger = ChallengeLedger()
        gen = ChallengeGenerator(ledger=ledger)
        ch = gen.generate("s1", make_skg())

        assert ledger.consume(ch.id).id == ch.id
        with pytest.raises(ChallengeError, match="already been used"):
            ledger.consume(ch.id)

    def test_expiry_enforced(self):
        ledger = ChallengeLedger(ttl_seconds=0)
        gen = ChallengeGenerator(ledger=ledger, ttl_seconds=0)
        ch = gen.generate("s1", make_skg())
        time.sleep(0.01)
        with pytest.raises(ChallengeError, match="expired"):
            ledger.consume(ch.id)

    def test_unknown_id_rejected(self):
        with pytest.raises(ChallengeError, match="Unknown challenge"):
            ChallengeLedger().consume("chg_nope")

    def test_history_tracked_per_speaker(self):
        gen = ChallengeGenerator()
        gen.generate("s1", make_skg("s1"))
        assert len(gen.ledger.previously_asked("s1")) == 1
        assert not gen.ledger.previously_asked("s2")

    def test_forget_speaker(self):
        gen = ChallengeGenerator()
        ch = gen.generate("s1", make_skg())
        gen.ledger.forget_speaker("s1")
        assert gen.ledger.get(ch.id) is None
        assert not gen.ledger.previously_asked("s1")

    def test_purge_expired(self):
        ledger = ChallengeLedger(ttl_seconds=0)
        gen = ChallengeGenerator(ledger=ledger, ttl_seconds=0)
        gen.generate("s1", make_skg())
        time.sleep(0.01)
        assert ledger.purge_expired() == 1


class TestChallengeGeneration:
    def test_template_mode_needs_no_api(self):
        ch = ChallengeGenerator().generate("s1", make_skg())
        assert ch.question_text
        assert ch.generator == "template"
        assert ch.is_valid

    def test_public_dict_hides_the_answer(self):
        """Leaking the expected answer would hand over the knowledge factor.

        The leak check runs over the *text* fields only, not `str(public)`.
        That was the original form and it was intermittently wrong: the dict
        carries `issued_at` and `expires_at` as Unix timestamps, and one of the
        SKG facts here is a hostel room number, "214". A float timestamp is ten
        digits, so it contains any given three-digit run every few hundred
        seconds -- the test failed roughly once in a few full-suite runs, on a
        collision with the clock rather than on anything the code did.

        Substring-searching a serialised structure for a secret is the wrong
        shape of assertion in general: it is simultaneously too weak (an answer
        split across fields, or case-folded, slips through) and too strong (any
        numeric field can collide). Checking the fields that could actually
        carry it is both.
        """
        ch = ChallengeGenerator().generate("s1", make_skg())
        public = ch.public_dict()

        assert "expected_answer" not in public
        assert "expected_predicate" not in public

        text_fields = {
            k: v for k, v in public.items() if isinstance(v, str)
        }
        answer = ch.expected_answer.casefold()
        assert answer, "The challenge has no expected answer; nothing was tested."
        for key, value in text_fields.items():
            assert answer not in value.casefold(), (
                f"public_dict()[{key!r}] contains the expected answer "
                f"{ch.expected_answer!r}, which hands over the knowledge factor."
            )

    def test_llm_failure_falls_back_to_template(self):
        """A login must not fail because the LLM is unreachable."""

        class BrokenClient:
            class messages:
                @staticmethod
                def create(**_):
                    raise RuntimeError("network down")

        ch = ChallengeGenerator(llm_client=BrokenClient()).generate("s1", make_skg())
        assert ch.question_text
        assert ch.generator == "template_after_error"

    def test_llm_refusal_falls_back(self):
        class RefusingClient:
            class messages:
                @staticmethod
                def create(**_):
                    class R:
                        stop_reason = "refusal"
                        content: list = []
                        stop_details = None
                    return R()

        ch = ChallengeGenerator(llm_client=RefusingClient()).generate("s1", make_skg())
        assert ch.generator == "template_after_refusal"

    def test_llm_path_used_when_working(self):
        class GoodClient:
            class messages:
                @staticmethod
                def create(**_):
                    class Block:
                        type = "text"
                        text = "உங்க hostel room number என்ன?"

                    class R:
                        stop_reason = "end_turn"
                        content = [Block()]
                    return R()

        ch = ChallengeGenerator(llm_client=GoodClient()).generate("s1", make_skg())
        assert ch.generator == "llm"
        assert "hostel" in ch.question_text


# ==========================================================================
# Answer matching
# ==========================================================================


class TestNormalisation:
    def test_lowercase_and_strip_punctuation(self):
        assert normalise("Thanjavur!") == "thanjavur"

    def test_collapse_whitespace(self):
        assert normalise("kothu   parotta") == "kothu parotta"

    def test_preserves_tamil(self):
        assert "தஞ" in normalise("தஞ்சாவூர்")


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("abc", "abc") == 0

    def test_substitution(self):
        assert levenshtein("cat", "bat") == 1

    def test_empty(self):
        assert levenshtein("", "abc") == 3

    def test_symmetric(self):
        assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")


class TestMatchers:
    def test_exact_match_baseline(self):
        assert exact_match("Thanjavur", "thanjavur") == 1.0
        assert exact_match("Thanjavoor", "Thanjavur") == 0.0

    def test_exact_fails_on_conversational_answer(self):
        """The failure this module exists to fix -- and a paper result."""
        assert exact_match("naan Thanjavur-la irundhu varen", "Thanjavur") == 0.0

    def test_entity_match_handles_conversational_answer(self):
        assert entity_match("naan Thanjavur-la irundhu varen", "Thanjavur") == 1.0

    def test_entity_partial_credit_for_multiword(self):
        score = entity_match("naan kothu saapten", "kothu parotta")
        assert 0.0 < score < 1.0

    def test_phonetic_survives_spelling_variation(self):
        """ASR misspells Tamil proper nouns constantly."""
        assert phonetic_match("Thanjavoor", "Thanjavur") > 0.8
        assert phonetic_match("Tanjore", "Thanjavur") > 0.4

    def test_phonetic_key_folds_digraphs(self):
        assert phonetic_key("Thanjavur") == phonetic_key("Tanjavur")

    def test_phonetic_finds_entity_inside_long_answer(self):
        assert phonetic_match("naan Thanjavoor la irukken", "Thanjavur") > 0.8

    def test_phonetic_rejects_unrelated(self):
        assert phonetic_match("Chennai", "Thanjavur") < 0.5

    def test_empty_inputs_score_zero(self):
        assert phonetic_match("", "Thanjavur") == 0.0
        assert entity_match("", "Thanjavur") == 0.0


class TestAnswerMatcher:
    def test_accepts_exact(self):
        result = AnswerMatcher().match("Thanjavur", "Thanjavur")
        assert result.score == pytest.approx(1.0)
        assert result.method == "exact"

    def test_accepts_conversational(self):
        result = AnswerMatcher().match("naan Thanjavur-la irundhu varen", "Thanjavur")
        assert result.score > 0.9

    def test_accepts_misspelled(self):
        assert AnswerMatcher().match("Thanjavoor", "Thanjavur").score > 0.7

    def test_rejects_wrong_answer(self):
        assert AnswerMatcher().match("Coimbatore", "Thanjavur").score < 0.7

    def test_reports_unavailable_semantic_matcher(self):
        """Unavailable must be distinguishable from a confident mismatch."""
        result = AnswerMatcher(semantic_matcher=None).match("x", "y")
        assert result.available["semantic"] is False
        assert result.available["phonetic"] is True

    def test_explain_is_readable(self):
        text = AnswerMatcher().match("Thanjavur", "Thanjavur").explain()
        assert "exact" in text and "phonetic" in text


# ==========================================================================
# Fusion
# ==========================================================================


def branch(b: Branch, score: float, threshold: float = 0.5, available: bool = True):
    return BranchScore(branch=b, score=score, threshold=threshold, weight=0.0, available=available)


class TestFusion:
    def test_all_branches_pass_accepts(self):
        result = fuse([
            branch(Branch.SPEAKER, 0.9),
            branch(Branch.CSBG, 0.8),
            branch(Branch.KNOWLEDGE, 0.95),
        ])
        assert result.decision is Decision.ACCEPT

    def test_all_branches_fail_rejects(self):
        result = fuse([
            branch(Branch.SPEAKER, 0.2),
            branch(Branch.CSBG, 0.1),
            branch(Branch.KNOWLEDGE, 0.0),
        ])
        assert result.decision is Decision.REJECT

    def test_borderline_reported_not_guessed(self):
        policy = FusionPolicy(threshold=0.5, borderline_margin=0.05)
        result = fuse(
            [branch(Branch.SPEAKER, 0.51), branch(Branch.CSBG, 0.51), branch(Branch.KNOWLEDGE, 0.51)],
            policy,
        )
        assert result.decision is Decision.BORDERLINE
        assert any("inconclusive" in line for line in result.explanation)

    def test_liveness_failure_overrides_everything(self):
        """A replayed challenge cannot be outvoted by a strong voice match."""
        result = fuse([
            branch(Branch.SPEAKER, 1.0),
            branch(Branch.CSBG, 1.0),
            branch(Branch.KNOWLEDGE, 1.0),
            build_liveness_branch(
                challenge_valid=False, matched_challenge=True, response_latency_sec=1.0
            ),
        ])
        assert result.decision is Decision.REJECT
        assert not result.liveness_ok
        assert any("replay" in line for line in result.explanation)

    def test_unavailable_branch_is_excluded_not_zeroed(self):
        """Scoring a missing measurement as 0.0 would fabricate evidence."""
        with_unavailable = fuse([
            branch(Branch.SPEAKER, 0.9),
            branch(Branch.CSBG, 0.0, available=False),
            branch(Branch.KNOWLEDGE, 0.9),
        ])
        with_zero = fuse([
            branch(Branch.SPEAKER, 0.9),
            branch(Branch.CSBG, 0.0, available=True),
            branch(Branch.KNOWLEDGE, 0.9),
        ])
        assert with_unavailable.fused_score > with_zero.fused_score
        assert with_unavailable.decision is Decision.ACCEPT
        assert "csbg" not in with_unavailable.contributing_branches

    def test_weights_renormalise_when_branch_missing(self):
        result = fuse([
            branch(Branch.SPEAKER, 0.8),
            branch(Branch.CSBG, 0.0, available=False),
            branch(Branch.KNOWLEDGE, 0.8),
        ])
        assert result.fused_score == pytest.approx(0.8, abs=1e-6)

    def test_no_measurable_branch_rejects_as_system_failure(self):
        result = fuse([
            branch(Branch.SPEAKER, 0.0, available=False),
            branch(Branch.CSBG, 0.0, available=False),
            branch(Branch.KNOWLEDGE, 0.0, available=False),
        ])
        assert result.decision is Decision.REJECT
        assert any("system failure" in line for line in result.explanation)

    def test_require_knowledge_gate(self):
        policy = FusionPolicy(require_knowledge=True)
        result = fuse(
            [branch(Branch.SPEAKER, 1.0), branch(Branch.CSBG, 1.0), branch(Branch.KNOWLEDGE, 0.1)],
            policy,
        )
        assert result.decision is Decision.REJECT

    def test_knowledge_not_gated_by_default(self):
        """Attack A4 assumes the attacker HAS the answer; hard-gating knowledge
        would mask whether the CSBG caught them."""
        assert not FusionPolicy().require_knowledge

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            FusionPolicy(weights={Branch.SPEAKER: 0.5, Branch.CSBG: 0.9})

    def test_explanation_names_the_weakest_branch(self):
        # CSBG weakest but above the veto floor, so this exercises ordinary
        # weighted fusion rather than the veto path.
        result = fuse([
            branch(Branch.SPEAKER, 0.9),
            branch(Branch.CSBG, 0.45),
            branch(Branch.KNOWLEDGE, 0.9),
        ])
        assert "csbg" in " ".join(result.explanation)

    def test_a_branch_below_its_veto_floor_rejects_alone(self):
        """The A4 defence: strong contrary evidence is not outvoted.

        Two branches at 0.9 put the weighted score above threshold, and a
        0.30-weighted CSBG cannot pull it back down by averaging. The veto is
        what makes the rejection possible at all -- see the fusion module
        docstring for the arithmetic.
        """
        scores = [
            branch(Branch.SPEAKER, 0.9),
            branch(Branch.CSBG, 0.05),
            branch(Branch.KNOWLEDGE, 0.9),
        ]
        assert fuse(scores, FusionPolicy(veto_thresholds={})).decision is Decision.ACCEPT

        vetoed = fuse(scores)
        assert vetoed.decision is Decision.REJECT
        assert "veto floor" in vetoed.explanation[0]
        assert vetoed.fused_score > vetoed.threshold  # evidence reported honestly

    def test_an_unavailable_branch_cannot_veto(self):
        """A probe too short to score must never reject a genuine speaker."""
        result = fuse([
            branch(Branch.SPEAKER, 0.9),
            BranchScore(Branch.CSBG, 0.0, 0.5, 0.3, available=False),
            branch(Branch.KNOWLEDGE, 0.9),
        ])
        assert result.decision is Decision.ACCEPT


class TestLivenessBranch:
    def test_fresh_response_passes(self):
        b = build_liveness_branch(
            challenge_valid=True, matched_challenge=True, response_latency_sec=5.0
        )
        assert b.passed

    def test_expired_challenge_fails(self):
        b = build_liveness_branch(
            challenge_valid=False, matched_challenge=True, response_latency_sec=5.0
        )
        assert not b.passed
        assert "expired" in b.detail

    def test_slow_response_fails(self):
        b = build_liveness_branch(
            challenge_valid=True, matched_challenge=True, response_latency_sec=120.0
        )
        assert not b.passed


class TestCalibrator:
    def test_fits_weights_summing_to_one(self):
        rng = np.random.default_rng(0)
        samples, labels = [], []
        for _ in range(120):
            genuine = rng.random() < 0.5
            base = 0.8 if genuine else 0.3
            samples.append({
                Branch.SPEAKER: base + 0.1 * rng.normal(),
                Branch.CSBG: base + 0.2 * rng.normal(),
                Branch.KNOWLEDGE: base + 0.1 * rng.normal(),
            })
            labels.append(1 if genuine else 0)

        weights = Calibrator().fit(samples, labels)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights.values())

    def test_imputes_missing_branches(self):
        """A missing branch must not be encoded as 0.0."""
        rng = np.random.default_rng(1)
        samples, labels = [], []
        for i in range(80):
            genuine = i % 2 == 0
            base = 0.8 if genuine else 0.3
            s = {Branch.SPEAKER: base, Branch.KNOWLEDGE: base + 0.05 * rng.normal()}
            if i % 3:
                s[Branch.CSBG] = base
            samples.append(s)
            labels.append(1 if genuine else 0)

        weights = Calibrator().fit(samples, labels)
        assert all(np.isfinite(w) for w in weights.values())

    def test_single_class_rejected(self):
        with pytest.raises(ValueError, match="both genuine and impostor"):
            Calibrator().fit([{Branch.SPEAKER: 0.5}] * 10, [1] * 10)

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="labels"):
            Calibrator().fit([{Branch.SPEAKER: 0.5}] * 5, [1, 0])

    def test_to_policy_requires_fit(self):
        with pytest.raises(RuntimeError, match="fit"):
            Calibrator().to_policy()

    def test_fitted_policy_is_usable(self):
        rng = np.random.default_rng(2)
        samples = [
            {
                Branch.SPEAKER: (0.8 if i % 2 else 0.3) + 0.05 * rng.normal(),
                Branch.CSBG: (0.8 if i % 2 else 0.3) + 0.05 * rng.normal(),
                Branch.KNOWLEDGE: (0.8 if i % 2 else 0.3) + 0.05 * rng.normal(),
            }
            for i in range(100)
        ]
        labels = [i % 2 for i in range(100)]
        cal = Calibrator()
        cal.fit(samples, labels)
        result = fuse(
            [branch(Branch.SPEAKER, 0.9), branch(Branch.CSBG, 0.9), branch(Branch.KNOWLEDGE, 0.9)],
            cal.to_policy(),
        )
        assert result.decision is Decision.ACCEPT
