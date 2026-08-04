"""Tests for CSBG scoring -- the likelihood-ratio verifier.

The headline test is `TestDiscrimination::test_separates_synthetic_speakers`,
which checks that the estimator recovers a known idiolect difference. Read the
warning in `kavach.simulation` first: this validates the *implementation*, not
the research hypothesis. Real speakers are the week-2 pilot's job.

`test_null_control_fails_to_discriminate` is its necessary companion -- a
scorer that separates identical speakers is broken, and without that control
a passing discrimination test proves nothing.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from kavach.csbg.graph import CSBG
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.scoring import (
    CohortNormaliser,
    ScoringWeights,
    build_background_model,
    discriminative_classes,
    score_jsd,
    score_llr,
)
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.eval.metrics import compute_eer, evaluate
from kavach.simulation import make_corpus, make_population, sample_session, sample_utterance

TA, EN = Language.TA, Language.EN


def make_utterance(pairs: list[tuple[Language, SemanticClass]], uid: str = "u") -> UtteranceTokens:
    return UtteranceTokens(
        utterance_id=uid,
        tokens=[
            Token(text=f"w{i}", language=lang, semantic_class=cls_)
            for i, (lang, cls_) in enumerate(pairs)
        ],
    )


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(n_speakers=20, seed=42, separation=0.6, consistency=0.85)


@pytest.fixture(scope="module")
def graphs(corpus):
    return {
        sid: CSBG.build(sid, utts) for sid, utts in corpus.enrolment.items()
    }


class TestScoringWeights:
    def test_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(lexical=0.5, transition=0.3, metrics=0.3)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            ScoringWeights(lexical=1.2, transition=-0.2, metrics=0.0)


class TestBackgroundModel:
    def test_empty_rejected(self):
        """A UBM over nothing is uniform, silently turning the LLR into a
        language-plausibility check. Must fail loudly."""
        with pytest.raises(ValueError, match="zero graphs"):
            build_background_model([])

    def test_pools_counts_not_probabilities(self, graphs):
        """Pooling must weight by evidence, so UBM counts equal the sum."""
        subset = list(graphs.values())[:5]
        ubm = build_background_model(subset)
        expected = sum(g.lexical_counts.sum() for g in subset)
        assert ubm.lexical_counts.sum() == pytest.approx(expected)

    def test_normalised(self, graphs):
        ubm = build_background_model(list(graphs.values()))
        np.testing.assert_allclose(ubm.lexical_probs.sum(axis=1), 1.0, rtol=1e-9)


class TestLLRBasics:
    def test_matching_habits_score_positive(self):
        """Tokens following the reference speaker's habits are evidence for them."""
        ref = CSBG.build("ref", [make_utterance([(EN, SemanticClass.NUMBER)] * 100)])
        other = CSBG.build("oth", [make_utterance([(TA, SemanticClass.NUMBER)] * 100)])
        ubm = build_background_model([ref, other])

        probe = [make_utterance([(EN, SemanticClass.NUMBER)] * 10)]
        assert score_llr(probe, ref, ubm).raw_score > 0

    def test_contradicting_habits_score_negative(self):
        ref = CSBG.build("ref", [make_utterance([(EN, SemanticClass.NUMBER)] * 100)])
        other = CSBG.build("oth", [make_utterance([(TA, SemanticClass.NUMBER)] * 100)])
        ubm = build_background_model([ref, other])

        probe = [make_utterance([(TA, SemanticClass.NUMBER)] * 10)]
        assert score_llr(probe, ref, ubm).raw_score < 0

    def test_population_typical_behaviour_is_uninformative(self):
        """Behaviour everyone shares must score ~0.

        This is what makes the LLR a biometric rather than a plausibility
        check: if all speakers say numbers in English, saying numbers in
        English identifies nobody.
        """
        graphs = [
            CSBG.build(f"s{i}", [make_utterance([(EN, SemanticClass.NUMBER)] * 100)])
            for i in range(5)
        ]
        ubm = build_background_model(graphs)
        probe = [make_utterance([(EN, SemanticClass.NUMBER)] * 10)]
        assert abs(score_llr(probe, graphs[0], ubm).raw_score) < 0.05

    def test_length_normalised(self):
        """A 10-token and a 40-token probe of identical composition must score alike."""
        ref = CSBG.build("ref", [make_utterance([(EN, SemanticClass.NUMBER)] * 100)])
        other = CSBG.build("oth", [make_utterance([(TA, SemanticClass.NUMBER)] * 100)])
        ubm = build_background_model([ref, other])

        short = score_llr([make_utterance([(EN, SemanticClass.NUMBER)] * 10)], ref, ubm)
        long = score_llr([make_utterance([(EN, SemanticClass.NUMBER)] * 40)], ref, ubm)
        assert short.lexical_llr == pytest.approx(long.lexical_llr, rel=0.01)

    def test_empty_probe_scores_zero_not_crash(self):
        """Failed ASR produces empty token lists; this must degrade gracefully."""
        ref = CSBG.build("ref", [make_utterance([(EN, SemanticClass.NUMBER)] * 50)])
        ubm = build_background_model([ref])
        score = score_llr([], ref, ubm)
        assert score.raw_score == 0.0
        assert score.n_scored_tokens == 0
        assert not score.is_reliable

    def test_reliability_flag(self):
        ref = CSBG.build("ref", [make_utterance([(EN, SemanticClass.NUMBER)] * 50)])
        ubm = build_background_model([ref])
        assert not score_llr([make_utterance([(EN, SemanticClass.NUMBER)] * 3)], ref, ubm).is_reliable
        assert score_llr([make_utterance([(EN, SemanticClass.NUMBER)] * 20)], ref, ubm).is_reliable

    def test_low_signal_classes_excluded_by_default(self):
        """FUNCTION_WORD carries no between-speaker signal and must not score."""
        ref = CSBG.build("ref", [make_utterance([(TA, SemanticClass.FUNCTION_WORD)] * 100)])
        other = CSBG.build("oth", [make_utterance([(EN, SemanticClass.FUNCTION_WORD)] * 100)])
        ubm = build_background_model([ref, other])

        probe = [make_utterance([(TA, SemanticClass.FUNCTION_WORD)] * 20)]
        assert score_llr(probe, ref, ubm).n_scored_tokens == 0
        assert score_llr(probe, ref, ubm, include_low_signal=True).n_scored_tokens == 20


class TestExplainability:
    def test_contributions_sorted_worst_first(self):
        ref = CSBG.build(
            "ref",
            [
                make_utterance(
                    [(EN, SemanticClass.NUMBER)] * 60 + [(TA, SemanticClass.FOOD)] * 60
                )
            ],
        )
        other = CSBG.build(
            "oth",
            [
                make_utterance(
                    [(TA, SemanticClass.NUMBER)] * 60 + [(EN, SemanticClass.FOOD)] * 60
                )
            ],
        )
        ubm = build_background_model([ref, other])

        # Matches on FOOD, contradicts on NUMBER.
        probe = [
            make_utterance(
                [(TA, SemanticClass.FOOD)] * 5 + [(TA, SemanticClass.NUMBER)] * 5
            )
        ]
        score = score_llr(probe, ref, ubm)
        assert score.contributions[0].semantic_class is SemanticClass.NUMBER
        assert score.contributions[0].is_evidence_against

        against = score.top_evidence_against(k=1)
        assert against[0].semantic_class is SemanticClass.NUMBER
        assert against[0].expected_language is EN
        assert against[0].observed_language is TA


class TestDiscrimination:
    """The headline behavioural tests."""

    def test_separates_synthetic_speakers(self, corpus, graphs):
        """The estimator must recover a known idiolect difference.

        Validates the implementation, NOT the research hypothesis -- these
        speakers differ by construction. See kavach.simulation.
        """
        genuine, impostor = [], []
        speaker_ids = list(graphs)

        for sid in speaker_ids:
            # Leave-one-out UBM: including the target speaker in the background
            # model would leak their statistics into the denominator.
            ubm = build_background_model([g for k, g in graphs.items() if k != sid])
            for trial_sid in speaker_ids:
                for utt in corpus.trials[trial_sid]:
                    s = score_llr([utt], graphs[sid], ubm).raw_score
                    (genuine if trial_sid == sid else impostor).append(s)

        eer, _ = compute_eer(np.asarray(genuine), np.asarray(impostor))
        assert eer < 0.30, f"EER {eer:.3f} -- estimator failed to recover known differences"
        assert np.mean(genuine) > np.mean(impostor)

    def test_null_control_fails_to_discriminate(self):
        """With identical speakers, EER must be near chance.

        The essential companion to the test above. A scorer that "separates"
        speakers who differ only by sampling noise is finding structure that
        is not there, and would produce a spuriously good EER on real data
        too.
        """
        corpus = make_corpus(
            n_speakers=10, seed=1, separation=0.0, consistency=1.0, trial_utterances=6
        )
        graphs = {sid: CSBG.build(sid, u) for sid, u in corpus.enrolment.items()}

        genuine, impostor = [], []
        for sid in graphs:
            ubm = build_background_model([g for k, g in graphs.items() if k != sid])
            for trial_sid in graphs:
                for utt in corpus.trials[trial_sid]:
                    s = score_llr([utt], graphs[sid], ubm).raw_score
                    (genuine if trial_sid == sid else impostor).append(s)

        eer, _ = compute_eer(np.asarray(genuine), np.asarray(impostor))
        assert eer > 0.35, f"EER {eer:.3f} on identical speakers -- scorer is finding phantom signal"

    def test_more_enrolment_data_improves_separation(self, corpus):
        """Precursor to the paper's CSBG stability curve (proposal 5.3)."""
        speakers = list(corpus.enrolment)[:10]

        def eer_with(n_utts: int) -> float:
            graphs = {
                sid: CSBG.build(sid, corpus.enrolment[sid][:n_utts]) for sid in speakers
            }
            genuine, impostor = [], []
            for sid in speakers:
                ubm = build_background_model([g for k, g in graphs.items() if k != sid])
                for trial_sid in speakers:
                    for utt in corpus.trials[trial_sid][:5]:
                        s = score_llr([utt], graphs[sid], ubm).raw_score
                        (genuine if trial_sid == sid else impostor).append(s)
            return compute_eer(np.asarray(genuine), np.asarray(impostor))[0]

        assert eer_with(30) < eer_with(3) + 0.02, "more enrolment data did not help"

    def test_longer_probes_score_more_reliably(self, corpus, graphs):
        """Discriminability must improve with probe length -- via variance, not mean.

        Scores are per-token length-normalised LLRs, so E[score] does NOT
        depend on probe length; only Var[score] shrinks, roughly as 1/N. The
        mean genuine-impostor gap is therefore flat in N, and asserting it
        grows would be asserting that length normalisation is broken.

        The quantity that actually improves is sensitivity index d':

            d' = (mu_genuine - mu_impostor) / sqrt((var_genuine + var_impostor) / 2)
        """
        speakers = list(graphs)[:10]
        profiles = {p.speaker_id: p for p in corpus.profiles}

        def d_prime(n_tokens: int) -> float:
            rng = random.Random(99)  # same draws across conditions
            genuine, impostor = [], []
            for sid in speakers:
                ubm = build_background_model([g for k, g in graphs.items() if k != sid])
                for trial_sid in speakers:
                    for _ in range(8):
                        utt = sample_utterance(profiles[trial_sid], rng, n_tokens=n_tokens)
                        s = score_llr([utt], graphs[sid], ubm).raw_score
                        (genuine if trial_sid == sid else impostor).append(s)
            g, i = np.asarray(genuine), np.asarray(impostor)
            pooled = np.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2.0)
            return float((g.mean() - i.mean()) / pooled)

        assert d_prime(40) > d_prime(6)

    def test_length_normalisation_keeps_mean_score_stable(self, corpus, graphs):
        """The flip side: mean genuine score must NOT drift with probe length.

        If it did, a fixed decision threshold would behave differently for a
        terse answer than a chatty one -- an unacceptable property for an
        authentication system, where response length is not under our control.
        """
        speakers = list(graphs)[:10]
        profiles = {p.speaker_id: p for p in corpus.profiles}

        def mean_genuine(n_tokens: int) -> float:
            rng = random.Random(7)
            scores = []
            for sid in speakers:
                ubm = build_background_model([g for k, g in graphs.items() if k != sid])
                for _ in range(10):
                    utt = sample_utterance(profiles[sid], rng, n_tokens=n_tokens)
                    scores.append(score_llr([utt], graphs[sid], ubm).raw_score)
            return float(np.mean(scores))

        assert mean_genuine(40) == pytest.approx(mean_genuine(8), abs=0.1)


class TestCohortNormalisation:
    def test_zero_variance_does_not_produce_infinity(self):
        norm = CohortNormaliser()
        norm.fit_speaker("s1", [0.5, 0.5, 0.5])
        assert np.isfinite(norm.apply("s1", 0.7))

    def test_too_few_samples_falls_back_to_identity(self):
        norm = CohortNormaliser()
        norm.fit_speaker("s1", [0.5])
        assert norm.apply("s1", 1.5) == pytest.approx(1.5)

    def test_centres_impostor_distribution(self):
        norm = CohortNormaliser()
        norm.fit_speaker("s1", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert norm.apply("s1", 3.0) == pytest.approx(0.0)

    def test_unknown_speaker_is_identity(self):
        assert CohortNormaliser().apply("never_seen", 2.5) == pytest.approx(2.5)

    def test_roundtrip(self):
        norm = CohortNormaliser()
        norm.fit_speaker("s1", [1.0, 2.0, 3.0])
        restored = CohortNormaliser.from_dict(norm.to_dict())
        assert restored.apply("s1", 2.0) == pytest.approx(norm.apply("s1", 2.0))

    def test_improves_or_holds_eer(self, corpus, graphs):
        """Z-norm should not make separation worse.

        Report EER both with and without it in the paper.
        """
        speaker_ids = list(graphs)[:12]
        raw_g, raw_i, norm_g, norm_i = [], [], [], []

        for sid in speaker_ids:
            ubm = build_background_model([g for k, g in graphs.items() if k != sid])
            impostor_scores = [
                score_llr([u], graphs[sid], ubm).raw_score
                for other in speaker_ids
                if other != sid
                for u in corpus.enrolment[other][:3]
            ]
            norm = CohortNormaliser()
            norm.fit_speaker(sid, impostor_scores)

            for trial_sid in speaker_ids:
                for utt in corpus.trials[trial_sid][:5]:
                    raw = score_llr([utt], graphs[sid], ubm).raw_score
                    z = norm.apply(sid, raw)
                    if trial_sid == sid:
                        raw_g.append(raw)
                        norm_g.append(z)
                    else:
                        raw_i.append(raw)
                        norm_i.append(z)

        raw_eer = compute_eer(np.asarray(raw_g), np.asarray(raw_i))[0]
        norm_eer = compute_eer(np.asarray(norm_g), np.asarray(norm_i))[0]
        assert norm_eer <= raw_eer + 0.05


class TestJSD:
    def test_identical_graphs_have_zero_divergence(self):
        g = CSBG.build("s1", [make_utterance([(TA, SemanticClass.FOOD)] * 50)])
        assert score_jsd(g, g)[0] == pytest.approx(0.0, abs=1e-9)

    def test_opposite_graphs_have_high_divergence(self):
        a = CSBG.build("a", [make_utterance([(TA, SemanticClass.FOOD)] * 100)])
        b = CSBG.build("b", [make_utterance([(EN, SemanticClass.FOOD)] * 100)])
        assert score_jsd(a, b)[0] > 0.5

    def test_bounded(self):
        a = CSBG.build("a", [make_utterance([(TA, SemanticClass.FOOD)] * 100)])
        b = CSBG.build("b", [make_utterance([(EN, SemanticClass.FOOD)] * 100)])
        mean, divs = score_jsd(a, b)
        assert 0.0 <= mean <= 1.0
        assert all(0.0 <= d.jsd <= 1.0 for d in divs)

    def test_skips_undersampled_classes(self):
        """A class only one speaker exercised carries no comparative meaning."""
        a = CSBG.build("a", [make_utterance([(TA, SemanticClass.FOOD)] * 50)])
        b = CSBG.build("b", [make_utterance([(EN, SemanticClass.NUMBER)] * 50)])
        assert score_jsd(a, b, min_count=3.0)[1] == []

    def test_symmetric(self):
        a = CSBG.build("a", [make_utterance([(TA, SemanticClass.FOOD)] * 60)])
        b = CSBG.build("b", [make_utterance([(EN, SemanticClass.FOOD)] * 60)])
        assert score_jsd(a, b)[0] == pytest.approx(score_jsd(b, a)[0])


class TestDiscriminativeClasses:
    def test_finds_the_deviating_class(self):
        """Adaptive challenge targeting must surface where a speaker stands out.

        Population says numbers in English; this speaker says them in Tamil.
        NUMBER should be their most discriminative class.
        """
        population = [
            CSBG.build(f"s{i}", [make_utterance([(EN, SemanticClass.NUMBER)] * 80)])
            for i in range(6)
        ]
        odd = CSBG.build("odd", [make_utterance([(TA, SemanticClass.NUMBER)] * 80)])
        ubm = build_background_model(population)

        top = discriminative_classes(odd, ubm, top_k=3)
        assert top[0][0] is SemanticClass.NUMBER
        assert top[0][1] > 0.3

    def test_respects_min_count(self):
        g = CSBG.build("s", [make_utterance([(TA, SemanticClass.FOOD)] * 3)])
        ubm = build_background_model([CSBG.build("u", [make_utterance([(EN, SemanticClass.FOOD)] * 80)])])
        assert discriminative_classes(g, ubm, min_count=10.0) == []


class TestEvaluationMetrics:
    def test_perfect_separation(self):
        m = evaluate([1.0] * 50, [0.0] * 50)
        assert m.eer == pytest.approx(0.0, abs=0.01)
        assert m.auc == pytest.approx(1.0)

    def test_no_separation(self):
        rng = np.random.default_rng(0)
        m = evaluate(rng.normal(0, 1, 500).tolist(), rng.normal(0, 1, 500).tolist())
        assert 0.4 < m.eer < 0.6
        assert m.auc == pytest.approx(0.5, abs=0.06)

    def test_requires_both_classes(self):
        with pytest.raises(ValueError, match="genuine and impostor"):
            evaluate([1.0, 2.0], [])

    def test_reliability_flag(self):
        assert not evaluate([1.0] * 5, [0.0] * 5).is_reliable
        assert evaluate([1.0] * 50, [0.0] * 50).is_reliable

    def test_bootstrap_ci_brackets_point_estimate(self):
        from kavach.eval.metrics import bootstrap_eer_ci

        rng = np.random.default_rng(3)
        g = rng.normal(1.0, 1.0, 60).tolist()
        i = rng.normal(0.0, 1.0, 60).tolist()
        point, lo, hi = bootstrap_eer_ci(g, i, n_resamples=200, seed=1)
        assert lo <= point <= hi
        assert hi > lo
