"""Offline evaluation tests.

Most of these are leak tests. An evaluation harness that leaks does not
crash, does not warn, and produces a *better* number than an honest one --
which is the worst possible failure mode, because the result looks like
success. So the properties asserted here are mostly negative: what the fitted
policy must NOT have seen, which trials must NOT be counted, and which score
must NOT stand in for a missing measurement.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from kavach.csbg.graph import CSBG
from kavach.eval.ablation import (
    CONFIGURATIONS,
    UNAVAILABLE,
    VETOED,
    Trial,
    ablate_policy,
    ablate_scoring,
    apply_cohort_norm,
    build_trials,
    cohort_fitting_trials,
    evaluate_configuration,
    fairness_slices,
    fit_cohort_normaliser,
    fit_fusion_weights,
    fit_policy,
    fit_veto_floor,
    fused_scores,
    run_ablation,
    split_by_speaker,
    stability_curve,
)
from kavach.eval.metrics import bootstrap_eer_ci, format_rate
from kavach.fusion import Branch, FusionPolicy
from kavach.simulation import make_corpus

BRANCHES = (Branch.SPEAKER, Branch.CSBG, Branch.KNOWLEDGE)


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(
        n_speakers=16, seed=3, separation=0.65, consistency=0.85,
        enrolment_utterances=25, trial_utterances=6,
    )


@pytest.fixture(scope="module")
def graphs(corpus):
    return {sid: CSBG.build(sid, utts) for sid, utts in corpus.enrolment.items()}


def acoustic(rng_seed: int = 0):
    """A stand-in acoustic branch: genuine trials score high, impostors low.

    Not a model of ECAPA -- a *separable* branch, so the fusion machinery has
    something with real discriminative power to weight against the CSBG. The
    overlap is deliberate: two perfectly separable branches would make every
    fusion configuration score 0% EER and the table would say nothing.
    """
    def fn(probe: str, claimed: str, _utts) -> float:
        rng = random.Random(f"{rng_seed}:{probe}:{claimed}")
        base = 0.78 if probe == claimed else 0.55
        return max(0.0, min(1.0, rng.gauss(base, 0.12)))

    return fn


def knowledge(rng_seed: int = 0):
    """A stand-in knowledge branch: near-binary, as the real matcher is."""
    def fn(probe: str, claimed: str, _utts) -> float:
        rng = random.Random(f"k{rng_seed}:{probe}:{claimed}")
        if probe == claimed:
            return 1.0 if rng.random() < 0.95 else 0.3
        return 1.0 if rng.random() < 0.05 else 0.05

    return fn


def make_trial(**kw) -> Trial:
    base = dict(
        probe_speaker="a", claimed_speaker="a", is_genuine=True,
        speaker_score=0.9, csbg_score=0.9, knowledge_score=0.9,
    )
    base.update(kw)
    return Trial(**base)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


class TestSplit:
    def test_no_speaker_appears_on_both_sides(self, graphs, corpus) -> None:
        """The whole point. A speaker's graph is in every trial claiming them."""
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, seed=0)
        assert set(split.dev_speakers) & set(split.test_speakers) == set()

    def test_every_dev_trial_is_dev_on_both_ends(self, graphs, corpus) -> None:
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, seed=0)
        dev = set(split.dev_speakers)
        for t in split.dev:
            assert t.probe_speaker in dev and t.claimed_speaker in dev

    def test_cross_partition_trials_are_discarded(self, graphs, corpus) -> None:
        """A dev model scoring a test probe has one foot in each side.

        Assigning it to test means the threshold was fitted partly on that
        model. Dropping it costs trials; keeping it costs the result.
        """
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, dev_fraction=0.4, seed=0)
        assert len(split.dev) + len(split.test) < len(trials)

        test = set(split.test_speakers)
        for t in split.test:
            assert t.probe_speaker in test and t.claimed_speaker in test

    def test_both_sides_keep_enough_speakers_for_a_background_model(
        self, graphs, corpus
    ) -> None:
        trials = build_trials(graphs, corpus.trials)
        for fraction in (0.1, 0.3, 0.5, 0.9):
            split = split_by_speaker(trials, dev_fraction=fraction, seed=1)
            assert len(split.dev_speakers) >= 2
            assert len(split.test_speakers) >= 2

    def test_too_few_speakers_is_an_error_not_a_silent_split(self) -> None:
        trials = [
            Trial(probe_speaker=a, claimed_speaker=b, is_genuine=a == b)
            for a in "abc" for b in "abc"
        ]
        with pytest.raises(ValueError, match="at least 4 speakers"):
            split_by_speaker(trials)


# --------------------------------------------------------------------------
# Trial construction
# --------------------------------------------------------------------------


class TestTrials:
    def test_all_pairs_are_scored(self, graphs, corpus) -> None:
        trials = build_trials(graphs, corpus.trials)
        n, k = len(graphs), len(next(iter(corpus.trials.values())))
        assert len(trials) == n * n * k
        assert sum(t.is_genuine for t in trials) == n * k

    def test_an_omitted_branch_is_unavailable_not_zero(self, graphs, corpus) -> None:
        """0.0 on a [0,1] scale is maximal evidence against.

        A branch that was never measured must not be indistinguishable from
        one that fired and said 'definitely not them'.
        """
        trials = build_trials(graphs, corpus.trials)
        assert all(math.isnan(t.speaker_score) for t in trials)
        assert all(not t.available(Branch.SPEAKER) for t in trials)
        assert not any(t.speaker_score == 0.0 for t in trials)

    def test_a_short_probe_marks_the_csbg_unavailable(self, graphs) -> None:
        from kavach.csbg.ontology import Language, SemanticClass
        from kavach.csbg.tokens import Token, UtteranceTokens

        terse = UtteranceTokens(
            utterance_id="terse",
            tokens=[Token("ok", Language.EN, SemanticClass.POLITENESS)],
        )
        sid = next(iter(graphs))
        trials = build_trials(graphs, {sid: [terse]}, min_scored_tokens=5)
        assert all(not t.available(Branch.CSBG) for t in trials)

    def test_the_background_model_excludes_the_claimed_speaker(self, graphs, corpus) -> None:
        """Otherwise a speaker is partly compared against themselves.

        Genuine scores would be pulled toward zero -- the LLR would be
        measuring how unusual the speaker is against a population that
        includes them.
        """
        trials = build_trials(graphs, corpus.trials)
        genuine = np.mean([t.csbg_raw for t in trials if t.is_genuine])
        impostor = np.mean([t.csbg_raw for t in trials if not t.is_genuine])
        assert genuine > impostor
        assert genuine > 0 > impostor


# --------------------------------------------------------------------------
# Fusion over trials
# --------------------------------------------------------------------------


class TestFusedScores:
    def test_weights_renormalise_over_measured_branches(self) -> None:
        policy = FusionPolicy(veto_thresholds={})
        both = make_trial(speaker_score=0.8, csbg_score=0.4, knowledge_score=0.6)
        only_speaker = make_trial(
            speaker_score=0.8, csbg_score=UNAVAILABLE, knowledge_score=UNAVAILABLE
        )
        scores = fused_scores([both, only_speaker], policy, BRANCHES)
        assert scores[0] == pytest.approx(0.4 * 0.8 + 0.3 * 0.4 + 0.3 * 0.6)
        assert scores[1] == pytest.approx(0.8), "a lone branch must not be diluted"

    def test_a_veto_is_a_score_no_threshold_accepts(self) -> None:
        """Which is what lets a vetoed system still have a DET curve."""
        policy = FusionPolicy(veto_thresholds={Branch.CSBG: 0.35})
        scores = fused_scores([make_trial(csbg_score=0.05)], policy, BRANCHES)
        assert scores[0] == VETOED

    def test_an_unmeasured_branch_cannot_veto(self) -> None:
        policy = FusionPolicy(veto_thresholds={Branch.CSBG: 0.35})
        scores = fused_scores([make_trial(csbg_score=UNAVAILABLE)], policy, BRANCHES)
        assert math.isfinite(scores[0])

    def test_a_trial_with_nothing_measured_is_not_accepted(self) -> None:
        policy = FusionPolicy(veto_thresholds={})
        trial = make_trial(
            speaker_score=UNAVAILABLE, csbg_score=UNAVAILABLE, knowledge_score=UNAVAILABLE
        )
        assert fused_scores([trial], policy, BRANCHES)[0] == VETOED


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


class TestFitting:
    def test_weights_are_fitted_and_the_source_recorded(self, graphs, corpus) -> None:
        """'Fitted weights' and 'the defaults' are different claims."""
        trials = build_trials(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
        )
        split = split_by_speaker(trials, seed=2)
        weights, source = fit_fusion_weights(split.dev, BRANCHES)
        assert set(weights) == set(BRANCHES)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert source == "logistic regression on dev"

    def test_a_failed_fit_says_so_rather_than_passing_off_priors(self) -> None:
        genuine_only = [make_trial(is_genuine=True) for _ in range(10)]
        weights, source = fit_fusion_weights(genuine_only, BRANCHES)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert "NOT FITTED" in source

    def test_the_veto_floor_respects_its_false_reject_budget(self, graphs, corpus) -> None:
        """A veto is a false-reject risk and the search must price it."""
        trials = build_trials(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
        )
        split = split_by_speaker(trials, seed=2)
        fitted = fit_policy(split.dev, BRANCHES, max_veto_frr_cost=0.02)
        assert fitted.veto_frr_cost <= 0.02 + 1e-9
        if fitted.veto_floor is not None:
            assert fitted.veto_far_benefit > 0, "a veto that catches nothing is dead code"

    def test_a_veto_that_buys_nothing_is_discarded(self) -> None:
        """Reporting 'fitted and discarded' is a result, not an omission."""
        policy = FusionPolicy(weights={Branch.SPEAKER: 0.5, Branch.CSBG: 0.5}, threshold=0.5)
        # CSBG carries no information: both classes sit at the same score.
        dev = [
            make_trial(
                probe_speaker=f"s{i}", claimed_speaker=f"s{i}",
                is_genuine=i % 2 == 0,
                speaker_score=0.9 if i % 2 == 0 else 0.1,
                csbg_score=0.6,
                knowledge_score=UNAVAILABLE,
            )
            for i in range(40)
        ]
        floor, benefit, cost = fit_veto_floor(dev, policy, BRANCHES)
        assert floor is None
        assert benefit == 0.0

    def test_a_tighter_budget_never_yields_a_higher_floor(self, graphs, corpus) -> None:
        trials = build_trials(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
        )
        split = split_by_speaker(trials, seed=2)
        loose = fit_policy(split.dev, BRANCHES, max_veto_frr_cost=0.10).veto_floor
        tight = fit_policy(split.dev, BRANCHES, max_veto_frr_cost=0.001).veto_floor
        if loose is not None and tight is not None:
            assert tight <= loose

    def test_cohort_norm_is_fitted_on_impostors_only(self, graphs, corpus) -> None:
        """Genuine trials would drag the mean toward the target distribution."""
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, seed=2)
        normaliser = fit_cohort_normaliser(split.dev)

        for speaker in split.dev_speakers:
            if speaker not in normaliser.means:
                continue
            impostors = [
                t.csbg_raw for t in split.dev
                if t.claimed_speaker == speaker and not t.is_genuine
            ]
            assert normaliser.means[speaker] == pytest.approx(float(np.mean(impostors)))

    def test_cohort_norm_centres_impostors_and_lifts_genuine(self, graphs, corpus) -> None:
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, seed=2)
        apply_cohort_norm(split.test, fit_cohort_normaliser(split.dev))

        genuine = np.mean([t.csbg_score for t in split.test if t.is_genuine])
        impostor = np.mean([t.csbg_score for t in split.test if not t.is_genuine])
        assert genuine > impostor
        assert 0.0 < impostor < 1.0


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class TestEvaluation:
    def test_a_configuration_with_no_measured_branch_is_omitted(
        self, graphs, corpus
    ) -> None:
        """An empty row beats an EER of 0.5 that reads like a result."""
        trials = build_trials(graphs, corpus.trials)
        policy = FusionPolicy(veto_thresholds={})
        assert (
            evaluate_configuration("ECAPA alone", trials, policy, (Branch.SPEAKER,))
            is None
        )

    def test_vetoed_trials_are_counted_and_do_not_break_the_metrics(self) -> None:
        policy = FusionPolicy(
            weights={Branch.SPEAKER: 0.5, Branch.CSBG: 0.5},
            threshold=0.5,
            veto_thresholds={Branch.CSBG: 0.35},
        )
        trials = [
            make_trial(
                probe_speaker=f"s{i}", claimed_speaker=f"s{i}",
                is_genuine=i % 2 == 0,
                speaker_score=0.9 if i % 2 == 0 else 0.4,
                csbg_score=0.9 if i % 2 == 0 else 0.1,
                knowledge_score=UNAVAILABLE,
            )
            for i in range(40)
        ]
        result = evaluate_configuration(
            "full", trials, policy, (Branch.SPEAKER, Branch.CSBG), bootstrap=50
        )
        assert result is not None
        assert result.n_vetoed == 20
        assert math.isfinite(result.metrics.eer)
        assert result.metrics.eer == pytest.approx(0.0, abs=1e-9)

    def test_the_full_system_beats_its_own_branches(self, graphs, corpus) -> None:
        """Not guaranteed on real data -- asserted here because the stand-in
        branches are independent by construction, so fusion must help. If this
        ever fails on real data that is a finding, not a bug in the harness."""
        report = run_ablation(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
            seed=4, bootstrap=100,
        )
        by_name = {c.name: c.metrics.eer for c in report.configurations}
        assert "Full fusion" in by_name
        assert by_name["Full fusion"] <= min(
            by_name["ECAPA alone"], by_name["CSBG alone"]
        )

    def test_every_configuration_is_reported_when_all_branches_are_measured(
        self, graphs, corpus
    ) -> None:
        report = run_ablation(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
            seed=4, bootstrap=50,
        )
        assert {c.name for c in report.configurations} == set(CONFIGURATIONS)

    def test_intervals_are_reported_not_just_point_estimates(
        self, graphs, corpus
    ) -> None:
        """With 16 speakers a bare EER overstates its own precision."""
        report = run_ablation(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
            seed=4, bootstrap=100,
        )
        for c in report.configurations:
            lo, hi = c.eer_ci
            assert lo <= c.metrics.eer <= hi
            assert hi > lo

    def test_the_report_records_the_split_and_the_missing_branches(
        self, graphs, corpus
    ) -> None:
        """The property a reviewer checks first."""
        report = run_ablation(graphs, corpus.trials, seed=4, bootstrap=50)
        assert set(report.split.dev_speakers) & set(report.split.test_speakers) == set()
        assert "speaker_embedding" in " ".join(report.caveats)
        assert report.measured_branches == (Branch.CSBG,)

    def test_an_unattainable_operating_point_is_not_reported_as_a_rate(self) -> None:
        """A veto sets a floor on FRR that no threshold can go below.

        "FAR at 1% FRR" then does not exist. Reporting it as 1.0 prints
        "100.00" beside a baseline's "46.70", and a reader compares them as
        measured rates -- concluding fusion is catastrophic at an operating
        point fusion cannot occupy.
        """
        from kavach.eval.metrics import format_rate

        policy = FusionPolicy(
            weights={Branch.SPEAKER: 0.5, Branch.CSBG: 0.5},
            threshold=0.5,
            veto_thresholds={Branch.CSBG: 0.35},
        )
        # Ten genuine trials, three of them vetoed: FRR can never fall below 30%.
        trials = [
            make_trial(
                probe_speaker=f"g{i}", claimed_speaker=f"g{i}", is_genuine=True,
                speaker_score=0.9, csbg_score=0.05 if i < 3 else 0.9,
                knowledge_score=UNAVAILABLE,
            )
            for i in range(10)
        ] + [
            make_trial(
                probe_speaker=f"i{i}", claimed_speaker=f"g{i}", is_genuine=False,
                speaker_score=0.4, csbg_score=0.4, knowledge_score=UNAVAILABLE,
            )
            for i in range(10)
        ]
        result = evaluate_configuration(
            "full", trials, policy, (Branch.SPEAKER, Branch.CSBG), bootstrap=20
        )
        assert result is not None
        assert math.isnan(result.metrics.far_at_frr_1pct)
        assert format_rate(result.metrics.far_at_frr_1pct) == "n/a"
        assert "n/a" in result.row()
        assert "3 vetoed" in result.row()

    def test_markdown_renders(self, graphs, corpus) -> None:
        report = run_ablation(
            graphs, corpus.trials,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
            seed=4, bootstrap=50,
        )
        text = report.to_markdown()
        assert "Configuration | EER" in text
        assert "minDCF parameters" in text
        assert "dev:" in text and "test:" in text


# --------------------------------------------------------------------------
# Ablations, stability, fairness
# --------------------------------------------------------------------------


class TestAblations:
    def test_the_veto_ablation_carries_its_own_health_warning(self) -> None:
        """EER can improve without the veto while the attack table worsens.

        A reader who sees "vetoes disabled: EER -0.4%" and nothing else will
        conclude the veto is harmful. It trades false rejects for stopped
        attacks, and EER over genuine-vs-impostor trials cannot see the trade.
        """
        policy = FusionPolicy(
            weights={Branch.SPEAKER: 0.5, Branch.CSBG: 0.5},
            threshold=0.5,
            veto_thresholds={Branch.CSBG: 0.35},
        )
        from kavach.eval.ablation import FittedPolicy

        fitted = FittedPolicy(
            policy=policy, weights_source="test", threshold_source="test",
            veto_floor=0.35, veto_frr_cost=0.0, veto_far_benefit=0.1,
        )
        trials = [
            make_trial(
                probe_speaker=f"s{i}", claimed_speaker=f"s{i}",
                is_genuine=i % 2 == 0,
                speaker_score=0.9 if i % 2 == 0 else 0.6,
                csbg_score=0.8 if i % 2 == 0 else 0.2,
                knowledge_score=UNAVAILABLE,
            )
            for i in range(40)
        ]
        rows = ablate_policy(trials, fitted, (Branch.SPEAKER, Branch.CSBG))
        veto_row = next(r for r in rows if "veto" in r.name)
        assert "attack table" in veto_row.note


class TestStability:
    def test_more_enrolment_speech_lowers_the_error(self, corpus) -> None:
        """The curve the paper reads in both directions (§5.1.2, §5.3)."""
        points = stability_curve(
            corpus.enrolment, corpus.trials, budgets=(2, 5, 25), bootstrap=50
        )
        assert len(points) == 3
        assert points[-1].eer < points[0].eer

    def test_a_budget_no_speaker_can_meet_is_skipped(self, corpus) -> None:
        points = stability_curve(
            corpus.enrolment, corpus.trials, budgets=(5, 10_000), bootstrap=50
        )
        assert [p.n_utterances for p in points] == [5]


class TestFairness:
    def test_small_groups_are_dropped_not_shown_with_wide_bars(
        self, graphs, corpus
    ) -> None:
        """A fairness table is read as a comparison; four trials cannot support one."""
        groups = {sid: {"device": "unique" if i == 0 else "common"}
                  for i, sid in enumerate(graphs)}
        trials = build_trials(graphs, corpus.trials, groups=groups)
        policy = FusionPolicy(weights={Branch.CSBG: 1.0}, threshold=0.5, veto_thresholds={})
        slices = fairness_slices(trials, policy, (Branch.CSBG,), min_trials=20)
        assert {s.group for s in slices} == {"common"}

    def test_slices_are_reported_per_condition(self, graphs, corpus) -> None:
        rng = random.Random(11)
        groups = {
            sid: {
                "device": rng.choice(["phone-a", "phone-b"]),
                "environment": rng.choice(["quiet", "noisy"]),
            }
            for sid in graphs
        }
        trials = build_trials(graphs, corpus.trials, groups=groups)
        policy = FusionPolicy(weights={Branch.CSBG: 1.0}, threshold=0.5, veto_thresholds={})
        slices = fairness_slices(trials, policy, (Branch.CSBG,))
        assert {s.condition for s in slices} == {"device", "environment"}


class TestCohortCoverage:
    """The normaliser must reach the speakers it is reported as normalising.

    `fit_cohort_normaliser(split.dev)` produces statistics for dev speakers
    only. Every test trial then misses the lookup and falls back to
    (mean 0, std 1), so the test table said "cohort z-normalised" while the
    test scores were not normalised at all. `CohortNormaliser.apply` cannot
    detect this -- a miss is indistinguishable from a speaker whose cohort
    happens to be centred -- so it has to be asserted here.
    """

    def test_dev_alone_covers_no_test_speaker(self, graphs, corpus) -> None:
        """The bug, pinned so it cannot come back quietly."""
        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        stale = fit_cohort_normaliser(split.dev)
        assert not set(split.test_speakers) & set(stale.means)

    def test_cohort_fitting_covers_every_test_speaker(self, graphs, corpus) -> None:
        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        normaliser = fit_cohort_normaliser(cohort_fitting_trials(split))
        assert set(split.test_speakers) <= set(normaliser.means)
        assert set(split.dev_speakers) <= set(normaliser.means)

    def test_no_fitting_trial_uses_a_test_probe(self, graphs, corpus) -> None:
        """The leak the cohort split exists to avoid.

        A dev model normalised using test speech would reach the fitted
        weights, threshold and veto floor -- all of which are fitted on dev
        trials that the dev statistics rescale.
        """
        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        dev = set(split.dev_speakers)
        for t in cohort_fitting_trials(split):
            assert t.probe_speaker in dev

    def test_cohort_holds_the_trials_the_split_discards(self, graphs, corpus) -> None:
        trials = build_trials(graphs, corpus.trials)
        split = split_by_speaker(trials, dev_fraction=0.4, seed=0)
        assert len(split.dev) + len(split.test) + len(split.cohort) == len(trials)
        assert split.cohort

    def test_cohort_trials_are_all_impostors(self, graphs, corpus) -> None:
        """They straddle the boundary, so probe and claimed differ by construction."""
        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        assert all(not t.is_genuine for t in split.cohort)

    def test_normalisation_actually_changes_test_scores(self, graphs, corpus) -> None:
        split = split_by_speaker(build_trials(graphs, corpus.trials), seed=0)
        before = [t.csbg_score for t in split.test]
        apply_cohort_norm(split.test, fit_cohort_normaliser(cohort_fitting_trials(split)))
        after = [t.csbg_score for t in split.test]
        assert any(a != pytest.approx(b) for a, b in zip(after, before))

    def test_coverage_is_reported_in_the_caveats(self, graphs, corpus) -> None:
        report = run_ablation(graphs, corpus.trials, bootstrap=20, scoring_ablations=False)
        assert any("test speakers" in c for c in report.caveats)


class TestScoringAblations:
    """Ablations that need a fresh `build_trials`, unlike `ablate_policy`."""

    def test_rows_are_scoped_to_the_branch_they_change(self, graphs, corpus) -> None:
        """Scored on the full system they all read +0.00, and the zero lies.

        Every switch here changes only the CSBG score. A knowledge branch that
        separates far better pins the fused EER, so the delta measures that
        dominance rather than the switch.
        """
        rows = ablate_scoring(
            graphs, corpus.trials, BRANCHES,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
        )
        assert rows
        assert all(r.scope == Branch.CSBG.value for r in rows)

    def test_the_low_signal_hypothesis_is_actually_tested(self, graphs, corpus) -> None:
        """`ontology.LOW_SIGNAL_CLASSES` defers to this ablation by name."""
        rows = ablate_scoring(graphs, corpus.trials, (Branch.CSBG,))
        assert any("low-signal" in r.name for r in rows)

    def test_every_scoring_stream_gets_a_row(self, graphs, corpus) -> None:
        names = {r.name for r in ablate_scoring(graphs, corpus.trials, (Branch.CSBG,))}
        assert "transition stream removed" in names
        assert "lexical stream removed" in names
        assert "graph metrics removed" in names
        assert any("z-norm" in n for n in names)

    def test_removing_the_lexical_stream_hurts(self, graphs, corpus) -> None:
        """Lexical choice is the best-estimated stream; if dropping it helps,
        either the weights or the scorer is wrong."""
        rows = ablate_scoring(graphs, corpus.trials, (Branch.CSBG,))
        lexical = next(r for r in rows if r.name == "lexical stream removed")
        assert lexical.delta > 0

    def test_deltas_are_against_a_baseline_built_the_same_way(
        self, graphs, corpus
    ) -> None:
        """`eer - delta` must reconstruct one shared baseline for every row."""
        rows = ablate_scoring(graphs, corpus.trials, (Branch.CSBG,))
        baselines = {round(r.eer - r.delta, 9) for r in rows}
        assert len(baselines) == 1

    def test_notes_do_not_break_the_markdown_table(self, graphs, corpus) -> None:
        """A `|` in a note silently splits the row into extra columns."""
        rows = ablate_scoring(graphs, corpus.trials, (Branch.CSBG,))
        for r in rows:
            assert "|" not in r.note, f"{r.name} note contains a pipe"
            assert "|" not in r.name

    def test_scoring_ablations_can_be_switched_off(self, graphs, corpus) -> None:
        report = run_ablation(graphs, corpus.trials, bootstrap=20, scoring_ablations=False)
        assert all(r.scope == "full system" for r in report.ablations)

    def test_run_ablation_includes_both_kinds(self, graphs, corpus) -> None:
        report = run_ablation(
            graphs, corpus.trials, bootstrap=20,
            speaker_score_fn=acoustic(), knowledge_score_fn=knowledge(),
        )
        scopes = {r.scope for r in report.ablations}
        assert "full system" in scopes
        assert Branch.CSBG.value in scopes

    def test_markdown_carries_the_scope_column(self, graphs, corpus) -> None:
        report = run_ablation(graphs, corpus.trials, bootstrap=20)
        text = report.to_markdown()
        assert "| Removed | Scope | EER % |" in text


class TestStabilityWiring:
    def test_run_ablation_computes_the_curve_when_given_enrolment(
        self, graphs, corpus
    ) -> None:
        """`stability_curve` existed but nothing called it."""
        report = run_ablation(
            graphs, corpus.trials, bootstrap=20, scoring_ablations=False,
            enrolment=corpus.enrolment, stability_budgets=(2, 5, 20),
        )
        assert [p.n_utterances for p in report.stability] == [2, 5, 20]
        assert "stability against enrolment budget" in report.to_markdown()

    def test_missing_enrolment_is_a_caveat_not_a_silent_gap(
        self, graphs, corpus
    ) -> None:
        """A section that vanishes reads as 'not applicable', not 'not run'."""
        report = run_ablation(graphs, corpus.trials, bootstrap=20, scoring_ablations=False)
        assert report.stability == []
        assert any("stability curve was skipped" in c for c in report.caveats)

    def test_an_empty_curve_says_why(self, graphs, corpus) -> None:
        report = run_ablation(
            graphs, corpus.trials, bootstrap=20, scoring_ablations=False,
            enrolment=corpus.enrolment, stability_budgets=(10_000,),
        )
        assert report.stability == []
        assert any("stability curve came out empty" in c for c in report.caveats)


class TestBootstrapShortcut:
    def test_zero_resamples_returns_nan_bounds_not_a_crash(self) -> None:
        """Ablation rows report no interval; an empty quantile used to raise."""
        genuine = np.array([0.8, 0.7, 0.9, 0.75])
        impostor = np.array([0.2, 0.3, 0.25, 0.4])
        point, lo, hi = bootstrap_eer_ci(genuine, impostor, n_resamples=0)
        assert math.isnan(lo) and math.isnan(hi)
        assert not math.isnan(point)

    def test_nan_bounds_render_as_not_available(self) -> None:
        """Trap 3: a sentinel that prints as a rate is worse than no number."""
        assert format_rate(math.nan) == "n/a"
