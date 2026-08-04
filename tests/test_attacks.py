"""Tests for the A1-A5 attack suite.

No TTS model, no API key, no audio files. Replays are signal-processed,
clones use `EchoCloner`, and attacker text comes from the template fallback,
so the whole file runs offline in a second -- which is exactly why the suite
refuses to call any of it paper-ready.
"""

from __future__ import annotations

import numpy as np
import pytest

from kavach.attacks import AttackType, StyleSource
from kavach.attacks.clone import (
    MIN_REFERENCE_SEC,
    CloneBatchStats,
    EchoCloner,
    SynthesisRequest,
    check_language_support,
    screen_clone,
)
from kavach.attacks.replay import (
    ReplayChannel,
    ReplayDetector,
    SpectralCues,
    band_energy_ratios,
    cue_separation,
    envelope_similarity,
    fit_spectral_thresholds,
    log_energy_envelope,
    simulate_replay,
)
from kavach.attacks.splice import (
    SpliceConfig,
    detect_clicks,
    detect_digital_silence,
    detect_splice,
    splice_segments,
)
from kavach.attacks.suite import (
    MIN_TRIALS_PER_CELL,
    AttackSuite,
    AttackTrial,
    SystemConfig,
    policy_for,
    wilson_interval,
)
from kavach.attacks.text import (
    AttackTextGenerator,
    StyleProfile,
    describe_style,
    estimate_style_from_speech,
)
from kavach.audio import Audio, AudioError
from kavach.csbg.graph import CSBG
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.embedding import SpeakerEmbedding, SpeakerTemplate
from kavach.fusion import Branch, Decision, FusionPolicy

SR = 16000


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def speechlike(seconds: float, *, seed: int = 0, sr: int = SR, noise_floor: float = 0.001) -> Audio:
    """Amplitude-modulated harmonics plus a breath-noise band.

    Crude, but it has four properties the code under test needs: syllables
    with quiet gaps between them, energy across the full band rather than a
    few low harmonics, a *seed-dependent syllable rate* so two clips are
    genuinely different performances rather than the same rhythm with a
    different pitch, and a non-zero noise floor.

    The floor is on by default because every real recording has one -- a
    microphone cannot produce exact zeros. A fixture without it would test
    the splice detectors on input they will never see, and
    `detect_digital_silence` exists precisely because that input means the
    file was edited.

    Not speech, and no test here claims it is.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr

    f0 = 110.0 + 60.0 * rng.random()
    carrier = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 12))

    # Fricative-like broadband component, so the high band is not empty --
    # the spectral cues have nothing to measure on pure low harmonics.
    breath = rng.standard_normal(len(t)) * 0.15

    syllable_hz = 2.5 + 3.0 * rng.random()
    syllable = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_hz * t + rng.random() * 6.28)
    envelope = syllable**3  # sharpen into distinct syllables with quiet gaps

    samples = ((carrier + breath) * envelope * 0.2).astype(np.float32)
    if noise_floor:
        samples = samples + (rng.standard_normal(len(samples)) * noise_floor).astype(np.float32)
    return Audio(samples.astype(np.float32), sr, f"speechlike_{seed}")


def utterance(uid: str, pairs: list[tuple[str, Language, SemanticClass]]) -> UtteranceTokens:
    return UtteranceTokens(
        utterance_id=uid,
        tokens=[Token(text=t, language=lang, semantic_class=cls_) for t, lang, cls_ in pairs],
    )


def tamil_leaning_utterances(n: int = 12) -> list[UtteranceTokens]:
    """A speaker who names family in Tamil and numbers in English."""
    out = []
    for i in range(n):
        out.append(
            utterance(
                f"u{i}",
                [
                    ("amma", Language.TA, SemanticClass.KINSHIP),
                    ("appa", Language.TA, SemanticClass.KINSHIP),
                    ("thatha", Language.TA, SemanticClass.KINSHIP),
                    ("three", Language.EN, SemanticClass.NUMBER),
                    ("twelve", Language.EN, SemanticClass.NUMBER),
                    ("morning", Language.EN, SemanticClass.TIME_DATE),
                ],
            )
        )
    return out


@pytest.fixture
def victim_graph() -> CSBG:
    return CSBG.build("victim", tamil_leaning_utterances(), total_duration_sec=180.0)


# --------------------------------------------------------------------------
# A1 -- replay
# --------------------------------------------------------------------------


class TestReplaySimulation:
    def test_band_limiting_removes_high_frequency_energy(self):
        clean = speechlike(2.0, seed=1)
        replayed = simulate_replay(clean, ReplayChannel(reverb_rt60_sec=0.0, snr_db=float("inf")))

        _, clean_hf = band_energy_ratios(clean)
        _, replay_hf = band_energy_ratios(replayed)
        assert replay_hf < clean_hf

    def test_sub_bass_is_attenuated(self):
        clean = speechlike(2.0, seed=2)
        replayed = simulate_replay(clean, ReplayChannel(low_cut_hz=200.0, snr_db=float("inf")))

        clean_lf, _ = band_energy_ratios(clean)
        replay_lf, _ = band_energy_ratios(replayed)
        assert replay_lf <= clean_lf

    def test_replay_is_reproducible_given_a_seed(self):
        clean = speechlike(1.0, seed=3)
        a = simulate_replay(clean, seed=42)
        b = simulate_replay(clean, seed=42)
        assert np.allclose(a.samples, b.samples)

    def test_replay_does_not_clip(self):
        clean = speechlike(1.5, seed=4)
        assert simulate_replay(clean).peak <= 0.99

    def test_output_is_marked_as_simulated(self):
        assert "simulated_replay" in simulate_replay(speechlike(1.0)).source


class TestReplayDetection:
    def test_exact_resubmission_is_caught(self):
        clip = speechlike(2.0, seed=5)
        det = ReplayDetector()
        det.remember(clip, label="enrolment_1")

        report = det.check(clip)
        assert report.is_replay
        assert report.exact_hash_match
        assert report.matched_source == "enrolment_1"
        assert report.score == 1.0

    def test_unseen_recording_is_clean(self):
        det = ReplayDetector()
        det.remember(speechlike(2.0, seed=6))
        report = det.check(speechlike(2.0, seed=7))
        assert not report.is_replay

    def test_envelope_survives_the_playback_channel(self):
        """The point of using the envelope rather than the waveform.

        A replay is filtered, reverberated and noisy, so sample-level
        correlation with the original is destroyed. The rhythm is not, and
        that is what identifies the same performance.
        """
        original = speechlike(3.0, seed=8)
        replayed = simulate_replay(original, seed=8)

        env_sim = envelope_similarity(
            log_energy_envelope(original), log_energy_envelope(replayed)
        )
        raw_sim = envelope_similarity(original.samples, replayed.samples)
        assert env_sim > raw_sim
        assert env_sim > 0.8

    def test_different_performances_do_not_match(self):
        a = log_energy_envelope(speechlike(3.0, seed=9))
        b = log_energy_envelope(speechlike(3.0, seed=10))
        assert envelope_similarity(a, b) < 0.9

    def test_empty_audio_is_rejected_not_scored(self):
        with pytest.raises(AudioError):
            ReplayDetector().check(Audio(np.zeros(0, dtype=np.float32), SR))

    def test_spectral_cues_are_off_by_default(self):
        """They are device-dependent and unfitted; opting in must be explicit."""
        det = ReplayDetector()
        assert not det.use_spectral_cues
        # A heavily band-limited clip is not called a replay without them.
        squashed = simulate_replay(speechlike(2.0, seed=11), ReplayChannel(high_cut_hz=3000.0))
        assert not det.check(squashed).is_replay

    def test_fitting_thresholds_needs_both_classes(self):
        with pytest.raises(ValueError):
            fit_spectral_thresholds([speechlike(1.0)], [])

    def test_fitted_thresholds_land_between_the_classes(self):
        genuine = [speechlike(2.0, seed=s) for s in range(20, 25)]
        replayed = [simulate_replay(g, ReplayChannel(high_cut_hz=3000.0)) for g in genuine]

        fitted = fit_spectral_thresholds(genuine, replayed)
        gen_hf = np.mean([band_energy_ratios(g)[1] for g in genuine])
        rep_hf = np.mean([band_energy_ratios(r)[1] for r in replayed])

        assert rep_hf < fitted.min_high_ratio < gen_hf

    def test_cue_separation_distinguishes_a_working_cue_from_a_dead_one(self):
        """Fitting a threshold on a cue that carries no information is silent.

        Band-limiting at 3 kHz destroys the high band, so that cue separates
        the classes cleanly. It barely touches the sub-bass, so that cue does
        not. `fit_spectral_thresholds` places a threshold for both regardless
        -- `cue_separation` is what shows which one to believe.
        """
        genuine = [speechlike(2.0, seed=s) for s in range(30, 36)]
        replayed = [simulate_replay(g, ReplayChannel(high_cut_hz=3000.0), seed=1) for g in genuine]

        d = cue_separation(genuine, replayed)
        assert d["high_band"] > 2.0
        assert d["sub_bass"] < d["high_band"]

    def test_one_strong_cue_catches_replays_when_the_bar_allows_it(self):
        genuine = [speechlike(2.0, seed=s) for s in range(30, 36)]
        replayed = [simulate_replay(g, ReplayChannel(high_cut_hz=3000.0), seed=1) for g in genuine]
        fitted = fit_spectral_thresholds(genuine, replayed)

        det = ReplayDetector(cues=fitted, use_spectral_cues=True, spectral_threshold=0.3)
        assert all(det.check(r).is_replay for r in replayed)

    def test_by_default_a_single_cue_is_not_enough(self):
        """A lone weak cue also describes a cheap microphone.

        Rejecting on one cue would lock genuine users out of their own
        accounts, so the default bar sits above what either contributes.
        """
        replayed = simulate_replay(speechlike(2.0, seed=31), ReplayChannel(high_cut_hz=3000.0))
        lf, _ = band_energy_ratios(replayed)

        # High-band cue fires; sub-bass cue deliberately cannot.
        one_cue = SpectralCues(min_sub_bass_ratio=lf / 10.0, min_high_ratio=0.001)
        det = ReplayDetector(cues=one_cue, use_spectral_cues=True)  # default 0.5 bar

        report = det.check(replayed)
        assert not report.is_replay
        assert any("band-limited" in r for r in report.reasons)
        assert any("not conclusive" in r for r in report.reasons)

    def test_both_cues_together_clear_the_default_bar(self):
        replayed = simulate_replay(speechlike(2.0, seed=32), ReplayChannel(high_cut_hz=3000.0))
        lf, hf = band_energy_ratios(replayed)

        both = SpectralCues(min_sub_bass_ratio=lf * 10.0, min_high_ratio=hf * 10.0)
        det = ReplayDetector(cues=both, use_spectral_cues=True)
        assert det.check(replayed).is_replay


# --------------------------------------------------------------------------
# A2 -- splice
# --------------------------------------------------------------------------


class TestSpliceGeneration:
    def test_splice_length_is_the_sum_of_the_parts(self):
        parts = [speechlike(0.5, seed=s) for s in range(3)]
        out = splice_segments(parts, SpliceConfig(crossfade_ms=0.0, gap_ms=0.0))
        assert out.duration_sec == pytest.approx(1.5, abs=0.01)

    def test_gaps_lengthen_the_result(self):
        parts = [speechlike(0.5, seed=s) for s in range(3)]
        plain = splice_segments(parts, SpliceConfig(gap_ms=0.0))
        gapped = splice_segments(parts, SpliceConfig(gap_ms=100.0))
        assert gapped.duration_sec > plain.duration_sec

    def test_level_matching_equalises_segment_loudness(self):
        loud = Audio(speechlike(0.5, seed=1).samples * 4.0, SR)
        quiet = Audio(speechlike(0.5, seed=2).samples * 0.25, SR)

        matched = splice_segments([loud, quiet], SpliceConfig(match_levels=True))
        first = matched.slice_seconds(0.0, 0.5).rms
        second = matched.slice_seconds(0.5, 1.0).rms
        assert second / max(first, 1e-9) > 0.5

    def test_empty_and_mismatched_inputs_are_rejected(self):
        with pytest.raises(AudioError):
            splice_segments([])
        with pytest.raises(AudioError):
            splice_segments([speechlike(0.5), Audio(np.zeros(100, dtype=np.float32), 8000)])

    def test_output_is_marked(self):
        assert splice_segments([speechlike(0.5), speechlike(0.5, seed=1)]).source == "spliced"


class TestSpliceDetection:
    def _hard_cut(self) -> Audio:
        """Two clips joined mid-waveform, no crossfade: a guaranteed step."""
        a = Audio(np.full(SR // 2, 0.4, dtype=np.float32), SR)
        b = Audio(np.full(SR // 2, -0.4, dtype=np.float32), SR)
        return splice_segments([a, b], SpliceConfig.naive())

    def test_hard_cut_produces_a_click(self):
        assert detect_clicks(self._hard_cut())

    def test_crossfade_removes_the_click(self):
        a = Audio(np.full(SR // 2, 0.4, dtype=np.float32), SR)
        b = Audio(np.full(SR // 2, -0.4, dtype=np.float32), SR)
        faded = splice_segments([a, b], SpliceConfig(crossfade_ms=25.0))
        assert not detect_clicks(faded)

    def test_continuous_recording_has_no_clicks(self):
        assert not detect_clicks(speechlike(3.0, seed=12))

    def test_hard_cut_between_real_takes_is_caught(self):
        a = speechlike(1.0, seed=40)
        b = speechlike(1.0, seed=41)
        assert detect_clicks(splice_segments([a, b], SpliceConfig.naive()))

    def test_inserted_digital_silence_is_caught_outright(self):
        """The naive gap-padding attacker leaves exact zeros behind."""
        parts = [speechlike(1.0, seed=42), speechlike(1.0, seed=43)]
        padded = splice_segments(parts, SpliceConfig(gap_ms=120.0, gap_fill="silence"))

        runs = detect_digital_silence(padded)
        assert runs
        assert runs[0][1] - runs[0][0] == pytest.approx(0.120, abs=0.01)

        report = detect_splice(padded)
        assert report.is_spliced
        assert any("exact zeros" in r for r in report.reasons)

    def test_room_tone_padding_leaves_no_digital_silence(self):
        """The competent attacker pads with real background instead."""
        parts = [speechlike(1.0, seed=44), speechlike(1.0, seed=45)]
        padded = splice_segments(parts, SpliceConfig(gap_ms=120.0, gap_fill="room_tone"))
        assert not detect_digital_silence(padded)

    def test_genuine_recording_has_no_digital_silence(self):
        assert not detect_digital_silence(speechlike(3.0, seed=46))

    def test_background_mismatch_is_detected_across_a_pause(self):
        """The durable cue: segments from different rooms.

        Two takes with very different noise floors, joined with a pause and a
        crossfade. The click is gone; the background step is not.
        """
        quiet_room = speechlike(1.2, seed=13, noise_floor=0.0005)
        noisy_room = speechlike(1.2, seed=14, noise_floor=0.02)
        spliced = splice_segments([quiet_room, noisy_room], SpliceConfig.careful())

        report = detect_splice(spliced)
        assert report.is_spliced
        assert report.boundaries

    def test_same_room_splice_is_harder(self):
        """Documents the limitation rather than pretending it away.

        Two phrases cut from *one* recording, crossfaded and padded with that
        recording's own room tone. The background genuinely matches, because
        it genuinely is the same background, so the signal-level cues have
        nothing to find. The detector must not claim otherwise: overstating
        it here would let the A2 row take credit the system has not earned.
        Catching this attacker is the CSBG's job.
        """
        take = speechlike(4.0, seed=15, noise_floor=0.004)
        a = take.slice_seconds(0.0, 1.2)
        b = take.slice_seconds(2.0, 3.2)

        report = detect_splice(splice_segments([a, b], SpliceConfig.careful()))
        assert not report.is_spliced

    def test_genuine_recording_with_pauses_is_not_flagged(self):
        assert not detect_splice(speechlike(4.0, seed=17, noise_floor=0.002)).is_spliced

    def test_empty_audio_is_rejected(self):
        with pytest.raises(AudioError):
            detect_splice(Audio(np.zeros(0, dtype=np.float32), SR))

    def test_report_serialises(self):
        d = detect_splice(speechlike(2.0, seed=18)).to_dict()
        assert {"is_spliced", "score", "boundaries", "reasons"} <= set(d)


# --------------------------------------------------------------------------
# A3/A4/A5 -- clone audio
# --------------------------------------------------------------------------


class StubEmbedder:
    """A log-band spectral fingerprint standing in for ECAPA.

    Must be invariant to gain and to silence trimming, because
    `screen_clone` runs `prepare_for_embedding` internally -- an embedder
    keyed on raw amplitude would score a clip against a trimmed copy of
    *itself* as a different speaker, and the screening tests would be
    measuring the stub rather than the screening.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, audio: Audio) -> SpeakerEmbedding:
        spectrum = np.abs(np.fft.rfft(audio.samples, n=8192))
        edges = np.geomspace(1, len(spectrum), self.dim + 1).astype(int)
        bands = np.array(
            [np.log(spectrum[edges[i] : max(edges[i + 1], edges[i] + 1)].mean() + 1e-12)
             for i in range(self.dim)]
        )
        return SpeakerEmbedding(bands - bands.mean())


class TestCloneBackends:
    def test_echo_cloner_returns_reference_audio(self):
        ref = speechlike(8.0, seed=19)
        out = EchoCloner().synthesise(SynthesisRequest("en amma peru Lakshmi"), [ref])
        assert out.source == "echo_clone"
        assert out.duration_sec == pytest.approx(ref.duration_sec)

    def test_short_reference_is_refused(self):
        with pytest.raises(AudioError, match="reference audio"):
            EchoCloner().synthesise(SynthesisRequest("hello"), [speechlike(2.0)])

    def test_reference_clips_are_joined(self):
        clips = [speechlike(4.0, seed=20), speechlike(4.0, seed=21)]
        out = EchoCloner().synthesise(SynthesisRequest("hello"), clips)
        assert out.duration_sec > MIN_REFERENCE_SEC

    def test_empty_text_is_refused(self):
        with pytest.raises(ValueError):
            SynthesisRequest("   ")

    def test_no_reference_is_refused(self):
        with pytest.raises(AudioError):
            EchoCloner().synthesise(SynthesisRequest("hello"), [])

    def test_language_support_check(self):
        ok, msg = check_language_support(EchoCloner(), "ta")
        assert ok and "ta" in msg
        bad, msg = check_language_support(EchoCloner(), "kn")
        assert not bad and "does not list" in msg


class TestCloneScreening:
    def _template(self, embedder: StubEmbedder, clips: list[Audio]) -> SpeakerTemplate:
        return SpeakerTemplate.from_embeddings("victim", [embedder.embed(c) for c in clips])

    def test_perfect_clone_is_admissible(self):
        emb = StubEmbedder()
        ref = speechlike(8.0, seed=22)
        template = self._template(emb, [ref])
        clone = EchoCloner().synthesise(SynthesisRequest("hi"), [ref])

        report = screen_clone(clone, template, emb, threshold=0.5)
        assert report.admissible
        assert "fooled" in report.reason

    def test_bad_clone_is_inadmissible_and_says_why(self):
        emb = StubEmbedder()
        template = self._template(emb, [speechlike(8.0, seed=23)])
        unrelated = speechlike(8.0, seed=24)

        report = screen_clone(unrelated, template, emb, threshold=0.99)
        assert not report.admissible
        assert "yield" in report.reason

    def test_report_serialises(self):
        emb = StubEmbedder()
        ref = speechlike(8.0, seed=25)
        d = screen_clone(ref, self._template(emb, [ref]), emb, threshold=0.5).to_dict()
        assert {"similarity", "admissible", "threshold"} <= set(d)


class TestCloneBatchStats:
    def test_yield_is_admissible_over_attempted(self):
        stats = CloneBatchStats(attempted=40, synthesised=38, admissible=30)
        assert stats.yield_rate == pytest.approx(0.75)
        assert stats.synthesis_rate == pytest.approx(0.95)

    def test_empty_batch_does_not_divide_by_zero(self):
        stats = CloneBatchStats()
        assert stats.yield_rate == 0.0
        assert stats.mean_similarity == 0.0

    def test_summary_mentions_yield(self):
        assert "yield" in CloneBatchStats(attempted=10, admissible=4).summary()


# --------------------------------------------------------------------------
# Attacker text
# --------------------------------------------------------------------------


class TestStyleProfile:
    def test_oracle_profile_captures_the_speakers_habits(self, victim_graph):
        profile = describe_style(victim_graph)
        assert profile.source is StyleSource.ORACLE
        assert profile.dominant_by_class[SemanticClass.KINSHIP][0] is Language.TA
        assert profile.dominant_by_class[SemanticClass.NUMBER][0] is Language.EN

    def test_oracle_profile_reports_no_observation_budget(self, victim_graph):
        """An oracle attacker did not eavesdrop; that field must not imply they did."""
        assert describe_style(victim_graph).observed_seconds == 0.0

    def test_sparse_classes_are_omitted(self, victim_graph):
        """An attacker who saw a class twice has not learned a habit."""
        strict = describe_style(victim_graph, min_count=1000.0)
        assert strict.dominant_by_class == {}

    def test_prompt_rendering_mentions_language_and_topic(self, victim_graph):
        prompt = describe_style(victim_graph).to_prompt()
        assert "Tamil" in prompt
        assert "KINSHIP" in prompt

    def test_empty_profile_renders_honestly(self):
        empty = StyleProfile(speaker_id="x", source=StyleSource.OBSERVED)
        assert "No reliable observations" in empty.to_prompt()

    def test_observed_profile_respects_the_eavesdropping_budget(self):
        """The A5-observed axis: less overheard speech, less style stolen."""
        utts = tamil_leaning_utterances(40)
        small = estimate_style_from_speech("victim", utts, budget_seconds=5.0)
        large = estimate_style_from_speech("victim", utts, budget_seconds=60.0)

        assert small.source is StyleSource.OBSERVED
        assert small.observed_seconds < large.observed_seconds
        assert small.n_classes <= large.n_classes

    def test_profile_serialises(self, victim_graph):
        d = describe_style(victim_graph).to_dict()
        assert d["source"] == "oracle"
        assert "dominant_by_class" in d


class TestAttackText:
    def _gen(self) -> AttackTextGenerator:
        return AttackTextGenerator(use_llm=False, seed=0)

    def test_replay_and_splice_do_not_compose_text(self):
        for attack in (AttackType.A1_REPLAY, AttackType.A2_SPLICE):
            with pytest.raises(ValueError, match="recorded audio"):
                self._gen().generate(
                    attack=attack, question="Who?", true_answer="Lakshmi"
                )

    def test_a5_without_a_style_profile_is_refused(self):
        """A5 *is* the style imitation; there is no A5 without a profile."""
        with pytest.raises(ValueError, match="StyleProfile"):
            self._gen().generate(
                attack=AttackType.A5_STYLE_ADAPTIVE,
                question="Who?",
                true_answer="Lakshmi",
            )

    def test_a3_does_not_know_the_answer(self):
        answer = self._gen().generate(
            attack=AttackType.A3_CLONE, question="Mother's name?", true_answer="Lakshmi"
        )
        assert not answer.knows_answer
        assert not answer.contains_fact

    def test_a4_carries_the_fact(self):
        answer = self._gen().generate(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            question="Mother's name?",
            true_answer="Lakshmi",
            target_class=SemanticClass.KINSHIP,
        )
        assert answer.knows_answer
        assert answer.contains_fact
        assert "Lakshmi" in answer.text

    def test_a4_wraps_the_fact_rather_than_stating_it_bare(self):
        """The whole premise: the attacker knows the fact, not the sentence."""
        answer = self._gen().generate(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            question="Mother's name?",
            true_answer="Lakshmi",
        )
        assert answer.text.strip() != "Lakshmi"
        assert len(answer.text.split()) > 1

    def test_a5_uses_the_victims_frame_language(self, victim_graph):
        style = describe_style(victim_graph)
        answer = self._gen().generate(
            attack=AttackType.A5_STYLE_ADAPTIVE,
            question="Mother's name?",
            true_answer="Lakshmi",
            target_class=SemanticClass.KINSHIP,
            style=style,
        )
        assert answer.knows_answer
        assert answer.style_source is StyleSource.ORACLE
        assert answer.contains_fact

    def test_generator_path_is_recorded(self):
        answer = self._gen().generate(
            attack=AttackType.A4_CLONE_KNOWLEDGE, question="Q?", true_answer="Lakshmi"
        )
        assert answer.provenance["generator"] == "template"

    def test_fact_check_is_strict_about_paraphrase(self):
        """A paraphrased-away fact turns an A4 trial into a mislabelled A3."""
        from kavach.attacks.text import _contains

        assert _contains("my mother is Lakshmi", "Lakshmi")
        assert _contains("My  Mother  Is  LAKSHMI", "lakshmi")
        assert not _contains("my mother is Laxmi", "Lakshmi")
        assert not _contains("anything", "")


# --------------------------------------------------------------------------
# The money table
# --------------------------------------------------------------------------


def trial(
    tid: str,
    attack: AttackType,
    *,
    speaker: float,
    csbg: float,
    knowledge: float,
    speaker_id: str = "victim",
    liveness_ok: bool = True,
    **kw,
) -> AttackTrial:
    return AttackTrial(
        trial_id=tid,
        attack=attack,
        speaker_id=speaker_id,
        speaker_score=speaker,
        speaker_threshold=0.5,
        csbg_score=csbg,
        csbg_threshold=0.5,
        knowledge_score=knowledge,
        knowledge_threshold=0.5,
        liveness_ok=liveness_ok,
        **kw,
    )


class TestPolicyRestriction:
    def test_weights_renormalise_to_one(self):
        for config in SystemConfig:
            policy = policy_for(config)
            assert sum(policy.weights.values()) == pytest.approx(1.0)
            assert set(policy.weights) <= set(config.branches)

    def test_baseline_does_not_inherit_the_csbg_veto(self):
        """Crediting 'ECAPA alone' with a CSBG mechanism would fake the baseline."""
        assert Branch.CSBG not in policy_for(SystemConfig.ECAPA_ONLY).veto_thresholds
        assert Branch.CSBG in policy_for(SystemConfig.FULL).veto_thresholds

    def test_shared_branch_weight_ratios_are_preserved(self):
        """Columns must differ in evidence available, not in how it is weighted."""
        base = FusionPolicy()
        restricted = policy_for(SystemConfig.ECAPA_KNOWLEDGE, base)
        assert restricted.weights[Branch.SPEAKER] / restricted.weights[Branch.KNOWLEDGE] == (
            pytest.approx(base.weights[Branch.SPEAKER] / base.weights[Branch.KNOWLEDGE])
        )

    def test_baseline_gets_no_liveness_gate(self):
        assert not SystemConfig.ECAPA_ONLY.uses_liveness
        assert SystemConfig.ECAPA_KNOWLEDGE.uses_liveness


class TestHeadlineClaim:
    """The A4 row: this is what the paper argues."""

    def _a4(self, tid: str = "t0") -> AttackTrial:
        # Acoustic branch fooled, answer scraped, code-switching wrong.
        return trial(tid, AttackType.A4_CLONE_KNOWLEDGE, speaker=0.85, csbg=0.05, knowledge=0.90)

    def test_ecapa_alone_accepts_the_clone(self):
        suite = AttackSuite()
        outcomes = {o.config: o for o in suite.run(self._a4())}
        assert outcomes[SystemConfig.ECAPA_ONLY].attacker_succeeded

    def test_knowledge_does_not_help_when_the_answer_was_scraped(self):
        suite = AttackSuite()
        outcomes = {o.config: o for o in suite.run(self._a4())}
        assert outcomes[SystemConfig.ECAPA_KNOWLEDGE].attacker_succeeded

    def test_full_system_rejects(self):
        suite = AttackSuite()
        outcomes = {o.config: o for o in suite.run(self._a4())}
        result = outcomes[SystemConfig.FULL].result
        assert result.decision is Decision.REJECT
        assert not outcomes[SystemConfig.FULL].attacker_succeeded

    def test_rejection_names_the_csbg(self):
        suite = AttackSuite()
        outcomes = {o.config: o for o in suite.run(self._a4())}
        text = " ".join(outcomes[SystemConfig.FULL].result.explanation)
        assert "csbg" in text.lower()

    def test_weighted_average_alone_could_not_have_done_this(self):
        """Documents *why* the veto exists, so nobody removes it as redundant.

        With vetoes disabled the fused score stays above threshold no matter
        how anomalous the code-switching, because a 0.30-weighted branch
        cannot overturn 0.61 from the other two. That arithmetic is the
        reason `veto_thresholds` is in FusionPolicy.
        """
        no_veto = FusionPolicy(veto_thresholds={})
        suite = AttackSuite(no_veto)
        outcomes = {o.config: o for o in suite.run(self._a4())}
        assert outcomes[SystemConfig.FULL].attacker_succeeded

    def test_genuine_speaker_with_matching_style_is_accepted(self):
        """The veto must not simply reject everyone."""
        suite = AttackSuite()
        genuine = trial("g0", AttackType.A4_CLONE_KNOWLEDGE, speaker=0.85, csbg=0.80, knowledge=0.90)
        outcomes = {o.config: o for o in suite.run(genuine)}
        assert outcomes[SystemConfig.FULL].attacker_succeeded  # i.e. accepted

    def test_short_probe_cannot_veto(self):
        """An unmeasurable CSBG must never reject a genuine terse answer."""
        suite = AttackSuite()
        terse = trial(
            "s0", AttackType.A4_CLONE_KNOWLEDGE,
            speaker=0.85, csbg=0.0, knowledge=0.90, csbg_reliable=False,
        )
        outcomes = {o.config: o for o in suite.run(terse)}
        assert outcomes[SystemConfig.FULL].attacker_succeeded


class TestReplayAndSpliceRows:
    def test_replay_is_stopped_by_the_liveness_gate_not_by_ecapa(self):
        suite = AttackSuite()
        replay = trial(
            "r0", AttackType.A1_REPLAY,
            speaker=0.95, csbg=0.70, knowledge=0.20, liveness_ok=False,
        )
        outcomes = {o.config: o for o in suite.run(replay)}
        assert outcomes[SystemConfig.ECAPA_ONLY].attacker_succeeded
        assert not outcomes[SystemConfig.ECAPA_KNOWLEDGE].attacker_succeeded
        assert not outcomes[SystemConfig.FULL].attacker_succeeded


class TestAttackTable:
    def _fill(self, suite: AttackSuite, attack: AttackType, n: int, **kw) -> list[AttackTrial]:
        trials = [trial(f"{attack.value}_{i}", attack, **kw) for i in range(n)]
        suite.run_all(trials)
        return trials

    def test_iapmr_counts_accepts_over_admissible_trials(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A4_CLONE_KNOWLEDGE, 40,
                   speaker=0.85, csbg=0.05, knowledge=0.90)
        table = suite.table()

        assert table.cell(AttackType.A4_CLONE_KNOWLEDGE, SystemConfig.ECAPA_ONLY).iapmr == 1.0
        assert table.cell(AttackType.A4_CLONE_KNOWLEDGE, SystemConfig.FULL).iapmr == 0.0

    def test_inadmissible_clones_are_excluded_not_counted_as_rejections(self):
        """A clone ECAPA stopped never tested the CSBG."""
        suite = AttackSuite()
        suite.run_all(
            [
                trial(f"x{i}", AttackType.A3_CLONE,
                      speaker=0.85, csbg=0.05, knowledge=0.10, admissible=False)
                for i in range(10)
            ]
        )
        cell = suite.table().cell(AttackType.A3_CLONE, SystemConfig.FULL)
        assert cell.n_trials == 0
        assert cell.n_excluded_inadmissible == 10
        assert cell.iapmr == 0.0

    def test_borderline_is_not_counted_as_a_breach(self):
        suite = AttackSuite()
        # Fused score lands within borderline_margin of the threshold.
        suite.run(trial("b0", AttackType.A4_CLONE_KNOWLEDGE,
                        speaker=0.55, csbg=0.55, knowledge=0.55))
        cell = suite.table().cell(AttackType.A4_CLONE_KNOWLEDGE, SystemConfig.FULL)
        assert cell.n_accepted == 0
        assert cell.n_borderline == 1

    def test_underpowered_cells_are_flagged(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A4_CLONE_KNOWLEDGE, 5,
                   speaker=0.85, csbg=0.05, knowledge=0.90)
        ready, problems = suite.table().paper_ready()
        assert not ready
        assert any("below" in p for p in problems)

    def test_simulated_trials_disqualify_a_row(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A1_REPLAY, MIN_TRIALS_PER_CELL + 5,
                   speaker=0.85, csbg=0.5, knowledge=0.5, simulated=True)
        ready, problems = suite.table().paper_ready()
        assert not ready
        assert any("simulated" in p.lower() for p in problems)

    def test_template_written_a5_disqualifies(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A5_STYLE_ADAPTIVE, MIN_TRIALS_PER_CELL + 5,
                   speaker=0.85, csbg=0.6, knowledge=0.9, text_generator="template")
        ready, problems = suite.table().paper_ready()
        assert not ready
        assert any("capable adversary" in p for p in problems)

    def test_llm_written_a5_with_enough_trials_passes(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A5_STYLE_ADAPTIVE, MIN_TRIALS_PER_CELL + 5,
                   speaker=0.85, csbg=0.6, knowledge=0.9, text_generator="llm")
        ready, problems = suite.table().paper_ready()
        assert ready, problems

    def test_low_clone_yield_is_flagged_as_a_tts_finding(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A4_CLONE_KNOWLEDGE, MIN_TRIALS_PER_CELL + 5,
                   speaker=0.85, csbg=0.05, knowledge=0.9)
        suite.record_yield(
            AttackType.A4_CLONE_KNOWLEDGE,
            CloneBatchStats(attempted=100, synthesised=100, admissible=12),
        )
        ready, problems = suite.table().paper_ready()
        assert not ready
        assert any("TTS-quality" in p for p in problems)

    def test_markdown_has_a_row_per_attack_and_the_caveats(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A4_CLONE_KNOWLEDGE, 3,
                   speaker=0.85, csbg=0.05, knowledge=0.90)
        md = suite.table().to_markdown()

        assert "ECAPA alone" in md
        assert "Voice clone + knowledge" in md
        assert "Wilson" in md
        assert "Not yet reportable" in md

    def test_table_serialises(self):
        suite = AttackSuite()
        self._fill(suite, AttackType.A3_CLONE, 3, speaker=0.8, csbg=0.5, knowledge=0.1)
        d = suite.table().to_dict()
        assert d["paper_ready"] is False
        assert d["cells"]

    def test_per_speaker_breakdown_exposes_the_worst_case(self):
        """A mean hides the speaker nobody is protecting."""
        suite = AttackSuite()
        trials = [
            trial(f"p{i}", AttackType.A4_CLONE_KNOWLEDGE,
                  speaker=0.85, csbg=0.05, knowledge=0.9, speaker_id=f"spk{i}")
            for i in range(4)
        ]
        # One speaker whose CSBG does not separate them at all.
        weak = trial("pW", AttackType.A4_CLONE_KNOWLEDGE,
                     speaker=0.85, csbg=0.60, knowledge=0.9, speaker_id="spkW")
        trials.append(weak)
        suite.run_all(trials)

        breakdown = suite.per_speaker_breakdown(trials)
        assert breakdown["spkW"]["iapmr"] == 1.0
        assert breakdown["spk0"]["iapmr"] == 0.0


class TestWilsonInterval:
    def test_zero_of_n_still_has_an_upper_bound(self):
        """0/40 is not 'zero risk', and the interval must say so."""
        lo, hi = wilson_interval(0, 40)
        assert lo == 0.0
        assert 0.0 < hi < 0.15

    def test_interval_brackets_the_estimate(self):
        lo, hi = wilson_interval(20, 40)
        assert lo < 0.5 < hi

    def test_interval_narrows_with_more_trials(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(50, 100)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_no_trials_gives_no_information(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_interval_stays_inside_the_unit_range(self):
        for k, n in ((0, 5), (5, 5), (1, 3)):
            lo, hi = wilson_interval(k, n)
            assert 0.0 <= lo <= hi <= 1.0


class TestVetoPolicyValidation:
    def test_veto_at_or_above_the_threshold_is_refused(self):
        with pytest.raises(ValueError, match="veto"):
            FusionPolicy(threshold=0.55, veto_thresholds={Branch.CSBG: 0.55})

    def test_vetoes_can_be_disabled_for_the_ablation(self):
        assert FusionPolicy(veto_thresholds={}).veto_thresholds == {}
