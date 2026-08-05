"""Corpus layer tests.

Three properties carry most of the weight here, and each has a failure mode
that produces numbers rather than errors:

1. **The split is by session.** A within-session split flatters every result
   downstream and looks exactly like a working system. It must take an
   explicit argument to obtain, and must announce itself when it happens.

2. **Un-annotated is not empty.** An utterance with no tags must refuse to
   become an `UtteranceTokens`, because an empty token list builds a graph
   sitting at the smoothing prior that scores near-chance against everything
   -- a load failure wearing the appearance of a weak result.

3. **Simulated speech cannot be reported by accident.** `from_simulation`
   exists to make the simulator convenient, which is precisely why the object
   it returns must keep saying it is not evidence.
"""

from __future__ import annotations

import json

import pytest

from kavach import corpus as C
from kavach.csbg.graph import CSBG, SmoothingConfig
from kavach.csbg.ontology import ELICITABLE_CLASSES, Language, SemanticClass
from kavach.csbg.tokens import Token
from kavach.eval.ablation import build_trials, run_ablation
from kavach.simulation import make_corpus


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def tok(cls_: SemanticClass, lang: Language = Language.TA, conf: float = 1.0) -> Token:
    return Token(text=f"{lang.value}_{cls_.value}", language=lang, semantic_class=cls_,
                 lid_confidence=conf)


def utterance(
    uid: str,
    session_id: str,
    speaker_id: str,
    *,
    n: int = 8,
    cls_: SemanticClass = SemanticClass.FOOD,
    prompt_id: str = "p02_food",
    annotated: bool = True,
) -> C.UtteranceRecord:
    return C.UtteranceRecord(
        utterance_id=uid,
        session_id=session_id,
        speaker_id=speaker_id,
        prompt_id=prompt_id,
        duration_sec=n * 0.4,
        transcript=" ".join(["w"] * n),
        tokens=[tok(cls_) for _ in range(n)] if annotated else None,
        annotation_source=C.AnnotationSource.HUMAN if annotated else None,
    )


@pytest.fixture
def two_session_corpus() -> C.Corpus:
    """Three speakers, two sessions each, everything annotated and consented."""
    cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
    for s in ("spk1", "spk2", "spk3"):
        cp.speakers.append(C.SpeakerRecord(speaker_id=s, consent_ref=f"consent/{s}"))
        for idx, day in enumerate(("2026-01-01", "2026-02-01")):
            sid = f"{s}_s{idx}"
            cp.sessions.append(
                C.SessionRecord(
                    session_id=sid, speaker_id=s, recorded_on=day,
                    device="phone", environment=C.Environment.QUIET_ROOM,
                )
            )
            cp.utterances.extend(
                utterance(f"{sid}_u{j}", sid, s) for j in range(4)
            )
    return cp


@pytest.fixture(scope="module")
def simulated() -> C.Corpus:
    sim = make_corpus(
        n_speakers=10, seed=5, enrolment_utterances=20, trial_utterances=6
    )
    return C.from_simulation(sim, sessions_per_speaker=2)


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


class TestProtocol:
    def test_every_elicitable_class_has_a_prompt(self):
        """A class nothing elicits cannot be estimated.

        The ontology marks 13 classes elicitable because a challenge is only
        useful if it reliably produces tokens of the class it targets.
        Discovering after collection that nothing asked about TRANSPORT costs
        a recording round, so this is asserted at import time in `corpus` and
        checked here as well.
        """
        targeted = {c for p in C.PROTOCOL_V1 for c in p.target_classes}
        missing = sorted(c.value for c in set(ELICITABLE_CLASSES) - targeted)
        assert not missing, f"no prompt elicits {missing}"

    def test_prompt_ids_are_unique(self):
        ids = [p.prompt_id for p in C.PROTOCOL_V1]
        assert len(ids) == len(set(ids))

    def test_protocol_carries_a_one_word_control(self):
        """§5.1.1: a one-word answer has no wrapper for the CSBG to read.

        Such a prompt is not a mistake -- it is the control that shows the
        null result belongs to the question. But it must be *labelled*, or its
        result gets read as evidence against the hypothesis.
        """
        controls = [p for p in C.PROTOCOL_V1 if not p.elicits_phrase]
        assert controls, "the protocol needs at least one one-word control"

    def test_prompts_are_bilingual(self):
        for p in C.PROTOCOL_V1:
            assert p.text_en.strip(), f"{p.prompt_id} has no English text"
            assert p.text_ta.strip(), f"{p.prompt_id} has no Tamil text"


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class TestValidate:
    def test_clean_corpus_validates(self, two_session_corpus):
        assert two_session_corpus.validate() == []

    def test_recorded_speaker_without_consent_is_invalid(self, two_session_corpus):
        """Consent is enforced, not documented.

        `data/` is git-ignored precisely because it holds voiceprints next to
        hometowns and family names. A manifest that lets an unconsented
        speaker through makes the ignore rule the only protection there is.
        """
        two_session_corpus.speakers[0].consent_ref = ""
        problems = two_session_corpus.validate()
        assert any("consent" in p for p in problems)

    def test_simulated_corpus_does_not_need_consent(self, simulated):
        assert not any("consent" in p for p in simulated.validate())

    def test_utterance_pointing_at_another_speakers_session_is_caught(
        self, two_session_corpus
    ):
        """The cross-reference that silently corrupts a split.

        An utterance claiming spk1 but filed under spk2's session lands in
        spk2's held-out probes while being scored as spk1's speech. Both
        speakers' numbers move and nothing raises.
        """
        two_session_corpus.utterances[0].speaker_id = "spk2"
        problems = two_session_corpus.validate()
        assert any("but its session belongs to" in p for p in problems)

    def test_unknown_session_reference_is_caught(self, two_session_corpus):
        two_session_corpus.utterances[0].session_id = "nope"
        assert any("unknown session" in p for p in two_session_corpus.validate())

    def test_unknown_prompt_is_caught(self, two_session_corpus):
        two_session_corpus.utterances[0].prompt_id = "p99_invented"
        assert any("unknown prompt" in p for p in two_session_corpus.validate())

    def test_duplicate_utterance_ids_are_caught(self, two_session_corpus):
        two_session_corpus.utterances[1].utterance_id = (
            two_session_corpus.utterances[0].utterance_id
        )
        assert any("duplicate utterance" in p for p in two_session_corpus.validate())

    def test_guessed_tokens_without_annotation_is_incoherent(self, two_session_corpus):
        two_session_corpus.utterances[0].tokens = None
        two_session_corpus.utterances[0].n_guessed_tokens = 3
        assert any("guessed tokens but has no annotation" in p
                   for p in two_session_corpus.validate())

    def test_audio_check_is_opt_in(self, two_session_corpus, tmp_path):
        """A manifest is usually read on a machine that does not hold the audio."""
        two_session_corpus.utterances[0].audio_path = "missing.wav"
        two_session_corpus.root = tmp_path
        assert two_session_corpus.validate() == []
        assert any("missing audio" in p
                   for p in two_session_corpus.validate(check_audio=True))


# --------------------------------------------------------------------------
# Reportability
# --------------------------------------------------------------------------


class TestReportability:
    def test_clean_recorded_corpus_is_reportable(self, two_session_corpus):
        assert two_session_corpus.reportability() == []

    def test_simulated_corpus_is_never_reportable(self, simulated):
        """The single property this module exists to guarantee."""
        reasons = simulated.reportability()
        assert reasons
        assert any("SIMULATED" in r for r in reasons)

    def test_unannotated_utterances_block_reporting(self, two_session_corpus):
        two_session_corpus.utterances[0].tokens = None
        two_session_corpus.utterances[0].annotation_source = None
        assert any("not annotated" in r for r in two_session_corpus.reportability())

    def test_guessed_tokens_block_reporting(self, two_session_corpus):
        """Same rule as `PipelineStats.is_corpus_grade`, not a second one."""
        two_session_corpus.utterances[0].n_guessed_tokens = 2
        two_session_corpus.utterances[0].annotation_source = C.AnnotationSource.RULES
        assert any("guessed" in r for r in two_session_corpus.reportability())

    def test_single_session_speakers_block_the_stability_claim(self):
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        cp.speakers.append(C.SpeakerRecord(speaker_id="a", consent_ref="c/a"))
        cp.sessions.append(C.SessionRecord(session_id="a_s0", speaker_id="a"))
        cp.utterances.append(utterance("a_s0_u0", "a_s0", "a"))
        assert any("one session" in r for r in cp.reportability())

    def test_ontology_drift_blocks_reporting(self, two_session_corpus):
        two_session_corpus.ontology_version = "0.9"
        assert any("ontology" in r for r in two_session_corpus.reportability())

    def test_augmented_is_not_reportable(self, two_session_corpus):
        two_session_corpus.provenance = C.Provenance.AUGMENTED
        assert any("not independent speakers" in r
                   for r in two_session_corpus.reportability())


# --------------------------------------------------------------------------
# The session split
# --------------------------------------------------------------------------


class TestSplitSessions:
    def test_enrolment_and_probes_share_no_session(self, two_session_corpus):
        """The property the whole split exists for.

        Utterances from one sitting share a microphone, a room, a topic and a
        mood. If any session appears on both sides, the EER measured is partly
        a measure of how well the estimator memorised a recording session.
        """
        split = C.split_sessions(two_session_corpus)
        for sid in split.enrolment:
            overlap = set(split.enrolment_sessions[sid]) & set(split.probe_sessions[sid])
            assert not overlap, f"{sid} shares sessions across the split: {overlap}"

    def test_enrolment_and_probes_share_no_utterance(self, two_session_corpus):
        split = C.split_sessions(two_session_corpus)
        for sid in split.enrolment:
            enrol = {u.utterance_id for u in split.enrolment[sid]}
            probe = {u.utterance_id for u in split.probes[sid]}
            assert not (enrol & probe)

    def test_probe_session_is_the_most_recent(self, two_session_corpus):
        """§5.3 is a claim about time, so the gap must be as long as available."""
        split = C.split_sessions(two_session_corpus)
        for sid in split.enrolment:
            probes = split.probe_sessions[sid]
            latest = two_session_corpus.sessions_of(sid)[-1].session_id
            assert probes == [latest]

    def test_single_session_speaker_is_dropped_not_downgraded(self):
        """Dropping is better than silently mixing two kinds of trial.

        A corpus containing both cross-session and within-session trials scored
        on one scale hides the mixture inside a single EER, and the more
        flattering trials are invisible.
        """
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        cp.speakers.append(C.SpeakerRecord(speaker_id="solo", consent_ref="c"))
        cp.sessions.append(C.SessionRecord(session_id="solo_s0", speaker_id="solo"))
        cp.utterances.extend(utterance(f"u{i}", "solo_s0", "solo") for i in range(6))

        split = C.split_sessions(cp)
        assert split.dropped_speakers == ["solo"]
        assert split.enrolment == {}
        assert split.cross_session is True

    def test_within_session_split_must_be_asked_for_and_announces_itself(self):
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        cp.speakers.append(C.SpeakerRecord(speaker_id="solo", consent_ref="c"))
        cp.sessions.append(C.SessionRecord(session_id="solo_s0", speaker_id="solo"))
        cp.utterances.extend(utterance(f"u{i}", "solo_s0", "solo") for i in range(10))

        split = C.split_sessions(cp, allow_within_session=True)
        assert split.dropped_speakers == []
        assert split.cross_session is False
        assert "within-session" in split.summary()

    def test_dropped_speakers_are_reported_in_the_summary(self, two_session_corpus):
        two_session_corpus.speakers.append(
            C.SpeakerRecord(speaker_id="solo", consent_ref="c")
        )
        two_session_corpus.sessions.append(
            C.SessionRecord(session_id="solo_s0", speaker_id="solo")
        )
        two_session_corpus.utterances.append(utterance("su0", "solo_s0", "solo"))
        split = C.split_sessions(two_session_corpus)
        assert "dropped 1" in split.summary()

    def test_unannotated_utterances_do_not_reach_the_split(self, two_session_corpus):
        for u in two_session_corpus.utterances:
            if u.session_id == "spk1_s0":
                u.tokens = None
        split = C.split_sessions(two_session_corpus)
        # spk1 loses its whole enrolment session, so it cannot be split.
        assert "spk1" in split.dropped_speakers

    def test_n_probe_sessions_must_be_positive(self, two_session_corpus):
        with pytest.raises(ValueError, match="n_probe_sessions"):
            C.split_sessions(two_session_corpus, n_probe_sessions=0)

    def test_requesting_more_probe_sessions_than_exist_drops_the_speaker(
        self, two_session_corpus
    ):
        split = C.split_sessions(two_session_corpus, n_probe_sessions=2)
        assert sorted(split.dropped_speakers) == ["spk1", "spk2", "spk3"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


class TestLoading:
    def test_unannotated_utterance_refuses_to_become_tokens(self):
        """`None` tokens must raise, not return an empty list.

        `CSBG.build` accepts an empty utterance list and returns a valid graph
        sitting entirely at the smoothing prior. That graph scores near-chance
        against everything, so an un-annotated corpus would produce an EER of
        roughly 50% and look like a negative result about the hypothesis
        rather than a pipeline that never ran.
        """
        rec = utterance("u", "s", "spk", annotated=False)
        with pytest.raises(ValueError, match="no annotation"):
            rec.to_utterance_tokens()

    def test_tokens_by_speaker_skips_unannotated(self, two_session_corpus):
        two_session_corpus.utterances[0].tokens = None
        by_speaker = two_session_corpus.tokens_by_speaker()
        assert len(by_speaker["spk1"]) == 7

    def test_empty_token_list_is_not_unannotated(self):
        """`[]` means "tagged, nothing scoreable"; `None` means "not tagged"."""
        rec = C.UtteranceRecord(
            utterance_id="u", session_id="s", speaker_id="spk", tokens=[]
        )
        assert rec.is_annotated
        assert rec.to_utterance_tokens().tokens == []

    def test_tokens_by_speaker_can_restrict_to_sessions(self, two_session_corpus):
        by_speaker = two_session_corpus.tokens_by_speaker(session_ids=["spk1_s0"])
        assert set(by_speaker) == {"spk1"}
        assert len(by_speaker["spk1"]) == 4

    def test_groups_include_speaker_and_session_attributes(self, two_session_corpus):
        groups = two_session_corpus.groups()
        assert groups["spk1"]["device"] == "phone"
        assert groups["spk1"]["environment"] == "QUIET_ROOM"

    def test_conflicting_session_attribute_is_omitted_not_guessed(
        self, two_session_corpus
    ):
        """A speaker recorded on two devices has no single device.

        Asserting one would file every one of their trials into a slice that
        half their audio contradicts, which is worse than having no slice.
        """
        two_session_corpus.sessions[0].device = "laptop"
        groups = two_session_corpus.groups()
        assert "device" not in groups["spk1"]
        assert groups["spk2"]["device"] == "phone"

    def test_empty_attributes_are_dropped(self):
        spk = C.SpeakerRecord(speaker_id="a", gender="", dominant_language="TA")
        assert spk.attributes() == {"dominant_language": "TA"}


# --------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------


class TestManifestIO:
    def test_round_trip_is_lossless(self, two_session_corpus, tmp_path):
        path = C.save_manifest(two_session_corpus, tmp_path / "manifest.json")
        back = C.load_manifest(path)
        assert back.to_dict() == two_session_corpus.to_dict()

    def test_round_trip_preserves_token_annotations(self, two_session_corpus, tmp_path):
        two_session_corpus.utterances[0].tokens = [
            Token(text="வீடு", language=Language.TA,
                  semantic_class=SemanticClass.PLACE_LOCAL,
                  lid_confidence=0.75, start_ms=10, end_ms=430)
        ]
        path = C.save_manifest(two_session_corpus, tmp_path / "m.json")
        loaded = C.load_manifest(path).utterances[0].tokens
        assert loaded is not None
        assert loaded[0].text == "வீடு"
        assert loaded[0].language is Language.TA
        assert loaded[0].semantic_class is SemanticClass.PLACE_LOCAL
        assert loaded[0].lid_confidence == 0.75
        assert (loaded[0].start_ms, loaded[0].end_ms) == (10, 430)

    def test_unknown_manifest_version_refuses_to_load(self, tmp_path):
        """Guessing at an unfamiliar layout produces a corpus that is wrong."""
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"manifest_version": "99.0"}), encoding="utf-8")
        with pytest.raises(ValueError, match="manifest version"):
            C.load_manifest(path)

    def test_root_is_the_manifest_directory(self, two_session_corpus, tmp_path):
        nested = tmp_path / "corpus"
        path = C.save_manifest(two_session_corpus, nested / "manifest.json")
        assert C.load_manifest(path).root == nested

    def test_manifest_is_utf8_and_keeps_tamil_readable(
        self, two_session_corpus, tmp_path
    ):
        two_session_corpus.utterances[0].transcript = "நான் சாப்பிட்டேன்"
        path = C.save_manifest(two_session_corpus, tmp_path / "m.json")
        assert "நான்" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


class TestCoverage:
    def test_own_evidence_threshold_tracks_the_smoothing_config(self):
        """Derived from `class_alpha`, never written here.

        PROJECT.md §5.1 records two thresholds that were wrong because the
        constant lived in a module that could not observe what it thresholded.
        This one reads the pseudo-count it is a threshold on.
        """
        smoothing = SmoothingConfig(class_alpha=9.0)
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        report = C.coverage_report(cp, smoothing=smoothing)
        assert report.min_tokens_for_own_evidence == 9.0

    def test_a_class_below_alpha_has_no_own_evidence(self):
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        cp.speakers.append(C.SpeakerRecord(speaker_id="a", consent_ref="c"))
        cp.sessions.append(C.SessionRecord(session_id="a_s0", speaker_id="a"))
        cp.utterances.append(
            C.UtteranceRecord(
                utterance_id="u", session_id="a_s0", speaker_id="a",
                tokens=[tok(SemanticClass.FOOD) for _ in range(3)],
            )
        )
        report = C.coverage_report(cp, smoothing=SmoothingConfig(class_alpha=4.0))
        food = next(c for c in report.classes if c.semantic_class is SemanticClass.FOOD)
        assert food.n_tokens == 3
        assert food.n_speakers_with_own_evidence == 0

    def test_starved_classes_names_elicitable_classes_only(self, two_session_corpus):
        """Every class but FOOD is starved in the fixture; only elicitable ones
        are actionable, because nothing in the protocol can fix the others."""
        report = two_session_corpus.coverage()
        starved = {c.semantic_class for c in report.starved_classes()}
        assert SemanticClass.FOOD not in starved
        assert starved <= set(ELICITABLE_CLASSES)
        assert SemanticClass.TECH_DIGITAL in starved

    def test_language_independent_tokens_do_not_count_as_evidence(self):
        """A named entity is not a language choice, so it is not coverage."""
        cp = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
        cp.speakers.append(C.SpeakerRecord(speaker_id="a", consent_ref="c"))
        cp.sessions.append(C.SessionRecord(session_id="a_s0", speaker_id="a"))
        cp.utterances.append(
            C.UtteranceRecord(
                utterance_id="u", session_id="a_s0", speaker_id="a",
                tokens=[tok(SemanticClass.FOOD, Language.NAMED_ENTITY)
                        for _ in range(20)],
            )
        )
        report = cp.coverage()
        assert report.total_tokens == 20
        assert report.total_choice_tokens == 0

    def test_prompt_hit_rate_flags_a_broken_question(self, two_session_corpus):
        """A prompt whose targets never appear is a question that does not work.

        Worth finding at five speakers rather than thirty.
        """
        for u in two_session_corpus.utterances:
            u.prompt_id = "p05_phone"  # targets TECH_DIGITAL; fixture emits FOOD
        report = two_session_corpus.coverage()
        phone = next(p for p in report.prompts if p.prompt_id == "p05_phone")
        assert phone.hit_rate == 0.0

    def test_prompt_hit_rate_is_one_when_the_question_works(self, two_session_corpus):
        report = two_session_corpus.coverage()
        food = next(p for p in report.prompts if p.prompt_id == "p02_food")
        assert food.hit_rate == 1.0

    def test_wrapperless_prompts_exclude_the_labelled_control(
        self, two_session_corpus
    ):
        """The one-word control is *expected* to be short.

        Listing it beside genuinely broken prompts would train the reader to
        ignore the list.
        """
        for u in two_session_corpus.utterances:
            u.prompt_id = "p14_control_name"
            u.tokens = [tok(SemanticClass.KINSHIP)]
        report = two_session_corpus.coverage()
        assert report.wrapperless_prompts() == []

    def test_wrapperless_prompts_catch_a_phrase_prompt_answered_in_one_word(
        self, two_session_corpus
    ):
        for u in two_session_corpus.utterances:
            u.tokens = [tok(SemanticClass.FOOD)]
        report = two_session_corpus.coverage()
        assert [p.prompt_id for p in report.wrapperless_prompts()] == ["p02_food"]

    def test_markdown_renders(self, two_session_corpus):
        text = two_session_corpus.coverage().to_markdown()
        assert "Corpus coverage" in text
        assert "FOOD" in text


# --------------------------------------------------------------------------
# The simulation bridge
# --------------------------------------------------------------------------


class TestSimulationBridge:
    def test_bridge_produces_a_valid_corpus(self, simulated):
        assert simulated.validate() == []

    def test_bridge_marks_provenance_simulated(self, simulated):
        assert simulated.provenance is C.Provenance.SIMULATED
        assert not simulated.provenance.is_reportable

    def test_bridge_notes_that_sessions_are_a_fiction(self, simulated):
        """The one thing a reader of this corpus must not conclude.

        i.i.d. draws from one profile show none of the drift a real second
        session would, so a §5.3 number read off this corpus would be an
        upper bound presented as a measurement.
        """
        assert "fiction" in simulated.notes

    def test_bridge_gives_every_speaker_multiple_sessions(self, simulated):
        for sid in simulated.speaker_ids:
            assert len(simulated.sessions_of(sid)) >= 2

    def test_bridge_output_splits_without_dropping_anyone(self, simulated):
        split = C.split_sessions(simulated)
        assert split.dropped_speakers == []
        assert len(split.enrolment) == len(simulated.speaker_ids)

    def test_sessions_per_speaker_must_be_positive(self):
        sim = make_corpus(n_speakers=3, seed=1, enrolment_utterances=4,
                          trial_utterances=2)
        with pytest.raises(ValueError, match="sessions_per_speaker"):
            C.from_simulation(sim, sessions_per_speaker=0)

    def test_trial_utterances_become_the_held_out_session(self, simulated):
        """The simulator's trial set is already disjoint from enrolment, and
        the bridge must preserve that rather than reshuffling it."""
        split = C.split_sessions(simulated)
        probe_ids = {u.utterance_id for u in split.probes["sim_000"]}
        assert probe_ids
        assert all(i.startswith("trial_") for i in probe_ids)


# --------------------------------------------------------------------------
# Integration with the evaluation
# --------------------------------------------------------------------------


class TestEvaluationIntegration:
    def test_split_feeds_build_trials_unchanged(self, simulated):
        """The requirement the corpus layer was specified against.

        `eval/ablation.py` must need no changes: the loader emits exactly the
        `dict[str, list[UtteranceTokens]]` shape `build_trials` already takes.
        """
        split = C.split_sessions(simulated)
        graphs = {sid: CSBG.build(sid, utts) for sid, utts in split.enrolment.items()}
        trials = build_trials(graphs, split.probes, groups=simulated.groups())
        assert trials
        assert any(t.is_genuine for t in trials)
        assert any(not t.is_genuine for t in trials)

    def test_genuine_trials_never_score_a_probe_against_its_own_session(
        self, simulated
    ):
        split = C.split_sessions(simulated)
        for sid in split.enrolment:
            enrol = {u.utterance_id for u in split.enrolment[sid]}
            probe = {u.utterance_id for u in split.probes[sid]}
            assert not (enrol & probe)

    def test_run_ablation_accepts_a_corpus_split(self, simulated):
        split = C.split_sessions(simulated)
        graphs = {sid: CSBG.build(sid, utts) for sid, utts in split.enrolment.items()}
        report = run_ablation(
            graphs, split.probes, groups=simulated.groups(), bootstrap=25
        )
        assert report.configurations
        assert "Verification results" in report.to_markdown()

    def test_fairness_groups_survive_into_the_trials(self, simulated):
        split = C.split_sessions(simulated)
        graphs = {sid: CSBG.build(sid, utts) for sid, utts in split.enrolment.items()}
        trials = build_trials(graphs, split.probes, groups=simulated.groups())
        assert any(t.group.get("device") == "simulated" for t in trials)
