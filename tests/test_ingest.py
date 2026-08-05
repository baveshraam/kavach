"""Tests for the ingest step.

The interesting cases here are not the happy path. They are the three ways a
returned folder quietly corrupts a corpus: a filename that resolves to the
wrong prompt, a participant's real name reaching the manifest, and a missing
file that loads as a valid corpus with a hole in it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from kavach import corpus as C
from kavach import ingest as I
from kavach.audio import Audio, save_wav

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "participant_scripts"


def _tone_wav(path: Path, seconds: float = 4.0, lead: float = 0.5) -> None:
    """A wav with `lead` seconds of quiet, then speech-ish noise."""
    sr = 16_000
    quiet = np.random.default_rng(0).normal(0, 1e-4, int(lead * sr))
    t = np.arange(int((seconds - lead) * sr)) / sr
    voiced = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    save_wav(Audio(np.concatenate([quiet, voiced]).astype(np.float32), sr), path)


@pytest.fixture
def returned_folder(tmp_path: Path) -> Path:
    """One participant's folder, named after them, with all 14 answers."""
    folder = tmp_path / "recordings" / "priya"
    folder.mkdir(parents=True)
    for i in range(1, 15):
        _tone_wav(folder / f"{i:02d}.wav")
    return tmp_path / "recordings"


@pytest.fixture
def one_speaker() -> list[I.SpeakerSpec]:
    return [
        I.SpeakerSpec(
            speaker_id="S01",
            folder="priya",
            script_id="A",
            consent_ref="consent/2026-08-06-S01",
            device="Redmi Note 12",
            environment="QUIET_ROOM",
            recorded_on="2026-08-06",
        )
    ]


class TestResolvePromptId:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("1.m4a", "p01_family"),
            ("01.m4a", "p01_family"),
            ("Q1.ogg", "p01_family"),
            ("03.wav", "p03_commute"),
            ("Recording 03.m4a", "p03_commute"),
            ("14.m4a", "p14_control_name"),
            ("p07_festival.m4a", "p07_festival"),
        ],
    )
    def test_accepts_the_forms_people_actually_send(self, filename, expected):
        assert I.resolve_prompt_id(filename) == expected

    @pytest.mark.parametrize("filename", ["notes.txt", "recording.m4a", "0.m4a", "15.m4a"])
    def test_returns_none_rather_than_guessing(self, filename):
        """A file assigned to the wrong prompt still validates. Refuse instead."""
        assert I.resolve_prompt_id(filename) is None

    def test_a_bracketed_duplicate_counter_is_not_the_prompt_number(self):
        """`03 (1).m4a` is a re-download of prompt 3, not prompt 1."""
        assert I.resolve_prompt_id("03 (1).m4a") == "p03_commute"

    def test_position_not_prompt_id(self):
        """The number is an ordinal into PROTOCOL_V1, which is what makes
        reordering the protocol a silent relabelling of collected files."""
        for position, prompt in enumerate(C.PROTOCOL_V1, start=1):
            assert I.resolve_prompt_id(f"{position:02d}.m4a") == prompt.prompt_id


class TestParseScript:
    def test_every_shipped_script_yields_all_fourteen_answers(self):
        """A script missing an answer sends a participant a form with a hole."""
        scripts = sorted(SCRIPTS_DIR.glob("SPEAKER_*.md"))
        assert len(scripts) == 10, "expected SPEAKER_A..J"
        for path in scripts:
            parsed = I.parse_script(path)
            missing = [p.prompt_id for p in C.PROTOCOL_V1 if p.prompt_id not in parsed]
            assert not missing, f"{path.name} has no text for {missing}"

    def test_drops_the_instructions_addressed_to_the_reader(self):
        parsed = I.parse_script(SCRIPTS_DIR / "SPEAKER_A.md")
        assert "Shorter" not in parsed["p10_numbers"]
        assert "Very short" not in parsed["p14_control_name"]

    def test_keeps_the_answer_and_not_the_preamble(self):
        parsed = I.parse_script(SCRIPTS_DIR / "SPEAKER_A.md")
        assert parsed["p14_control_name"] == "En amma peru Vasanthi."
        assert "wait one second" not in " ".join(parsed.values()).lower()

    def test_scripts_differ_from_each_other(self):
        """Ten identical scripts produce ten identical CSBGs and a 50% EER.
        This is the assertion that the corpus can return a result at all."""
        bodies = {
            path.stem: " ".join(I.parse_script(path).values())
            for path in sorted(SCRIPTS_DIR.glob("SPEAKER_*.md"))
        }
        assert len(set(bodies.values())) == len(bodies)

    def test_find_script_is_case_insensitive_and_returns_none_when_absent(self):
        assert I.find_script("a", scripts_dir=SCRIPTS_DIR) is not None
        assert I.find_script("Z", scripts_dir=SCRIPTS_DIR) is None
        assert I.find_script("", scripts_dir=SCRIPTS_DIR) is None


class TestSpeakerCsv:
    def test_missing_required_columns_is_an_error_not_a_default(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text("speaker_id,gender\nS01,F\n")
        with pytest.raises(ValueError, match="folder"):
            I.read_speaker_csv(path)

    def test_template_lists_every_folder_and_assigns_pseudonyms(self, tmp_path):
        root = tmp_path / "recordings"
        for name in ("zoya", "arun", "meera"):
            (root / name).mkdir(parents=True)
        out = tmp_path / "speakers.csv"
        folders = I.write_template_csv(root, out)

        rows = list(csv.DictReader(out.open()))
        assert folders == ["arun", "meera", "zoya"]
        assert [r["speaker_id"] for r in rows] == ["S01", "S02", "S03"]
        assert [r["folder"] for r in rows] == ["arun", "meera", "zoya"]
        assert all(r["consent_ref"] == "" for r in rows)

    def test_blank_rows_are_skipped(self, tmp_path):
        path = tmp_path / "s.csv"
        path.write_text("speaker_id,folder\nS01,a\n,\nS02,b\n")
        assert [s.speaker_id for s in I.read_speaker_csv(path)] == ["S01", "S02"]


class TestIngest:
    def test_builds_a_valid_corpus_from_a_returned_folder(
        self, returned_folder, one_speaker, tmp_path
    ):
        corpus, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert corpus.validate() == []
        assert len(corpus.utterances) == len(C.PROTOCOL_V1)
        assert len(report.ingested) == len(C.PROTOCOL_V1)
        assert not report.problems

    def test_the_participants_name_never_reaches_the_manifest(
        self, returned_folder, one_speaker, tmp_path
    ):
        """The folder is a person's first name sitting next to a voiceprint.
        The CSV maps it to a pseudonym and that mapping stays out of here."""
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        blob = json.dumps(corpus.to_dict())
        assert "priya" not in blob.lower()
        assert all("priya" not in u.audio_path.lower() for u in corpus.utterances)

    def test_audio_paths_are_relative_to_the_corpus_root(
        self, returned_folder, one_speaker, tmp_path
    ):
        root = tmp_path / "corpus"
        corpus, _ = I.ingest(returned_folder, one_speaker, root, scripts_dir=SCRIPTS_DIR)
        for u in corpus.utterances:
            assert not Path(u.audio_path).is_absolute()
            assert (root / u.audio_path).exists()

    def test_reference_transcript_comes_from_the_speakers_own_script(
        self, returned_folder, one_speaker, tmp_path
    ):
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        control = next(u for u in corpus.utterances if u.prompt_id == "p14_control_name")
        assert control.reference_transcript == "En amma peru Vasanthi."

    def test_the_asr_transcript_field_is_left_empty_for_annotate_to_fill(
        self, returned_folder, one_speaker, tmp_path
    ):
        """Writing the script into `transcript` would make it the annotation
        input, and any word error rate measured against it zero."""
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert all(u.transcript == "" for u in corpus.utterances)
        assert all(u.asr_wer is None for u in corpus.utterances)

    def test_transcripts_are_absent_when_not_requested(
        self, returned_folder, one_speaker, tmp_path
    ):
        corpus, _ = I.ingest(
            returned_folder,
            one_speaker,
            tmp_path / "corpus",
            scripts_dir=SCRIPTS_DIR,
            reference_transcripts=False,
        )
        assert all(u.reference_transcript == "" for u in corpus.utterances)

    def test_a_missing_answer_is_a_problem_not_a_silent_gap(
        self, returned_folder, one_speaker, tmp_path
    ):
        (returned_folder / "priya" / "07.wav").unlink()
        corpus, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert len(corpus.utterances) == len(C.PROTOCOL_V1) - 1
        assert any("p07_festival" in p for p in report.problems)

    def test_two_files_claiming_one_prompt_leaves_the_first_and_reports(
        self, returned_folder, one_speaker, tmp_path
    ):
        _tone_wav(returned_folder / "priya" / "03 (1).wav")
        _, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert any("already taken" in s for s in report.skipped)

    def test_unparseable_filenames_are_skipped_loudly(
        self, returned_folder, one_speaker, tmp_path
    ):
        _tone_wav(returned_folder / "priya" / "intro.wav")
        _, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert any("intro.wav" in s for s in report.skipped)

    def test_a_missing_folder_is_reported_against_the_speaker(self, tmp_path):
        specs = [I.SpeakerSpec(speaker_id="S09", folder="nobody", consent_ref="c")]
        corpus, report = I.ingest(tmp_path, specs, tmp_path / "corpus")
        assert corpus.utterances == []
        assert any("S09" in p and "nobody" in p for p in report.problems)

    def test_dry_run_writes_nothing(self, returned_folder, one_speaker, tmp_path):
        root = tmp_path / "corpus"
        corpus, report = I.ingest(
            returned_folder, one_speaker, root, scripts_dir=SCRIPTS_DIR, copy_audio=False
        )
        assert not root.exists()
        assert len(report.ingested) == len(C.PROTOCOL_V1)
        assert all(u.audio_path == "" for u in corpus.utterances)

    def test_an_unknown_environment_is_reported_rather_than_dropped(
        self, returned_folder, tmp_path
    ):
        specs = [
            I.SpeakerSpec(
                speaker_id="S01", folder="priya", consent_ref="c", environment="BEDROOM"
            )
        ]
        _, report = I.ingest(returned_folder, specs, tmp_path / "corpus")
        assert any("BEDROOM" in p for p in report.problems)

    def test_default_provenance_is_scripted_and_therefore_unreportable(
        self, returned_folder, one_speaker, tmp_path
    ):
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert corpus.provenance is C.Provenance.SCRIPTED
        assert any("SCRIPTED" in r for r in corpus.reportability())


class TestSignalChecks:
    def test_a_file_that_starts_mid_syllable_is_flagged(
        self, returned_folder, one_speaker, tmp_path
    ):
        _tone_wav(returned_folder / "priya" / "05.wav", lead=0.0)
        _, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        outcome = next(f for f in report.ingested if f.prompt_id == "p05_phone")
        assert any("beat of silence" in w for w in outcome.warnings)

    def test_a_voice_note_extension_is_flagged_as_transit_encoded(self):
        """Opus in an .ogg is what WhatsApp turns a voice note into, and it is
        a different codec from every other speaker's recorder -- which makes
        the codec itself speaker-identifying."""
        assert ".ogg" in I.TRANSIT_ENCODED_SUFFIXES
        assert ".m4a" not in I.TRANSIT_ENCODED_SUFFIXES

    def test_warnings_do_not_stop_a_file_being_ingested(
        self, returned_folder, one_speaker, tmp_path
    ):
        """Rejecting a file means asking a participant to record again, which
        mostly means losing the participant. Flag, load, let a human decide."""
        _tone_wav(returned_folder / "priya" / "05.wav", lead=0.0)
        corpus, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert len(corpus.utterances) == len(C.PROTOCOL_V1)
        assert not report.is_clean

    def test_report_markdown_names_every_flagged_file(
        self, returned_folder, one_speaker, tmp_path
    ):
        _tone_wav(returned_folder / "priya" / "05.wav", lead=0.0)
        _, report = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        markdown = report.to_markdown()
        assert "p05_phone" in markdown
        assert "S01" in markdown


class TestConsent:
    def test_a_scripted_corpus_still_requires_consent(
        self, returned_folder, tmp_path
    ):
        """Consent covers the voiceprint, not the words. A speaker who read
        invented text is exactly as clonable as one who did not."""
        specs = [I.SpeakerSpec(speaker_id="S01", folder="priya", consent_ref="")]
        corpus, _ = I.ingest(
            returned_folder, specs, tmp_path / "corpus", provenance=C.Provenance.SCRIPTED
        )
        assert any("consent" in p for p in corpus.validate())

    def test_consent_reference_satisfies_it(self, returned_folder, one_speaker, tmp_path):
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        assert not any("consent" in p for p in corpus.validate())


class TestRoundTrip:
    def test_script_id_survives_save_and_load(
        self, returned_folder, one_speaker, tmp_path
    ):
        """The script letter is the answer key for a scripted run. Losing it in
        serialisation means the corpus can be scored but not checked."""
        corpus, _ = I.ingest(
            returned_folder, one_speaker, tmp_path / "corpus", scripts_dir=SCRIPTS_DIR
        )
        path = C.save_manifest(corpus, tmp_path / "corpus" / "manifest.json")
        reloaded = C.load_manifest(path)
        assert reloaded.speaker("S01").script_id == "A"
        assert reloaded.provenance is C.Provenance.SCRIPTED
