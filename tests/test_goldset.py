"""Gold-set tests.

The Track 1 number is word-level LID accuracy against human labels, and almost
every way of getting it wrong yields a *higher* number rather than an error:

- prefilling the annotator's file with the system's own guesses turns labelling
  into agreeing, so the file records that it was prefilled and the score
  carries the caveat;
- scoring gold labels against a re-transcribed utterance compares different
  words at the same index, so misalignment is detected and excluded rather than
  counted;
- treating a blank row as a mistake would punish "I am not sure"; treating it
  as correct would inflate the number. It is dropped and counted separately;
- sampling uniformly from a 4-speaker corpus can put most tokens on one voice,
  and LID accuracy is a per-speaker property here.
"""

from __future__ import annotations

import json

import pytest

from kavach import corpus as C
from kavach import goldset as G
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _tok(text, lang, cls=SemanticClass.OTHER):
    return Token(text=text, language=lang, semantic_class=cls)


def _corpus(n_speakers: int = 3, n_utterances: int = 4) -> C.Corpus:
    corpus = C.Corpus(name="gold-t", provenance=C.Provenance.RECORDED)
    for s in range(n_speakers):
        sid = f"S{s:02d}"
        corpus.speakers.append(C.SpeakerRecord(speaker_id=sid, consent_ref=f"c/{sid}"))
        session_id = f"{sid}_s1"
        corpus.sessions.append(C.SessionRecord(session_id=session_id, speaker_id=sid))
        for u in range(n_utterances):
            corpus.utterances.append(
                C.UtteranceRecord(
                    utterance_id=f"{session_id}_u{u}",
                    session_id=session_id,
                    speaker_id=sid,
                    transcript="நான் morning six மணிக்கு",
                    tokens=[
                        _tok("நான்", Language.TA, SemanticClass.FUNCTION_WORD),
                        _tok("morning", Language.EN, SemanticClass.TIME_DATE),
                        _tok("six", Language.EN, SemanticClass.NUMBER),
                        _tok("மணிக்கு", Language.TA, SemanticClass.TIME_DATE),
                    ],
                )
            )
    return corpus


def _write_gold(tmp_path, rows, *, prefilled=False, name="gold.tsv"):
    """rows: (utterance_id, index, token, lang, class)."""
    path = tmp_path / name
    header = [
        f"# {G.GOLDSET_FORMAT}",
        f"# prefilled: {'yes' if prefilled else 'no'}",
        "\t".join(G.COLUMNS),
    ]
    body = ["\t".join([*(str(c) for c in r), ""]) for r in rows]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


class TestExport:
    def test_writes_one_row_per_token(self, tmp_path):
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv")
        rows = [l for l in out.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
        assert rows[0].split("\t") == list(G.COLUMNS)
        assert len(rows) == 5  # header + 4 tokens

    def test_labels_are_blank_by_default(self, tmp_path):
        """The default has to be blind labelling: a corrector who sees 'TA'
        agrees with it far more often than a labeller starting from nothing."""
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv")
        gold = G.load(out)
        assert all(t.language is None for t in gold.tokens)
        assert gold.prefilled is False

    def test_prefill_fills_the_labels_and_says_so(self, tmp_path):
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv", prefill=True)
        gold = G.load(out)
        assert gold.prefilled is True
        assert [t.language for t in gold.tokens] == [
            Language.TA, Language.EN, Language.EN, Language.TA
        ]

    def test_prefilled_file_carries_a_warning_a_human_will_see(self, tmp_path):
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv", prefill=True)
        assert "overstates accuracy" in out.read_text(encoding="utf-8")

    def test_header_tells_the_labeller_about_transliteration(self, tmp_path):
        """The one instruction a bilingual labeller will not guess: label what
        was said, not the script it was written in."""
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv")
        text = out.read_text(encoding="utf-8")
        assert "not the script" in text
        assert "மானிங்க்" in text

    def test_header_lists_every_label(self, tmp_path):
        out = G.export(_corpus(1, 1), tmp_path / "g.tsv")
        text = out.read_text(encoding="utf-8")
        for cls in SemanticClass:
            assert cls.value in text
        for lang in Language:
            assert lang.value in text

    def test_sample_is_spread_across_speakers(self, tmp_path):
        """A random 6 from a 3-speaker corpus can land 5 on one voice, and one
        speaker's romanisation habit is not the corpus's."""
        corpus = _corpus(n_speakers=3, n_utterances=8)
        chosen = G.sample_utterances(corpus, n=6, seed=1)
        counts = {}
        for u in chosen:
            counts[u.speaker_id] = counts.get(u.speaker_id, 0) + 1
        assert set(counts) == {"S00", "S01", "S02"}
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_sample_smaller_than_the_speaker_count_still_works(self):
        chosen = G.sample_utterances(_corpus(4, 3), n=2, seed=0)
        assert len(chosen) == 2
        assert len({u.speaker_id for u in chosen}) == 2

    def test_sample_caps_at_what_exists(self):
        assert len(G.sample_utterances(_corpus(2, 2), n=99, seed=0)) == 4

    def test_a_corpus_with_no_transcripts_says_run_asr(self, tmp_path):
        corpus = _corpus(1, 1)
        corpus.utterances[0].transcript = ""
        with pytest.raises(G.GoldsetError, match="--stage asr"):
            G.export(corpus, tmp_path / "g.tsv")

    def test_export_is_deterministic_for_a_seed(self, tmp_path):
        corpus = _corpus(3, 5)
        a = G.export(corpus, tmp_path / "a.tsv", n_utterances=4, seed=7)
        b = G.export(corpus, tmp_path / "b.tsv", n_utterances=4, seed=7)
        assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


class TestLoad:
    def test_round_trips_an_export(self, tmp_path):
        out = G.export(_corpus(2, 2), tmp_path / "g.tsv", prefill=True)
        gold = G.load(out)
        assert len(gold.tokens) == 16
        assert len(gold.utterance_ids) == 4

    def test_blank_rows_are_kept_but_not_labelled(self, tmp_path):
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "morning", "", ""),
        ])
        gold = G.load(path)
        assert len(gold.tokens) == 2
        assert len(gold.labelled) == 1

    def test_a_half_labelled_row_is_not_labelled(self, tmp_path):
        """Language accuracy over the rows where the annotator also got round
        to the class is a different population from language accuracy."""
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "TA", "")])
        assert G.load(path).labelled == []

    def test_labels_are_case_insensitive(self, tmp_path):
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "ta", "function_word")])
        assert G.load(path).labelled[0].language is Language.TA

    def test_an_unknown_label_names_the_line_and_the_valid_values(self, tmp_path):
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "morning", "TAMIL", "TIME_DATE"),
        ])
        with pytest.raises(G.GoldsetError, match=r"gold\.tsv:5: lang='TAMIL'"):
            G.load(path)

    def test_the_error_lists_the_valid_labels(self, tmp_path):
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "x", "XX", "OTHER")])
        with pytest.raises(G.GoldsetError, match="NAMED_ENTITY"):
            G.load(path)

    def test_a_shifted_header_is_refused(self, tmp_path):
        path = tmp_path / "g.tsv"
        path.write_text("utterance_id\ttoken\tindex\tlang\tclass\tnote\n", encoding="utf-8")
        with pytest.raises(G.GoldsetError, match="Re-export"):
            G.load(path)

    def test_a_non_numeric_index_names_the_line(self, tmp_path):
        path = _write_gold(tmp_path, [("S00_s1_u0", "one", "நான்", "TA", "OTHER")])
        with pytest.raises(G.GoldsetError, match="is not a number"):
            G.load(path)

    def test_a_comments_only_file_is_refused(self, tmp_path):
        path = tmp_path / "g.tsv"
        path.write_text("# nothing here\n", encoding="utf-8")
        with pytest.raises(G.GoldsetError, match="no rows"):
            G.load(path)

    def test_trailing_blank_lines_are_ignored(self, tmp_path):
        """Spreadsheets add them on export and they are not an error."""
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "TA", "OTHER")])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n\n", encoding="utf-8")
        assert len(G.load(path).tokens) == 1


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------


class TestScore:
    def test_perfect_agreement_is_one(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "morning", "EN", "TIME_DATE"),
            ("S00_s1_u0", 2, "six", "EN", "NUMBER"),
            ("S00_s1_u0", 3, "மணிக்கு", "TA", "TIME_DATE"),
        ])
        result = G.score(G.load(path), corpus)
        assert result.n_tokens == 4
        assert result.language_accuracy == 1.0
        assert result.class_accuracy == 1.0
        assert result.joint_accuracy == 1.0

    def test_counts_language_and_class_errors_separately(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "EN", "FUNCTION_WORD"),   # language wrong
            ("S00_s1_u0", 1, "morning", "EN", "NUMBER"),        # class wrong
            ("S00_s1_u0", 2, "six", "EN", "NUMBER"),            # both right
            ("S00_s1_u0", 3, "மணிக்கு", "TA", "TIME_DATE"),     # both right
        ])
        result = G.score(G.load(path), corpus)
        assert result.language_accuracy == 0.75
        assert result.class_accuracy == 0.75
        assert result.joint_accuracy == 0.5

    def test_records_the_confusion_direction(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "EN", "FUNCTION_WORD")])
        result = G.score(G.load(path), corpus)
        assert result.worst_confusions() == [("EN", "TA", 1)]

    def test_blank_rows_are_dropped_not_scored(self, tmp_path):
        """A blank row is 'I am not sure', which is neither a hit nor a miss."""
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "morning", "", ""),
            ("S00_s1_u0", 2, "six", "", ""),
        ])
        result = G.score(G.load(path), corpus)
        assert result.n_tokens == 1
        assert result.n_unlabelled == 2
        assert result.language_accuracy == 1.0

    def test_misaligned_utterances_are_excluded_not_scored(self, tmp_path):
        """The failure this exists for: re-running ASR after export makes token
        5 a different word, and scoring anyway produces a plausible number."""
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "evening", "EN", "TIME_DATE"),  # corpus says "morning"
        ])
        result = G.score(G.load(path), corpus)
        assert result.misaligned == ["S00_s1_u0"]
        assert result.n_tokens == 0

    def test_an_index_past_the_end_is_misalignment_not_a_crash(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [("S00_s1_u0", 99, "நான்", "TA", "OTHER")])
        result = G.score(G.load(path), corpus)
        assert result.misaligned == ["S00_s1_u0"]

    def test_an_utterance_not_in_the_corpus_is_reported(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [("ghost", 0, "நான்", "TA", "OTHER")])
        result = G.score(G.load(path), corpus)
        assert result.missing_utterances == ["ghost"]

    def test_an_untagged_utterance_is_reported_not_scored_as_wrong(self, tmp_path):
        corpus = _corpus(1, 1)
        corpus.utterances[0].tokens = None
        path = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD")])
        result = G.score(G.load(path), corpus)
        assert result.missing_utterances == ["S00_s1_u0"]
        assert result.n_tokens == 0

    def test_counts_speakers_and_utterances(self, tmp_path):
        corpus = _corpus(2, 2)
        path = _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD"),
            ("S01_s1_u1", 0, "நான்", "TA", "FUNCTION_WORD"),
        ])
        result = G.score(G.load(path), corpus)
        assert result.n_utterances == 2
        assert result.n_speakers == 2

    def test_prefilled_flag_survives_into_the_score(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(
            tmp_path, [("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD")], prefilled=True
        )
        result = G.score(G.load(path), corpus)
        assert result.prefilled is True
        assert "may not be reported as blind" in result.to_markdown()


class TestTransliterationSlice:
    """The finding this tooling has to be able to measure: Whisper writes some
    English in Tamil script, and those tokens land on NUMBER and TIME_DATE."""

    def _corpus_with_transliteration(self, predicted: Language) -> C.Corpus:
        corpus = _corpus(1, 1)
        u = corpus.utterances[0]
        u.transcript = "நாம் மானிங்க் சிக்ஸ்"
        u.tokens = [
            _tok("நாம்", Language.TA, SemanticClass.FUNCTION_WORD),
            _tok("மானிங்க்", predicted, SemanticClass.TIME_DATE),
            _tok("சிக்ஸ்", predicted, SemanticClass.NUMBER),
        ]
        return corpus

    def _gold(self, tmp_path):
        return _write_gold(tmp_path, [
            ("S00_s1_u0", 0, "நாம்", "TA", "FUNCTION_WORD"),
            ("S00_s1_u0", 1, "மானிங்க்", "EN", "TIME_DATE"),
            ("S00_s1_u0", 2, "சிக்ஸ்", "EN", "NUMBER"),
        ])

    def test_counts_tamil_script_tokens_the_gold_calls_english(self, tmp_path):
        result = G.score(
            G.load(self._gold(tmp_path)), self._corpus_with_transliteration(Language.EN)
        )
        assert result.transliterated_total == 2
        assert result.transliterated_correct == 2
        assert result.transliteration_recall == 1.0

    def test_a_pipeline_that_trusts_script_scores_zero_recall(self, tmp_path):
        """Script evidence winning every transliterated token is exactly the
        bug the override was added for, and this is how it shows up."""
        result = G.score(
            G.load(self._gold(tmp_path)), self._corpus_with_transliteration(Language.TA)
        )
        assert result.transliterated_total == 2
        assert result.transliterated_correct == 0
        assert result.transliteration_recall == 0.0

    def test_the_report_says_what_a_miss_costs(self, tmp_path):
        md = G.score(
            G.load(self._gold(tmp_path)), self._corpus_with_transliteration(Language.TA)
        ).to_markdown()
        assert "Tamil choice the speaker did not make" in md

    def test_latin_script_english_is_not_counted_as_transliteration(self, tmp_path):
        corpus = _corpus(1, 1)
        path = _write_gold(tmp_path, [("S00_s1_u0", 1, "morning", "EN", "TIME_DATE")])
        assert G.score(G.load(path), corpus).transliterated_total == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestCLI:
    def test_export_then_score_round_trip(self, tmp_path, capsys):
        manifest = C.save_manifest(_corpus(2, 2), tmp_path / "manifest.json")
        gold = tmp_path / "gold.tsv"
        assert G.main([
            "export", "--manifest", str(manifest), "--out", str(gold),
            "--utterances", "2", "--prefill",
        ]) == 0
        assert G.main([
            "score", "--manifest", str(manifest), "--gold", str(gold),
            "--json", str(tmp_path / "s.json"),
        ]) == 0
        written = json.loads((tmp_path / "s.json").read_text())
        assert written["language_accuracy"] == 1.0
        assert written["prefilled"] is True

    def test_export_warns_on_stderr_when_prefilling(self, tmp_path, capsys):
        manifest = C.save_manifest(_corpus(1, 1), tmp_path / "manifest.json")
        G.main(["export", "--manifest", str(manifest), "--out",
                str(tmp_path / "g.tsv"), "--prefill"])
        assert "adjudication" in capsys.readouterr().err

    def test_scoring_an_all_blank_file_fails_loudly(self, tmp_path, capsys):
        """Returning 0 here would print a 0.0% accuracy that reads as a broken
        tagger rather than an unlabelled file."""
        manifest = C.save_manifest(_corpus(1, 1), tmp_path / "manifest.json")
        gold = tmp_path / "gold.tsv"
        G.main(["export", "--manifest", str(manifest), "--out", str(gold)])
        assert G.main(["score", "--manifest", str(manifest), "--gold", str(gold)]) == 2
        assert "no gold token could be scored" in capsys.readouterr().err

    def test_a_bad_label_exits_two_rather_than_traceback(self, tmp_path, capsys):
        manifest = C.save_manifest(_corpus(1, 1), tmp_path / "manifest.json")
        bad = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "TAMIL", "OTHER")])
        assert G.main(["score", "--manifest", str(manifest), "--gold", str(bad)]) == 2
        assert "not one of" in capsys.readouterr().err

    def test_report_is_written_when_asked(self, tmp_path):
        manifest = C.save_manifest(_corpus(1, 1), tmp_path / "manifest.json")
        gold = _write_gold(tmp_path, [("S00_s1_u0", 0, "நான்", "TA", "FUNCTION_WORD")])
        report = tmp_path / "out" / "lid.md"
        assert G.main([
            "score", "--manifest", str(manifest), "--gold", str(gold),
            "--report", str(report),
        ]) == 0
        assert "language accuracy" in report.read_text(encoding="utf-8")


class TestExportRowCount:
    def test_the_cli_counts_tokens_not_lines(self, tmp_path, capsys):
        """The instruction header is ~22 comment lines; counting them
        overstates the labelling job by that much, and this is the number
        someone uses to decide whether to accept the task."""
        manifest = C.save_manifest(_corpus(1, 1), tmp_path / "m.json")
        G.main(["export", "--manifest", str(manifest), "--out", str(tmp_path / "g.tsv")])
        out = capsys.readouterr().out
        assert "4 tokens to label" in out
