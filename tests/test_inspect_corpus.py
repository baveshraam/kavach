"""Inspector tests.

The inspector's job is to make a bad corpus *look* bad, so the properties worth
asserting are the ones where a silently-degenerate corpus would otherwise pass
for a healthy one:

- a rules-only pass puts every token in OTHER, which the class table must show
  as one row rather than as a full-looking corpus;
- untagged utterances must be named, not silently excluded from every table;
- metrics computed over a handful of tokens must be flagged, because CMI over
  12 tokens looks exactly like CMI over 1200;
- four speakers with the same Tamil share is the go/no-go answer, and it has to
  be stated rather than left for the reader to compute from the table.
"""

from __future__ import annotations

import pytest

from kavach import corpus as C
from kavach import inspect_corpus as I
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token


def _tok(text, lang, cls=SemanticClass.OTHER, conf=0.95):
    return Token(text=text, language=lang, semantic_class=cls, lid_confidence=conf)


def _corpus(*, tagged: bool = True, n_speakers: int = 2, ta_bias: bool = False):
    """Two speakers, 24 choice tokens each.

    With `ta_bias`, S00 says numbers in English and S01 says them in Tamil --
    the difference the CSBG exists to find. Without it they are identical,
    which is the case the summary has to call out.
    """
    corpus = C.Corpus(name="t", provenance=C.Provenance.RECORDED)
    for s in range(n_speakers):
        sid = f"S{s:02d}"
        corpus.speakers.append(C.SpeakerRecord(speaker_id=sid, consent_ref=f"c/{sid}"))
        corpus.sessions.append(C.SessionRecord(session_id=f"{sid}_s1", speaker_id=sid))
        for u in range(2):
            number_lang = (
                (Language.EN if s == 0 else Language.TA) if ta_bias else Language.TA
            )
            tokens = (
                [
                    _tok("நான்", Language.TA, SemanticClass.FUNCTION_WORD),
                    _tok("போனேன்", Language.TA, SemanticClass.ACTION_VERB),
                    _tok("six", number_lang, SemanticClass.NUMBER),
                    _tok("மணி", Language.TA, SemanticClass.TIME_DATE),
                    _tok("Chennai", Language.NAMED_ENTITY, SemanticClass.NAMED_ENTITY),
                    _tok("bus", Language.EN, SemanticClass.TRANSPORT),
                ]
                * 4
            )
            corpus.utterances.append(
                C.UtteranceRecord(
                    utterance_id=f"{sid}_s1_u{u}",
                    session_id=f"{sid}_s1",
                    speaker_id=sid,
                    prompt_id=f"p{u:02d}",
                    duration_sec=12.0,
                    transcript="நான் போனேன் six மணி Chennai bus",
                    tokens=tokens if tagged else None,
                )
            )
    return corpus


class TestSummary:
    def test_counts_speakers_and_utterances(self):
        out = I.summary(_corpus())
        assert "2 speakers" in out
        assert "4 utterances, 4 tagged" in out

    def test_names_untagged_utterances_rather_than_dropping_them(self):
        """They are excluded from every table below; a reader who does not know
        that reads the tables as covering the whole corpus."""
        corpus = _corpus()
        corpus.utterances[0].tokens = None
        out = I.summary(corpus)
        assert "1 utterances have no tokens" in out
        assert "S00_s1_u0" in out

    def test_an_untagged_corpus_says_run_the_tag_stage(self):
        out = I.summary(_corpus(tagged=False))
        assert "--stage tag" in out

    def test_a_rules_only_corpus_shows_one_class(self):
        """The single most important thing this tool catches: a pass with no
        LLM puts every token in OTHER, and the CSBG then has one class
        containing everything and every speaker's graph is identical."""
        corpus = _corpus()
        for u in corpus.utterances:
            u.tokens = [
                Token(text=t.text, language=t.language,
                      semantic_class=SemanticClass.OTHER)
                for t in (u.tokens or [])
            ]
        out = I.summary(corpus)
        class_rows = [l for l in out.splitlines() if l.startswith("| OTHER |")]
        assert len(class_rows) == 1
        assert "| NUMBER |" not in out

    def test_flags_metrics_computed_over_too_few_tokens(self):
        """CMI over 12 tokens renders identically to CMI over 1200."""
        corpus = _corpus()
        for u in corpus.utterances:
            u.tokens = (u.tokens or [])[:4]
        out = I.summary(corpus)
        assert "sampling noise" in out
        assert "*" in out

    def test_says_so_when_speakers_do_not_differ(self):
        """This is the go/no-go answer and it must be stated, not left for the
        reader to compute from the table."""
        out = I.summary(_corpus(ta_bias=False))
        assert "very little to separate on" in out

    def test_says_so_when_they_do(self):
        out = I.summary(_corpus(ta_bias=True))
        assert "a real difference to separate on" in out
        assert "very little to separate on" not in out

    def test_named_entities_do_not_count_as_a_language_choice(self):
        """Saying a proper noun is not a switch. If NE leaked into the choice
        count the Tamil share would move with how many place names a prompt
        happened to elicit."""
        out = I.summary(_corpus())
        # 6 tokens x 4 repeats x 2 utterances = 48 tokens, of which 8 are NE.
        assert "| 48 | 40" in out

    def test_class_table_marks_thin_cells(self):
        out = I.summary(_corpus(ta_bias=True))
        # NUMBER has 8 tokens per speaker, TRANSPORT 8 -- both above the floor.
        assert f"fewer than {I.THIN_CLASS_TOKENS} tokens" in out


class TestTranscripts:
    def test_groups_by_speaker(self):
        out = I.transcripts(_corpus())
        assert "## S00" in out and "## S01" in out

    def test_filters_to_one_speaker(self):
        out = I.transcripts(_corpus(), speaker="S01")
        assert "## S01" in out and "## S00" not in out

    def test_shows_the_prompt_and_duration(self):
        out = I.transcripts(_corpus())
        assert "`p00`" in out and "12.0s" in out

    def test_an_empty_transcript_is_visible_not_blank(self):
        corpus = _corpus()
        corpus.utterances[0].transcript = ""
        assert "_(no transcript)_" in I.transcripts(corpus)

    def test_shows_the_read_script_when_there_is_one(self):
        corpus = _corpus()
        corpus.utterances[0].reference_transcript = "Naan ponen six mani"
        assert "script: Naan ponen six mani" in I.transcripts(corpus)


class TestTags:
    def test_lists_tokens_with_language_and_class(self):
        out = I.tags(_corpus())
        assert "| six | TA |" in out
        assert "NUMBER" in out

    def test_reports_the_script_of_each_token(self):
        out = I.tags(_corpus())
        assert "| tamil |" in out and "| latin |" in out

    def test_marks_tamil_script_tokens_labelled_english(self):
        """The disagreement worth an eyeball: either transliteration recovered,
        or an English label on a word the speaker said in Tamil."""
        corpus = _corpus()
        corpus.utterances[0].tokens = [_tok("சிக்ஸ்", Language.EN, SemanticClass.NUMBER)]
        assert "**<-**" in I.tags(corpus)

    def test_does_not_mark_agreement(self):
        assert "**<-**" not in I.tags(_corpus())

    def test_limit_stops_early(self):
        out = I.tags(_corpus(), limit=1)
        assert out.count("## S") == 1

    def test_a_filter_matching_nothing_says_so(self):
        assert "Nothing tagged matches" in I.tags(_corpus(), speaker="nobody")


class TestTransliteration:
    def test_lists_tamil_script_tokens_tagged_english(self):
        corpus = _corpus()
        corpus.utterances[0].tokens = [
            _tok("மானிங்க்", Language.EN, SemanticClass.TIME_DATE),
            _tok("சிக்ஸ்", Language.EN, SemanticClass.NUMBER),
            _tok("நான்", Language.TA, SemanticClass.FUNCTION_WORD),
        ]
        out = I.transliteration(corpus)
        assert "2 tokens" in out
        assert "மானிங்க்" in out and "சிக்ஸ்" in out
        assert "நான்" not in out

    def test_latin_english_is_not_listed(self):
        """`bus` is English in Latin script -- not transliteration."""
        assert "| bus |" not in I.transliteration(_corpus())

    def test_none_found_explains_both_possible_reasons(self):
        """Zero can mean the ASR wrote no transliteration, or that the pipeline
        is not recovering any. Those need different responses."""
        out = I.transliteration(_corpus())
        assert "None found" in out
        assert "not recovering it" in out

    def test_groups_by_class(self):
        corpus = _corpus()
        corpus.utterances[0].tokens = [
            _tok("சிக்ஸ்", Language.EN, SemanticClass.NUMBER),
            _tok("செவன்", Language.EN, SemanticClass.NUMBER),
        ]
        assert "| NUMBER | 2 |" in I.transliteration(corpus)


class TestCLI:
    def test_summary_is_the_default(self, tmp_path, capsys):
        path = C.save_manifest(_corpus(), tmp_path / "m.json")
        assert I.main(["--manifest", str(path)]) == 0
        assert "Code-mixing profile" in capsys.readouterr().out

    def test_views_are_mutually_exclusive(self, tmp_path):
        with pytest.raises(SystemExit):
            I.build_parser().parse_args(["--manifest", "m", "--text", "--tags"])

    def test_writes_to_a_file_when_asked(self, tmp_path):
        path = C.save_manifest(_corpus(), tmp_path / "m.json")
        out = tmp_path / "nested" / "report.md"
        assert I.main(["--manifest", str(path), "--translit", "--out", str(out)]) == 0
        assert "Transliterated English" in out.read_text(encoding="utf-8")

    def test_each_view_runs(self, tmp_path, capsys):
        path = C.save_manifest(_corpus(), tmp_path / "m.json")
        for flag, marker in (
            ("--text", "Transcripts --"),
            ("--tags", "Tags --"),
            ("--translit", "Transliterated English"),
            ("--summary", "Code-mixing profile"),
        ):
            assert I.main(["--manifest", str(path), flag]) == 0
            assert marker in capsys.readouterr().out


class TestAcoustic:
    """Do a speaker's own recordings agree that they are one person?

    Not a verification result -- no probes, no impostors, no EER. It settles a
    narrower question first, and it is the check to run on a folder that
    arrived from someone else: a clip filed under the wrong speaker, a second
    voice in the room, and two devices months apart all look like unexplained
    EER later and like a low number here.
    """

    class _Embedder:
        """Templates whose self-consistency the test dictates."""

        def __init__(self, consistency: dict[str, float], *, skip: int = 0) -> None:
            self.consistency = consistency
            self.skip = skip

        def enrol(self, speaker_id, clips, **kw):
            import types

            n = max(1, len(clips) - self.skip)
            return types.SimpleNamespace(
                speaker_id=speaker_id,
                embeddings=list(range(n)),
                self_consistency=self.consistency.get(speaker_id, 0.9),
            )

    def _corpus_with_audio(self, tmp_path, missing: set[str] | None = None):
        import wave

        import numpy as np

        corpus = C.Corpus(name="a", provenance=C.Provenance.RECORDED, root=tmp_path)
        for s in range(2):
            sid = f"S{s:02d}"
            corpus.speakers.append(C.SpeakerRecord(speaker_id=sid, consent_ref="c"))
            corpus.sessions.append(C.SessionRecord(session_id=f"{sid}_s1", speaker_id=sid))
            for u in range(3):
                uid = f"{sid}_s1_u{u}"
                rel = f"audio/{uid}.wav"
                if uid not in (missing or set()):
                    path = tmp_path / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    n = 16000 * 2
                    tone = (0.3 * np.sin(2 * np.pi * 200 * np.arange(n) / 16000)
                            * 32767).astype("<i2")
                    with wave.open(str(path), "wb") as fh:
                        fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(16000)
                        fh.writeframes(tone.tobytes())
                corpus.utterances.append(
                    C.UtteranceRecord(utterance_id=uid, session_id=f"{sid}_s1",
                                      speaker_id=sid, audio_path=rel)
                )
        return corpus

    def _run(self, monkeypatch, corpus, embedder):
        from kavach import embedding as E

        monkeypatch.setattr(E, "ECAPAEmbedder", lambda **kw: embedder)
        return I.acoustic(corpus)

    def test_reports_self_consistency_per_speaker(self, tmp_path, monkeypatch):
        out = self._run(monkeypatch, self._corpus_with_audio(tmp_path),
                        self._Embedder({"S00": 0.94, "S01": 0.88}))
        assert "0.940" in out and "0.880" in out
        assert "No problems" in out

    def test_flags_a_speaker_below_the_floor(self, tmp_path, monkeypatch):
        out = self._run(monkeypatch, self._corpus_with_audio(tmp_path),
                        self._Embedder({"S00": 0.41}))
        assert "**!**" in out
        assert "wrong speaker" in out and "second voice" in out

    def test_counts_missing_files_without_crashing(self, tmp_path, monkeypatch):
        corpus = self._corpus_with_audio(tmp_path, missing={"S00_s1_u1"})
        out = self._run(monkeypatch, corpus, self._Embedder({}))
        assert "| S00 | 2 |" in out

    def test_a_speaker_with_no_usable_audio_is_a_problem_not_a_blank_row(
        self, tmp_path, monkeypatch
    ):
        corpus = self._corpus_with_audio(
            tmp_path, missing={"S00_s1_u0", "S00_s1_u1", "S00_s1_u2"}
        )
        out = self._run(monkeypatch, corpus, self._Embedder({}))
        assert "S00: no usable audio" in out

    def test_reports_clips_the_embedder_skipped(self, tmp_path, monkeypatch):
        """`embed_many` drops clips too short to embed rather than failing, so
        the count that matters is what survived, not what was handed over."""
        out = self._run(monkeypatch, self._corpus_with_audio(tmp_path),
                        self._Embedder({}, skip=1))
        assert "too short to embed" in out

    def test_a_corpus_with_no_audio_paths_says_so(self, tmp_path, monkeypatch):
        corpus = _corpus()
        out = self._run(monkeypatch, corpus, self._Embedder({}))
        assert "No utterance has an `audio_path`" in out

    def test_the_cli_exposes_it(self):
        assert I.build_parser().parse_args(["--manifest", "m", "--acoustic"]).acoustic
