"""Real-branch tests.

The branches themselves are thin -- cosine similarity and a string matcher.
What is worth asserting is everything around them, because each of these
failures produces a plausible-looking number rather than an error:

- an unmeasurable trial must score `nan`, not 0.0. On a [0, 1] scale 0.0 is
  maximal evidence against the claim, so a missing recording would look like a
  confident impostor detection and *improve* the reported EER;
- the cosine map must be monotone, or the branch reorders trials and the EER
  stops describing the model;
- coverage must be counted, because a branch that scored nothing and a branch
  that scored everything and found nothing produce the same fusion table;
- scores must be cached, since `build_trials` calls each branch once per
  claimed speaker per ablation and an uncached ECAPA forward pass makes the run
  quadratic in speakers.

Both models are stubbed. These tests must run in CI, which has neither the
speechbrain checkpoint nor LaBSE, and stubbing is also what lets the coverage
and caching assertions be exact.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass

import numpy as np
import pytest

from kavach import corpus as C
from kavach.csbg.ontology import CLASS_ORDER, Language
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.eval import branches as B
from kavach.eval.ablation import UNAVAILABLE


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_wav(path, seconds: float = 2.0, sr: int = 16000) -> None:
    """A real readable WAV, so `load_audio` is exercised rather than mocked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / sr) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(tone.tobytes())


def _corpus(tmp_path, n_speakers: int = 3, n_prompts: int = 4, audio: bool = True):
    """A pilot corpus with two sessions per speaker over the same prompt list.

    Two sessions, deliberately: the knowledge branch needs the claimed speaker
    to have answered the probe's prompt at enrolment, and only a cross-session
    protocol gives it that.
    """
    corpus = C.Corpus(name="t", provenance=C.Provenance.SCRIPTED, root=tmp_path)
    for s in range(n_speakers):
        sid = f"S{s:02d}"
        corpus.speakers.append(
            C.SpeakerRecord(speaker_id=sid, consent_ref=f"c/{sid}", script_id="ABC"[s])
        )
        for sess in (1, 2):
            session_id = f"{sid}_s{sess}"
            corpus.sessions.append(
                C.SessionRecord(session_id=session_id, speaker_id=sid)
            )
            for p in range(n_prompts):
                uid = f"{session_id}_p{p:02d}"
                rel = f"audio/{uid}.wav"
                if audio:
                    _write_wav(tmp_path / rel)
                corpus.utterances.append(
                    C.UtteranceRecord(
                        utterance_id=uid,
                        session_id=session_id,
                        speaker_id=sid,
                        prompt_id=f"p{p:02d}",
                        audio_path=rel if audio else "",
                        duration_sec=2.0,
                        transcript=f"speaker {sid} answering prompt {p}",
                        tokens=[
                            Token(
                                text=f"w{i}",
                                language=Language.TA if (i + s) % 3 else Language.EN,
                                semantic_class=CLASS_ORDER[i % len(CLASS_ORDER)],
                            )
                            for i in range(12)
                        ],
                        annotation_source=C.AnnotationSource.SYNTHETIC,
                    )
                )
    return corpus


def _enrolment(corpus, session_suffix: str = "_s1"):
    out: dict[str, list[UtteranceTokens]] = {}
    for u in corpus.utterances:
        if u.session_id.endswith(session_suffix):
            out.setdefault(u.speaker_id, []).append(
                UtteranceTokens(
                    utterance_id=u.utterance_id,
                    tokens=u.tokens or [],
                    speaker_id=u.speaker_id,
                    transcript=u.transcript,
                )
            )
    return out


@dataclass
class _StubEmbedding:
    """Stands in for SpeakerEmbedding: a unit vector keyed by speaker."""

    vector: np.ndarray

    def similarity(self, other) -> float:
        return float(np.dot(self.vector, other.vector))


@dataclass
class _StubTemplate:
    speaker_id: str
    centroid: _StubEmbedding

    def score(self, probe) -> float:
        return self.centroid.similarity(probe)


class _StubEmbedder:
    """Embeds by the speaker id in the filename, so identity is exact and the
    call count is meaningful."""

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    @staticmethod
    def _vector(speaker_id: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(speaker_id)) % (2**31))
        v = rng.normal(size=8)
        return v / np.linalg.norm(v)

    def embed(self, audio):
        source = str(getattr(audio, "source", ""))
        self.calls.append(source)
        speaker = source.split("/")[-1][:3]
        if speaker in self.fail_on:
            from kavach.audio import AudioError

            raise AudioError("too short")
        return _StubEmbedding(self._vector(speaker))

    def enrol(self, speaker_id, clips, **kw):
        if speaker_id in self.fail_on:
            from kavach.audio import AudioError

            raise AudioError("no usable embeddings")
        return _StubTemplate(speaker_id, _StubEmbedding(self._vector(speaker_id)))


class _StubMatcher:
    """1.0 when the two strings are equal, 0.1 otherwise -- and it counts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def match(self, answer: str, expected: str):
        self.calls.append((answer, expected))

        @dataclass
        class _R:
            score: float

        return _R(1.0 if answer == expected else 0.1)


def _probe(utterance_id: str, speaker_id: str, transcript: str = "x"):
    return UtteranceTokens(
        utterance_id=utterance_id, tokens=[], speaker_id=speaker_id,
        transcript=transcript,
    )


# --------------------------------------------------------------------------
# The cosine map
# --------------------------------------------------------------------------


class TestCosineToUnit:
    def test_maps_the_endpoints(self):
        assert B.cosine_to_unit(-1.0) == 0.0
        assert B.cosine_to_unit(1.0) == 1.0
        assert B.cosine_to_unit(0.0) == 0.5

    def test_is_strictly_monotone_including_below_zero(self):
        """Clamping negatives to 0 is the tempting alternative, and it collapses
        the impostor tail that the veto threshold is fitted on."""
        raw = [-0.9, -0.5, -0.1, 0.0, 0.3, 0.8]
        mapped = [B.cosine_to_unit(c) for c in raw]
        assert mapped == sorted(mapped)
        assert len(set(mapped)) == len(raw)

    def test_preserves_ordering_of_arbitrary_scores(self):
        raw = sorted(np.random.default_rng(0).uniform(-1, 1, 50))
        mapped = [B.cosine_to_unit(c) for c in raw]
        assert mapped == sorted(mapped)

    def test_clamps_out_of_range_input(self):
        """Floating-point dot products of normalised vectors can land a hair
        outside [-1, 1]; the branch scale must not."""
        assert B.cosine_to_unit(1.0000001) == 1.0
        assert B.cosine_to_unit(-1.0000001) == 0.0


# --------------------------------------------------------------------------
# Acoustic branch
# --------------------------------------------------------------------------


class TestAcousticBranch:
    def test_enrols_every_speaker(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        assert set(branch.templates) == {"S00", "S01", "S02"}

    def test_genuine_scores_above_impostor(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        genuine = branch.score("S00", "S00", [_probe("S00_s2_p00", "S00")])
        impostor = branch.score("S00", "S01", [_probe("S00_s2_p00", "S00")])
        assert genuine > impostor

    def test_scores_land_in_the_unit_interval(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        for claimed in ("S00", "S01", "S02"):
            s = branch.score("S00", claimed, [_probe("S00_s2_p00", "S00")])
            assert 0.0 <= s <= 1.0

    def test_missing_audio_is_nan_not_zero(self, tmp_path):
        """The property that matters most in this file. 0.0 would read as
        maximal evidence against the claim and flatter the EER."""
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        score = branch.score("S00", "S00", [_probe("does_not_exist", "S00")])
        assert math.isnan(score)
        assert score != 0.0

    def test_unenrolled_claimed_speaker_is_nan(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        assert math.isnan(branch.score("S00", "S99", [_probe("S00_s2_p00", "S00")]))

    def test_unembeddable_clip_is_nan(self, tmp_path):
        """A clip that is present but too short is still not a measurement."""
        corpus = _corpus(tmp_path)
        embedder = _StubEmbedder(fail_on={"S02"})
        branch = B.acoustic_branch(corpus, _enrolment(corpus), embedder=embedder)
        # S02 drops out of enrolment too, so `claimed` here is S00 -- a speaker
        # that *did* enrol. The nan is about the probe, not the template.
        assert "S02" not in branch.templates
        assert math.isnan(branch.score("S02", "S00", [_probe("S02_s2_p00", "S02")]))

    def test_embeds_each_utterance_once_across_many_claimed_speakers(self, tmp_path):
        """`build_trials` scores every probe against every enrolled speaker and
        `run_ablation` re-scores per ablation. Without the cache the ECAPA
        forward pass runs N_speakers x N_ablations times per probe."""
        corpus = _corpus(tmp_path)
        embedder = _StubEmbedder()
        branch = B.acoustic_branch(corpus, _enrolment(corpus), embedder=embedder)
        before = len(embedder.calls)

        probe = [_probe("S00_s2_p00", "S00")]
        for _ in range(3):
            for claimed in ("S00", "S01", "S02"):
                branch.score("S00", claimed, probe)

        assert len(embedder.calls) - before == 1

    def test_caches_the_failure_too(self, tmp_path):
        """Re-attempting a clip that cannot be read, once per claimed speaker
        per ablation, is pure cost -- it will not start working."""
        corpus = _corpus(tmp_path)
        embedder = _StubEmbedder()
        branch = B.acoustic_branch(corpus, _enrolment(corpus), embedder=embedder)
        before = len(embedder.calls)
        for _ in range(5):
            branch.score("S00", "S00", [_probe("missing", "S00")])
        assert len(embedder.calls) == before

    def test_counts_coverage(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        branch.score("S00", "S00", [_probe("S00_s2_p00", "S00")])
        branch.score("S00", "S00", [_probe("gone", "S00")])
        assert branch.coverage.measured == 1
        assert branch.coverage.unavailable == 1
        assert branch.coverage.rate == 0.5

    def test_a_corpus_with_no_audio_raises_rather_than_scoring_nan(self, tmp_path):
        """Silently returning an all-nan branch would be indistinguishable from
        a branch that ran and found nothing."""
        corpus = _corpus(tmp_path, audio=False)
        with pytest.raises(ValueError, match="Could not enrol a single speaker"):
            B.acoustic_branch(corpus, _enrolment(corpus), embedder=_StubEmbedder())

    def test_that_error_says_what_to_check(self, tmp_path):
        corpus = _corpus(tmp_path, audio=False)
        with pytest.raises(ValueError, match="audio_path"):
            B.acoustic_branch(corpus, _enrolment(corpus), embedder=_StubEmbedder())

    def test_empty_probe_is_nan(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.acoustic_branch(
            corpus, _enrolment(corpus), embedder=_StubEmbedder()
        )
        assert math.isnan(branch.score("S00", "S00", []))


# --------------------------------------------------------------------------
# Knowledge branch
# --------------------------------------------------------------------------


class TestKnowledgeBranch:
    def test_matches_the_claimed_speakers_answer_to_the_same_prompt(self, tmp_path):
        corpus = _corpus(tmp_path)
        matcher = _StubMatcher()
        branch = B.knowledge_branch(corpus, _enrolment(corpus), matcher=matcher)
        branch.score("S00", "S00", [_probe("S00_s2_p01", "S00")])
        answer, expected = matcher.calls[-1]
        assert "prompt 1" in expected
        assert "S00" in expected

    def test_impostor_is_compared_against_the_claimed_speaker_not_their_own(
        self, tmp_path
    ):
        """The whole point of the branch: S01 claiming to be S00 is checked
        against what S00 said, never against what S01 said."""
        corpus = _corpus(tmp_path)
        matcher = _StubMatcher()
        branch = B.knowledge_branch(corpus, _enrolment(corpus), matcher=matcher)
        branch.score("S01", "S00", [_probe("S01_s2_p01", "S01")])
        _, expected = matcher.calls[-1]
        assert "S00" in expected and "S01" not in expected

    def test_scores_land_in_the_unit_interval(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.knowledge_branch(
            corpus, _enrolment(corpus), matcher=_StubMatcher()
        )
        s = branch.score("S00", "S01", [_probe("S00_s2_p00", "S00")])
        assert 0.0 <= s <= 1.0

    def test_unanswered_prompt_is_nan_not_zero(self, tmp_path):
        """A prompt the claimed speaker never answered is a gap in the corpus.
        Scoring it 0.0 would manufacture evidence of impersonation."""
        corpus = _corpus(tmp_path)
        branch = B.knowledge_branch(
            corpus, _enrolment(corpus), matcher=_StubMatcher()
        )
        corpus.utterances.append(
            C.UtteranceRecord(
                utterance_id="orphan", session_id="S00_s2", speaker_id="S00",
                prompt_id="p99", transcript="an answer to a prompt nobody enrolled",
            )
        )
        branch.prompt_of["orphan"] = "p99"
        branch.transcripts["orphan"] = "an answer"
        assert math.isnan(branch.score("S00", "S00", [_probe("orphan", "S00")]))

    def test_within_session_split_leaves_the_branch_unmeasured(self, tmp_path):
        """A prompt held out as a probe is by construction absent from that
        speaker's enrolment, so the branch cannot run at all. This must be
        visible in coverage rather than showing up as scoring noise."""
        corpus = _corpus(tmp_path, n_prompts=4)
        enrolment = _enrolment(corpus)
        # Hold p03 out of every speaker's enrolment, as a within-session split
        # does.
        for sid in enrolment:
            enrolment[sid] = [u for u in enrolment[sid] if not u.utterance_id.endswith("p03")]

        branch = B.knowledge_branch(corpus, enrolment, matcher=_StubMatcher())
        for claimed in ("S00", "S01"):
            branch.score("S00", claimed, [_probe("S00_s1_p03", "S00")])

        assert branch.coverage.measured == 0
        assert "never answered this prompt" in branch.coverage.summary()

    def test_unenrolled_claimed_speaker_is_nan(self, tmp_path):
        corpus = _corpus(tmp_path)
        branch = B.knowledge_branch(
            corpus, _enrolment(corpus), matcher=_StubMatcher()
        )
        assert math.isnan(branch.score("S00", "S99", [_probe("S00_s2_p00", "S00")]))

    def test_matches_each_pair_once(self, tmp_path):
        """The semantic matcher is a LaBSE forward pass; re-running it per
        ablation is the expensive mistake here."""
        corpus = _corpus(tmp_path)
        matcher = _StubMatcher()
        branch = B.knowledge_branch(corpus, _enrolment(corpus), matcher=matcher)
        probe = [_probe("S00_s2_p00", "S00")]
        for _ in range(4):
            branch.score("S00", "S01", probe)
        assert len(matcher.calls) == 1

    def test_prefers_a_corrected_transcript_over_asr(self, tmp_path):
        """The branch measures whether the speaker knew the answer, not whether
        Whisper heard it."""
        corpus = _corpus(tmp_path)
        target = next(u for u in corpus.utterances if u.utterance_id == "S00_s1_p00")
        target.reference_transcript = "the hand corrected answer"

        matcher = _StubMatcher()
        branch = B.knowledge_branch(corpus, _enrolment(corpus), matcher=matcher)
        branch.score("S00", "S00", [_probe("S00_s2_p00", "S00")])
        _, expected = matcher.calls[-1]
        assert expected == "the hand corrected answer"

    def test_blank_transcript_is_nan(self, tmp_path):
        corpus = _corpus(tmp_path)
        for u in corpus.utterances:
            if u.utterance_id == "S00_s2_p00":
                u.transcript = "   "
        branch = B.knowledge_branch(
            corpus, _enrolment(corpus), matcher=_StubMatcher()
        )
        probe = UtteranceTokens(utterance_id="S00_s2_p00", speaker_id="S00", transcript="")
        assert math.isnan(branch.score("S00", "S00", [probe]))


# --------------------------------------------------------------------------
# Coverage bookkeeping
# --------------------------------------------------------------------------


class TestBranchCoverage:
    def test_never_called_says_so(self):
        assert B.BranchCoverage("x").summary() == "x: never called"

    def test_summary_names_the_reasons(self):
        cov = B.BranchCoverage("x")
        cov._hit(0.5)
        cov._miss("no audio")
        cov._miss("no audio")
        cov._miss("no template")
        assert "1/4" in cov.summary()
        assert "no audio x2" in cov.summary()

    def test_miss_returns_the_unavailable_sentinel(self):
        assert math.isnan(B.BranchCoverage("x")._miss("r"))
        assert math.isnan(UNAVAILABLE)

    def test_to_dict_is_json_safe(self):
        import json

        cov = B.BranchCoverage("x")
        cov._hit(1.0)
        cov._miss("r")
        assert json.loads(json.dumps(cov.to_dict()))["coverage"] == 0.5

    def test_rate_of_an_uncalled_branch_is_zero_not_a_crash(self):
        assert B.BranchCoverage("x").rate == 0.0
