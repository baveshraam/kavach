"""Real acoustic and knowledge branches, built from a corpus.

`run_ablation` takes two `ScoreFn` callables it knows nothing about. Until now
the only things passed in were the documented stand-ins in `experiments.py`,
which draw from a fixed distribution and mark the whole run unreportable. This
module supplies the real ones: ECAPA-TDNN cosine similarity against an enrolled
template, and the cross-lingual answer matcher against the claimed speaker's
enrolled answer.

Three properties matter more than the scoring itself:

* **Every score is cached by utterance.** `build_trials` scores each probe
  against *every* enrolled speaker, and `run_ablation` re-scores from scratch
  for each scoring ablation. Embedding per call would run the ECAPA forward
  pass N_speakers x N_ablations times per probe. It is computed once.
* **What could not be measured is `nan`, never 0.0.** On a [0, 1] branch scale
  0.0 is maximal evidence against the claim, so scoring a missing recording as
  0.0 turns an operational gap into a fabricated impostor signal.
* **Coverage is counted and returned.** A branch that quietly measured nothing
  looks exactly like a branch that measured everything and found no signal, and
  the second is a result while the first is a bug. `BranchCoverage` is what
  `experiments.py` turns into a caveat.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..csbg.tokens import UtteranceTokens
from .ablation import UNAVAILABLE, ScoreFn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..corpus import Corpus


@dataclass(slots=True)
class BranchCoverage:
    """How much of a branch actually ran.

    Report `measured` alongside any number the branch contributed to. A branch
    with `measured == 0` contributed nothing, and every fusion row that names
    it is really a row without it.
    """

    name: str
    measured: int = 0
    unavailable: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    """Why scores came back unavailable, most common first when reported."""

    @property
    def total(self) -> int:
        return self.measured + self.unavailable

    @property
    def rate(self) -> float:
        return self.measured / self.total if self.total else 0.0

    def _miss(self, reason: str) -> float:
        self.unavailable += 1
        self.reasons[reason] += 1
        return UNAVAILABLE

    def _hit(self, score: float) -> float:
        self.measured += 1
        return score

    def summary(self) -> str:
        if not self.total:
            return f"{self.name}: never called"
        head = f"{self.name}: {self.measured}/{self.total} trials scored ({self.rate:.1%})"
        if not self.reasons:
            return head
        why = ", ".join(f"{r} x{n}" for r, n in self.reasons.most_common(4))
        return f"{head}; unmeasured: {why}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "measured": self.measured,
            "unavailable": self.unavailable,
            "coverage": round(self.rate, 4),
            "reasons": dict(self.reasons.most_common()),
        }


def _prompt_index(corpus: Corpus) -> dict[str, str]:
    """utterance_id -> prompt_id."""
    return {u.utterance_id: u.prompt_id for u in corpus.utterances}


def _audio_index(corpus: Corpus) -> dict[str, Path]:
    """utterance_id -> resolved audio path, for the utterances that have one."""
    root = corpus.root or Path()
    return {
        u.utterance_id: root / u.audio_path
        for u in corpus.utterances
        if u.audio_path
    }


def _transcript_index(corpus: Corpus) -> dict[str, str]:
    """utterance_id -> transcript, preferring a hand-corrected reference.

    The knowledge branch is measuring whether the speaker knew the answer, not
    whether Whisper heard it, so a corrected transcript is used where one
    exists. Where one does not, ASR output is what there is -- and the gap
    between the two is itself the ASR-robustness result.
    """
    out: dict[str, str] = {}
    for u in corpus.utterances:
        out[u.utterance_id] = u.reference_transcript or u.transcript
    return out


# --------------------------------------------------------------------------
# Acoustic branch
# --------------------------------------------------------------------------


def cosine_to_unit(cosine: float) -> float:
    """Map a cosine similarity in [-1, 1] onto the branch's [0, 1] scale.

    An affine map, deliberately: it is strictly monotone, so it moves no
    trial past another and leaves EER and every ROC point exactly where the
    raw cosines put them. Clamping negatives to 0 would be the tempting
    alternative and it destroys the ordering among the impostors that sit
    below zero -- precisely the tail a veto threshold is fitted on.
    """
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


@dataclass(slots=True)
class AcousticBranch:
    """ECAPA-TDNN cosine similarity against the claimed speaker's template.

    This is the baseline KAVACH has to beat under attack, so it is configured
    the way a speaker-verification paper would configure it -- centroid of the
    enrolment embeddings, identical preprocessing on both sides -- rather than
    in whatever way happens to flatter the CSBG. See `embedding.py`.
    """

    templates: dict[str, Any]
    """speaker_id -> SpeakerTemplate, for the speakers who enrolled."""

    embedder: Any
    audio: dict[str, Path]
    coverage: BranchCoverage = field(
        default_factory=lambda: BranchCoverage("acoustic (ECAPA)")
    )
    _cache: dict[str, Any] = field(default_factory=dict)
    """utterance_id -> SpeakerEmbedding, or None for a clip that cannot be
    embedded. The None is cached too: a clip that is too short stays too short,
    and re-attempting it once per claimed speaker per ablation is pure cost."""

    def _embed(self, utterance_id: str) -> Any:
        if utterance_id in self._cache:
            return self._cache[utterance_id]

        from ..audio import AudioError, load_audio

        path = self.audio.get(utterance_id)
        result = None
        if path is not None and path.exists():
            try:
                result = self.embedder.embed(load_audio(path))
            except (AudioError, ValueError, OSError):
                result = None
        self._cache[utterance_id] = result
        return result

    def score(
        self, probe_speaker: str, claimed: str, utterances: list[UtteranceTokens]
    ) -> float:
        template = self.templates.get(claimed)
        if template is None:
            return self.coverage._miss("claimed speaker has no template")
        if not utterances:
            return self.coverage._miss("empty probe")

        sims: list[float] = []
        for utt in utterances:
            embedding = self._embed(utt.utterance_id)
            if embedding is not None:
                sims.append(template.score(embedding))
        if not sims:
            return self.coverage._miss("probe audio missing or unembeddable")

        # Mean over the probe's clips. `build_trials` passes one at a time
        # today; averaging keeps this correct if it ever passes more.
        return self.coverage._hit(cosine_to_unit(sum(sims) / len(sims)))


def acoustic_branch(
    corpus: Corpus,
    enrolment: dict[str, list[UtteranceTokens]],
    *,
    embedder: Any = None,
    device: str = "cpu",
) -> AcousticBranch:
    """Enrol every speaker and return a ready-to-use acoustic branch.

    Enrolment runs eagerly. A missing checkpoint, an unreadable recording or a
    speaker whose clips are all too short is a fact the operator needs before a
    multi-minute ablation, not sixty seconds into one.

    Raises:
        ImportError: If speechbrain/torch are not installed.
        ValueError: If not one speaker could be enrolled -- that is a broken
            corpus path or a broken checkpoint, not a weak baseline.
    """
    from ..audio import AudioError, load_audio
    from ..embedding import ECAPAEmbedder

    embedder = embedder or ECAPAEmbedder(device=device)
    audio = _audio_index(corpus)
    coverage = BranchCoverage("acoustic (ECAPA)")

    templates: dict[str, Any] = {}
    for speaker_id, utterances in enrolment.items():
        clips = []
        for utt in utterances:
            path = audio.get(utt.utterance_id)
            if path is None or not path.exists():
                continue
            try:
                clips.append(load_audio(path))
            except (AudioError, OSError):
                continue
        if not clips:
            coverage.reasons[f"no enrolment audio for {speaker_id}"] += 1
            continue
        try:
            templates[speaker_id] = embedder.enrol(speaker_id, clips)
        except (AudioError, ValueError):
            coverage.reasons[f"enrolment produced no embedding for {speaker_id}"] += 1

    if not templates:
        raise ValueError(
            "Could not enrol a single speaker acoustically. Every clip was "
            "missing, unreadable or too short. Check that `audio_path` in the "
            "manifest resolves against the manifest's directory, then run "
            "`audio.check_quality` on one file by hand. Reasons seen: "
            f"{dict(coverage.reasons)}"
        )

    return AcousticBranch(
        templates=templates, embedder=embedder, audio=audio, coverage=coverage
    )


# --------------------------------------------------------------------------
# Knowledge branch
# --------------------------------------------------------------------------


@dataclass(slots=True)
class KnowledgeBranch:
    """Does the probe answer match what the claimed speaker said at enrolment?

    The deployed challenge flow matches a spoken answer against a `Fact` value
    extracted into the speaker's SKG. Offline there is no interviewer, so the
    stand-in for the stored fact is the claimed speaker's own enrolment answer
    to *the same elicitation prompt*: both sides are then the same question put
    to two people, which is exactly the comparison the branch makes live.

    That substitution has a consequence worth stating plainly. It scores whole
    conversational answers rather than extracted entity values, so genuine and
    impostor answers to the same prompt share filler and function words and the
    floor sits well above zero. It measures the matcher's discrimination on
    real code-mixed speech, and it is not a measurement of the deployed
    challenge flow.

    It also requires the claimed speaker to have answered the probe's prompt at
    enrolment. Under a cross-session protocol -- every speaker records the same
    prompt list twice -- that always holds. Under a within-session split it
    never does, because a prompt held out as a probe is by construction absent
    from that speaker's enrolment. In that case the branch measures nothing,
    which `coverage` reports rather than hides.
    """

    expected: dict[str, dict[str, str]]
    """speaker_id -> prompt_id -> their enrolment answer."""

    prompt_of: dict[str, str]
    """utterance_id -> prompt_id."""

    transcripts: dict[str, str]
    matcher: Any
    coverage: BranchCoverage = field(
        default_factory=lambda: BranchCoverage("knowledge (answer matcher)")
    )
    _cache: dict[tuple[str, str], float] = field(default_factory=dict)
    """(probe utterance_id, claimed speaker) -> score. The matcher is called
    once per pair however many ablations re-score; the semantic matcher is a
    LaBSE forward pass and is the expensive part."""

    def score(
        self, probe_speaker: str, claimed: str, utterances: list[UtteranceTokens]
    ) -> float:
        if not utterances:
            return self.coverage._miss("empty probe")
        answers = self.expected.get(claimed)
        if not answers:
            return self.coverage._miss("claimed speaker has no enrolled answers")

        scores: list[float] = []
        for utt in utterances:
            prompt_id = self.prompt_of.get(utt.utterance_id, "")
            if not prompt_id:
                continue
            reference = answers.get(prompt_id)
            if not reference:
                continue
            spoken = self.transcripts.get(utt.utterance_id) or utt.transcript
            if not spoken.strip():
                continue

            key = (utt.utterance_id, claimed)
            if key not in self._cache:
                self._cache[key] = self.matcher.match(spoken, reference).score
            scores.append(self._cache[key])

        if not scores:
            return self.coverage._miss(
                "claimed speaker never answered this prompt at enrolment"
            )
        return self.coverage._hit(max(0.0, min(1.0, sum(scores) / len(scores))))


def knowledge_branch(
    corpus: Corpus,
    enrolment: dict[str, list[UtteranceTokens]],
    *,
    matcher: Any = None,
    semantic: bool = True,
) -> KnowledgeBranch:
    """Index the enrolment answers and return a ready-to-use knowledge branch.

    Args:
        semantic: Load the LaBSE sentence-embedding matcher. It is the only one
            of the four that handles a cross-language answer, so leaving it off
            understates the matcher; off is for a fast pass. Its absence is
            recorded on every `MatchResult.available`, so a run without it is
            never mistakable for a run with it.
    """
    from ..matcher import AnswerMatcher, SemanticMatcher

    if matcher is None:
        matcher = AnswerMatcher(
            semantic_matcher=SemanticMatcher() if semantic else None
        )

    prompt_of = _prompt_index(corpus)
    transcripts = _transcript_index(corpus)

    expected: dict[str, dict[str, str]] = {}
    for speaker_id, utterances in enrolment.items():
        answers: dict[str, str] = {}
        for utt in utterances:
            prompt_id = prompt_of.get(utt.utterance_id, "")
            text = transcripts.get(utt.utterance_id) or utt.transcript
            if prompt_id and text.strip():
                # One answer per prompt. A repeated prompt within an enrolment
                # session keeps the first: later takes are usually retries
                # after a recording problem, not better answers.
                answers.setdefault(prompt_id, text)
        if answers:
            expected[speaker_id] = answers

    return KnowledgeBranch(
        expected=expected,
        prompt_of=prompt_of,
        transcripts=transcripts,
        matcher=matcher,
        coverage=BranchCoverage("knowledge (answer matcher)"),
    )


def unavailable_is_nan(score: float) -> bool:
    """True if `score` is the unavailable sentinel. `nan != nan`, so compare
    with this rather than `== UNAVAILABLE`."""
    return math.isnan(score)


__all__ = [
    "AcousticBranch",
    "BranchCoverage",
    "KnowledgeBranch",
    "ScoreFn",
    "acoustic_branch",
    "cosine_to_unit",
    "knowledge_branch",
    "unavailable_is_nan",
]
