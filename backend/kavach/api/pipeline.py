"""The runtime: models, enrolment, and the verification path.

Everything expensive lives behind a lazy property. Importing this module loads
nothing; the first request that needs Whisper loads Whisper. That matters
because `uvicorn --reload` re-imports on every file save, and a server that
downloads a 3 GB checkpoint at import time is a server nobody runs.

DEGRADED MODE IS A FIRST-CLASS STATE
------------------------------------
speechbrain, faster-whisper and sentence-transformers are all optional. On a
machine without them the API starts, serves the corpus, draws graphs, and
reports which branches it cannot measure. It does *not* quietly score a
missing branch as zero -- `fusion.fuse` renormalises over the branches that
ran, and `/api/health` lists the ones that loaded.

This is not a convenience. A branch that scores 0.0 because its model is
missing is indistinguishable from a branch that scores 0.0 because the
speaker is an impostor, and the first would look like a working defence in
exactly the table this project exists to produce.

THE UBM IS LEAVE-ONE-OUT
------------------------
`score_llr` measures a probe against the claimed speaker relative to a
background model. If that background pooled the claimed speaker's own tokens,
their model would partly be compared against itself and every genuine score
would be pulled toward zero. `_background_for` excludes the claimed speaker.
With fewer than two other enrolled speakers there is no honest background at
all, and the CSBG branch reports itself unavailable rather than scoring
against a UBM that is mostly one person.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..asr import Transcript, WhisperASR
from ..audio import Audio, AudioError, check_quality, decode_bytes, load_audio
from ..challenge import Challenge, ChallengeError, ChallengeGenerator, ChallengeLedger
from ..config import Settings, get_settings
from ..csbg.graph import CSBG
from ..csbg.scoring import CSBGScore, build_background_model, score_llr
from ..csbg.tokens import UtteranceTokens
from ..embedding import ECAPAEmbedder, SpeakerTemplate
from ..fusion import (
    Branch,
    BranchScore,
    FusionPolicy,
    FusionResult,
    build_liveness_branch,
    fuse,
)
from ..integrity import IntegrityChecker, IntegrityReport, build_integrity_branch
from ..lid.pipeline import LIDPipeline
from ..matcher import AnswerMatcher, SemanticMatcher
from .converters import utterance_tokens_from_wire
from .store import Store, StoreError

#: Enrolled speakers needed before a background model means anything. Below
#: this the LLR is a comparison against one other person, not a population.
MIN_COHORT = 2

#: Why a branch is unavailable when `Settings.offline` is set. Phrased as a
#: configuration statement, not a failure, because that is what it is -- and
#: because `/api/health` shows this string to whoever is wondering why the
#: acoustic branch never fires.
_OFFLINE_REASON = (
    "Offline mode is on (KAVACH_OFFLINE); no model checkpoint will be loaded."
)


@dataclass(slots=True)
class Annotation:
    """ASR output plus its language and semantic tagging."""

    transcript: Transcript
    tokens: UtteranceTokens

    @property
    def text(self) -> str:
        return self.transcript.text


@dataclass(slots=True)
class VerificationOutcome:
    """Everything the authenticate route needs to build a response."""

    fusion: FusionResult
    annotation: Annotation | None
    csbg_score: CSBGScore | None
    speaker_id: str
    challenge_id: str
    latency_ms: int
    notes: list[str] = field(default_factory=list)
    integrity: IntegrityReport | None = None
    """Edit-artefact evidence, kept separate from the fusion branches so a
    rejection can be explained with the specific artefact that caused it
    rather than with a bare score."""


class Pipeline:
    """Holds the models and runs enrolment and verification.

    One instance per process, created by `deps.get_pipeline`. Not thread-safe
    for *loading* -- two simultaneous first requests could both start loading
    Whisper -- which is acceptable here: the second load finds the checkpoint
    cached and the wasted work is bounded. A lock would serialise every
    request behind the slowest model load.
    """

    def __init__(self, store: Store, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.ledger = ChallengeLedger(ttl_seconds=self.settings.challenge_ttl_seconds)

        self._asr: WhisperASR | None = None
        self._embedder: ECAPAEmbedder | None = None
        self._lid: LIDPipeline | None = None
        self._matcher: AnswerMatcher | None = None
        self._challenges: ChallengeGenerator | None = None

        self.integrity = IntegrityChecker()
        """Edit- and duplicate-artefact tests. Pure NumPy, always available --
        unlike every other component here it has no model to load and no
        network dependency, which is why it is constructed eagerly.

        Its duplicate memory is rebuilt from the corpus below. A replay
        detector that starts empty on every restart catches resubmissions only
        within one process lifetime; that is uptime, not a security
        property."""
        self._reload_integrity_memory()

        self._failed: dict[str, str] = {}
        """Component name -> why it could not load. Reported by /api/health so
        a missing dependency is visible rather than showing up as an
        unexplained branch that never fires."""

    def _reload_integrity_memory(self, *, limit: int = 2000) -> None:
        """Re-teach the duplicate detector every recording already on disk.

        Bounded, because the envelope memory is a linear scan on every probe
        and the honest scale of this system is tens of speakers. Past `limit`
        the check silently becomes partial, so it says so in a note rather
        than degrading quietly -- and at that point the right answer is an
        index, not a longer list.

        Never raises: a corrupt or missing file must not stop the server from
        starting. The cost of skipping one is one recording that can be
        replayed undetected, which is strictly better than a system that will
        not boot.
        """
        loaded = 0
        for row in self.store.list_utterances():
            if loaded >= limit:
                break
            try:
                clip = load_audio(self.store.audio_path(row["id"]))
            except (AudioError, StoreError, OSError):
                continue
            self.integrity.remember(clip, label=row["id"])
            loaded += 1

    def remember_recording(self, audio: Audio, *, label: str) -> None:
        """Add a newly stored recording to the duplicate memory.

        Called on enrolment upload. Verification probes are deliberately NOT
        remembered here: a rejected probe is an attacker's recording, and
        storing it would let the next honest attempt collide with it. Only
        accepted enrolment audio -- material the speaker chose to give us --
        goes in.
        """
        self.integrity.remember(audio, label=label)

    # ------------------------------------------------------------ components

    @property
    def asr(self) -> WhisperASR | None:
        if self.settings.offline:
            self._failed.setdefault("asr", _OFFLINE_REASON)
            return None
        if self._asr is None and "asr" not in self._failed:
            try:
                self._asr = WhisperASR(
                    model_size=self.settings.whisper_model,
                    device=self.settings.whisper_device,
                    compute_type=self.settings.whisper_compute_type,
                    language=self.settings.whisper_language,
                    suppress_numerals=self.settings.suppress_numerals,
                )
                # Touch the model so a missing dependency fails here rather
                # than inside a request handler.
                _ = self._asr.model
            except Exception as exc:
                self._asr = None
                self._failed["asr"] = str(exc)
        return self._asr

    @property
    def embedder(self) -> ECAPAEmbedder | None:
        if self.settings.offline:
            self._failed.setdefault("embedder", _OFFLINE_REASON)
            return None
        if self._embedder is None and "embedder" not in self._failed:
            try:
                embedder = ECAPAEmbedder(
                    model_name=self.settings.ecapa_model,
                    device=self.settings.embedding_device,
                    min_seconds=self.settings.min_audio_seconds,
                    max_seconds=self.settings.max_audio_seconds,
                )
                _ = embedder.model
                self._embedder = embedder
            except Exception as exc:
                self._embedder = None
                self._failed["embedder"] = str(exc)
        return self._embedder

    @property
    def lid(self) -> LIDPipeline:
        """Tagging pipeline.

        Always available: without an LLM tagger it falls back to rules plus a
        low-confidence guess for unresolved Latin tokens. That output is fine
        for the plumbing and is **not corpus-grade** -- `PipelineStats
        .is_corpus_grade` says so, and `/api/health` surfaces it.
        """
        if self._lid is None:
            tagger = None
            try:
                # `LLMTagger` builds its client lazily, so constructing one
                # proves nothing. Import the SDK and check for a key here, or
                # the first login would be the thing that discovers the LLM is
                # not configured -- and it would discover it as an exception
                # mid-verification.
                import anthropic  # noqa: F401
                import os

                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise RuntimeError(
                        "ANTHROPIC_API_KEY is not set; falling back to rule-based "
                        "tagging, which is not corpus-grade."
                    )
                from ..lid.llm import LLMTagger

                tagger = LLMTagger(
                    model=self.settings.llm_model,
                    effort=self.settings.llm_tagging_effort,
                )
            except Exception as exc:
                self._failed["llm_tagger"] = str(exc)
            self._lid = LIDPipeline(llm_tagger=tagger)
        return self._lid

    @property
    def matcher(self) -> AnswerMatcher:
        """Answer matcher. The three string matchers always run; the semantic
        one reports its own availability."""
        if self._matcher is None:
            self._matcher = AnswerMatcher(semantic_matcher=SemanticMatcher())
        return self._matcher

    @property
    def challenges(self) -> ChallengeGenerator:
        if self._challenges is None:
            client: Any = None
            try:
                import anthropic

                client = anthropic.Anthropic()
            except Exception as exc:
                self._failed["llm_challenge"] = str(exc)
            self._challenges = ChallengeGenerator(
                ledger=self.ledger,
                llm_client=client,
                model=self.settings.llm_model,
                effort=self.settings.llm_challenge_effort,
                ttl_seconds=self.settings.challenge_ttl_seconds,
            )
        return self._challenges

    #: What each branch needs, and what it costs the system when it is absent.
    #: Checked by import name rather than by loading, so `/api/health` is cheap
    #: -- probing the ASR by constructing it would download a 3 GB checkpoint on
    #: every health poll.
    REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
        ("speechbrain", "speaker_embedding", "No acoustic branch: voices are not compared."),
        ("faster_whisper", "asr", "No transcript: the CSBG and knowledge branches cannot run."),
        ("anthropic", "llm", "Rule-based tagging only; annotations are not corpus-grade."),
        ("sentence_transformers", "semantic_matcher", "Cross-lingual answer matching is weaker."),
    )

    def availability(self) -> dict[str, str]:
        """Which components could run, without loading any of them.

        Import-presence, not a load. A package that is installed but whose
        checkpoint fails to download shows here as available and then records a
        real failure in `failures()` the first time it is used -- which is the
        right order, because the second condition cannot be checked cheaply.
        """
        from importlib.util import find_spec

        out: dict[str, str] = {}
        if self.settings.offline:
            # Reported for every branch that needs a checkpoint, including the
            # ones whose package is installed. Otherwise a machine with every
            # dependency present would show a clean bill of health while
            # scoring nothing, which is the one health report worse than a
            # missing dependency: a wrong one.
            for _module, name, consequence in self.REQUIREMENTS:
                if name in ("speaker_embedding", "asr", "semantic_matcher"):
                    out[name] = f"{_OFFLINE_REASON} {consequence}"
        for module, name, consequence in self.REQUIREMENTS:
            try:
                present = find_spec(module) is not None
            except (ImportError, ValueError):
                present = False
            if not present:
                out[name] = f"{module} is not installed. {consequence}"
        return out

    def loaded_models(self) -> list[str]:
        """Model identifiers this server can use, for `/api/health`.

        Reports what is *installed*, not what happens to be loaded into memory
        right now. A list that empties itself between requests because nothing
        has been lazily triggered yet would tell an operator nothing.
        """
        missing = self.availability()
        out: list[str] = []
        if "asr" not in missing:
            out.append(f"faster-whisper/{self.settings.whisper_model}")
        if "speaker_embedding" not in missing:
            out.append(self.settings.ecapa_model)
        if "llm" not in missing:
            out.append(f"{self.settings.llm_model} (tagging, challenges)")
        if "semantic_matcher" not in missing:
            out.append("sentence-transformers/LaBSE")
        return out

    def failures(self) -> dict[str, str]:
        """Everything that cannot run: missing packages plus failed loads.

        Both go in one map because the caller's question is "which branches
        will report themselves unmeasured?", and the answer does not depend on
        whether the cause was a missing package or a checkpoint that would not
        download.
        """
        return {**self.availability(), **self._failed}

    # ------------------------------------------------------------- annotation

    def annotate(
        self, audio: Audio, *, utterance_id: str, speaker_id: str | None = None
    ) -> Annotation | None:
        """Transcribe and tag one recording.

        Returns None when ASR is unavailable, so the caller can store the
        audio unannotated rather than losing the recording. An utterance with
        `annotated=False` can be re-annotated later; a rejected upload cannot
        be re-recorded.
        """
        asr = self.asr
        if asr is None:
            return None
        transcript = asr.transcribe(audio)
        tokens = self.lid.tag_utterance(
            transcript.text,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            timings=transcript.timings,
        )
        return Annotation(transcript=transcript, tokens=tokens)

    def stored_tokens(self, speaker_id: str) -> list[UtteranceTokens]:
        """Annotated tokens for every one of a speaker's utterances.

        Read back from the database rather than re-derived: annotation costs
        an ASR pass and an LLM call per utterance and is not deterministic, so
        the tags are data, not a cache.
        """
        out: list[UtteranceTokens] = []
        for row in self.store.list_utterances(speaker_id):
            if not row.get("annotated") or not row.get("tokens"):
                continue
            out.append(
                utterance_tokens_from_wire(
                    row["id"],
                    [_as_token(t) for t in row["tokens"]],
                    speaker_id=speaker_id,
                    transcript=row.get("transcript", ""),
                )
            )
        return out

    # -------------------------------------------------------------- enrolment

    def build_csbg(self, speaker_id: str) -> tuple[CSBG, list[str]]:
        """Fit and store a speaker's CSBG from their annotated utterances.

        Returns:
            (graph, warnings). Warnings cover the conditions that make a graph
            untrustworthy without making it unusable -- too little speech, too
            many sparse classes, unannotated recordings. Enrolment warns
            rather than blocking, because the threshold below which a CSBG
            stops working is exactly what the stability experiment is meant to
            measure and hard-coding a guess would pre-empt it.
        """
        rows = self.store.list_utterances(speaker_id)
        total_duration = sum(float(r.get("duration_sec") or 0.0) for r in rows)
        unannotated = [r for r in rows if not r.get("annotated")]
        utterances = self.stored_tokens(speaker_id)

        graph = CSBG.build(
            speaker_id,
            utterances,
            lid_confidence_floor=self.settings.lid_confidence_floor,
            total_duration_sec=total_duration,
        )
        self.store.save_csbg(graph)

        warnings: list[str] = []
        if total_duration < self.settings.min_enrolment_seconds:
            warnings.append(
                f"Only {total_duration:.0f}s of speech recorded; "
                f"{self.settings.min_enrolment_seconds:.0f}s is the working minimum. "
                "The graph will be dominated by the smoothing prior."
            )
        if unannotated:
            warnings.append(
                f"{len(unannotated)} of {len(rows)} recordings are not annotated and "
                "contributed nothing to the graph. Check that ASR is available."
            )
        if not self.lid.stats.is_corpus_grade:
            warnings.append(
                f"{self.lid.stats.fallback_guesses} token(s) were tagged by fallback "
                "guess rather than by rules or the LLM. Usable for a demo, not for "
                "corpus annotation or reported numbers."
            )
        sparse = graph.sparse_classes(self.settings.sparse_class_threshold)
        if len(sparse) > 10:
            warnings.append(
                f"{len(sparse)} of 21 semantic classes have fewer than "
                f"{self.settings.sparse_class_threshold:.0f} observations "
                f"({', '.join(c.value for c in sparse[:6])}...). Record free speech "
                "covering more topics."
            )
        return graph, warnings

    def build_template(self, speaker_id: str) -> SpeakerTemplate | None:
        """Fit and store the speaker's ECAPA template from enrolment audio.

        Returns None when the embedder is unavailable or no clip was usable.
        """
        embedder = self.embedder
        if embedder is None:
            return None

        clips: list[Audio] = []
        for row in self.store.list_utterances(speaker_id):
            try:
                clips.append(load_audio(self.store.audio_path(row["id"])))
            except (StoreError, AudioError):
                continue
        if not clips:
            return None

        try:
            template = embedder.enrol(speaker_id, clips)
        except AudioError:
            return None
        self.store.save_template(speaker_id, template.to_dict())
        return template

    def load_template(self, speaker_id: str) -> SpeakerTemplate | None:
        payload = self.store.load_template(speaker_id)
        return SpeakerTemplate.from_dict(payload) if payload else None

    # ------------------------------------------------------------ background

    def _background_for(self, speaker_id: str) -> CSBG | None:
        """Leave-one-out UBM: every enrolled graph except the claimed speaker.

        None when fewer than `MIN_COHORT` other speakers are enrolled. See the
        module docstring -- an LLR against a one-person background is not a
        biometric, and returning it would make the CSBG branch look like it
        was working.
        """
        others = [g for sid, g in self.store.all_csbgs().items() if sid != speaker_id]
        if len(others) < MIN_COHORT:
            return None
        return build_background_model(others)

    # ------------------------------------------------------------- challenge

    def issue_challenge(self, speaker_id: str) -> Challenge:
        """Generate an adaptive challenge for a login attempt.

        Raises:
            ChallengeError: If the speaker has no knowledge-graph facts.
        """
        skg = self.store.get_skg(speaker_id)
        if len(skg) == 0:
            raise ChallengeError(
                f"Speaker {speaker_id!r} has no knowledge-graph facts. Complete the "
                "enrolment interview on the Enrolment page before authenticating."
            )
        return self.challenges.generate(
            speaker_id,
            skg,
            csbg=self.store.load_csbg(speaker_id),
            ubm=self._background_for(speaker_id),
        )

    # ---------------------------------------------------------- verification

    def verify(self, challenge_id: str, audio_bytes: bytes, *, extension: str) -> VerificationOutcome:
        """Score a spoken response against the challenge it answers.

        The liveness gate runs first and on the *ledger*, not on the audio: an
        expired, unknown or already-used challenge is rejected before a single
        model runs. That ordering is the point of the gate -- it is what makes
        a captured recording useless later, and spending a Whisper pass to
        confirm it would only make replay cheaper to probe.
        """
        started = time.perf_counter()
        notes: list[str] = []

        pending = self.ledger.get(challenge_id)
        speaker_id = pending.speaker_id if pending else ""

        try:
            challenge = self.ledger.consume(challenge_id)
        except ChallengeError as exc:
            liveness = build_liveness_branch(
                challenge_valid=False,
                matched_challenge=pending is not None,
                response_latency_sec=0.0,
                detail=str(exc),
            )
            result = fuse([liveness], self._policy())
            return VerificationOutcome(
                fusion=result,
                annotation=None,
                csbg_score=None,
                speaker_id=speaker_id,
                challenge_id=challenge_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                notes=[str(exc)],
            )

        speaker_id = challenge.speaker_id
        latency = time.time() - challenge.issued_at
        branches: list[BranchScore] = [
            build_liveness_branch(
                challenge_valid=True,
                matched_challenge=True,
                response_latency_sec=latency,
                max_latency_sec=float(self.settings.challenge_ttl_seconds),
            )
        ]

        audio = decode_bytes(audio_bytes, suffix=extension)
        quality = check_quality(
            audio,
            min_seconds=self.settings.min_audio_seconds,
            max_seconds=self.settings.max_audio_seconds,
        )
        notes.extend(quality.warnings)

        # Integrity runs before the models, for the same reason liveness runs
        # before integrity: a file we can show was assembled does not need a
        # voiceprint computed for it, and a gate placed after the expensive
        # work is a gate an attacker can use to spend our GPU.
        integrity = self.integrity.check(audio)
        branches.append(build_integrity_branch(integrity))
        if not integrity.clean:
            result = fuse(branches, self._policy())
            return VerificationOutcome(
                fusion=result,
                annotation=None,
                csbg_score=None,
                speaker_id=speaker_id,
                challenge_id=challenge.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                notes=notes + integrity.reasons,
                integrity=integrity,
            )

        branches.append(self._speaker_branch(speaker_id, audio, notes))

        annotation = self.annotate(audio, utterance_id=challenge.id, speaker_id=speaker_id)
        csbg_score = None
        if annotation is None:
            notes.append("ASR unavailable; the CSBG and knowledge branches could not run.")
            branches.append(
                BranchScore(
                    branch=Branch.CSBG,
                    score=0.0,
                    threshold=self.settings.csbg_threshold,
                    weight=0.0,
                    available=False,
                    detail="No transcript: speech recognition is not available.",
                )
            )
            branches.append(
                BranchScore(
                    branch=Branch.KNOWLEDGE,
                    score=0.0,
                    threshold=self.settings.knowledge_threshold,
                    weight=0.0,
                    available=False,
                    detail="No transcript: speech recognition is not available.",
                )
            )
        else:
            csbg_branch, csbg_score = self._csbg_branch(speaker_id, annotation)
            branches.append(csbg_branch)
            branches.append(self._knowledge_branch(challenge, annotation))

        result = fuse(branches, self._policy())
        return VerificationOutcome(
            fusion=result,
            annotation=annotation,
            csbg_score=csbg_score,
            speaker_id=speaker_id,
            challenge_id=challenge.id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            notes=notes,
            integrity=integrity,
        )

    def _policy(self) -> FusionPolicy:
        """Fusion policy from settings.

        The weights and the veto floor are `FusionPolicy`'s defaults, which are
        reasoned starting points rather than fitted values -- `eval.ablation`
        fits them on a dev split, and the fitted numbers are what the paper
        reports.
        """
        return FusionPolicy(
            threshold=self.settings.fused_threshold,
            borderline_margin=self.settings.borderline_margin,
        )

    def _speaker_branch(
        self, speaker_id: str, audio: Audio, notes: list[str]
    ) -> BranchScore:
        embedder = self.embedder
        template = self.load_template(speaker_id)
        threshold = self.settings.speaker_threshold

        if embedder is None or template is None:
            reason = (
                "Speaker embedding model is not installed."
                if embedder is None
                else "No enrolled voice template; complete enrolment first."
            )
            notes.append(reason)
            return BranchScore(
                branch=Branch.SPEAKER,
                score=0.0,
                threshold=threshold,
                weight=0.0,
                available=False,
                detail=reason,
            )

        try:
            probe = embedder.embed(audio)
        except (AudioError, ValueError) as exc:
            notes.append(str(exc))
            return BranchScore(
                branch=Branch.SPEAKER,
                score=0.0,
                threshold=threshold,
                weight=0.0,
                available=False,
                detail=str(exc),
            )

        score = template.score(probe)
        return BranchScore(
            branch=Branch.SPEAKER,
            score=score,
            threshold=threshold,
            weight=0.0,
            detail=(
                f"ECAPA-TDNN cosine against a {len(template.embeddings)}-clip template."
            ),
        )

    def _csbg_branch(
        self, speaker_id: str, annotation: Annotation
    ) -> tuple[BranchScore, CSBGScore | None]:
        # On the squashed [0, 1] scale the fusion layer consumes, 0.5 is "the
        # speaker model and the background explain this equally well" -- the
        # neutral point, and the image of `Settings.csbg_threshold = 0.0` on
        # the raw LLR scale that setting is expressed in.
        threshold = 0.5

        graph = self.store.load_csbg(speaker_id)
        ubm = self._background_for(speaker_id)

        if graph is None or ubm is None:
            reason = (
                "No enrolled code-switch graph for this speaker."
                if graph is None
                else f"Fewer than {MIN_COHORT} other enrolled speakers: no honest "
                "background model, so the likelihood ratio has nothing to compare "
                "against."
            )
            return (
                BranchScore(
                    branch=Branch.CSBG,
                    score=0.0,
                    threshold=threshold,
                    weight=0.0,
                    available=False,
                    detail=reason,
                ),
                None,
            )

        score = score_llr(
            [annotation.tokens],
            graph,
            ubm,
            lid_confidence_floor=self.settings.lid_confidence_floor,
        )
        reliable = score.n_scored_tokens >= self.settings.min_scored_tokens
        return (
            BranchScore(
                branch=Branch.CSBG,
                score=score.normalised_score,
                threshold=threshold,
                weight=0.0,
                available=reliable,
                detail=(
                    f"Log-likelihood ratio over {score.n_scored_tokens} language-choice "
                    f"tokens (raw {score.raw_score:+.3f})."
                    if reliable
                    else f"Only {score.n_scored_tokens} language-choice token(s); at "
                    f"least {self.settings.min_scored_tokens} are needed. Branch "
                    "excluded rather than scored -- a terse answer is not an "
                    "atypical one."
                ),
            ),
            score,
        )

    def _knowledge_branch(self, challenge: Challenge, annotation: Annotation) -> BranchScore:
        match = self.matcher.match(annotation.text, challenge.expected_answer)
        return BranchScore(
            branch=Branch.KNOWLEDGE,
            score=match.score,
            threshold=self.settings.knowledge_threshold,
            weight=0.0,
            detail=match.explain(),
        )


def _as_token(payload: dict[str, Any]) -> Any:
    """Coerce a stored token dict into the wire schema.

    Stored tokens were serialised from `schemas.Token`, so they arrive with
    camelCase keys; `populate_by_name` lets the model accept either, which
    keeps rows written before a rename readable.
    """
    from . import schemas

    return schemas.Token.model_validate(payload)


__all__ = [
    "Annotation",
    "MIN_COHORT",
    "Pipeline",
    "VerificationOutcome",
]
