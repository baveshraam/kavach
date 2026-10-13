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

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..asr import FOREIGN_SCRIPT, Transcript, WhisperASR
from ..attacks.bank import BANK_FILE, CloneBank, resolve_speaker_id
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
    Decision,
    FusionPolicy,
    FusionResult,
    build_liveness_branch,
    fuse,
)
from ..integrity import IntegrityChecker, IntegrityReport, build_integrity_branch
from ..lid.lexicon import MIN_USEFUL_COVERAGE
from ..lid.pipeline import LIDPipeline
from ..matcher import AnswerMatcher, SemanticMatcher
from ..mixed_phrase import MIXED_EN_PROMPT, MIXED_TA_DEMO_PROMPT, MIXED_TA_PROMPT, match_mixed
from ..phrase import match_phrase
from ..tamil_words import BANK, demo_pool, load_recorded
from ..calibrate_voice import POLICY_FILE, PolicyError, VoicePolicy, load_voice_policy
from ..throttle import AttemptThrottle
from .converters import utterance_tokens_from_wire
from .store import Store, StoreError

logger = logging.getLogger("kavach.mixed")

#: The Tamil words a mixed challenge may use are restricted to those the recogniser heard from the enrolled
#: speaker (`kavach.tamil_dataset`) only when at least this many were. Below it the whole bank is used.
MIN_MIXED_POOL = 30

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
    degraded: str | None = None
    """Set when the LLM tagger failed (or none is configured). The tokens then
    came from the corpus lexicon, or from rules alone if that is missing."""

    fallback_coverage: float | None = None
    """Share of tokens the corpus lexicon knew, when it was the tagger. None
    means rules-only: every class is OTHER and the tokens must not be scored
    as a CSBG."""

    dropped_hallucinated: int = 0
    """Tokens removed because they contained a script this corpus cannot
    contain (`asr.FOREIGN_SCRIPT`)."""

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
    phrase: Any = None
    """The `PhraseMatch` of a phrase challenge (which shown words were heard), else None."""
    anonymous: bool = False
    """The challenge named no account: who was speaking is the pipeline's guess, which only an ACCEPT may reveal."""


#: A phrase login with no claimed identity: the voice itself says who is speaking. See `Pipeline._identify`.
ANONYMOUS_ID = "anonymous"


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
        self.voice_policy: VoicePolicy | None = None
        self.voice_policy_error: str = ""
        try:
            self.voice_policy = load_voice_policy(Path(self.settings.data_dir) / POLICY_FILE)
        except PolicyError as exc:
            # A damaged file must neither crash the login nor silently change it: the defaults
            # stay in force, and /api/health says why the calibrated policy is not in use.
            self.voice_policy_error = str(exc)
        self._step_up: dict[str, float] = {}
        """speaker_id -> when their last attempt was borderline (a step-up is on offer)."""
        self.throttle = AttemptThrottle(
            enabled=self.settings.throttle_enabled,
            free_attempts=self.settings.throttle_free_attempts,
            base_delay=self.settings.throttle_base_delay,
            max_delay=self.settings.throttle_max_delay,
        )

        self._asr: WhisperASR | None = None
        self._mixed_asr: WhisperASR | None = None
        self._embedder: ECAPAEmbedder | None = None
        self._lid: LIDPipeline | None = None
        self._matcher: AnswerMatcher | None = None
        self._challenges: ChallengeGenerator | None = None

        self.integrity = IntegrityChecker(check_splice=self.settings.integrity_check_splice)
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
    def mixed_asr(self) -> WhisperASR | None:
        """The recogniser for the mixed login: `mixed_whisper_model` if set, otherwise the English login's own."""
        model = self.settings.mixed_whisper_model
        if not model or model == self.settings.whisper_model:
            return self.asr
        if self.settings.offline:
            self._failed.setdefault("mixed_asr", _OFFLINE_REASON)
            return None
        if self._mixed_asr is None and "mixed_asr" not in self._failed:
            try:
                asr = WhisperASR(
                    model_size=model,
                    device=self.settings.whisper_device,
                    compute_type=self.settings.whisper_compute_type,
                    language=self.settings.whisper_language,
                    suppress_numerals=self.settings.suppress_numerals,
                )
                _ = asr.model
                self._mixed_asr = asr
            except Exception as exc:
                self._mixed_asr = None
                self._failed["mixed_asr"] = str(exc)
        return self._mixed_asr

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
                # `make_tagger` is the same selection the corpus CLI uses:
                # Anthropic when its key is present, otherwise the first
                # provider that has one. Hard-coding Anthropic here meant a
                # machine holding only a Gemini key ran the *API* rules-only
                # while `kavach.annotate` on the same machine tagged the corpus
                # fine -- two answers to "is the LLM configured" from one
                # `.env`. It also resolves eagerly: a tagger that built its
                # client lazily would make the first login the thing that
                # discovers the LLM is missing, as an exception mid-
                # verification.
                from ..lid.llm import LLMTagger, make_tagger

                tagger = make_tagger(self.settings.llm_provider)
                if tagger is None:
                    raise RuntimeError(
                        "no LLM provider key found (checked ANTHROPIC_API_KEY and "
                        "the OpenAI-compatible providers); falling back to "
                        "rule-based tagging, which is not corpus-grade."
                    )
                # `llm_model` names an Anthropic model by default, so it is
                # only applied to an Anthropic tagger; every other provider
                # keeps its own default rather than being handed a model id
                # from a different vendor's namespace.
                if isinstance(tagger, LLMTagger):
                    tagger.model = self.settings.llm_model
                    tagger.effort = self.settings.llm_tagging_effort
                tagger.max_attempts = self.settings.live_llm_attempts
            except Exception as exc:
                self._failed["llm_tagger"] = str(exc)
                tagger = None

            # Offline fallback: the corpus the LLM already tagged, as a lookup.
            # See `lid.lexicon`. Built once; a failure here only costs the
            # fallback, never the pipeline.
            fallback = None
            if self.settings.live_lexicon_fallback:
                try:
                    from ..lid.lexicon import LexiconTagger

                    fallback = LexiconTagger.from_data_dir(self.settings.data_dir) or None
                except Exception as exc:  # noqa: BLE001
                    self._failed["lexicon_fallback"] = str(exc)
            self._lid = LIDPipeline(
                llm_tagger=tagger, degrade_on_llm_failure=True, fallback_tagger=fallback
            )
        return self._lid

    @property
    def matcher(self) -> AnswerMatcher:
        """Answer matcher. The three string matchers always run; the semantic
        one reports its own availability."""
        if self._matcher is None:
            # Cache-only on the live path: see `SemanticMatcher.allow_download`.
            self._matcher = AnswerMatcher(semantic_matcher=SemanticMatcher(allow_download=False))
        return self._matcher

    @property
    def challenges(self) -> ChallengeGenerator:
        if self._challenges is None:
            # No client is built here. `ChallengeGenerator` selects its own
            # provider now, so constructing an Anthropic one first would just
            # reintroduce the restriction this indirection removed -- and it
            # would record a failure for "anthropic missing" on a machine that
            # is about to generate challenges perfectly well on a free tier.
            self._challenges = ChallengeGenerator(
                ledger=self.ledger,
                model=self.settings.llm_model,
                effort=self.settings.llm_challenge_effort,
                ttl_seconds=self.settings.challenge_ttl_seconds,
                provider=self.settings.llm_provider,
            )
        return self._challenges

    #: What each branch needs, and what it costs the system when it is absent.
    #: Checked by import name rather than by loading, so `/api/health` is cheap
    #: -- probing the ASR by constructing it would download a 3 GB checkpoint on
    #: every health poll.
    REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
        ("speechbrain", "speaker_embedding", "No acoustic branch: voices are not compared."),
        ("faster_whisper", "asr", "No transcript: the CSBG and knowledge branches cannot run."),
        ("sentence_transformers", "semantic_matcher", "Cross-lingual answer matching is weaker."),
    )
    #: The LLM is deliberately absent from that table. Every other row is a
    #: single package, but tagging works through any of several providers, so
    #: `find_spec("anthropic")` answers a narrower question than the one being
    #: asked -- it used to report "anthropic is not installed" on a machine
    #: whose Gemini key was tagging the corpus perfectly well. See `_llm_gap`.

    #: What the operator loses when no provider is reachable.
    _LLM_CONSEQUENCE = "Rule-based tagging only; annotations are not corpus-grade."

    def _llm_gap(self) -> str | None:
        """Why no LLM tagger can be built, or None if one can.

        Keys and import presence only -- no client is constructed and no
        request is made, because this runs on every health poll.
        """
        from importlib.util import find_spec

        from ..lid.llm import available_providers, load_api_keys

        load_api_keys()
        import os

        has_anthropic_key = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if has_anthropic_key and find_spec("anthropic") is not None:
            return None
        try:
            others = [p.name for p in available_providers()]
        except Exception:  # noqa: BLE001 - a health poll never raises
            others = []
        if others and find_spec("openai") is not None:
            return None

        # Distinguish "no key" from "key but no SDK" -- they need different fixes.
        if has_anthropic_key:
            return f"anthropic is not installed but ANTHROPIC_API_KEY is set. {self._LLM_CONSEQUENCE}"
        if others:
            return (
                f"openai is not installed but a key for {', '.join(others)} is set. "
                f"{self._LLM_CONSEQUENCE}"
            )
        return (
            "No LLM provider key found: set ANTHROPIC_API_KEY, or a key for one of the "
            f"OpenAI-compatible providers, in .env. {self._LLM_CONSEQUENCE}"
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
        gap = self._llm_gap()
        if gap is not None:
            out["llm"] = gap
        return out

    def resolved_llm_model(self) -> str:
        """The model that will actually tag, for the provenance record.

        Returning `settings.llm_model` unconditionally would name an Anthropic
        model on a machine where Gemini is doing the work. That string is what
        `/api/health` publishes as `reportable.llm_model` and what someone
        copies into a write-up, so a confidently wrong answer here outlives the
        demo it was taken from.

        Resolution is by construction only -- `make_tagger` builds no client
        and sends no request -- so this stays cheap enough for a health poll.
        """
        from ..lid.llm import LLMTagger, make_tagger

        try:
            tagger = make_tagger(self.settings.llm_provider)
        except Exception:  # noqa: BLE001 - a health poll never raises
            return self.settings.llm_model
        if tagger is None or isinstance(tagger, LLMTagger):
            return self.settings.llm_model
        return f"{tagger.provider.name}/{tagger.model}"

    def _llm_model_label(self) -> str:
        """`resolved_llm_model` plus what it is used for, for the models list.

        Both jobs are named unconditionally now. This used to append
        "challenges use templates" on any non-Anthropic provider, which was
        true while `ChallengeGenerator` spoke only the Anthropic Messages API
        and is a lie since it stopped. The distinction is worth keeping in mind
        rather than in this string: whether a *given* challenge came from a
        model or the template bank is per-challenge and already recorded on
        `Challenge.generator`, which is the honest place for it -- a provider
        that is reachable at health-poll time can still fail at login.
        """
        return f"{self.resolved_llm_model()} (tagging, challenges)"

    def loaded_models(self) -> list[str]:
        """Model identifiers this server can use, for `/api/health`.

        Reports what is *installed*, not what happens to be loaded into memory
        right now. A list that empties itself between requests because nothing
        has been lazily triggered yet would tell an operator nothing.
        """
        missing = self.availability()
        out: list[str] = []
        if "asr" not in missing:
            device = getattr(self._asr, "resolved_device", None) if self._asr else None
            out.append(f"faster-whisper/{self.settings.whisper_model}" + (f" ({device})" if device else ""))
        if "speaker_embedding" not in missing:
            out.append(self.settings.ecapa_model)
        if "llm" not in missing:
            out.append(self._llm_model_label())
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
        transcript = asr.transcribe(audio, fast=self.settings.live_fast_asr)
        tokens = self.lid.tag_utterance(
            transcript.text,
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            timings=transcript.timings,
        )
        # Whisper's third failure mode: Hangul / Cyrillic / CJK fragments in
        # Tamil-English speech (13 of the 161 stored transcripts carry some).
        # `Transcript.hallucinated_script` detects it; nothing acted on it, so
        # those fragments were tagged and scored like real words. Drop them.
        kept = [t for t in tokens.tokens if not FOREIGN_SCRIPT.search(t.text)]
        dropped = len(tokens.tokens) - len(kept)
        tokens.tokens = kept
        return Annotation(
            transcript=transcript,
            tokens=tokens,
            degraded=self.lid.last_llm_error,
            fallback_coverage=self.lid.last_fallback_coverage,
            dropped_hallucinated=dropped,
        )

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
                    # Hallucinated-script fragments are not words anyone said;
                    # see `annotate`. Filtered here too so a rebuilt graph
                    # drops the ones already stored.
                    [_as_token(t) for t in row["tokens"] if not FOREIGN_SCRIPT.search(t.get("text", ""))],
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

    def clone_bank(self) -> CloneBank | None:
        """The pre-generated clone bank, or None when the feature is off or no bank exists.

        Raises:
            BankError: A bank exists but cannot be trusted. Callers report it;
                they do not fall back silently, because a measured row quietly
                replaced by a modelled one looks exactly the same on screen.
        """
        if not self.settings.demo_attack_bank:
            return None
        cache = self.__dict__.setdefault("_bank_cache", {})
        for victim in self.settings.clone_victims:
            root = self.settings.attack_dir / "clones" / victim
            path = root / BANK_FILE
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            cached = cache.get(victim)
            if cached and cached[0] == mtime:
                return cached[1]
            bank = CloneBank.load(
                root,
                allowed_victims=self.settings.clone_victims,
                expected_speaker_id=resolve_speaker_id(self.store.list_speakers(), victim),
            )
            cache[victim] = (mtime, bank)
            return bank
        return None

    def warm_inference(self) -> list[str]:
        """Run one embedding and one transcription on synthetic audio, so the first login is not slow.

        Loading a checkpoint is not the same as running it: the first forward pass initialises
        kernels (measured on this machine: 8.8 s for the first login, 2-3 s after). Returns the
        problems met, never raises: a model that fails here fails visibly at login anyway, and
        must not take the server down at start-up. VAD is off for the transcription, because it
        would discard synthetic noise before the decoder ran.
        """
        import numpy as np

        problems: list[str] = []
        sr = self.settings.target_sample_rate
        noise = Audio((0.05 * np.random.default_rng(0).standard_normal(sr * 3)).astype(np.float32), sr, "warmup")
        try:
            if self.embedder is not None:
                self.embedder.embed(noise, preprocess=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"embedder warm-up failed: {exc}")
        try:
            if self.asr is not None:
                self.asr.transcribe(noise, language="en", initial_prompt="", fast=True, vad_filter=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"ASR warm-up failed: {exc}")
        return problems

    @property
    def voice_threshold(self) -> float:
        """The accept threshold on the voice score: calibrated if a policy file is in force."""
        return self.voice_policy.threshold if self.voice_policy else self.settings.speaker_threshold

    @property
    def voice_grey_margin(self) -> float:
        return self.voice_policy.grey_margin if self.voice_policy else self.settings.voice_grey_margin

    #: How long after a borderline attempt a step-up challenge may be requested.
    STEP_UP_WINDOW_SEC = 120.0

    def issue_challenge(
        self, speaker_id: str, *, kind: str = "question", step_up: bool = False
    ) -> Challenge:
        """Generate a challenge for a login attempt.

        `kind="phrase"` is the 'read these random words' login: it needs no facts and no
        language tagger. `kind="question"` is the adaptive personal-question login.

        `step_up=True` asks for the stricter second sample a borderline voice earns: the
        voice must pass outright, with no inconclusive band. Only the server knows whether
        the previous attempt was borderline, and it grants one step-up per borderline attempt.

        Raises:
            ChallengeError: If a question is asked of a speaker with no knowledge-graph
                facts, or a step-up is requested without a recent borderline attempt.
        """
        if step_up:
            if kind not in ("phrase", "mixed"):
                raise ChallengeError("A step-up challenge is a read-these-words challenge.")
            earned = self._step_up.pop(speaker_id, None)
            if earned is None or time.monotonic() - earned > self.STEP_UP_WINDOW_SEC:
                raise ChallengeError(
                    "A second sample is only offered after a borderline attempt; "
                    "there is none to follow up."
                )
        if kind == "phrase":
            return self.challenges.generate_phrase(
                speaker_id, n_words=self.settings.phrase_words, strict_voice=step_up
            )
        if kind == "mixed":
            if not self.settings.mixed_enabled:
                raise ChallengeError(
                    "The Tamil-English mixed login is an experiment and is switched off (KAVACH_MIXED_ENABLED)."
                )
            return self.challenges.generate_mixed(speaker_id, strict_voice=step_up, pool=self._mixed_pool())
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
        """Score a response, then tell the throttle how the attempt went.

        Only an attempt whose voice was actually measured counts: silence, a file that cannot
        be read, a dead challenge or a missing model say nothing about who was speaking, so
        they must not slow the owner down.
        """
        pending = self.ledger.get(challenge_id)
        outcome = self._verify(challenge_id, audio_bytes, extension=extension)
        measured = any(
            b.branch is Branch.SPEAKER and b.available for b in outcome.fusion.branches
        )
        # An anonymous login is throttled as one identity (a stranger's failures slow the next attempt
        # whoever makes it, and the owner's success clears them) AND against the account its voice
        # matched, so naming no account never buys a second budget of tries at that account.
        key = pending.speaker_id if pending is not None else outcome.speaker_id
        outcome.anonymous = pending is not None and pending.speaker_id == ANONYMOUS_ID
        keys = {k for k in (key, outcome.speaker_id if outcome.anonymous else "") if k}
        if measured:
            for k in keys:
                self.throttle.record(k, accepted=outcome.fusion.accepted)
                if outcome.fusion.decision is Decision.BORDERLINE:
                    self._step_up[k] = time.monotonic()
                else:
                    self._step_up.pop(k, None)
        return outcome

    def _verify(self, challenge_id: str, audio_bytes: bytes, *, extension: str) -> VerificationOutcome:
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

        # Decode BEFORE the challenge is consumed, but only for a challenge that
        # could still be consumed. A file that cannot be read is a 400 and an
        # invitation to retry on the same screen; if the challenge were spent
        # first, the retry would be rejected as "the signature of a replay".
        # Decoding is not a model, so the ledger-first rule (never spend a model
        # pass on a dead challenge) still holds: a dead challenge is not decoded.
        audio: Audio | None = None
        if pending is not None and not pending.consumed and not pending.is_expired:
            audio = decode_bytes(audio_bytes, suffix=extension)

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

        if audio is None:  # the challenge expired between the check above and consume
            audio = decode_bytes(audio_bytes, suffix=extension)
        quality = check_quality(
            audio,
            min_seconds=self.settings.min_audio_seconds,
            max_seconds=self.settings.max_audio_seconds,
        )
        notes.extend(quality.warnings)

        # A recording with nothing usable in it is rejected before any model
        # sees it. Whisper invents text for silence ("Thank you for watching"),
        # and that invention would be tagged, scored against the speaker's
        # graph and explained to the audience as if it had been said.
        if quality.is_silent or quality.is_too_short:
            if quality.is_silent:
                reason = "no speech was detected in the recording"
            else:
                reason = (
                    f"the recording is only {quality.duration_sec:.1f}s long and at "
                    f"least {self.settings.min_audio_seconds:.0f}s is needed"
                )
            result = fuse(branches, self._policy())
            result.explanation = [
                f"Rejected: {reason}. That is a problem with the recording, not "
                "evidence about the speaker -- record the answer again."
            ]
            return VerificationOutcome(
                fusion=result,
                annotation=None,
                csbg_score=None,
                speaker_id=speaker_id,
                challenge_id=challenge.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                notes=notes,
            )

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

        if challenge.kind == "phrase":
            return self._verify_phrase(challenge, audio, branches, notes, started, integrity)
        if challenge.kind == "mixed":
            return self._verify_mixed(challenge, audio, branches, notes, started, integrity)

        branches.append(self._speaker_branch(speaker_id, audio, notes))

        annotation = self.annotate(audio, utterance_id=challenge.id, speaker_id=speaker_id)
        if annotation is not None and annotation.dropped_hallucinated:
            notes.append(
                f"Dropped {annotation.dropped_hallucinated} transcript fragment(s) in a script "
                "no Tamil-English speaker produces (a speech-recognition hallucination)."
            )
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
        elif annotation.degraded and (annotation.fallback_coverage or 0.0) >= MIN_USEFUL_COVERAGE:
            # The LLM failed but the corpus lexicon knew most of the words, so
            # the tokens do carry semantic classes. Score it, and say how.
            notes.append(
                "The language tagger (LLM) was unavailable "
                f"({annotation.degraded}); words were tagged offline from the "
                f"corpus lexicon instead ({annotation.fallback_coverage:.0%} of words known)."
            )
            csbg_branch, csbg_score = self._csbg_branch(speaker_id, annotation)
            branches.append(csbg_branch)
            branches.append(self._knowledge_branch(challenge, annotation))
        elif annotation.degraded:
            # Rules-only tokens (or a lexicon that knew too few words) carry
            # no usable semantic class, so a CSBG scored on them would compare
            # "everything is OTHER" against the enrolled graph. Unmeasured is
            # the honest answer; the knowledge branch reads the transcript,
            # not the tags, so it still runs.
            coverage = (
                f"; the offline lexicon knew only {annotation.fallback_coverage:.0%} of the words"
                if annotation.fallback_coverage is not None else ""
            )
            notes.append(
                "The language tagger (LLM) was unavailable for this login "
                f"({annotation.degraded}{coverage}), so the code-switch graph was not scored."
            )
            branches.append(
                BranchScore(
                    branch=Branch.CSBG,
                    score=0.0,
                    threshold=self.settings.csbg_threshold,
                    weight=0.0,
                    available=False,
                    detail="LLM tagger unavailable; tokens have no semantic class.",
                )
            )
            branches.append(self._knowledge_branch(challenge, annotation))
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

    def _verify_phrase(
        self,
        challenge: Challenge,
        audio: Audio,
        branches: list[BranchScore],
        notes: list[str],
        started: float,
        integrity: IntegrityReport,
    ) -> VerificationOutcome:
        """Score a 'read these words' response: freshness (the words) and identity (the voice).

        No language tagger, no CSBG, no knowledge: nothing here touches the network, so
        this path works with the venue's wifi down. The words are transcribed as English
        and without the code-mixing prompt, which would bias Whisper toward writing
        English words in Tamil script and defeat a Latin-script match.
        """
        threshold = self.settings.phrase_min_match
        asr = self.asr
        annotation: Annotation | None = None
        match = None
        voice_audio, span_note = audio, ""
        if asr is None:
            reason = self._failed.get("asr", "speech recognition is not available")
            notes.append(f"Speech recognition is unavailable: {reason}")
            branches.append(
                BranchScore(
                    branch=Branch.PHRASE, score=0.0, threshold=threshold, weight=0.0,
                    available=False, detail=reason,
                )
            )
        else:
            transcript = asr.transcribe(
                audio, language="en", initial_prompt="", fast=self.settings.live_fast_asr
            )
            match = match_phrase(challenge.phrase, transcript.text, words=transcript.words)
            branches.append(
                BranchScore(
                    branch=Branch.PHRASE, score=match.score, threshold=threshold,
                    weight=0.0, detail=match.detail,
                )
            )
            annotation = Annotation(
                transcript=transcript,
                tokens=UtteranceTokens(
                    utterance_id=challenge.id, tokens=[], speaker_id=challenge.speaker_id,
                    transcript=transcript.text,
                ),
            )
            voice_audio, span_note = self._phrase_span_audio(audio, match.span_ms)

        speaker_id = challenge.speaker_id
        if speaker_id == ANONYMOUS_ID:
            speaker_id = self._identify(voice_audio)
            # No account id here: these notes go to an unauthenticated client, and a rejected stranger must not learn which account they sounded like.
            notes.append("The voice was compared with the enrolled accounts.")  # the same words whether or not one matched: no oracle
        branches.append(self._speaker_branch(speaker_id, voice_audio, notes, detail_suffix=span_note))
        result = fuse(branches, self._policy(strict_voice=challenge.strict_voice, phrase=True))
        return VerificationOutcome(
            fusion=result,
            annotation=annotation,
            csbg_score=None,
            speaker_id=speaker_id,
            challenge_id=challenge.id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            notes=notes,
            integrity=integrity,
            phrase=match,
        )

    def _mixed_pool(self) -> list[Any] | None:
        """The Tamil words a mixed challenge may use: the demo set, else bank words the recogniser heard the speaker say, else None (whole bank)."""
        if self.settings.mixed_demo_words:
            return demo_pool()
        try:
            heard = {r["word_id"] for r in load_recorded(self.settings.data_dir) if r.get("heard")}
        except (OSError, ValueError, KeyError):
            return None
        pool = [w for w in BANK if w.word_id in heard]
        return pool if len(pool) >= MIN_MIXED_POOL else None

    def _verify_mixed(
        self,
        challenge: Challenge,
        audio: Audio,
        branches: list[BranchScore],
        notes: list[str],
        started: float,
        integrity: IntegrityReport,
    ) -> VerificationOutcome:
        """Score a 'ten English words and two Tamil words' response: this challenge's words, then the voice.

        The same gates as `_verify_phrase` and nothing relaxed: liveness and integrity already ran in `_verify`,
        the words check is a gate that fails closed, and the voice is the hard gate with the English login's
        threshold. Differences: two decodes (the English one for the English words, the Tamil one for the Tamil
        words, because no single decode hears both), and a words check that names the stage it failed at.
        The voice score and the challenge-matching result are separate branches and are logged separately.
        """
        threshold = self.settings.phrase_min_match
        asr = self.mixed_asr
        annotation: Annotation | None = None
        match = None
        voice_audio, span_note = audio, ""
        if asr is None:
            reason = self._failed.get("mixed_asr") or self._failed.get("asr", "speech recognition is not available")
            notes.append(f"Speech recognition is unavailable: {reason}")
            branches.append(
                BranchScore(
                    branch=Branch.PHRASE, score=0.0, threshold=threshold, weight=0.0,
                    available=False, detail=reason,
                )
            )
        else:
            fast = self.settings.live_fast_asr
            english = asr.transcribe(audio, language="en", initial_prompt=MIXED_EN_PROMPT, fast=fast)
            ta_prompt = MIXED_TA_DEMO_PROMPT if self.settings.mixed_tamil_prompt else MIXED_TA_PROMPT
            tamil = asr.transcribe(audio, language="ta", initial_prompt=ta_prompt, fast=fast)
            match = match_mixed(challenge.phrase, english, tamil, min_english=threshold, check_tamil=self.settings.mixed_check_tamil)
            if not self.settings.mixed_check_tamil:
                heard = [w for w, s in match.tamil_status.items() if s == "heard"]
                notes.append(f"Tamil words: {len(heard)} of {match.tamil_total} heard ({', '.join(f'{w}: {s.replace(chr(95), chr(32))}' for w, s in match.tamil_status.items())}).")
            branches.append(
                BranchScore(
                    branch=Branch.PHRASE, score=match.score, threshold=threshold,
                    weight=0.0, detail=match.detail,
                )
            )
            if match.failure_stage:
                why = {
                    "tamil_words": "A Tamil word was not written by the Tamil decode: either another word was said or the recogniser "
                                   "failed on it, and this check cannot tell which. It refuses either way.",
                    "english_words": "Too few of the English words were heard in order.",
                    "order": "A Tamil word was not heard between the English words that surround it.",
                    "no_timings": "A decode returned no word timings, so the order of the words cannot be verified.",
                }.get(match.failure_stage, "")
                notes.append(f"Words check failed at the {match.failure_stage.replace('_', ' ')} stage. {why}".strip())
            notes.append(f"Tamil decode heard: {tamil.text or '(nothing)'}")
            annotation = Annotation(
                transcript=english,
                tokens=UtteranceTokens(
                    utterance_id=challenge.id, tokens=[], speaker_id=challenge.speaker_id,
                    transcript=english.text,
                ),
            )
            voice_audio, span_note = self._phrase_span_audio(audio, match.span_ms)

        speaker_id = challenge.speaker_id
        if speaker_id == ANONYMOUS_ID:
            speaker_id = self._identify(voice_audio)
            notes.append("The voice was compared with the enrolled accounts.")  # same words whether or not one matched: no oracle
        voice = self._speaker_branch(speaker_id, voice_audio, notes, detail_suffix=span_note)
        branches.append(voice)
        result = fuse(branches, self._policy(strict_voice=challenge.strict_voice, phrase=True))
        # Two separate facts, logged side by side and never merged: how the voice scored, and what the words check found.
        logger.info("mixed_login %s", json.dumps({
            "challenge": challenge.id,
            "decision": result.decision.value,
            "voice": {"score": round(voice.score, 4) if voice.available else None, "threshold": round(voice.threshold, 4), "available": voice.available},
            "challenge_match": match.to_log() if match is not None else None,
        }))
        self._log_mixed_attempt(challenge, result, voice, match, english if asr is not None else None, tamil if asr is not None else None)
        return VerificationOutcome(
            fusion=result,
            annotation=annotation,
            csbg_score=None,
            speaker_id=speaker_id,
            challenge_id=challenge.id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            notes=notes,
            integrity=integrity,
            phrase=match,
        )

    def _log_mixed_attempt(self, challenge: Challenge, result: FusionResult, voice: BranchScore, match: Any, english: Any, tamil: Any) -> None:
        """Append what each decode heard beside the words check and the voice score, for tuning. Local, git-ignored (`data/`)."""
        try:
            path = Path(self.settings.data_dir) / "tamil_words" / "live_mixed.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "at": time.strftime("%Y-%m-%d %H:%M:%S"), "decision": result.decision.value, "challenge": list(challenge.phrase),
                "english_decode": getattr(english, "text", None), "tamil_decode": getattr(tamil, "text", None),
                "words_check": match.to_log() if match is not None else None,
                "voice": {"score": round(voice.score, 4) if voice.available else None, "threshold": round(voice.threshold, 4)},
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass  # diagnostics must never break a login

    #: Margin kept around the phrase when the voice is scored on it alone.
    SPAN_MARGIN_SEC = 0.2

    def _phrase_span_audio(self, audio: Audio, span_ms: tuple[int, int] | None) -> tuple[Audio, str]:
        """The part of the recording where the shown words were spoken, for the voiceprint.

        Whatever else was said before or after (a judge: "let me try") is not part of the sample,
        so it can neither dilute the owner's voice nor stand in for it. If the span is too short
        to embed, or there are no word timings, the whole recording is used.
        """
        if span_ms is None:
            return audio, ""
        start = max(0.0, span_ms[0] / 1000.0 - self.SPAN_MARGIN_SEC)
        end = min(audio.duration_sec, span_ms[1] / 1000.0 + self.SPAN_MARGIN_SEC)
        if end - start < self.settings.min_audio_seconds or end - start >= audio.duration_sec - 0.05:
            return audio, ""
        return (
            audio.slice_seconds(start, end),
            f"; scored on the {end - start:.1f} s span where the shown words were spoken ({start:.1f}-{end:.1f} s of {audio.duration_sec:.1f} s)",
        )

    def _policy(self, *, strict_voice: bool = False, phrase: bool = False) -> FusionPolicy:
        """Fusion policy from settings.

        `strict_voice` is a step-up challenge: the voice must pass outright, so the
        inconclusive band is closed. `phrase` is a read-these-words login, where the voice is the only
        weighted branch: the decision is then exactly the voice zones, so the generic fused threshold and
        its borderline margin (0.55 +/- 0.05) must not decide anything -- with a calibrated threshold of
        0.58 they turned a passing voice of 0.59 into BORDERLINE.

        The weights and the veto floor are `FusionPolicy`'s defaults, which are
        reasoned starting points rather than fitted values -- `eval.ablation`
        fits them on a dev split, and the fitted numbers are what the paper
        reports.
        """
        policy = FusionPolicy(
            threshold=self.voice_threshold if phrase else self.settings.fused_threshold,
            borderline_margin=0.0 if phrase else self.settings.borderline_margin,
            voice_gate=self.settings.voice_gate,
            voice_grey_margin=0.0 if strict_voice else self.voice_grey_margin,
        )
        if not self.settings.csbg_veto_enabled:
            # The offline run fitted the veto on dev and discarded it: no floor
            # bought any FAR reduction inside the 2% FRR budget. See
            # `Settings.csbg_veto_enabled`.
            policy.veto_thresholds = {}
        return policy

    def _identify(self, audio: Audio) -> str:
        """Which enrolled account does this voice sound most like? "" when there is none to compare.

        Only accounts enrolled from Studio sessions (the demo laptop's own microphone path) are
        candidates: the voice threshold was measured for that path, and the pilot speakers'
        phone-built templates were not, so letting them compete would multiply the false-accept
        chances by a threshold nobody calibrated. The best candidate still has to pass the same
        voice gate as a claimed identity; this only names who to test.
        """
        embedder = self.embedder
        if embedder is None:
            return ""
        try:
            probe = embedder.embed(audio)
        except (AudioError, ValueError):
            return ""
        best, best_score = "", float("-inf")
        for row in self.store.list_speakers():
            payload = self.store.load_template(row["id"])
            if not payload or (payload.get("provenance") or {}).get("source") != "studio":
                continue
            score = SpeakerTemplate.from_dict(payload).score(probe)
            if score > best_score:
                best, best_score = row["id"], score
        return best

    def _speaker_branch(
        self, speaker_id: str, audio: Audio, notes: list[str], *, detail_suffix: str = ""
    ) -> BranchScore:
        embedder = self.embedder
        template = self.load_template(speaker_id)
        threshold = self.voice_threshold

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
                f"ECAPA-TDNN cosine against a {len(template.embeddings)}-clip template{detail_suffix}."
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
