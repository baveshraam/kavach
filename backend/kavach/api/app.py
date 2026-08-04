"""FastAPI application: the 19 endpoints the frontend in `kavach/` calls.

The route list is not a design decision -- it is transcribed from
`kavach/src/api/client.ts`, which was written first. Where the frontend and
the research code disagreed, the mapping lives in `converters`, not in a
compromise on either side.

WHAT THE ROUTES DO NOT DO
-------------------------
No route computes anything. Each one loads records, hands them to
`Pipeline`, and converts the result. Any logic that appears here would be
logic the test suite reaches only through HTTP, which is the most expensive
way to test a likelihood ratio.

TWO THINGS A REVIEWER WILL LOOK FOR
-----------------------------------
1.  **`/api/challenge` does not return the answer.** `types.ts` says it does.
    See `schemas` and `Settings.demo_reveal_answers`.

2.  **`/api/attacks/generate` returns simulated numbers and says so.** The
    Attack Lab drives `attacks.suite` over signal-processed audio, which is
    for developing detectors, not for the paper's table. Every run carries
    `simulated: true` and the `paper_ready()` problems in `notes`, so a
    screenshot of that page carries its own disclaimer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..audio import AudioError, decode_bytes
from ..challenge import ChallengeError
from ..config import Settings, get_settings
from ..csbg.metrics import compute_all_metrics
from . import converters as conv
from . import schemas
from .attacks import run_attack
from .pipeline import Pipeline
from .store import Store, StoreError

_store: Store | None = None
_pipeline: Pipeline | None = None


def get_store() -> Store:
    """Process-wide store singleton."""
    global _store
    if _store is None:
        settings = get_settings()
        settings.ensure_dirs()
        _store = Store(settings.db_path, settings.audio_dir)
    return _store


def get_pipeline() -> Pipeline:
    """Process-wide pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(get_store(), get_settings())
    return _pipeline


def reset_state() -> None:
    """Drop the singletons. Tests only."""
    global _store, _pipeline
    if _store is not None:
        _store.close()
    _store = None
    _pipeline = None


StoreDep = Annotated[Store, Depends(get_store)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Override the process settings. Tests pass a `Settings` with
            temporary paths so a test run never touches the real corpus.
    """
    settings = settings or get_settings()
    app = FastAPI(
        title="KAVACH",
        description=(
            "Code-switch behaviour graphs for speaker verification in "
            "Tamil-English code-mixed speech."
        ),
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- health

    @app.get("/api/health", response_model=schemas.Health)
    def health(pipeline: PipelineDep, cfg: SettingsDep) -> schemas.Health:
        """Which models loaded, on what device, under which settings.

        `reportable` carries the exact configuration behind every number this
        server produces. It is here so a screenshot of the UI can be traced
        back to a configuration, rather than being a number with no provenance.
        """
        failures = pipeline.failures()
        return schemas.Health(
            status="degraded" if failures else "connected",
            models=pipeline.loaded_models(),
            device=cfg.embedding_device,
            version=app.version,
            demo_reveal_answers=cfg.demo_reveal_answers,
            reportable={**cfg.reportable(), "unavailable": failures},
        )

    # -------------------------------------------------------------- speakers

    @app.get("/api/speakers", response_model=list[schemas.Speaker])
    def list_speakers(store: StoreDep, pipeline: PipelineDep) -> list[schemas.Speaker]:
        return [_speaker(store, pipeline, row) for row in store.list_speakers()]

    @app.get("/api/speakers/{speaker_id}", response_model=schemas.Speaker)
    def get_speaker(
        speaker_id: str, store: StoreDep, pipeline: PipelineDep
    ) -> schemas.Speaker:
        row = _require_speaker(store, speaker_id)
        return _speaker(store, pipeline, row)

    @app.post("/api/speakers", response_model=schemas.Speaker, status_code=201)
    def create_speaker(
        payload: schemas.SpeakerCreate, store: StoreDep, pipeline: PipelineDep
    ) -> schemas.Speaker:
        """Register a speaker.

        Consent is recorded but not enforced here: the ethics constraint is
        that no *recording* happens without it, and blocking the row would
        just move the same decision to a place with less context. The UI gates
        the recorder on this field.
        """
        row = store.create_speaker(payload.model_dump())
        return _speaker(store, pipeline, row)

    @app.delete("/api/speakers/{speaker_id}", response_model=schemas.Deleted)
    def delete_speaker(speaker_id: str, store: StoreDep, pipeline: PipelineDep) -> schemas.Deleted:
        """Erase a speaker and everything derived from them.

        This is the data-subject deletion path: audio, transcripts, tokens,
        knowledge-graph facts, the CSBG, the voice template and the login
        history all go. Challenges held in memory go too -- a live challenge
        for a deleted speaker would otherwise still be answerable.
        """
        _require_speaker(store, speaker_id)
        pipeline.ledger.forget_speaker(speaker_id)
        store.delete_speaker(speaker_id)
        return schemas.Deleted()

    @app.get(
        "/api/speakers/{speaker_id}/utterances", response_model=list[schemas.Utterance]
    )
    def speaker_utterances(speaker_id: str, store: StoreDep) -> list[schemas.Utterance]:
        _require_speaker(store, speaker_id)
        return [conv.utterance_to_wire(r) for r in store.list_utterances(speaker_id)]

    @app.get("/api/speakers/{speaker_id}/csbg", response_model=schemas.CSBGGraph)
    def speaker_csbg(
        speaker_id: str,
        store: StoreDep,
        cfg: SettingsDep,
        include_transitions: bool = Query(False),
    ) -> schemas.CSBGGraph:
        """The speaker's code-switch behaviour graph as nodes and edges.

        Returns an empty graph rather than 404 when enrolment has not
        produced one yet: during enrolment "no graph" is the normal state, and
        an error there would read as a fault.
        """
        _require_speaker(store, speaker_id)
        graph = store.load_csbg(speaker_id)
        if graph is None:
            return conv.empty_csbg(speaker_id)
        return conv.csbg_to_wire(
            graph,
            include_transitions=include_transitions,
            sparse_threshold=cfg.sparse_class_threshold,
        )

    @app.get("/api/speakers/{speaker_id}/skg", response_model=list[schemas.Triple])
    def speaker_skg(speaker_id: str, store: StoreDep) -> list[schemas.Triple]:
        _require_speaker(store, speaker_id)
        return conv.skg_to_triples(store.get_skg(speaker_id))

    @app.put("/api/speakers/{speaker_id}/skg", response_model=list[schemas.Triple])
    def update_skg(
        speaker_id: str, triples: list[schemas.Triple], store: StoreDep
    ) -> list[schemas.Triple]:
        """Replace the speaker's facts with the edited set."""
        _require_speaker(store, speaker_id)
        kg = conv.skg_from_triples(speaker_id, triples)
        return conv.skg_to_triples(store.put_skg(kg))

    @app.post(
        "/api/speakers/{speaker_id}/enrol/complete",
        response_model=schemas.EnrolmentResult,
    )
    def complete_enrolment(
        speaker_id: str, store: StoreDep, pipeline: PipelineDep, cfg: SettingsDep
    ) -> schemas.EnrolmentResult:
        """Fit the speaker's CSBG and voice template from their recordings.

        Both are fitted here rather than incrementally per upload, because a
        template is a centroid and a CSBG is a smoothed distribution -- each
        is a function of the whole set, and rebuilding from scratch is the
        only way deleting a bad recording actually removes its influence.
        """
        _require_speaker(store, speaker_id)
        graph, warnings = pipeline.build_csbg(speaker_id)

        template = pipeline.build_template(speaker_id)
        if template is None:
            warnings.append(
                "No voice template was built: the speaker-embedding model is "
                "unavailable or no recording was usable. The acoustic branch will "
                "report itself unmeasured at login rather than scoring zero."
            )
        elif template.self_consistency < 0.5:
            warnings.append(
                f"Enrolment recordings only agree with each other at "
                f"{template.self_consistency:.2f} mean similarity. Usually this means "
                "one clip is mislabelled, has a second speaker in it, or was recorded "
                "in very different conditions. Check before trusting the template."
            )

        n_others = len([s for s in store.all_csbgs() if s != speaker_id])
        if n_others < 2:
            warnings.append(
                f"Only {n_others} other speaker(s) enrolled. The CSBG branch needs a "
                "background model built from several others; until then it will "
                "report itself unmeasured at login."
            )

        return schemas.EnrolmentResult(
            csbg=conv.csbg_to_wire(graph, sparse_threshold=cfg.sparse_class_threshold),
            warnings=warnings,
        )

    # ------------------------------------------------------------ utterances

    @app.post("/api/utterances", response_model=schemas.Utterance, status_code=201)
    async def upload_utterance(
        store: StoreDep,
        pipeline: PipelineDep,
        audio: Annotated[UploadFile, File()],
        speakerId: Annotated[str, Form()],
        type: Annotated[str, Form()] = "free-speech",
    ) -> schemas.Utterance:
        """Store a recording and annotate it.

        Annotation is best-effort: if ASR is unavailable the audio is still
        stored with `annotated: false`. Losing a recording is unrecoverable;
        annotating one later is not.
        """
        _require_speaker(store, speakerId)
        raw = await audio.read()
        if not raw:
            raise HTTPException(400, "The uploaded audio is empty.")

        suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
        try:
            decoded = decode_bytes(raw, suffix=suffix)
        except AudioError as exc:
            raise HTTPException(400, str(exc)) from exc

        utt_id_placeholder = f"pending_{int(time.time() * 1000)}"
        annotation = pipeline.annotate(
            decoded, utterance_id=utt_id_placeholder, speaker_id=speakerId
        )

        row = store.add_utterance(
            speaker_id=speakerId,
            type=type,
            audio_bytes=raw,
            extension=suffix,
            duration_sec=decoded.duration_sec,
            sample_rate=decoded.sample_rate,
            transcript=annotation.text if annotation else "",
            tokens=(
                [t.model_dump(by_alias=True) for t in conv.tokens_to_wire(annotation.tokens)]
                if annotation
                else []
            ),
            annotated=annotation is not None,
        )
        # Only now, with an id assigned: a duplicate reported against
        # `pending_1234` names nothing a human can look up.
        pipeline.remember_recording(decoded, label=row["id"])
        return conv.utterance_to_wire(row)

    @app.get("/api/utterances", response_model=list[schemas.Utterance])
    def list_utterances(store: StoreDep) -> list[schemas.Utterance]:
        return [conv.utterance_to_wire(r) for r in store.list_utterances()]

    @app.delete("/api/utterances/{utterance_id}", response_model=schemas.Deleted)
    def delete_utterance(utterance_id: str, store: StoreDep) -> schemas.Deleted:
        if not store.delete_utterance(utterance_id):
            raise HTTPException(404, f"Unknown utterance {utterance_id!r}.")
        return schemas.Deleted()

    @app.get("/api/audio/{utterance_id}")
    def get_audio(utterance_id: str, store: StoreDep) -> FileResponse:
        """Serve a stored recording.

        Not in `client.ts`: the frontend gets this URL inside every
        `Utterance.audioUrl` and hands it to an `<audio>` element, so the
        route has to exist even though nothing calls it by name.
        """
        try:
            path = store.audio_path(utterance_id)
        except StoreError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path)

    # ------------------------------------------------- challenge and auth

    @app.post("/api/challenge", response_model=schemas.Challenge)
    def issue_challenge(
        payload: schemas.ChallengeRequest,
        store: StoreDep,
        pipeline: PipelineDep,
        cfg: SettingsDep,
    ) -> schemas.Challenge:
        """Issue a single-use, expiring, adaptively-targeted challenge."""
        _require_speaker(store, payload.speaker_id)
        try:
            challenge = pipeline.issue_challenge(payload.speaker_id)
        except ChallengeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return conv.challenge_to_wire(challenge, reveal_answer=cfg.demo_reveal_answers)

    @app.post("/api/authenticate", response_model=schemas.AuthResult)
    async def authenticate(
        store: StoreDep,
        pipeline: PipelineDep,
        audio: Annotated[UploadFile, File()],
        challengeId: Annotated[str, Form()],
    ) -> schemas.AuthResult:
        """Score a spoken response and return the decision with its reasoning."""
        raw = await audio.read()
        if not raw:
            raise HTTPException(400, "The uploaded audio is empty.")
        suffix = Path(audio.filename or "response.webm").suffix or ".webm"

        try:
            outcome = pipeline.verify(challengeId, raw, extension=suffix)
        except AudioError as exc:
            raise HTTPException(400, str(exc)) from exc

        result = conv.auth_result_to_wire(
            result_id=f"auth_{int(time.time() * 1000)}",
            speaker_id=outcome.speaker_id,
            challenge_id=outcome.challenge_id,
            transcript=outcome.annotation.text if outcome.annotation else "",
            tokens=outcome.annotation.tokens.tokens if outcome.annotation else [],
            fusion=outcome.fusion,
            contributions=outcome.csbg_score.contributions if outcome.csbg_score else [],
            latency_ms=outcome.latency_ms,
        )
        if outcome.notes:
            result.explanation = list(result.explanation) + outcome.notes
        if outcome.speaker_id:
            store.record_auth(outcome.speaker_id, result.model_dump(by_alias=True))
        return result

    @app.get("/api/auth-history", response_model=list[schemas.AuthResult])
    def auth_history(
        store: StoreDep, limit: int = Query(50, ge=1, le=500)
    ) -> list[schemas.AuthResult]:
        return [schemas.AuthResult.model_validate(r) for r in store.list_auth(limit)]

    # --------------------------------------------------------------- attacks

    @app.post("/api/attacks/generate", response_model=schemas.AttackRun)
    def generate_attack(
        payload: schemas.AttackRequest, store: StoreDep, pipeline: PipelineDep
    ) -> schemas.AttackRun:
        """Run one attack row of the evaluation table against a speaker.

        **These are simulated attacks.** See `api.attacks` and the module
        docstring: the response carries `simulated: true` and the
        `paper_ready()` problems, and neither is decoration.
        """
        _require_speaker(store, payload.target_speaker_id)
        try:
            attack = conv.attack_from_wire(payload.attack_type)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        run = run_attack(
            attack=attack,
            speaker_id=payload.target_speaker_id,
            trials=payload.trials,
            store=store,
            pipeline=pipeline,
        )
        store.record_attack(payload.target_speaker_id, run.model_dump(by_alias=True))
        return run

    @app.get("/api/attacks", response_model=list[schemas.AttackRun])
    def list_attacks(store: StoreDep) -> list[schemas.AttackRun]:
        return [schemas.AttackRun.model_validate(r) for r in store.list_attacks()]

    # ------------------------------------------------------------ evaluation

    @app.get("/api/evaluation", response_model=schemas.EvalMetrics)
    def evaluation(store: StoreDep, pipeline: PipelineDep) -> schemas.EvalMetrics:
        """Verification metrics across configurations.

        Computed from the login history in this database, which is a *demo*
        evaluation, not the paper's: it scores whatever trials happen to have
        been run through the UI, with no genuine/impostor design and no dev/
        test split. `eval.ablation` produces the reportable numbers offline.
        An empty result is the honest answer until enough trials exist.
        """
        from .evaluation import evaluate_history

        return evaluate_history(store, pipeline)

    return app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _require_speaker(store: Store, speaker_id: str) -> dict[str, Any]:
    row = store.get_speaker(speaker_id)
    if row is None:
        raise HTTPException(404, f"Unknown speaker {speaker_id!r}.")
    return row


def _speaker(store: Store, pipeline: Pipeline, row: dict[str, Any]) -> schemas.Speaker:
    """Attach derived code-mixing statistics to a stored speaker row.

    Metrics come from the stored CSBG when there is one, and from the raw
    tokens otherwise -- a speaker mid-enrolment should still see their CMI
    move as they record, before they have pressed "complete enrolment".
    """
    speaker_id = row["id"]
    utterances = store.list_utterances(speaker_id)
    graph = store.load_csbg(speaker_id)

    if graph is not None:
        metrics = graph.metrics
        density = graph.density
    else:
        metrics = compute_all_metrics(pipeline.stored_tokens(speaker_id))
        density = 0.0

    return conv.speaker_to_wire(
        row,
        metrics=metrics,
        csbg_density=density,
        utterance_count=len(utterances),
        total_duration_sec=sum(float(u.get("duration_sec") or 0.0) for u in utterances),
    )


app = create_app()

__all__ = ["app", "create_app", "get_pipeline", "get_store", "reset_state"]
