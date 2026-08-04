"""API layer tests.

The load-bearing one is `TestFrontendContract`. TypeScript cannot check a JSON
payload at runtime, so nothing in the frontend build catches a renamed field
-- the failure appears as a blank panel in a demo. Parsing `types.ts` and
comparing it to the Pydantic aliases moves that failure to `pytest`.

Everything else here runs in **degraded mode on purpose**: this environment
has no speechbrain and no faster-whisper, so the acoustic and CSBG branches
report themselves unmeasured. That is the configuration most likely to be
running on a fresh clone, and the property that matters is that it produces
honest "not measured" output rather than silently scoring a missing model as
a zero.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.api import schemas
from kavach.api.app import create_app, get_pipeline, get_settings, get_store
from kavach.api.attacks import DEFEATED_BY, run_attack
from kavach.api.converters import (
    ATTACK_FROM_WIRE,
    ATTACK_TO_WIRE,
    CONFIG_TO_WIRE,
    challenge_to_wire,
    contribution_to_wire,
    csbg_to_wire,
    skg_from_triples,
    skg_to_triples,
)
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store, StoreError
from kavach.attacks import AttackType
from kavach.attacks.suite import SystemConfig
from kavach.audio import Audio, save_wav
from kavach.challenge import Challenge
from kavach.config import Settings
from kavach.csbg.graph import CSBG
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.scoring import build_background_model, score_llr
from kavach.csbg.tokens import Token, UtteranceTokens

TYPES_TS = Path(__file__).resolve().parents[1] / "kavach" / "src" / "api" / "types.ts"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def tokens(pattern: str, *, utterance_id: str = "u") -> UtteranceTokens:
    """Build an utterance from a compact 'CLASS:LANG' spec.

        tokens("NUMBER:EN FOOD:TA FOOD:TA")

    Timings are synthesised at 400 ms per token so duration-dependent code
    (the A5 eavesdropping budget, the stability curve) has something real to
    work with.
    """
    out: list[Token] = []
    for i, spec in enumerate(pattern.split()):
        cls_, lang = spec.split(":")
        out.append(
            Token(
                text=f"{cls_.lower()}{i}",
                language=Language[lang],
                semantic_class=SemanticClass[cls_],
                start_ms=i * 400,
                end_ms=(i + 1) * 400,
            )
        )
    return UtteranceTokens(utterance_id=utterance_id, tokens=out)


def speaker_utterances(profile: dict[SemanticClass, Language], n: int = 8) -> list[UtteranceTokens]:
    """Repeat a per-class language profile into enough utterances to fit a CSBG."""
    spec = " ".join(f"{c.name}:{l.name}" for c, l in profile.items())
    return [tokens(spec, utterance_id=f"u{i}") for i in range(n)]


TAMIL_NUMBERS = {
    SemanticClass.NUMBER: Language.TA,
    SemanticClass.FOOD: Language.TA,
    SemanticClass.TECH_DIGITAL: Language.EN,
    SemanticClass.KINSHIP: Language.TA,
    SemanticClass.EDU_WORK: Language.EN,
}

ENGLISH_NUMBERS = {
    SemanticClass.NUMBER: Language.EN,
    SemanticClass.FOOD: Language.EN,
    SemanticClass.TECH_DIGITAL: Language.EN,
    SemanticClass.KINSHIP: Language.EN,
    SemanticClass.EDU_WORK: Language.EN,
}


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
    )


@pytest.fixture()
def store(settings: Settings) -> Store:
    s = Store(settings.db_path, settings.audio_dir)
    yield s
    s.close()


@pytest.fixture()
def pipeline(store: Store, settings: Settings) -> Pipeline:
    return Pipeline(store, settings)


@pytest.fixture()
def client(store: Store, pipeline: Pipeline, settings: Settings) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def wav_bytes(seconds: float = 2.0, sr: int = 16_000, seed: int = 0) -> bytes:
    """A short synthetic clip as 16-bit PCM WAV, decodable without ffmpeg."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * 140 * t) + 0.02 * rng.standard_normal(len(t))
    tmp = Path(__file__).parent / f"_tmp_{seed}.wav"
    save_wav(Audio(wave.astype(np.float32), sr), tmp)
    try:
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def make_speaker(client: TestClient, name: str, **kw) -> dict:
    body = {"displayName": name, "consentGiven": True, **kw}
    response = client.post("/api/speakers", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def enrol_tokens(store: Store, speaker_id: str, utterances: list[UtteranceTokens]) -> CSBG:
    """Insert annotated utterances directly and fit the graph.

    Bypasses the upload route because ASR is unavailable here, and the point
    of these tests is the layer above annotation.
    """
    for utt in utterances:
        row = store.add_utterance(
            speaker_id=speaker_id,
            type="free-speech",
            audio_bytes=b"RIFF",
            extension=".wav",
            duration_sec=len(utt.tokens) * 0.4,
            sample_rate=16_000,
        )
        store.update_annotation(
            row["id"],
            transcript=" ".join(t.text for t in utt.tokens),
            tokens=[
                {
                    "text": t.text,
                    "language": t.language.value,
                    "semanticClass": t.semantic_class.value,
                    "lidConfidence": t.lid_confidence,
                    "startMs": t.start_ms,
                    "endMs": t.end_ms,
                }
                for t in utt.tokens
            ],
        )
    graph = CSBG.build(speaker_id, utterances, total_duration_sec=sum(
        len(u.tokens) * 0.4 for u in utterances
    ))
    store.save_csbg(graph)
    return graph


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def strip_line_comments(source: str) -> str:
    """Blank out `//` comments, preserving line structure and offsets.

    Necessary because prose is not syntax: an English sentence in a comment
    can contain `word:` and every field parser here keys on exactly that. A
    comment reading "these are gates, not weighted factors: they carry weight
    0" contributes a phantom field named `factors`, and the contract test then
    fails against a field that does not exist in either the schema or the
    interface. Replacing the comment body with spaces rather than deleting it
    keeps every subsequent index the same, which matters because the callers
    below walk the string by offset.

    Only `//` is handled. `/* ... */` does not appear in `types.ts`, and a
    half-correct block-comment stripper that silently eats a real field would
    be worse than not having one.
    """
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        idx = line.find("//")
        if idx == -1:
            out.append(line)
            continue
        newline = len(line) - len(line.rstrip("\r\n"))
        body = line[idx : len(line) - newline]
        out.append(line[:idx] + " " * len(body) + line[len(line) - newline :])
    return "".join(out)


def parse_typescript_interfaces(source: str) -> dict[str, set[str]]:
    """Field names of every `export interface` in a TypeScript file.

    Only depth-1 fields are recorded, so a nested `Array<{ far: number }>`
    does not contribute `far` to its parent. Nested literals are picked up
    separately by `parse_nested_literals`.
    """
    source = strip_line_comments(source)
    out: dict[str, set[str]] = {}
    for match in re.finditer(r"export interface (\w+)\s*\{", source):
        name = match.group(1)
        depth = 1
        i = match.end()
        fields: set[str] = set()
        while i < len(source) and depth:
            ch = source[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1:
                field = re.match(r"\s*(\w+)\??\s*:", source[i:])
                if field and (i == 0 or source[i - 1] in "{;\n "):
                    fields.add(field.group(1))
                    i += field.end() - 1
            i += 1
        out[name] = fields
    return out


def parse_nested_literals(source: str, interface: str) -> list[set[str]]:
    """Field sets of each `{...}` object literal nested inside an interface.

    `EvalMetrics` declares its rows inline rather than as named interfaces, so
    the schemas that mirror them (`EvalConfiguration`, `StabilityPoint`, ...)
    have nothing to compare against without this.
    """
    source = strip_line_comments(source)
    start = source.index(f"export interface {interface}")
    depth = 0
    literals: list[set[str]] = []
    current: list[str] = []
    stack: list[list[str]] = []
    i = source.index("{", start)
    while i < len(source):
        ch = source[i]
        if ch == "{":
            depth += 1
            stack.append([])
        elif ch == "}":
            body = stack.pop()
            if depth > 1:
                literals.append({f for f in body})
            depth -= 1
            if depth == 0:
                break
        else:
            field = re.match(r"\s*(\w+)\??\s*:", source[i:])
            if field and stack and (i == 0 or source[i - 1] in "{;\n ,"):
                stack[-1].append(field.group(1))
                i += field.end() - 1
        i += 1
    return literals


def aliases(model: type[schemas.Model]) -> set[str]:
    return {f.alias or name for name, f in model.model_fields.items()}


class TestFrontendContract:
    """`schemas` must match `kavach/src/api/types.ts` field for field."""

    def test_types_file_exists(self) -> None:
        assert TYPES_TS.exists(), (
            f"{TYPES_TS} is missing. The API schemas are defined against it; without "
            "it nothing checks that the backend and the UI agree."
        )

    @pytest.mark.parametrize(
        ("interface", "model", "backend_extras"),
        [
            ("Speaker", schemas.Speaker, set()),
            ("Token", schemas.Token, set()),
            ("Utterance", schemas.Utterance, set()),
            ("Triple", schemas.Triple, set()),
            ("Challenge", schemas.Challenge, set()),
            ("BranchScore", schemas.BranchScore, set()),
            ("ClassDivergence", schemas.ClassDivergence, set()),
            ("AuthResult", schemas.AuthResult, set()),
            ("CSBGNode", schemas.CSBGNode, set()),
            ("CSBGEdge", schemas.CSBGEdge, set()),
            ("CSBG", schemas.CSBGGraph, set()),
            # The three extras are backend-only. TypeScript ignores unknown
            # keys, and omitting them would let the Attack Lab's simulated
            # numbers be screenshotted without their caveats.
            ("AttackRun", schemas.AttackRun, {"simulated", "yieldRate", "notes"}),
        ],
    )
    def test_schema_matches_interface(
        self, interface: str, model: type[schemas.Model], backend_extras: set[str]
    ) -> None:
        declared = parse_typescript_interfaces(TYPES_TS.read_text(encoding="utf-8"))
        assert interface in declared, f"types.ts has no interface {interface}"
        assert aliases(model) - backend_extras == declared[interface], (
            f"{model.__name__} and types.ts:{interface} have diverged. "
            f"Backend only: {sorted(aliases(model) - backend_extras - declared[interface])}; "
            f"frontend only: {sorted(declared[interface] - aliases(model))}."
        )

    def test_eval_metrics_top_level_matches(self) -> None:
        declared = parse_typescript_interfaces(TYPES_TS.read_text(encoding="utf-8"))
        assert aliases(schemas.EvalMetrics) == declared["EvalMetrics"]

    def test_eval_metrics_rows_match_their_inline_literals(self) -> None:
        literals = parse_nested_literals(
            TYPES_TS.read_text(encoding="utf-8"), "EvalMetrics"
        )
        expected = [
            aliases(schemas.EvalConfiguration),
            aliases(schemas.DETPoint),
            aliases(schemas.StabilityPoint),
            aliases(schemas.FairnessSlice),
            aliases(schemas.ScoreDistribution),
        ]
        for want in expected:
            assert want in literals, (
                f"No inline literal in EvalMetrics matches {sorted(want)}. "
                f"types.ts declares {[sorted(s) for s in literals]}."
            )

    def test_attack_type_literal_matches(self) -> None:
        source = TYPES_TS.read_text(encoding="utf-8")
        declared = set(re.findall(r"'(A\d_[A-Z_]+)'", source))
        assert declared == set(ATTACK_TO_WIRE.values())

    def test_semantic_classes_match_the_ontology(self) -> None:
        """The UI's `SemanticClass` union must be the ontology, exactly.

        A class the frontend does not know about arrives as a string it cannot
        render; one it knows about that the backend dropped is a filter that
        silently matches nothing.
        """
        source = TYPES_TS.read_text(encoding="utf-8")
        block = source[source.index("export type SemanticClass") : source.index(";", source.index("export type SemanticClass"))]
        declared = set(re.findall(r"'([A-Z_]+)'", block))
        assert declared == {c.value for c in SemanticClass}

    def test_config_keys_match_the_frontend_record(self) -> None:
        """`successRateByConfig`'s four keys are exactly `SystemConfig`."""
        source = TYPES_TS.read_text(encoding="utf-8")
        block = source[source.index("successRateByConfig") :]
        block = block[: block.index(">")]
        declared = set(re.findall(r"'(\w+)'", block))
        assert declared == set(CONFIG_TO_WIRE.values())
        assert set(CONFIG_TO_WIRE) == set(SystemConfig)

    def test_attack_mapping_is_a_bijection(self) -> None:
        assert set(ATTACK_TO_WIRE) == set(AttackType)
        assert len(ATTACK_FROM_WIRE) == len(ATTACK_TO_WIRE)


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class TestStore:
    def test_speaker_round_trip(self, store: Store) -> None:
        row = store.create_speaker(
            {"display_name": "Aarthi", "other_languages": ["Hindi"], "consent_given": True}
        )
        fetched = store.get_speaker(row["id"])
        assert fetched is not None
        assert fetched["display_name"] == "Aarthi"
        assert fetched["other_languages"] == ["Hindi"]
        assert fetched["consent_given"] is True

    def test_utterance_needs_an_existing_speaker(self, store: Store) -> None:
        with pytest.raises(StoreError, match="Unknown speaker"):
            store.add_utterance(
                speaker_id="nobody",
                type="free-speech",
                audio_bytes=b"x",
                extension=".wav",
                duration_sec=1.0,
                sample_rate=16_000,
            )

    def test_deleting_a_speaker_removes_their_audio(self, store: Store) -> None:
        """A deletion request that leaves the recordings behind is not honoured."""
        speaker = store.create_speaker({"display_name": "Test"})
        row = store.add_utterance(
            speaker_id=speaker["id"],
            type="free-speech",
            audio_bytes=b"RIFFdata",
            extension=".wav",
            duration_sec=1.0,
            sample_rate=16_000,
        )
        path = store.audio_path(row["id"])
        assert path.exists()

        store.put_skg(skg_from_triples(speaker["id"], [
            schemas.Triple(subject="s", predicate="hometown", object="Thanjavur")
        ]))

        store.delete_speaker(speaker["id"])

        assert not path.exists()
        assert store.list_utterances(speaker["id"]) == []
        assert len(store.get_skg(speaker["id"])) == 0

    def test_skg_round_trip_drops_empty_values(self, store: Store) -> None:
        """Clearing a field must delete the fact, not assert an empty one.

        An empty expected value would be "matched" by anything the normaliser
        reduces to nothing, which turns the knowledge branch into a pass.
        """
        speaker = store.create_speaker({"display_name": "Test"})
        kg = skg_from_triples(
            speaker["id"],
            [
                schemas.Triple(subject="x", predicate="hometown", object="Thanjavur"),
                schemas.Triple(subject="x", predicate="school", object="   "),
            ],
        )
        stored = store.put_skg(kg)
        assert {f.predicate for f in stored.facts} == {"hometown"}

    def test_put_skg_replaces_rather_than_merges(self, store: Store) -> None:
        speaker = store.create_speaker({"display_name": "Test"})
        store.put_skg(skg_from_triples(speaker["id"], [
            schemas.Triple(subject="x", predicate="hometown", object="Madurai"),
            schemas.Triple(subject="x", predicate="school", object="St Joseph"),
        ]))
        store.put_skg(skg_from_triples(speaker["id"], [
            schemas.Triple(subject="x", predicate="hometown", object="Madurai"),
        ]))
        assert {f.predicate for f in store.get_skg(speaker["id"]).facts} == {"hometown"}

    def test_csbg_round_trip(self, store: Store) -> None:
        speaker = store.create_speaker({"display_name": "Test"})
        graph = CSBG.build(speaker["id"], speaker_utterances(TAMIL_NUMBERS))
        store.save_csbg(graph)
        loaded = store.load_csbg(speaker["id"])
        assert loaded is not None
        assert np.allclose(loaded.lexical_counts, graph.lexical_counts)

    def test_graph_from_a_different_ontology_is_not_loaded(self, store: Store) -> None:
        """Rather than comparing different class indices and calling it a score."""
        speaker = store.create_speaker({"display_name": "Test"})
        graph = CSBG.build(speaker["id"], speaker_utterances(TAMIL_NUMBERS))
        payload = graph.to_dict()
        payload["ontology_version"] = "0.9"
        import json

        with store._tx() as conn:  # noqa: SLF001 - fabricating a stale row
            conn.execute(
                "INSERT INTO graphs (speaker_id, payload, built_at) VALUES (?, ?, ?)",
                (speaker["id"], json.dumps(payload), "2020-01-01T00:00:00+00:00"),
            )
        assert store.load_csbg(speaker["id"]) is None


# --------------------------------------------------------------------------
# Converters
# --------------------------------------------------------------------------


class TestConverters:
    def test_csbg_omits_classes_with_no_evidence(self) -> None:
        """An edge sitting at the smoothing prior is the absence of data.

        Drawing all 21 classes would show a graph far denser than the speech
        behind it supports, which is the single most misleading thing this
        figure could do.
        """
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        wire = csbg_to_wire(graph)

        class_nodes = [n for n in wire.nodes if n.kind == "class"]
        assert {n.label for n in class_nodes} == {c.name for c in TAMIL_NUMBERS}
        assert all(n.token_count > 0 for n in class_nodes)

    def test_csbg_has_both_language_nodes_and_lexical_edges(self) -> None:
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        wire = csbg_to_wire(graph)

        assert {n.label for n in wire.nodes if n.kind == "language"} == {"TA", "EN"}
        assert all(e.edge_type == "lexical_choice" for e in wire.edges)
        # Two edges per drawn class: one to each language.
        assert len(wire.edges) == 2 * len(TAMIL_NUMBERS)

    def test_edge_probabilities_for_a_class_sum_to_one(self) -> None:
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        wire = csbg_to_wire(graph)
        by_source: dict[str, float] = {}
        for e in wire.edges:
            by_source[e.source] = by_source.get(e.source, 0.0) + e.probability
        assert all(abs(v - 1.0) < 1e-9 for v in by_source.values())

    def test_transitions_are_off_by_default(self) -> None:
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        assert not any(
            e.edge_type == "switch_transition" for e in csbg_to_wire(graph).edges
        )
        assert any(
            e.edge_type == "switch_transition"
            for e in csbg_to_wire(graph, include_transitions=True).edges
        )

    def test_empty_graph_is_a_graph_not_an_error(self) -> None:
        graph = CSBG.build("spk", [])
        wire = csbg_to_wire(graph)
        assert [n for n in wire.nodes if n.kind == "class"] == []
        assert wire.speaker_id == "spk"

    def test_challenge_does_not_leak_the_answer(self) -> None:
        """The whole knowledge branch lives or dies on this one field."""
        challenge = Challenge(
            id="chg_1",
            speaker_id="spk",
            question_text="உங்க native place எது?",
            target_class=SemanticClass.PLACE_LOCAL,
            expected_predicate="hometown",
            expected_answer="Thanjavur",
            issued_at=1.0,
            expires_at=61.0,
        )
        assert challenge_to_wire(challenge).expected_answer_entity == ""
        assert (
            challenge_to_wire(challenge, reveal_answer=True).expected_answer_entity
            == "Thanjavur"
        )

    def test_divergence_is_low_when_the_probe_matches_the_speaker(self) -> None:
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        ubm = build_background_model([CSBG.build("other", speaker_utterances(ENGLISH_NUMBERS))])
        score = score_llr([tokens("NUMBER:TA NUMBER:TA FOOD:TA FOOD:TA")], graph, ubm)

        divergences = {
            d.semantic_class: d for d in map(contribution_to_wire, score.contributions)
        }
        assert divergences["NUMBER"].jsd < 0.15

    def test_divergence_is_high_when_the_probe_contradicts_the_speaker(self) -> None:
        graph = CSBG.build("spk", speaker_utterances(TAMIL_NUMBERS))
        ubm = build_background_model([CSBG.build("other", speaker_utterances(ENGLISH_NUMBERS))])
        score = score_llr([tokens("NUMBER:EN NUMBER:EN NUMBER:EN NUMBER:EN")], graph, ubm)

        divergences = {
            d.semantic_class: d for d in map(contribution_to_wire, score.contributions)
        }
        assert divergences["NUMBER"].jsd > 0.5
        assert divergences["NUMBER"].expected_language == "TA"
        assert divergences["NUMBER"].observed_language == "EN"

    def test_skg_triples_round_trip(self) -> None:
        kg = skg_from_triples(
            "spk",
            [
                schemas.Triple(subject="ignored", predicate="hometown", object="Erode"),
                schemas.Triple(subject="ignored", predicate="commute", object="bus"),
            ],
        )
        triples = skg_to_triples(kg)
        assert {(t.predicate, t.object) for t in triples} == {
            ("hometown", "Erode"),
            ("commute", "bus"),
        }
        # The subject comes from the speaker id, never from the request body:
        # otherwise editing a string in the form rewrites someone else's facts.
        assert all("speaker_spk" in t.subject for t in triples)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


class TestRoutes:
    def test_health_reports_missing_models_rather_than_claiming_ready(
        self, client: TestClient
    ) -> None:
        """A server with no acoustic model must not report itself connected.

        This environment has no speechbrain and no faster-whisper. If health
        said "connected" anyway, an operator would read an empty `models` list
        as a display bug rather than as two absent branches.
        """
        body = client.get("/api/health").json()
        assert body["status"] == "degraded"
        unavailable = body["reportable"]["unavailable"]
        assert "speaker_embedding" in unavailable
        assert "asr" in unavailable
        # Each entry must say what is lost, not just what is absent.
        assert "acoustic branch" in unavailable["speaker_embedding"]

    def test_health_announces_a_demo_build(self, client: TestClient) -> None:
        """`demo_reveal_answers` leaks the knowledge factor; it must be visible."""
        body = client.get("/api/health").json()
        assert body["demoRevealAnswers"] is False
        assert body["reportable"]["speaker_threshold"] == 0.62

    def test_speaker_lifecycle(self, client: TestClient) -> None:
        created = make_speaker(client, "Divya", gender="F", device="Redmi Note 12")
        assert created["displayName"] == "Divya"
        assert created["utteranceCount"] == 0

        listed = client.get("/api/speakers").json()
        assert [s["id"] for s in listed] == [created["id"]]

        fetched = client.get(f"/api/speakers/{created['id']}").json()
        assert fetched["device"] == "Redmi Note 12"

        assert client.delete(f"/api/speakers/{created['id']}").json() == {"deleted": True}
        assert client.get(f"/api/speakers/{created['id']}").status_code == 404

    def test_unknown_speaker_is_404_everywhere(self, client: TestClient) -> None:
        for path in (
            "/api/speakers/nope",
            "/api/speakers/nope/utterances",
            "/api/speakers/nope/csbg",
            "/api/speakers/nope/skg",
        ):
            assert client.get(path).status_code == 404, path

    def test_csbg_before_enrolment_is_empty_not_an_error(self, client: TestClient) -> None:
        speaker = make_speaker(client, "New")
        body = client.get(f"/api/speakers/{speaker['id']}/csbg").json()
        assert body["nodes"] == []
        assert body["speakerId"] == speaker["id"]
        assert len(body["sparseClasses"]) == 21

    def test_skg_put_then_get(self, client: TestClient) -> None:
        speaker = make_speaker(client, "Karthik")
        payload = [
            {"subject": "x", "predicate": "hometown", "object": "Thanjavur"},
            {"subject": "x", "predicate": "favouriteFood", "object": "kothu parotta"},
        ]
        put = client.put(f"/api/speakers/{speaker['id']}/skg", json=payload).json()
        assert len(put) == 2
        got = client.get(f"/api/speakers/{speaker['id']}/skg").json()
        assert {t["predicate"] for t in got} == {"hometown", "favouriteFood"}

    def test_upload_and_serve_audio(self, client: TestClient, store: Store) -> None:
        speaker = make_speaker(client, "Meena")
        response = client.post(
            "/api/utterances",
            files={"audio": ("clip.wav", wav_bytes(), "audio/wav")},
            data={"speakerId": speaker["id"], "type": "free-speech"},
        )
        assert response.status_code == 201, response.text
        utterance = response.json()
        assert utterance["durationSec"] == pytest.approx(2.0, abs=0.05)
        assert utterance["sampleRate"] == 16_000
        # ASR is unavailable in this environment, so the recording is stored
        # unannotated rather than rejected.
        assert utterance["annotated"] is False
        assert utterance["audioUrl"] == f"/api/audio/{utterance['id']}"

        audio = client.get(utterance["audioUrl"])
        assert audio.status_code == 200
        assert len(audio.content) > 1000

        assert client.get(f"/api/speakers/{speaker['id']}/utterances").json()[0]["id"] == (
            utterance["id"]
        )

    def test_empty_upload_is_rejected(self, client: TestClient) -> None:
        speaker = make_speaker(client, "Meena")
        response = client.post(
            "/api/utterances",
            files={"audio": ("clip.wav", b"", "audio/wav")},
            data={"speakerId": speaker["id"], "type": "free-speech"},
        )
        assert response.status_code == 400

    def test_deleting_an_utterance_removes_the_file(
        self, client: TestClient, store: Store
    ) -> None:
        speaker = make_speaker(client, "Meena")
        utterance = client.post(
            "/api/utterances",
            files={"audio": ("clip.wav", wav_bytes(), "audio/wav")},
            data={"speakerId": speaker["id"], "type": "free-speech"},
        ).json()
        path = store.audio_path(utterance["id"])
        assert client.delete(f"/api/utterances/{utterance['id']}").json() == {"deleted": True}
        assert not path.exists()
        assert client.get(utterance["audioUrl"]).status_code == 404

    def test_challenge_needs_knowledge_graph_facts(self, client: TestClient) -> None:
        speaker = make_speaker(client, "Ravi")
        response = client.post("/api/challenge", json={"speakerId": speaker["id"]})
        assert response.status_code == 409
        assert "enrolment interview" in response.json()["detail"]

    def test_challenge_is_issued_and_hides_the_answer(self, client: TestClient) -> None:
        speaker = make_speaker(client, "Ravi")
        client.put(
            f"/api/speakers/{speaker['id']}/skg",
            json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
        )
        challenge = client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()
        assert challenge["targetClass"] == "PLACE_LOCAL"
        assert challenge["questionText"]
        assert challenge["expectedAnswerEntity"] == ""
        assert "Thanjavur" not in str(challenge)

    def test_a_challenge_cannot_be_answered_twice(
        self, client: TestClient, pipeline: Pipeline
    ) -> None:
        """Reuse is the replay attack; the liveness gate must fail closed."""
        speaker = make_speaker(client, "Ravi")
        client.put(
            f"/api/speakers/{speaker['id']}/skg",
            json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
        )
        challenge = client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()

        files = {"audio": ("resp.wav", wav_bytes(seconds=3.0), "audio/wav")}
        first = client.post(
            "/api/authenticate", files=files, data={"challengeId": challenge["id"]}
        ).json()
        second = client.post(
            "/api/authenticate",
            files={"audio": ("resp.wav", wav_bytes(seconds=3.0), "audio/wav")},
            data={"challengeId": challenge["id"]},
        ).json()

        assert second["decision"] == "REJECT"
        assert any("already been used" in line for line in second["explanation"])
        assert first["challengeId"] == challenge["id"]

    def test_an_unknown_challenge_is_rejected_not_accepted(self, client: TestClient) -> None:
        result = client.post(
            "/api/authenticate",
            files={"audio": ("resp.wav", wav_bytes(), "audio/wav")},
            data={"challengeId": "chg_nonexistent"},
        ).json()
        assert result["decision"] == "REJECT"
        assert any("Unknown challenge" in line for line in result["explanation"])

    def test_no_measurable_branch_rejects_and_says_why(self, client: TestClient) -> None:
        """Degraded mode must reject, and must not look like an impostor verdict.

        With no ASR and no embedder there is nothing to measure. The rejection
        has to name that as a system failure -- a user told "your voice did
        not match" when no model ran has been told something false.
        """
        speaker = make_speaker(client, "Ravi")
        client.put(
            f"/api/speakers/{speaker['id']}/skg",
            json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
        )
        challenge = client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()
        result = client.post(
            "/api/authenticate",
            files={"audio": ("resp.wav", wav_bytes(seconds=3.0), "audio/wav")},
            data={"challengeId": challenge["id"]},
        ).json()

        assert result["decision"] == "REJECT"
        assert "system failure" in " ".join(result["explanation"])

        # No *identity* branch may be reported when none of them ran. The two
        # gates are exempt and must still appear: liveness is decided from the
        # ledger and signal integrity is pure signal processing, so both are
        # real measurements even with every model absent. Listing them is the
        # point -- it is what distinguishes "we checked what we could and found
        # nothing wrong, but could not identify you" from "we checked nothing".
        identity = {"speaker_embedding", "csbg", "knowledge"}
        assert [b for b in result["branches"] if b["name"] in identity] == []
        assert {b["name"] for b in result["branches"]} == {"liveness", "signal_integrity"}

    def test_auth_history_is_recorded(self, client: TestClient) -> None:
        speaker = make_speaker(client, "Ravi")
        client.put(
            f"/api/speakers/{speaker['id']}/skg",
            json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
        )
        challenge = client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()
        client.post(
            "/api/authenticate",
            files={"audio": ("resp.wav", wav_bytes(seconds=3.0), "audio/wav")},
            data={"challengeId": challenge["id"]},
        )
        history = client.get("/api/auth-history?limit=10").json()
        assert len(history) == 1
        assert history[0]["speakerId"] == speaker["id"]
        assert schemas.AuthResult.model_validate(history[0])

    def test_evaluation_is_empty_before_there_are_trials(self, client: TestClient) -> None:
        """An honest empty page beats a fabricated curve."""
        body = client.get("/api/evaluation").json()
        assert body == {
            "configurations": [],
            "stabilityCurve": [],
            "fairness": [],
            "scoreDistributions": [],
        }

    def test_enrolment_warns_about_a_thin_corpus(
        self, client: TestClient, store: Store
    ) -> None:
        speaker = make_speaker(client, "Ravi")
        enrol_tokens(store, speaker["id"], speaker_utterances(TAMIL_NUMBERS, n=2))
        body = client.post(f"/api/speakers/{speaker['id']}/enrol/complete").json()

        warnings = " ".join(body["warnings"])
        assert "working minimum" in warnings
        assert "background model" in warnings
        assert body["csbg"]["speakerId"] == speaker["id"]
        assert len(body["csbg"]["nodes"]) > 0


# --------------------------------------------------------------------------
# Veto calibration
# --------------------------------------------------------------------------


class TestVetoCalibration:
    """The veto floor must be a number the scorer can actually reach.

    `FusionPolicy.veto_thresholds` is set on the branch's [0, 1] scale, but the
    values that scale actually takes are decided by `csbg.scoring._squash`.
    Those two facts live in different modules, and nothing else connects them:
    the fusion tests use hand-written branch scores, and the scorer tests never
    look at fusion. So a floor could sit below anything the scorer emits, the
    veto would silently never fire, and every test would still pass.

    It did. See `fusion.CSBG_VETO_FLOOR` for the arithmetic.
    """

    def _scores(self) -> tuple[float, float]:
        """(genuine, impostor) CSBG branch scores from the real scorer."""
        victim = CSBG.build("v", speaker_utterances(TAMIL_NUMBERS, n=10))
        ubm = build_background_model(
            [CSBG.build(f"o{i}", speaker_utterances(ENGLISH_NUMBERS, n=10)) for i in range(3)]
        )
        genuine = score_llr([speaker_utterances(TAMIL_NUMBERS, n=1)[0]], victim, ubm)
        impostor = score_llr([speaker_utterances(ENGLISH_NUMBERS, n=1)[0]], victim, ubm)
        return genuine.normalised_score, impostor.normalised_score

    def test_the_floor_is_above_a_decisive_impostor(self) -> None:
        from kavach.fusion import CSBG_VETO_FLOOR

        _, impostor = self._scores()
        assert impostor < CSBG_VETO_FLOOR, (
            f"An impostor contradicting the speaker in every measured class scores "
            f"{impostor:.3f}, at or above the veto floor of {CSBG_VETO_FLOOR}. The veto "
            "cannot fire on any real probe."
        )

    def test_the_floor_is_below_a_genuine_speaker(self) -> None:
        from kavach.fusion import CSBG_VETO_FLOOR

        genuine, _ = self._scores()
        assert genuine > CSBG_VETO_FLOOR + 0.2, (
            f"A genuine speaker scores {genuine:.3f} against a veto floor of "
            f"{CSBG_VETO_FLOOR}. Too little headroom -- this floor will reject real "
            "users. Measure the false-reject rate before shipping it."
        )

    def test_the_floor_matches_its_documented_likelihood_ratio(self) -> None:
        """0.35 is claimed to mean 'the background is ~3.5x more likely'."""
        import math

        from kavach.fusion import CSBG_VETO_FLOOR

        raw_llr = 2.0 * math.log(CSBG_VETO_FLOOR / (1.0 - CSBG_VETO_FLOOR))
        assert math.exp(-raw_llr) == pytest.approx(3.5, abs=0.2)


# --------------------------------------------------------------------------
# Attack lab
# --------------------------------------------------------------------------


class TestAttackLab:
    """The Attack Lab's numbers are simulated; these tests check it says so,
    and that the one column it computes honestly behaves as the paper claims."""

    def _corpus(self, store: Store) -> str:
        victim = store.create_speaker({"display_name": "Victim"})["id"]
        enrol_tokens(store, victim, speaker_utterances(TAMIL_NUMBERS, n=10))
        for i in range(3):
            other = store.create_speaker({"display_name": f"Other{i}"})["id"]
            enrol_tokens(store, other, speaker_utterances(ENGLISH_NUMBERS, n=10))
        return victim

    def test_a_run_declares_itself_simulated(self, store: Store, pipeline: Pipeline) -> None:
        victim = self._corpus(store)
        run = run_attack(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            speaker_id=victim,
            trials=40,
            store=store,
            pipeline=pipeline,
        )
        assert run.simulated is True
        assert any("Simulated attack" in n for n in run.notes)

    def test_a_run_without_a_cohort_explains_itself(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        victim = store.create_speaker({"display_name": "Alone"})["id"]
        enrol_tokens(store, victim, speaker_utterances(TAMIL_NUMBERS, n=4))
        run = run_attack(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            speaker_id=victim,
            trials=10,
            store=store,
            pipeline=pipeline,
        )
        assert run.trials == 0
        assert any("background model" in n for n in run.notes)

    def test_the_csbg_column_stops_a4_where_knowledge_cannot(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        """THE HEADLINE CLAIM, on the one branch computed for real.

        A4 has the voice and the answer, so ECAPA and knowledge both pass. The
        only thing left is how they code-switch, and it is not the victim's.
        """
        victim = self._corpus(store)
        run = run_attack(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            speaker_id=victim,
            trials=60,
            store=store,
            pipeline=pipeline,
        )
        rates = run.success_rate_by_config
        assert rates["ecapa_only"] > 0.9, "the attacker should own the acoustic branch"
        assert rates["plus_knowledge"] > 0.9, "A4 knows the answer, by definition"
        assert rates["full_fusion"] < 0.1, (
            "A4 got through the full system: the CSBG branch is not doing the work "
            f"the paper claims for it (rates={rates})."
        )

    def test_a1_is_stopped_by_freshness_not_by_the_csbg(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        """A replay is the victim's own speech, so the CSBG has nothing to see.

        This is the honest half of the table: the contribution stops A4, and
        the challenge protocol stops A1. Claiming the CSBG stops replays would
        be claiming credit for the liveness gate.
        """
        victim = self._corpus(store)
        run = run_attack(
            attack=AttackType.A1_REPLAY,
            speaker_id=victim,
            trials=40,
            store=store,
            pipeline=pipeline,
        )
        rates = run.success_rate_by_config
        assert rates["ecapa_only"] > 0.9
        assert rates["plus_csbg"] > 0.5, (
            "the CSBG should NOT stop a replay -- it is the victim's own speech"
        )
        assert rates["full_fusion"] == 0.0, "the liveness gate must stop every replay"

    def test_every_row_says_what_actually_defeats_it(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        """Three of five attacks are stopped by something that is not a branch.

        Without this the A2 row reads as "the system has no answer to a
        splice", and the A1 row reads as though the CSBG failed at something
        it was never asked to do.
        """
        victim = self._corpus(store)
        for attack in AttackType:
            run = run_attack(
                attack=attack, speaker_id=victim, trials=30,
                store=store, pipeline=pipeline,
            )
            assert any(DEFEATED_BY[attack] == n for n in run.notes), attack

    def test_a_budget_larger_than_the_corpus_is_labelled_oracle(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        """An unbinding eavesdropping budget is the oracle condition.

        The A5-observed row is a claim about what a real attacker reaches. If
        the budget exceeds everything the speaker has on record, the attacker
        was handed the lot -- that is the unrealisable upper bound, and filing
        it as "observed" would put the wrong number in the row a reviewer
        reads most carefully.
        """
        victim = self._corpus(store)  # 10 utterances x 5 tokens = 20s
        run = run_attack(
            attack=AttackType.A5_STYLE_ADAPTIVE,
            speaker_id=victim, trials=30, store=store, pipeline=pipeline,
        )
        assert any("A5-oracle" in n for n in run.notes), run.notes

    def test_a_binding_budget_is_labelled_observed(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        victim = store.create_speaker({"display_name": "Chatty"})["id"]
        # 200 utterances x 5 tokens x 0.4s = 400s, well past the 60s budget.
        enrol_tokens(store, victim, speaker_utterances(TAMIL_NUMBERS, n=200))
        for i in range(3):
            other = store.create_speaker({"display_name": f"O{i}"})["id"]
            enrol_tokens(store, other, speaker_utterances(ENGLISH_NUMBERS, n=10))

        run = run_attack(
            attack=AttackType.A5_STYLE_ADAPTIVE,
            speaker_id=victim, trials=30, store=store, pipeline=pipeline,
        )
        assert any("A5-observed condition" in n for n in run.notes), run.notes
        assert not any("A5-oracle" in n for n in run.notes)

    def test_a5_scores_between_a4_and_the_victim(
        self, store: Store, pipeline: Pipeline
    ) -> None:
        """The credibility test: an adaptive attacker does better than A4.

        If A5 scored no better than A4, the style estimate would be doing
        nothing and the A5 row would be theatre.
        """
        victim = self._corpus(store)
        a4 = run_attack(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            speaker_id=victim, trials=60, store=store, pipeline=pipeline,
        )
        a5 = run_attack(
            attack=AttackType.A5_STYLE_ADAPTIVE,
            speaker_id=victim, trials=60, store=store, pipeline=pipeline,
        )
        assert a5.success_rate_by_config["full_fusion"] >= (
            a4.success_rate_by_config["full_fusion"]
        )

    def test_clone_attacks_report_yield(self, store: Store, pipeline: Pipeline) -> None:
        """A success rate without a yield is uninterpretable."""
        victim = self._corpus(store)
        run = run_attack(
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            speaker_id=victim, trials=40, store=store, pipeline=pipeline,
        )
        assert run.yield_rate is not None
        assert 0.0 <= run.yield_rate <= 1.0

    def test_non_clone_attacks_have_no_yield(self, store: Store, pipeline: Pipeline) -> None:
        victim = self._corpus(store)
        run = run_attack(
            attack=AttackType.A1_REPLAY,
            speaker_id=victim, trials=40, store=store, pipeline=pipeline,
        )
        assert run.yield_rate is None

    def test_runs_are_reproducible(self, store: Store, pipeline: Pipeline) -> None:
        victim = self._corpus(store)
        kw = dict(
            attack=AttackType.A3_CLONE, speaker_id=victim, trials=30,
            store=store, pipeline=pipeline,
        )
        assert run_attack(**kw).success_rate_by_config == run_attack(**kw).success_rate_by_config

    def test_the_route_records_the_run(self, client: TestClient, store: Store) -> None:
        victim = self._corpus(store)
        response = client.post(
            "/api/attacks/generate",
            json={"attackType": "A4_CLONE_KNOWLEDGE", "targetSpeakerId": victim, "trials": 30},
        )
        assert response.status_code == 200, response.text
        assert response.json()["attackType"] == "A4_CLONE_KNOWLEDGE"

        listed = client.get("/api/attacks").json()
        assert len(listed) == 1
        assert schemas.AttackRun.model_validate(listed[0])

    def test_an_unknown_attack_type_is_a_400(self, client: TestClient, store: Store) -> None:
        victim = self._corpus(store)
        response = client.post(
            "/api/attacks/generate",
            json={"attackType": "A9_MIND_READING", "targetSpeakerId": victim, "trials": 10},
        )
        assert response.status_code == 422 or response.status_code == 400
