"""The clone bank's settings, routes and leak surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app, get_pipeline, get_settings, get_store
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.config import Settings


def build_client(tmp_path, **overrides):
    settings = Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
        **overrides,
    )
    store = Store(settings.db_path, settings.audio_dir)
    pipeline = Pipeline(store, settings)
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), store, pipeline, settings


class TestSettings:
    def test_nobody_may_be_cloned_and_the_bank_is_off_by_default(self) -> None:
        fields = Settings.model_fields
        assert fields["demo_attack_bank"].default is False
        assert fields["clone_victims"].default_factory() == []

    def test_the_flag_travels_with_reported_settings(self, tmp_path) -> None:
        _, _, _, settings = build_client(tmp_path)
        assert settings.reportable()["demo_attack_bank"] is False


class TestHealth:
    def test_health_announces_whether_the_bank_is_on(self, tmp_path) -> None:
        client, *_ = build_client(tmp_path)
        assert client.get("/api/health").json()["demoAttackBank"] is False

    def test_health_announces_it_when_on(self, tmp_path) -> None:
        client, *_ = build_client(tmp_path, demo_attack_bank=True)
        assert client.get("/api/health").json()["demoAttackBank"] is True


import json

from clone_helpers import add_clip, tone
from kavach.attacks.bank import CloneBank
from kavach.audio import save_wav

VICTIM = "S04"


def with_bank(tmp_path, **overrides):
    """A client with a challenge-able victim (one fact) and an empty bank in the
    place the pipeline looks for it: attack_dir / clones / S04."""
    client, store, pipeline, settings = build_client(
        tmp_path, demo_attack_bank=True, clone_victims=[VICTIM], **overrides
    )
    speaker = store.create_speaker({"display_name": f"{VICTIM} · Tester"})
    client.put(
        f"/api/speakers/{speaker['id']}/skg",
        json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
    )
    bank = CloneBank.new(
        settings.attack_dir / "clones" / VICTIM, VICTIM, speaker["id"], allowed_victims=[VICTIM]
    )
    return client, store, pipeline, settings, speaker, bank


def issue(client, speaker_id):
    return client.post("/api/challenge", json={"speakerId": speaker_id}).json()["id"]


def switch_off(client, pipeline, settings):
    off = settings.model_copy(update={"demo_attack_bank": False})
    pipeline.settings = off
    client.app.dependency_overrides[get_settings] = lambda: off


class TestGate:
    def test_every_route_is_a_404_while_the_flag_is_off_even_for_a_real_clip(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        switch_off(client, pipeline, settings)
        cid = issue(client, speaker["id"])
        assert client.get("/api/clone-bank").status_code == 404
        assert client.post("/api/clone-bank/match", json={"challengeId": cid}).status_code == 404
        assert client.get(f"/api/clone-bank/{clip.clip_id}/audio").status_code == 404

    def test_no_other_route_serves_the_clone_text_while_off(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        switch_off(client, pipeline, settings)
        for path in ("/api/health", "/api/speakers", f"/api/speakers/{speaker['id']}"):
            assert "naan hometown" not in client.get(path).text, path


class TestInfo:
    def test_no_bank_says_so_instead_of_failing(self, tmp_path) -> None:
        client, *_ = with_bank(tmp_path)
        body = client.get("/api/clone-bank").json()
        assert body["enabled"] is True and body["clips"] == []
        assert body["problems"]

    def test_the_listing_exposes_scores_but_no_transcript_tokens_or_answers(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        text = client.get("/api/clone-bank").text
        body = json.loads(text)
        assert body["coveredFacts"] == ["hometown"]
        assert body["clips"][0]["similarity"] == 0.8
        for secret in ("naan hometown", "Thanjavur", "transcript", "tokens", "answerScore"):
            assert secret not in text, secret


class TestMatch:
    def test_it_returns_the_clip_for_the_issued_challenges_fact(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        cid = issue(client, speaker["id"])
        r = client.post("/api/clone-bank/match", json={"challengeId": cid})
        assert r.status_code == 200, r.text
        assert r.json()["clipId"] == clip.clip_id
        audio = client.get(r.json()["audioUrl"])
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"

    def test_matching_does_not_consume_the_challenge(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        cid = issue(client, speaker["id"])
        client.post("/api/clone-bank/match", json={"challengeId": cid})
        assert pipeline.ledger.get(cid).consumed is False

    def test_an_unknown_challenge_is_a_404(self, tmp_path) -> None:
        client, *_ = with_bank(tmp_path)
        assert client.post("/api/clone-bank/match", json={"challengeId": "nope"}).status_code == 404

    def test_a_consumed_challenge_is_a_409(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        cid = issue(client, speaker["id"])
        pipeline.ledger.consume(cid)
        assert client.post("/api/clone-bank/match", json={"challengeId": cid}).status_code == 409

    def test_a_fact_no_clone_answers_names_what_the_bank_covers(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank, fact_key="college")  # the speaker's only fact is `hometown`
        bank.save()
        cid = issue(client, speaker["id"])
        r = client.post("/api/clone-bank/match", json={"challengeId": cid})
        assert r.status_code == 404
        assert "hometown" in r.json()["detail"] and "college" in r.json()["detail"]

    def test_an_unannotated_clip_is_never_served(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        clip = add_clip(bank, annotated=False)
        bank.save()
        cid = issue(client, speaker["id"])
        assert client.post("/api/clone-bank/match", json={"challengeId": cid}).status_code == 404
        assert client.get(f"/api/clone-bank/{clip.clip_id}/audio").status_code == 404


class TestBrokenBank:
    def test_a_tampered_clip_is_a_503_and_the_rest_of_the_app_keeps_working(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        save_wav(tone(seed=7), bank.root / clip.audio_path)  # changed after generation
        r = client.get("/api/clone-bank")
        assert r.status_code == 503 and "hash" in r.json()["detail"]
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/speakers").status_code == 200

    def test_a_victim_removed_from_the_allowlist_has_no_visible_bank(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        narrowed = settings.model_copy(update={"clone_victims": ["S09"]})
        pipeline.settings = narrowed
        client.app.dependency_overrides[get_settings] = lambda: narrowed
        body = client.get("/api/clone-bank").json()
        assert body["clips"] == [] and body["problems"]
