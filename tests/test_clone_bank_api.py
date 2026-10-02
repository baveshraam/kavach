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
