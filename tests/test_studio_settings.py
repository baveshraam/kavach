from kavach.config import Settings
from test_clone_bank_api import build_client


def test_the_studio_is_off_and_records_nobody_by_default() -> None:
    fields = Settings.model_fields
    assert fields["studio_enabled"].default is False
    assert fields["studio_speakers"].default_factory() == []


def test_health_announces_the_studio(tmp_path) -> None:
    client, *_ = build_client(tmp_path)
    assert client.get("/api/health").json()["studioEnabled"] is False
    (tmp_path / "b").mkdir()
    client2, *_ = build_client(tmp_path / "b", studio_enabled=True)
    assert client2.get("/api/health").json()["studioEnabled"] is True
