"""The calibrated voice policy: written by the calibration tool, read by the live pipeline.

A threshold chosen from measured scores must reach the login without anyone editing code, and a
damaged policy file must never take the login down or silently loosen it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.calibrate_voice import PolicyError, load_voice_policy, write_voice_policy, choose_operating_point

from test_phrase_login import (  # noqa: F401  (fixtures and helpers shared with the phrase tests)
    FakeASR, FakeEmbedder, client, issue, login, pipeline, probe_with_cosine, settings, speaker, store, wav_bytes,
)


def op():
    r = np.random.default_rng(0)
    return choose_operating_point(r.normal(0.85, 0.05, 200), r.normal(0.2, 0.1, 5000))


def test_a_missing_policy_means_the_defaults(tmp_path):
    assert load_voice_policy(tmp_path / "voice_policy.json") is None


def test_a_written_policy_round_trips(tmp_path):
    path = tmp_path / "voice_policy.json"
    o = op()
    write_voice_policy(path, o, sessions=["S1", "S2"], note="test")
    p = load_voice_policy(path)
    assert p.threshold == pytest.approx(o.threshold) and p.grey_margin == pytest.approx(o.grey_margin)
    assert p.provisional == o.provisional and p.sessions == ["S1", "S2"]


@pytest.mark.parametrize("payload", [
    "not json",
    json.dumps({"threshold": 0.2, "grey_margin": 0.08}),     # below the minimum: would admit anyone
    json.dumps({"threshold": 0.99, "grey_margin": 0.08}),    # nobody could pass
    json.dumps({"threshold": 0.7, "grey_margin": -0.1}),
    json.dumps({"threshold": 0.7, "grey_margin": 0.9}),
    json.dumps({"threshold": "high", "grey_margin": 0.05}),
    json.dumps({"grey_margin": 0.05}),
])
def test_a_damaged_policy_is_refused(tmp_path, payload):
    path = tmp_path / "voice_policy.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(PolicyError):
        load_voice_policy(path)


def test_nan_is_refused(tmp_path):
    path = tmp_path / "voice_policy.json"
    path.write_text('{"threshold": NaN, "grey_margin": 0.05}', encoding="utf-8")
    with pytest.raises(PolicyError):
        load_voice_policy(path)


class TestPipelineUsesIt:
    def write(self, settings, threshold, margin, **extra):
        (Path(settings.data_dir) / "voice_policy.json").write_text(
            json.dumps({"threshold": threshold, "grey_margin": margin, "provisional": False, "ready": True, **extra}),
            encoding="utf-8",
        )

    def fresh(self, store, settings):
        from kavach.api.pipeline import Pipeline
        from test_phrase_login import ExplodingLID
        p = Pipeline(store, settings)
        p._lid = ExplodingLID()
        return p

    def test_health_reports_the_calibrated_numbers(self, store, settings, speaker, tmp_path):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        self.write(settings, 0.80, 0.05)
        p = self.fresh(store, settings)
        app = create_app(settings)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: p
        app.dependency_overrides[get_settings] = lambda: settings
        h = TestClient(app).get("/api/health").json()
        assert h["voiceThreshold"] == pytest.approx(0.80) and h["voiceGreyMargin"] == pytest.approx(0.05)
        assert h["voicePolicySource"] == "calibrated"

    def test_a_login_is_judged_by_the_calibrated_threshold(self, store, settings, speaker, tmp_path):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        self.write(settings, 0.80, 0.05)
        p = self.fresh(store, settings)
        app = create_app(settings)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: p
        app.dependency_overrides[get_settings] = lambda: settings
        c = TestClient(app)

        def decision(cos):
            ch = issue(c, speaker)
            r, _ = login(c, p, tmp_path, ch, heard=" ".join(ch["phrase"]), cosine=cos)
            return r["decision"]

        assert decision(0.70) == "REJECT"      # under 0.80 - 0.05: a flat no, though the default 0.62 would have passed it
        assert decision(0.78) == "BORDERLINE"  # in the band
        assert decision(0.85) == "ACCEPT"

    def test_a_damaged_policy_falls_back_to_the_defaults_and_says_so(self, store, settings, speaker):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        (Path(settings.data_dir) / "voice_policy.json").write_text("{broken", encoding="utf-8")
        p = self.fresh(store, settings)
        app = create_app(settings)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: p
        app.dependency_overrides[get_settings] = lambda: settings
        h = TestClient(app).get("/api/health").json()
        assert h["voiceThreshold"] == pytest.approx(settings.speaker_threshold)
        assert h["voicePolicySource"] == "default"
        assert h["voicePolicyError"]


class TestEvidenceRoute:
    """What the panel is shown: the measured numbers behind the threshold, with their limits."""

    def app_for(self, store, settings, pipeline):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        app = create_app(settings)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: pipeline
        app.dependency_overrides[get_settings] = lambda: settings
        return TestClient(app)

    def test_without_a_policy_it_says_nothing_was_measured(self, store, settings, pipeline, speaker):
        body = self.app_for(store, settings, pipeline).get("/api/voice-policy").json()
        assert body["source"] == "default" and body["measured"] is False
        assert body["threshold"] == pytest.approx(settings.speaker_threshold)
        assert body["nGenuine"] == 0 and body["frr"] is None

    def test_with_a_policy_it_carries_the_measurements_and_their_intervals(self, store, settings, pipeline, speaker, tmp_path):
        r = np.random.default_rng(0)
        o = choose_operating_point(r.normal(0.85, 0.05, 200), r.normal(0.2, 0.1, 5000))
        write_voice_policy(Path(settings.data_dir) / "voice_policy.json", o, sessions=["S1", "S2", "S3"], cohorts=["libri-dev", "tamil-m"], limits="LIMITS...")
        from kavach.api.pipeline import Pipeline
        p = Pipeline(store, settings)
        body = self.app_for(store, settings, p).get("/api/voice-policy").json()
        assert body["source"] == "calibrated" and body["measured"] is True
        assert body["nGenuine"] == 200 and body["nImpostor"] == 5000
        assert body["frr"]["rate"] == 0.0 and body["frr"]["high"] > 0.0
        assert body["far"]["rate"] == 0.0 and body["far"]["high"] > 0.0
        assert body["sessions"] == ["S1", "S2", "S3"] and body["cohorts"] == ["libri-dev", "tamil-m"]
        assert body["limits"] == "LIMITS..."
        assert body["provisional"] is False

    def test_a_damaged_policy_is_reported_not_hidden(self, store, settings, speaker):
        from kavach.api.pipeline import Pipeline
        (Path(settings.data_dir) / "voice_policy.json").write_text("{broken", encoding="utf-8")
        p = Pipeline(store, settings)
        body = self.app_for(store, settings, p).get("/api/voice-policy").json()
        assert body["source"] == "default" and body["error"]


def test_the_provenance_block_reports_the_threshold_actually_in_force(store, settings, speaker):
    """`reportable` is what someone copies into a write-up; it must not name the default 0.62 while a
    calibrated threshold is judging the logins."""
    from kavach.api.app import create_app, get_pipeline, get_settings, get_store
    from kavach.api.pipeline import Pipeline

    (Path(settings.data_dir) / "voice_policy.json").write_text(
        json.dumps({"threshold": 0.71, "grey_margin": 0.06, "provisional": False}), encoding="utf-8")
    p = Pipeline(store, settings)
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: p
    app.dependency_overrides[get_settings] = lambda: settings
    rep = TestClient(app).get("/api/health").json()["reportable"]
    assert rep["speaker_threshold"] == pytest.approx(0.71)
    assert rep["voice_grey_margin"] == pytest.approx(0.06)


class TestPhraseDecisionFollowsTheVoiceThresholdAlone:
    """In a phrase login the voice is the only weighted branch, so the decision must be exactly the voice
    zones. The generic fused threshold (0.55 +/- 0.05) used to leak in: a calibrated threshold of 0.58
    turned a voice of 0.59, which passed, into BORDERLINE."""

    def run(self, store, settings, speaker, tmp_path, threshold, margin, cosine):
        from kavach.api.app import create_app, get_pipeline, get_settings, get_store
        from kavach.api.pipeline import Pipeline
        from test_phrase_login import ExplodingLID
        (Path(settings.data_dir) / "voice_policy.json").write_text(
            json.dumps({"threshold": threshold, "grey_margin": margin, "provisional": False}), encoding="utf-8")
        p = Pipeline(store, settings)
        p._lid = ExplodingLID()
        app = create_app(settings)
        app.dependency_overrides[get_store] = lambda: store
        app.dependency_overrides[get_pipeline] = lambda: p
        app.dependency_overrides[get_settings] = lambda: settings
        c = TestClient(app)
        ch = issue(c, speaker)
        r, _ = login(c, p, tmp_path, ch, heard=" ".join(ch["phrase"]), cosine=cosine)
        return r["decision"]

    def test_a_voice_just_over_a_low_calibrated_threshold_is_accepted(self, store, settings, speaker, tmp_path):
        assert self.run(store, settings, speaker, tmp_path, 0.58, 0.05, 0.59) == "ACCEPT"

    def test_a_voice_at_the_default_fused_threshold_does_not_decide_anything(self, store, settings, speaker, tmp_path):
        assert self.run(store, settings, speaker, tmp_path, 0.75, 0.05, 0.72) == "BORDERLINE"   # in the band: asked once more
        assert self.run(store, settings, speaker, tmp_path, 0.75, 0.05, 0.55) == "REJECT"       # under the floor, though over 0.55
