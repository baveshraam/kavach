"""The preflight knows about the phrase login, the voice gate and the calibrated policy."""

from __future__ import annotations

from kavach.demo_check import FAIL, PASS, WARN, TransportError, run_checks

from test_demo_check import HEALTH, Fake, by_name, good, levels

CAL = dict(HEALTH, voiceGate=True, voicePolicySource="calibrated", voicePolicyProvisional=False, voicePolicyError="")
VOICEPRINT_STUDIO = {"nClips": 52, "selfConsistency": 0.87,
                     "provenance": {"source": "studio", "sessions": ["S1", "S2", "S3"], "devices": ["DEMO_LAPTOP_MIC"], "n_clips": 52}}


def run(health=CAL, **routes):
    return by_name(run_checks(good({"/api/health": health, **routes}), presenter="S04"))


def test_the_voice_gate_must_be_on():
    assert run(dict(CAL, voiceGate=False))["voice is a hard gate"].level == FAIL
    assert run()["voice is a hard gate"].level == PASS


def test_an_old_server_that_does_not_say_is_a_warning_not_a_pass():
    h = {k: v for k, v in CAL.items() if k != "voiceGate"}
    assert run(h)["voice is a hard gate"].level == WARN


def test_the_default_threshold_is_a_starting_point_and_says_how_to_calibrate():
    c = run(dict(CAL, voicePolicySource="default"))["voice threshold is calibrated"]
    assert c.level == WARN and "calibrate_voice" in c.fix


def test_a_provisional_calibration_warns():
    assert run(dict(CAL, voicePolicyProvisional=True))["voice threshold is calibrated"].level == WARN


def test_a_calibrated_final_policy_passes():
    assert run()["voice threshold is calibrated"].level == PASS


def test_a_damaged_policy_file_fails_because_the_defaults_are_silently_in_force():
    c = run(dict(CAL, voicePolicySource="default", voicePolicyError="threshold 0.2 is outside"))["voice threshold is calibrated"]
    assert c.level == FAIL and "outside" in c.detail


def test_a_voiceprint_enrolled_from_studio_sessions_passes():
    c = run(**{"/api/speakers/spk_p/voiceprint": VOICEPRINT_STUDIO})["presenter's voiceprint matches the demo's microphone path"]
    assert c.level == PASS and "S1" in c.detail


def test_a_voiceprint_from_the_original_recordings_warns_and_names_the_fix():
    c = run(**{"/api/speakers/spk_p/voiceprint": {"nClips": 13, "selfConsistency": 0.84, "provenance": None}})["presenter's voiceprint matches the demo's microphone path"]
    assert c.level == WARN and "studio.enrol" in c.fix


def test_a_server_without_the_voiceprint_route_is_a_warning():
    assert run()["presenter's voiceprint matches the demo's microphone path"].level == WARN


def test_a_presenter_with_no_facts_only_loses_the_question_login():
    """The phrase login needs no facts, so the demo is not blocked on them."""
    c = run(**{"/api/speakers/spk_p/skg": []})["presenter has knowledge-graph facts"]
    assert c.level == WARN and "phrase" in c.detail.lower()


# ---- flows -----------------------------------------------------------------------


def phrase_routes(stranger_decision, stranger_voice_ok, old_decision, old_phrase_ok):
    def chall(body):
        assert body.get("kind") == "phrase"
        return {"id": "chg_1", "speakerId": "spk_p", "questionText": "read", "kind": "phrase",
                "phrase": ["tiger", "river", "mango", "window", "candle", "silver"], "targetClass": "OTHER",
                "expectedAnswerEntity": "", "issuedAt": "2026-01-01T00:00:00Z", "expiresAt": "2026-01-01T00:01:00Z", "stepUp": False}

    def result(decision, voice_ok, phrase_ok):
        return {"decision": decision, "explanation": ["x"], "latencyMs": 2000, "transcript": "",
                "branches": [
                    {"name": "phrase", "score": 1.0 if phrase_ok else 0.0, "threshold": 0.67, "weight": 0, "passed": phrase_ok},
                    {"name": "speaker_embedding", "score": 0.2 if not voice_ok else 0.9, "threshold": 0.62, "weight": 0.4, "passed": voice_ok},
                ]}

    return {
        ("POST", "/api/challenge"): chall,
        ("AUTH", "pstranger"): result(stranger_decision, stranger_voice_ok, True),
        ("AUTH", "pold"): result(old_decision, True, old_phrase_ok),
    }


def speak_ok(text, voice):
    return b"RIFFfake"


def flows(speak=speak_ok, **kw):
    routes = phrase_routes(**{"stranger_decision": "REJECT", "stranger_voice_ok": False, "old_decision": "REJECT", "old_phrase_ok": False, **kw})
    return by_name(run_checks(good({"/api/health": CAL, "/api/speakers/spk_p/skg": [], **routes}), presenter="S04", flows=True, speak=speak))


def test_a_synthetic_stranger_reading_the_right_words_must_be_rejected_on_the_voice():
    c = flows()["phrase login: a stranger's voice reading the shown words is rejected"]
    assert c.level == PASS


def test_a_stranger_who_is_accepted_is_a_failure():
    c = flows(stranger_decision="ACCEPT", stranger_voice_ok=True)["phrase login: a stranger's voice reading the shown words is rejected"]
    assert c.level == FAIL


def test_a_stranger_rejected_only_because_the_words_were_not_recognised_is_a_warning():
    """Then the voice gate was never exercised: speech recognition may be failing."""
    def r(body):
        return {"decision": "REJECT", "explanation": [], "latencyMs": 1, "transcript": "", "branches": [
            {"name": "phrase", "score": 0.0, "threshold": 0.67, "weight": 0, "passed": False},
            {"name": "speaker_embedding", "score": 0.1, "threshold": 0.62, "weight": 0.4, "passed": False}]}
    routes = phrase_routes("REJECT", False, "REJECT", False)
    routes[("AUTH", "pstranger")] = r(None)
    c = by_name(run_checks(good({"/api/health": CAL, "/api/speakers/spk_p/skg": [], **routes}), presenter="S04", flows=True, speak=speak_ok))[
        "phrase login: a stranger's voice reading the shown words is rejected"]
    assert c.level == WARN


def test_a_recording_of_other_words_must_be_rejected_on_the_words():
    c = flows()["phrase login: a recording of other words is rejected (replay)"]
    assert c.level == PASS


def test_a_replay_that_is_accepted_is_a_failure():
    c = flows(old_decision="ACCEPT", old_phrase_ok=True)["phrase login: a recording of other words is rejected (replay)"]
    assert c.level == FAIL


def test_without_a_text_to_speech_voice_the_phrase_flows_are_skipped_with_a_warning():
    got = flows(speak=lambda text, voice: None)
    assert got["phrase login can be rehearsed"].level == WARN
