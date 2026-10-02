from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from kavach.demo_check import FAIL, PASS, WARN, Check, TransportError, render, run_checks


def wav() -> bytes:
    n = 16000 * 12
    t = np.arange(n) / 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((0.3 * np.sin(2 * np.pi * 140 * t) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class Fake:
    """A scripted backend. Anything not scripted is a TransportError(404)."""

    def __init__(self, routes):
        self.routes = routes
        self.posted: list[str] = []

    def get(self, path):
        if path not in self.routes:
            raise TransportError(404, "not found")
        v = self.routes[path]
        if isinstance(v, Exception):
            raise v
        return v

    def get_bytes(self, path):
        return wav()

    def post_json(self, path, body):
        self.posted.append(path)
        v = self.routes.get(("POST", path))
        if isinstance(v, Exception):
            raise v
        if v is None:
            raise TransportError(404, "not scripted")
        return v(body) if callable(v) else v

    def post_audio(self, challenge_id, data, filename):
        self.posted.append(f"authenticate:{filename}")
        return self.routes[("AUTH", filename.split("_")[0].split(".")[0])]


HEALTH = {
    "status": "connected",
    "models": [
        "faster-whisper/small",
        "speechbrain/spkrec-ecapa-voxceleb",
        "gemini/gemini-3.1-flash-lite (tagging, challenges)",
        "sentence-transformers/LaBSE",
    ],
    "demoRevealAnswers": False,
    "demoAttackBank": False,
    "reportable": {"integrity_check_splice": False},
}
SPEAKERS = [
    {"id": "spk_p", "displayName": "S04 · Presenter", "totalDurationSec": 400.0, "utteranceCount": 14},
    {"id": "spk_a", "displayName": "S08 · A", "totalDurationSec": 300.0, "utteranceCount": 13},
    {"id": "spk_b", "displayName": "S09 · B", "totalDurationSec": 300.0, "utteranceCount": 13},
    {"id": "spk_c", "displayName": "S10 · C", "totalDurationSec": 300.0, "utteranceCount": 13},
]


def good(extra: dict | None = None) -> Fake:
    routes = {
        "/api/health": HEALTH,
        "/api/speakers": SPEAKERS,
        "/api/speakers/spk_p/skg": [{"subject": "x", "predicate": "hometown", "object": "T"}],
        "/api/speakers/spk_p/csbg": {"nodes": [], "edges": []},
    }
    routes.update(extra or {})
    return Fake(routes)


def by_name(checks: list[Check]) -> dict[str, Check]:
    return {c.name: c for c in checks}


def levels(checks):
    return {c.level for c in checks}


def test_a_complete_setup_has_no_failures() -> None:
    assert FAIL not in levels(run_checks(good(), presenter="S04"))


def test_an_unreachable_backend_fails_first_with_a_fix() -> None:
    checks = run_checks(Fake({"/api/health": OSError("refused")}), presenter="S04")
    assert checks[0].level == FAIL and "run_demo.ps1" in checks[0].fix


def test_a_degraded_backend_fails() -> None:
    h = dict(HEALTH, status="degraded")
    assert FAIL in levels(run_checks(good({"/api/health": h}), presenter="S04"))


def test_a_missing_voice_model_fails_and_a_missing_labse_only_warns() -> None:
    no_ecapa = dict(HEALTH, models=["faster-whisper/small"])
    assert FAIL in levels(run_checks(good({"/api/health": no_ecapa}), presenter="S04"))
    no_labse = dict(HEALTH, models=HEALTH["models"][:-1])
    got = run_checks(good({"/api/health": no_labse}), presenter="S04")
    assert FAIL not in levels(got) and WARN in levels(got)


def test_a_presenter_with_no_facts_cannot_be_challenged() -> None:
    got = by_name(run_checks(good({"/api/speakers/spk_p/skg": []}), presenter="S04"))
    fact = got["presenter has knowledge-graph facts"]
    assert fact.level == FAIL and "Speakers" in fact.fix


def test_an_unknown_presenter_fails() -> None:
    assert FAIL in levels(run_checks(good(), presenter="S99"))


def test_too_few_other_speakers_fails_because_the_csbg_needs_a_background() -> None:
    few = good({"/api/speakers": SPEAKERS[:2]})
    assert FAIL in levels(run_checks(few, presenter="S04"))


def test_a_leaky_or_splice_enabled_build_warns() -> None:
    h = dict(HEALTH, demoRevealAnswers=True, reportable={"integrity_check_splice": True})
    names = {
        c.name for c in run_checks(good({"/api/health": h}), presenter="S04") if c.level == WARN
    }
    assert any("answers" in n for n in names) and any("splice" in n for n in names)


def test_a_bank_that_misses_a_fact_warns_and_names_it() -> None:
    h = dict(HEALTH, demoAttackBank=True)
    bank = {"enabled": True, "clips": [], "coveredFacts": [], "yieldRate": None, "problems": []}
    got = by_name(run_checks(good({"/api/health": h, "/api/clone-bank": bank}), presenter="S04"))
    assert got["clone bank covers the presenter's facts"].level == WARN
    assert "hometown" in got["clone bank covers the presenter's facts"].detail


def auth(decision, branches, ms=5000):
    return {"decision": decision, "fusedScore": 0.6, "latencyMs": ms, "branches": branches}


def b(name, passed, score=0.9):
    return {"name": name, "score": score, "threshold": 0.5, "weight": 0.3, "passed": passed}


def flows(**auth_by_kind) -> Fake:
    routes = {
        "/api/speakers/spk_p/utterances": [{"audioUrl": "/api/audio/u1", "durationSec": 30.0}],
        "/api/speakers/spk_a/utterances": [{"audioUrl": "/api/audio/u2", "durationSec": 30.0}],
        ("POST", "/api/challenge"): {"id": "chal_1"},
        ("AUTH", "standin"): auth("ACCEPT", [b("signal_integrity", True), b("speaker_embedding", True)]),
        ("AUTH", "replay"): auth("REJECT", [b("signal_integrity", False, 0.0)]),
        ("AUTH", "impostor"): auth(
            "REJECT", [b("signal_integrity", True), b("speaker_embedding", False, 0.1)]
        ),
    }
    for kind, value in auth_by_kind.items():
        routes[("AUTH", kind)] = value
    return good(routes)


def test_the_three_flows_pass_when_each_behaves() -> None:
    got = run_checks(flows(), presenter="S04", flows=True)
    flow = [c for c in got if c.name.startswith("flow:")]
    assert len(flow) == 3 and all(c.level == PASS for c in flow), [(c.name, c.detail) for c in flow]


def test_a_replay_that_is_accepted_fails() -> None:
    bad = auth("ACCEPT", [b("signal_integrity", True)])
    got = by_name(run_checks(flows(replay=bad), presenter="S04", flows=True))
    assert got["flow: replay is rejected at the integrity gate"].level == FAIL


def test_an_impostor_rejected_by_the_integrity_gate_instead_of_the_voice_fails() -> None:
    """The demo wants the voiceprint to decide; the gate tripping first is the
    bug the splice tests caused."""
    bad = auth("REJECT", [b("signal_integrity", False, 0.0)])
    got = by_name(run_checks(flows(impostor=bad), presenter="S04", flows=True))
    assert got["flow: impostor is rejected by the voiceprint"].level == FAIL


def test_a_genuine_stand_in_whose_voice_fails_is_a_failure() -> None:
    bad = auth("REJECT", [b("signal_integrity", True), b("speaker_embedding", False)])
    got = by_name(run_checks(flows(standin=bad), presenter="S04", flows=True))
    assert got["flow: genuine stand-in passes the voiceprint"].level == FAIL


def test_a_stand_in_the_integrity_gate_rejects_is_a_failure() -> None:
    bad = auth("REJECT", [b("signal_integrity", False, 0.0)])
    got = by_name(run_checks(flows(standin=bad), presenter="S04", flows=True))
    assert got["flow: genuine stand-in passes the voiceprint"].level == FAIL


def test_a_borderline_stand_in_with_a_passing_voice_only_warns() -> None:
    """A stand-in is a cut of old audio and cannot answer the random challenge,
    so its knowledge branch scores low. BORDERLINE with the voiceprint passing is
    expected, not a defect -- but it must say the live answer needs rehearsing."""
    borderline = auth(
        "BORDERLINE", [b("signal_integrity", True), b("speaker_embedding", True), b("knowledge", False, 0.3)]
    )
    got = by_name(run_checks(flows(standin=borderline), presenter="S04", flows=True))
    c = got["flow: genuine stand-in passes the voiceprint"]
    assert c.level == WARN and "live" in c.fix, c


def test_a_slow_login_warns() -> None:
    slow = auth("ACCEPT", [b("signal_integrity", True)], ms=60000)
    got = run_checks(flows(standin=slow), presenter="S04", flows=True)
    assert any(c.level == WARN and "slow" in c.name for c in got)


def test_render_ends_in_a_verdict() -> None:
    ok = render(run_checks(good(), presenter="S04"))
    assert ok.strip().splitlines()[-1].startswith("READY")
    bad = render(run_checks(Fake({"/api/health": OSError("x")}), presenter="S04"))
    assert bad.strip().splitlines()[-1].startswith("NOT READY")


def test_an_empty_bank_never_reads_as_covered_even_when_there_are_no_facts() -> None:
    """Vacuous truth: with no facts to cover, 'nothing is missing' used to read
    as 'all covered', on a bank that did not exist."""
    h = dict(HEALTH, demoAttackBank=True)
    bank = {"enabled": True, "clips": [], "coveredFacts": [], "yieldRate": None, "problems": ["none yet"]}
    got = by_name(
        run_checks(
            good({"/api/health": h, "/api/clone-bank": bank, "/api/speakers/spk_p/skg": []}),
            presenter="S04",
        )
    )
    check = got["clone bank covers the presenter's facts"]
    assert check.level == WARN and "no usable clips" in check.detail, check
