"""Preflight for the live demo: is everything the demo needs actually there?

    PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.demo_check --presenter S04 --flows

Run it with the backend up, 30 minutes before going on stage. It checks what the
demo depends on and names the fix for anything missing, so a problem is found
at a desk and not in front of an audience. With `--flows` it also drives the
demo's logins end to end from stored audio -- a genuine stand-in, a replay, an
impostor and (when the bank is on) a clone -- and checks each does what the demo
script says it does.

It reads the backend over HTTP only. It never writes to the database, except
that each login it submits leaves a row in the auth history, as any login does.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

#: Milliseconds above which a login is called slow. A cold first login took 31 s
#: on this laptop; past this an audience reads it as a hang.
SLOW_LOGIN_MS = 45_000

#: Other speakers needed for the CSBG's background model (api.pipeline.MIN_COHORT).
MIN_OTHERS = 3

#: Staged clips must not read as replays. The replay detector's envelope
#: similarity is the peak cross-correlation normalised by the full energies,
#: which for a cut of a stored clip is about sqrt(cut / clip); it flags at 0.85,
#: i.e. a cut over ~72% of the clip. Cutting at most 40% scores ~0.63. Found by
#: running the preflight: a 20 s cut of a 24 s clip tripped the integrity gate
#: before the voiceprint ran. `tests/test_demo_staging.py` pins these to the
#: detector, and the UI's `stagingCut` (lib/audioClip.ts) mirrors them.
STAGING_MAX_SHARE = 0.4
STAGING_MIN_SOURCE_SEC = 20.0
STAGING_MAX_CUT_SEC = 12.0


def cut_plan(duration_sec: float) -> tuple[float, float] | None:
    """(start, length) in seconds for a staged clip cut from a stored one, or
    None when the stored clip is too short to cut safely."""
    if duration_sec < STAGING_MIN_SOURCE_SEC:
        return None
    length = min(STAGING_MAX_CUT_SEC, STAGING_MAX_SHARE * duration_sec)
    return min(2.0, duration_sec - length), length


@dataclass(slots=True)
class Check:
    name: str
    level: str
    detail: str
    fix: str = ""


class TransportError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class Transport(Protocol):
    def get(self, path: str) -> Any: ...

    def get_bytes(self, path: str) -> bytes: ...

    def post_json(self, path: str, body: dict[str, Any]) -> Any: ...

    def post_audio(self, challenge_id: str, data: bytes, filename: str) -> Any: ...


class HttpTransport:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def _open(self, req: urllib.request.Request, timeout: float = 120.0) -> bytes:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail", exc.reason)
            except Exception:  # noqa: BLE001
                detail = exc.reason
            raise TransportError(exc.code, str(detail)) from exc

    def get(self, path: str) -> Any:
        return json.loads(self._open(urllib.request.Request(self.base + path), 60))

    def get_bytes(self, path: str) -> bytes:
        return self._open(urllib.request.Request(self.base + path), 60)

    def post_json(self, path: str, body: dict[str, Any]) -> Any:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return json.loads(self._open(req))

    def post_audio(self, challenge_id: str, data: bytes, filename: str) -> Any:
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f'--{boundary}\r\nContent-Disposition: form-data; name="challengeId"\r\n\r\n{challenge_id}\r\n'.encode(),
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
                "Content-Type: audio/wav\r\n\r\n".encode(),
                data,
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(
            self.base + "/api/authenticate",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return json.loads(self._open(req, 600))


def _find(speakers: list[dict[str, Any]], pseudonym: str) -> dict[str, Any] | None:
    for s in speakers:
        name = s.get("displayName", "")
        if name == pseudonym or name.startswith(pseudonym + " "):
            return s
    return None


def _static_checks(
    t: Transport, presenter: str
) -> tuple[list[Check], dict[str, Any] | None, list[str]]:
    out: list[Check] = []
    try:
        health = t.get("/api/health")
    except (TransportError, OSError) as exc:
        out.append(
            Check(
                "backend is reachable",
                FAIL,
                str(exc),
                "start it: powershell -ExecutionPolicy Bypass -File .\\run_demo.ps1",
            )
        )
        return out, None, []

    status = health.get("status")
    out.append(
        Check(
            "backend reports connected",
            PASS if status == "connected" else FAIL,
            f"status {status!r}",
            "" if status == "connected" else "a model failed to load: see the backend console and /api/health",
        )
    )
    models = " ".join(health.get("models", [])).lower()
    for key, label, level in (
        ("whisper", "speech recognition (Whisper)", FAIL),
        ("speechbrain", "voiceprint model (ECAPA)", FAIL),
        ("labse", "semantic matcher (LaBSE)", WARN),
    ):
        ok = key in models
        out.append(
            Check(
                f"{label} is loaded",
                PASS if ok else level,
                "loaded" if ok else "not loaded",
                ""
                if ok
                else (
                    "the knowledge branch will use its string matchers only; "
                    "run_demo.ps1 -Prefetch once to download it"
                    if key == "labse"
                    else "pip install -r requirements.txt, then restart"
                ),
            )
        )
    llm_ok = any(k in models for k in ("gemini", "groq", "anthropic", "ollama"))
    out.append(
        Check(
            "a tagging model is configured",
            PASS if llm_ok else WARN,
            "configured" if llm_ok else "none",
            "" if llm_ok else "set GEMINI_API_KEY; without it logins fall back to the offline lexicon",
        )
    )
    if health.get("demoRevealAnswers"):
        out.append(
            Check(
                "challenge answers are hidden",
                WARN,
                "demoRevealAnswers is ON",
                "unset KAVACH_DEMO_REVEAL_ANSWERS: anyone can read the answer in the network tab",
            )
        )
    if health.get("reportable", {}).get("integrity_check_splice"):
        out.append(
            Check(
                "splice detection is off",
                WARN,
                "it is ON",
                "it rejects genuine phone audio; leave KAVACH_INTEGRITY_CHECK_SPLICE unset",
            )
        )

    speakers = t.get("/api/speakers")
    me = _find(speakers, presenter)
    if me is None:
        out.append(
            Check(
                "presenter exists",
                FAIL,
                f"no speaker {presenter!r}",
                "run_demo.ps1 -Seed rebuilds the 12-speaker demo database",
            )
        )
        return out, health, []
    out.append(Check("presenter exists", PASS, presenter))

    minutes = float(me.get("totalDurationSec", 0.0)) / 60.0
    out.append(
        Check(
            "presenter has enrolment audio",
            PASS if minutes >= 5 else WARN,
            f"{minutes:.1f} min",
            "" if minutes >= 5 else "add clips with the microphone top-up in the Speakers drawer",
        )
    )

    facts = t.get(f"/api/speakers/{me['id']}/skg")
    out.append(
        Check(
            "presenter has knowledge-graph facts",
            PASS if facts else FAIL,
            f"{len(facts)} fact(s)",
            ""
            if facts
            else "open Speakers, choose the presenter, add facts (hometown, college, ...): "
            "without one a login cannot be challenged",
        )
    )
    try:
        t.get(f"/api/speakers/{me['id']}/csbg")
        out.append(Check("presenter has a code-switch graph", PASS, "present"))
    except TransportError as exc:
        out.append(
            Check("presenter has a code-switch graph", FAIL, str(exc), "rebuild the presenter in Speakers")
        )

    others = [s for s in speakers if s["id"] != me["id"]]
    out.append(
        Check(
            "enough other speakers for the background model",
            PASS if len(others) >= MIN_OTHERS else FAIL,
            f"{len(others)} other speaker(s)",
            "" if len(others) >= MIN_OTHERS else "run_demo.ps1 -Seed",
        )
    )

    predicates = [f["predicate"] for f in facts]
    if health.get("demoAttackBank"):
        try:
            bank = t.get("/api/clone-bank")
        except TransportError as exc:
            out.append(
                Check(
                    "clone bank is usable",
                    FAIL,
                    str(exc),
                    "regenerate or re-annotate the bank (see DEMO_RUNBOOK.md)",
                )
            )
        else:
            covered = set(bank.get("coveredFacts", []))
            missing = [p for p in predicates if p not in covered]
            if not covered:
                # Never read an empty bank as covered: with no facts to cover,
                # "nothing is missing" would otherwise be vacuously true.
                detail = "the bank has no usable clips"
                if missing:
                    detail += "; no cloned answer for: " + ", ".join(missing)
                out.append(
                    Check(
                        "clone bank covers the presenter's facts",
                        WARN,
                        detail,
                        "generate and annotate the bank (see DEMO_RUNBOOK.md)",
                    )
                )
            else:
                out.append(
                    Check(
                        "clone bank covers the presenter's facts",
                        PASS if not missing else WARN,
                        "all covered" if not missing else "no cloned answer for: " + ", ".join(missing),
                        ""
                        if not missing
                        else "the Clone attack button will say so for those questions; "
                        "record those answers and regenerate",
                    )
                )
    else:
        out.append(
            Check(
                "clone bank is enabled",
                WARN,
                "the Clone attack button will not appear",
                "set KAVACH_DEMO_ATTACK_BANK=true and KAVACH_CLONE_VICTIMS in run_demo.ps1",
            )
        )
    return out, health, predicates


def _cut_wav(raw: bytes, start: float, seconds: float) -> bytes:
    from .audio import decode_bytes, save_wav

    audio = decode_bytes(raw, suffix=".wav")
    dur = len(audio.samples) / audio.sample_rate
    start = min(start, max(0.0, dur - 1.0))
    segment = audio.slice_seconds(start, min(start + seconds, dur))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cut.wav"
        save_wav(segment, path)
        return path.read_bytes()


def _branch(result: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((b for b in result.get("branches", []) if b["name"] == name), None)


def _staging_clip(t: Transport, speaker_id: str) -> tuple[str, float] | None:
    clips = [
        u
        for u in t.get(f"/api/speakers/{speaker_id}/utterances")
        if u["durationSec"] >= STAGING_MIN_SOURCE_SEC
    ]
    return (clips[0]["audioUrl"], float(clips[0]["durationSec"])) if clips else None


def _flow_checks(
    t: Transport, speakers: list[dict[str, Any]], me: dict[str, Any], health: dict[str, Any]
) -> list[Check]:
    out: list[Check] = []
    own_clip = _staging_clip(t, me["id"])
    others = [s for s in speakers if s["id"] != me["id"]]
    other_clip = next((c for c in (_staging_clip(t, s["id"]) for s in others) if c), None)
    if own_clip is None or other_clip is None:
        return [
            Check(
                "flows can be staged",
                FAIL,
                f"no stored clip of {STAGING_MIN_SOURCE_SEC:.0f} s or more",
                "enrol longer clips: a staged cut must stay well under the replay threshold",
            )
        ]
    own, other = t.get_bytes(own_clip[0]), t.get_bytes(other_clip[0])
    own_start, own_len = cut_plan(own_clip[1])
    other_start, other_len = cut_plan(other_clip[1])

    def login(wav: bytes, name: str) -> dict[str, Any]:
        cid = t.post_json("/api/challenge", {"speakerId": me["id"]})["id"]
        return t.post_audio(cid, wav, name)

    results: dict[str, dict[str, Any]] = {}
    for kind, wav in (
        ("standin", _cut_wav(own, own_start, own_len)),
        ("replay", own),
        ("impostor", _cut_wav(other, other_start, other_len)),
    ):
        results[kind] = login(wav, f"{kind}.wav")

    r = results["standin"]
    voice, gate = _branch(r, "speaker_embedding"), _branch(r, "signal_integrity")
    gate_ok = gate is None or gate["passed"]
    voice_ok = voice is not None and voice["passed"]
    if not gate_ok:
        level, fix = FAIL, "the integrity gate rejected a genuine clip: check splice detection is off"
    elif not voice_ok:
        level, fix = FAIL, "the voiceprint did not recognise the presenter: add microphone clips or re-enrol"
    elif r["decision"] == "ACCEPT":
        level, fix = PASS, ""
    else:
        # A stand-in is a cut of old audio: it cannot answer the random
        # challenge, so the knowledge branch scores low and the verdict can
        # legitimately be BORDERLINE. What matters is that the voiceprint
        # passed. The live spoken answer scores higher and must be rehearsed.
        level = WARN
        fix = (
            "a stand-in cannot answer the challenge, so its knowledge score is low; "
            "rehearse the live spoken answer once on the demo laptop"
        )
    out.append(
        Check(
            "flow: genuine stand-in passes the voiceprint",
            level,
            f"{r['decision']} at {r['fusedScore']:.3f}; voice "
            f"{'passed' if voice_ok else 'FAILED'}; integrity {'ok' if gate_ok else 'TRIPPED'}",
            fix,
        )
    )
    r = results["replay"]
    gate = _branch(r, "signal_integrity")
    ok = r["decision"] == "REJECT" and gate is not None and not gate["passed"]
    out.append(
        Check(
            "flow: replay is rejected at the integrity gate",
            PASS if ok else FAIL,
            f"{r['decision']}; integrity gate {'tripped' if gate and not gate['passed'] else 'did not trip'}",
            "" if ok else "the duplicate detector did not recognise a stored clip",
        )
    )
    r = results["impostor"]
    voice, gate = _branch(r, "speaker_embedding"), _branch(r, "signal_integrity")
    ok = (
        r["decision"] == "REJECT"
        and voice is not None
        and not voice["passed"]
        and (gate is None or gate["passed"])
    )
    out.append(
        Check(
            "flow: impostor is rejected by the voiceprint",
            PASS if ok else FAIL,
            f"{r['decision']}; voice {'failed' if voice and not voice['passed'] else 'ok'}; "
            f"integrity {'ok' if gate is None or gate['passed'] else 'TRIPPED'}",
            "" if ok else "the voiceprint must decide, not the integrity gate: check splice detection is off",
        )
    )

    if health.get("demoAttackBank"):
        matched = None
        for _ in range(8):  # challenges are random; keep asking until one has a clone
            cid = t.post_json("/api/challenge", {"speakerId": me["id"]})["id"]
            try:
                matched = (cid, t.post_json("/api/clone-bank/match", {"challengeId": cid}))
                break
            except TransportError:
                continue
        if matched is None:
            out.append(
                Check(
                    "flow: clone attack",
                    WARN,
                    "no issued challenge had a cloned answer",
                    "regenerate the bank to cover more of the presenter's facts",
                )
            )
        else:
            cid, m = matched
            res = t.post_audio(cid, t.get_bytes(m["audioUrl"]), "clone.wav")
            gate = _branch(res, "signal_integrity")
            ok = gate is None or gate["passed"]
            out.append(
                Check(
                    "flow: clone attack reaches the voiceprint",
                    PASS if ok else FAIL,
                    f"{res['decision']} at {res['fusedScore']:.3f} (an outcome, not a pass/fail)",
                    "" if ok else "the integrity gate rejected a clone before the voiceprint ran",
                )
            )
            results["clone"] = res

    for kind, res in results.items():
        ms = res.get("latencyMs") or 0
        if ms > SLOW_LOGIN_MS:
            out.append(
                Check(
                    f"slow login: {kind}",
                    WARN,
                    f"{ms / 1000:.0f} s",
                    "warm the models first (the first login is the slow one)",
                )
            )
    return out


def run_checks(t: Transport, *, presenter: str, flows: bool = False) -> list[Check]:
    checks, health, _ = _static_checks(t, presenter)
    if flows and health is not None and not any(c.level == FAIL for c in checks):
        speakers = t.get("/api/speakers")
        me = _find(speakers, presenter)
        if me is not None:
            checks.extend(_flow_checks(t, speakers, me, health))
    return checks


def render(checks: Sequence[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{c.level}] {c.name}: {c.detail}")
        if c.level != PASS and c.fix:
            lines.append(f"       fix: {c.fix}")
    fails = sum(c.level == FAIL for c in checks)
    warns = sum(c.level == WARN for c in checks)
    lines.append("")
    lines.append(
        f"NOT READY: {fails} failure(s), {warns} warning(s)"
        if fails
        else f"READY: {sum(c.level == PASS for c in checks)} passed, {warns} warning(s)"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m kavach.demo_check", description=__doc__.splitlines()[0]
    )
    p.add_argument("--presenter", default="S04")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--flows", action="store_true", help="Also drive the demo logins end to end.")
    args = p.parse_args(argv)
    checks = run_checks(HttpTransport(args.base), presenter=args.presenter, flows=args.flows)
    print(render(checks))
    return 1 if any(c.level == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
