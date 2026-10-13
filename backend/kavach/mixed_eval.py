"""End-to-end trials of the experimental Tamil-English login, on real recordings, with real models.

    python -m kavach.mixed_eval --speaker S04 --english                 # English-only regression
    python -m kavach.mixed_eval --speaker S04 --genuine S6,S7           # genuine mixed clips, + replay trials from them
    python -m kavach.mixed_eval --speaker S04 --splice --splice-trials 20   # collage built from the speaker's own words

Every trial goes through `Pipeline._verify`: liveness, integrity, recognition, the words check, the voice gate, the
fusion rule. Nothing is faked and nothing is relaxed. For each trial the record keeps THE VOICE SCORE and THE
CHALLENGE-MATCHING RESULT AS SEPARATE FIELDS, and the stage a refusal happened at, so "the voice was fine, the Tamil
word was not heard" can be told apart from the reverse.

What the categories are, and what each can and cannot show:

genuine_mixed     Studio clips of kind 'mixed' read by the enrolled speaker. This is the end-to-end test. A clip from a
                  session the voiceprint was built from is flagged: its voice score is not independent evidence.
replay_*          Derived from the genuine clips: the same recording answering a different fresh challenge, and
                  answering the same challenge twice.
splice_*          A challenge assembled from the speaker's own recorded words (English words cut from 'words' clips,
                  Tamil words from 'tamil' clips): the attack the Tamil word dataset makes possible if it leaks.
                  Run with the splice detector off (the demo's setting) and on.
impostor          Another person reading a mixed challenge: needs recordings that do not exist yet (`--impostors`).
english_regress.  The English-only login on the speaker's 'words' clips; the baseline must not move.

The work is read-only on the live database: it runs on a copy, and the throttle is off so one trial's refusal cannot
slow the next. Results go to data/tamil_words/eval/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence

from .audio import Audio, load_audio, save_wav, trim_silence
from .challenge import Challenge
from .csbg.ontology import SemanticClass
from .mixed_phrase import new_mixed_phrase
from .phrase import _same_word
from .tamil_words import BANK, by_id, is_tamil_token


def parse_phrase(text_hint: str) -> list[str]:
    """'Say: a, b, c' (the Studio's text for a words or mixed clip) -> [a, b, c]."""
    return [t.strip() for t in text_hint.split(":", 1)[-1].split(",") if t.strip()]


def failure_stage(out: Any) -> str:
    """Where the first refusal happened: liveness, integrity, words:<stage>, voice, or '' if accepted."""
    decision = out.fusion.decision.value
    if decision == "ACCEPT":
        return ""
    by = {b.branch.value: b for b in out.fusion.branches}
    if not by["liveness"].passed:
        return "liveness"
    integ = by.get("signal_integrity")
    if integ is not None and integ.available and not integ.passed:
        return "integrity"
    phrase = by.get("phrase")
    if phrase is not None and not phrase.passed:
        return "words:" + (getattr(out.phrase, "failure_stage", "") or ("unavailable" if not phrase.available else "phrase"))
    voice = by.get("speaker_embedding")
    if voice is None or not voice.available:
        return "voice_unmeasured"
    return "voice_borderline" if decision == "BORDERLINE" else "voice"


def trial_record(category: str, label: str, kind: str, out: Any, **extra: Any) -> dict[str, Any]:
    by = {b.branch.value: b for b in out.fusion.branches}
    voice, phrase, integ = by.get("speaker_embedding"), by.get("phrase"), by.get("signal_integrity")
    match = out.phrase
    return {
        "category": category, "label": label, "kind": kind,
        "decision": out.fusion.decision.value, "failure_stage": failure_stage(out),
        # Two separate facts. Never merged.
        "voice": {"score": round(voice.score, 4) if voice is not None and voice.available else None,
                  "threshold": round(voice.threshold, 4) if voice is not None else None,
                  "passed": bool(voice.passed) if voice is not None else None},
        "challenge_match": (match.to_log() if hasattr(match, "to_log") else
                            ({"score": round(phrase.score, 3), "passed": bool(phrase.passed)} if phrase is not None else None)),
        "integrity": {"score": round(integ.score, 3), "passed": bool(integ.passed)} if integ is not None and integ.available else None,
        "latency_ms": out.latency_ms,
        **extra,
    }


def submit(pipe: Any, speaker_id: str, tokens: Sequence[str], audio: bytes | Audio, *, kind: str, tmp: Path) -> Any:
    """Issue a challenge with exactly these words, answer it with this recording, return the outcome."""
    if isinstance(audio, Audio):
        path = tmp / f"probe_{time.time_ns()}.wav"
        save_wav(audio, path)
        audio = path.read_bytes()
    now = time.time()
    ch = Challenge(
        id=f"chg_eval{time.time_ns()}", speaker_id=speaker_id, question_text="", target_class=SemanticClass.OTHER,
        expected_predicate="", expected_answer="", issued_at=now, expires_at=now + 120, generator="eval",
        kind=kind, phrase=tuple(tokens),
    )
    pipe.ledger.record(ch)
    out = pipe._verify(ch.id, audio, extension=".wav")
    out.challenge_id = ch.id
    return out


def make_pipeline(cfg: Any, tmp: Path) -> Any:
    """A Pipeline on a COPY of the live database, throttle off, mixed login on. Models load lazily and are real."""
    from .api.pipeline import Pipeline
    from .api.store import Store

    copy = tmp / "kavach_eval.db"
    src, dst = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True), sqlite3.connect(str(copy))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    store = Store(copy, cfg.audio_dir)
    return Pipeline(store, cfg.model_copy(update={"throttle_enabled": False, "mixed_enabled": True, "db_path": copy}))


def speaker_of(pipe: Any, pseudonym: str) -> tuple[str, set[str]]:
    from .studio.enrol import _speaker_id

    sid = _speaker_id(pipe.store, pseudonym)
    payload = pipe.store.load_template(sid) or {}
    return sid, set((payload.get("provenance") or {}).get("sessions") or [])


# --------------------------------------------------------------------------- categories


def english_regression(pipe, sid, studio, enrolled, tmp, limit) -> list[dict]:
    rows = []
    for c in [c for c in studio.clips() if c.kind == "words"][:limit]:
        out = submit(pipe, sid, parse_phrase(c.text_hint), studio.verified_wav_path(c).read_bytes(), kind="phrase", tmp=tmp)
        rows.append(trial_record("english_regression", c.clip_id, "phrase", out, session=c.session_id, enrolment_session=c.session_id in enrolled))
    return rows


def genuine_and_replay(pipe, sid, studio, sessions, enrolled, tmp, seed) -> list[dict]:
    rows, rng = [], random.Random(seed)
    clips = [c for c in studio.clips() if c.kind == "mixed" and c.session_id in sessions]
    for c in clips:
        wav = studio.verified_wav_path(c).read_bytes()
        tokens = parse_phrase(c.text_hint)
        flag = dict(session=c.session_id, enrolment_session=c.session_id in enrolled)
        out = submit(pipe, sid, tokens, wav, kind="mixed", tmp=tmp)
        rows.append(trial_record("genuine_mixed", c.clip_id, "mixed", out, **flag))
        # The same recording answering a DIFFERENT fresh challenge: what a replayed login looks like.
        other, _ = new_mixed_phrase(rng=rng)
        rows.append(trial_record("replay_other_challenge", c.clip_id, "mixed", submit(pipe, sid, other, wav, kind="mixed", tmp=tmp), **flag))
        # The same challenge answered twice: the second use of a spent challenge.
        first = submit(pipe, sid, tokens, wav, kind="mixed", tmp=tmp)
        again = pipe._verify(first.challenge_id, wav, extension=".wav")
        rows.append(trial_record("replay_spent_challenge", c.clip_id, "mixed", again, **flag))
    return rows


def word_inventory(pipe, studio) -> tuple[dict[str, list[Audio]], dict[str, list[Audio]]]:
    """English words cut from the speaker's 'words' clips (by recogniser timestamps) and whole Tamil word clips."""
    asr = pipe.asr
    english: dict[str, list[Audio]] = defaultdict(list)
    for c in [c for c in studio.clips() if c.kind == "words"]:
        audio = load_audio(studio.verified_wav_path(c))
        want = [w.lower() for w in parse_phrase(c.text_hint)]
        t = asr.transcribe(audio, language="en", initial_prompt="", fast=True)
        import re

        heard = [(m, w) for w in t.words for m in re.findall(r"[a-z]+", w.text.lower())]
        pos = 0
        for w in want:
            hit = next((i for i in range(pos, len(heard)) if _same_word(w, heard[i][0])), None)
            if hit is None:
                continue
            pos = hit + 1
            a, b = heard[hit][1].start_ms / 1000 - 0.03, heard[hit][1].end_ms / 1000 + 0.03
            english[w].append(audio.slice_seconds(max(0.0, a), min(audio.duration_sec, b)))
    tamil: dict[str, list[Audio]] = defaultdict(list)
    for c in [c for c in studio.clips() if c.kind == "tamil"]:
        if c.prompt_id in by_id():
            tamil[by_id()[c.prompt_id].ta].append(trim_silence(load_audio(studio.verified_wav_path(c)), pad_ms=40))
    return english, tamil


def splice_trials(pipe, sid, studio, tmp, n, seed) -> list[dict]:
    from .attacks.splice import SpliceConfig, splice_segments

    english, tamil = word_inventory(pipe, studio)
    rows, rng = [], random.Random(seed)
    if not tamil:
        return [{"category": "splice_collage", "skipped": "no 'tamil' word clips recorded yet: the collage needs the Tamil words on tape"}]
    pool = [w for w in BANK if w.ta in tamil]  # the Tamil words the attacker has on tape
    if len(pool) < 2:
        return [{"category": "splice_collage", "skipped": "fewer than two distinct Tamil words recorded"}]
    done = skipped = 0
    while done < n and skipped < 20 * n:
        tokens, _ = new_mixed_phrase(rng=rng, pool=pool)
        if any(not is_tamil_token(t) and not english.get(t) for t in tokens):
            skipped += 1  # the attacker lacks a recording of one of the English words
            continue
        segs = [rng.choice(tamil[t] if is_tamil_token(t) else english[t]) for t in tokens]
        for name, cfg in (("naive", SpliceConfig.naive()), ("careful", SpliceConfig.careful())):
            audio = splice_segments(segs, cfg)
            for splice_on in (False, True):
                pipe.integrity.check_splice = splice_on
                out = submit(pipe, sid, tokens, audio, kind="mixed", tmp=tmp)
                rows.append(trial_record(f"splice_collage_{name}", f"t{done}", "mixed", out, splice_detector="on" if splice_on else "off"))
        done += 1
    pipe.integrity.check_splice = pipe.settings.integrity_check_splice
    if skipped:
        rows.append({"category": "splice_collage", "note": f"{skipped} candidate challenges skipped: the attacker's inventory lacked a word"})
    return rows


def impostor_trials(pipe, sid, path, tmp) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out = submit(pipe, sid, d["phrase"], Path(d["path"]).read_bytes(), kind="mixed", tmp=tmp)
            rows.append(trial_record("impostor", d.get("label", Path(d["path"]).name), "mixed", out))
    return rows


# --------------------------------------------------------------------------- report


def summarise(rows: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if "decision" in r:
            by[r["category"]].append(r)
    lines = []
    for cat, rs in by.items():
        acc = sum(r["decision"] == "ACCEPT" for r in rs)
        stages = Counter(r["failure_stage"] or "accepted" for r in rs)
        v = [r["voice"]["score"] for r in rs if r["voice"]["score"] is not None]
        lines.append(f"{cat:28} n={len(rs):3}  accepted {acc:3}  stages {dict(stages)}")
        if v:
            lines.append(f"{'':28} voice score min/median/max {min(v):.3f}/{median(v):.3f}/{max(v):.3f}  (threshold {rs[0]['voice']['threshold']})")
        held = [r for r in rs if r.get("enrolment_session") is True]
        if held:
            lines.append(f"{'':28} {len(held)} of these clips come from a session the voiceprint was built from: their voice scores are not independent evidence")
        sessions = {r["session"] for r in rs if r.get("session")}
        if sessions:
            lines.append(f"{'':28} sessions: {sorted(sessions)}")
    for r in rows:
        if "decision" not in r:
            lines.append(f"NOTE {r.get('category', '')}: {r.get('skipped') or r.get('note')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .config import Settings
    from .studio.store import StudioStore

    p = argparse.ArgumentParser(prog="python -m kavach.mixed_eval", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", default="S04")
    p.add_argument("--english", action="store_true", help="English-only login on the speaker's 'words' clips")
    p.add_argument("--genuine", default="", help="comma list of Studio sessions holding 'mixed' clips")
    p.add_argument("--splice", action="store_true")
    p.add_argument("--splice-trials", type=int, default=10)
    p.add_argument("--impostors", type=Path, default=None, help="jsonl of {path, phrase, label}: other people reading mixed phrases")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not (args.english or args.genuine or args.splice or args.impostors):
        p.error("choose at least one of --english --genuine --splice --impostors")

    cfg = Settings()
    studio = StudioStore(cfg.data_dir / "studio" / args.speaker, args.speaker)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        pipe = make_pipeline(cfg, tmp)
        sid, enrolled = speaker_of(pipe, args.speaker)
        try:
            if args.english:
                rows += english_regression(pipe, sid, studio, enrolled, tmp, args.limit)
            if args.genuine:
                rows += genuine_and_replay(pipe, sid, studio, set(args.genuine.split(",")), enrolled, tmp, args.seed)
            if args.splice:
                rows += splice_trials(pipe, sid, studio, tmp, args.splice_trials, args.seed)
            if args.impostors:
                rows += impostor_trials(pipe, sid, args.impostors, tmp)
        finally:
            pipe.store.close()
    out_dir = cfg.data_dir / "tamil_words" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"trials_{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"asr={pipe.settings.mixed_whisper_model or pipe.settings.whisper_model}  voice threshold={pipe.voice_threshold}  splice detector default="
          f"{pipe.settings.integrity_check_splice}  enrolment sessions={sorted(enrolled)}")
    print(summarise(rows))
    print(f"trials: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
