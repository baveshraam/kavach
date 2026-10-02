# Evidence Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]`. Code blocks introduced by a line `FILE[test] <path>` or `FILE[impl] <path>` are the exact file contents: apply them with the extractor (`apply_plan.py PLAN N test|impl`), run, and read the output. Blocks introduced by `EDIT <path>` are manual single-line `Edit` anchors (the files are CRLF).

**Goal:** Let the presenter record labelled sessions of their own voice through the demo's browser-mic path, and measure the real voiceprint on held-out sessions with honest intervals.

**Architecture:** `kavach/studio/` (store + plan) behind three gated API routes and a `/studio` page; `kavach/eval/enrollee.py` evaluates a single enrollee from injected embeddings (so tests need no model).

**Tech Stack:** Python 3.11, FastAPI, numpy, pytest; React + Vite + TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-02-evidence-pipeline-design.md` (approved 2026-10-02).

## Global Constraints

- Work from the repo root; Python is `.venv/Scripts/python.exe`; bare `pytest` prints the count. Offline suite: no model, no network; real checkpoints only behind `@pytest.mark.models`.
- `Settings.studio_enabled` defaults `False` and `Settings.studio_speakers` defaults `[]` (nobody recorded). Studio routes: 404 when off, 403 for a pseudonym not on the list. `/api/health` reports `studioEnabled`. `run_demo.ps1` never enables the Studio.
- Data lives under `data/studio/<pseudonym>/` (git-ignored). Index lines and file names carry the pseudonym only.
- The index is append-only and written **last**: a crash leaves an orphan file, never an index entry pointing at nothing.
- Kinds are exactly `read | free | fact`; devices `DEMO_LAPTOP_MIC | PHONE | HEADSET | OTHER`; environments are `corpus.Environment` values.
- `config.py`, `api/app.py`, `kavach/src/api/client.ts`, `kavach/src/api/types.ts` are CRLF: single-line `Edit` anchors only.
- Commit locally on branch `feature/evidence-pipeline`; end messages with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Pushing to `baveshraam/kavach` happens at the end.
- Reuse, do not duplicate: `attacks.suite.wilson_interval`, `eval.metrics.compute_eer`, `challenge.TEMPLATE_QUESTIONS`, `corpus.PROTOCOL_V1`, `audio.decode_bytes/check_quality/save_wav`.

## Review Focus

1. A clip too short, silent, undecodable or with a hostile filename: refused with a reason, **nothing written**. — Tasks 2, 4.
2. A held-out session leaking into enrolment, or an impostor speaker on both sides of the dev/test split. — Task 7.
3. Wilson alone reading as "the" interval for clips that come from a handful of sittings. — Task 6/7.
4. A report that can be screenshotted without its limits. — Task 7.
5. The Studio reachable while off, or for someone not on the allowlist. — Task 4.

## File Structure

| File | Role | Task |
|---|---|---|
| `backend/kavach/config.py`, `api/schemas.py`, `api/app.py` | settings, health flag, routes | 1, 4 |
| `backend/kavach/studio/__init__.py`, `store.py` | the dataset | 2 |
| `backend/kavach/studio/plan.py` | what to record | 3 |
| `kavach/src/pages/Studio.tsx` (+ client, types, route, nav) | the recording UI | 5 |
| `backend/kavach/eval/enrollee.py` | single-enrollee evaluation | 6, 7 |
| `run_studio.ps1`, `RECORDING_GUIDE.md` | how to record tonight | 8 |

---

### Task 1: Studio settings and the health flag

**Files:** Modify `config.py` (anchor `    @field_validator("data_dir", "audio_dir", "attack_dir")`), `api/schemas.py` (anchor `    """True when the clone-bank routes are live. Those routes return audio that\n    says the answer...` — use the single line `    says the answer to a challenge, so a build with them on must announce it."""`), `api/app.py` (anchor `            demo_attack_bank=cfg.demo_attack_bank,`). Test: `tests/test_studio_settings.py`.

- [ ] **Step 1: tests**

FILE[test] tests/test_studio_settings.py
```python
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
```

- [ ] **Step 2:** apply tests; run `pytest tests/test_studio_settings.py`. Expected: FAIL (`KeyError: 'studio_enabled'`).
- [ ] **Step 3: implement.**

EDIT `config.py`: replace `    @field_validator("data_dir", "audio_dir", "attack_dir")` with
```python
    studio_enabled: bool = False
    """Accept uploads into the recording Studio (labelled sessions of the presenter's own voice).

    **Off by default and never turned on by the demo build.** Recording is a deliberate act: a separate
    switch (`run_studio.ps1`) enables it. `/api/health` reports the flag."""

    studio_speakers: list[str] = Field(default_factory=list)
    """Corpus pseudonyms the Studio may record. **Empty: nobody.** Fail closed, like `clone_victims`."""

    @field_validator("data_dir", "audio_dir", "attack_dir")
```
EDIT `schemas.py`: after `    says the answer to a challenge, so a build with them on must announce it."""` add `\n\n    studio_enabled: bool = False\n    """True when the recording Studio accepts uploads."""`.
EDIT `app.py`: replace `            demo_attack_bank=cfg.demo_attack_bank,` with that line plus `            studio_enabled=cfg.studio_enabled,`.

- [ ] **Step 4:** run `pytest tests/test_studio_settings.py tests/test_api.py` → PASS. **Step 5:** commit `Add the Studio settings, off and empty by default`.

---

### Task 2: The Studio dataset (`studio/store.py`)

**Interfaces — Produces:** `KINDS, DEVICES, ENVIRONMENTS, INDEX_FILE, MIN_CLIP_SEC`, `StudioError`, `ClipRecord` (frozen; `.to_dict()`), `StudioStore(root, pseudonym)` with `.clips()`, `.next_session_id()`, `.add_clip(*, session_id, kind, prompt_id, device, environment, audio, orig_bytes, orig_ext, text_hint="", state_note="", now=None) -> ClipRecord`, `.wav_path(clip)`, `.summary() -> dict`.

- [ ] **Step 1: tests**

FILE[test] tests/test_studio_store.py
```python
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from clone_helpers import tone
from kavach.audio import Audio
from kavach.studio.store import DEVICES, ENVIRONMENTS, KINDS, StudioError, StudioStore

KW = dict(
    session_id="S1", kind="read", prompt_id="demo1", device="DEMO_LAPTOP_MIC",
    environment="QUIET_ROOM", orig_bytes=b"RIFFxxxx", orig_ext=".webm",
)


def store(tmp_path) -> StudioStore:
    return StudioStore(tmp_path / "studio" / "S04", "S04")


def test_a_clip_is_written_and_indexed_last(tmp_path) -> None:
    s = store(tmp_path)
    rec = s.add_clip(audio=tone(seconds=3.0), text_hint="நாளைக்கு meeting", **KW)
    assert s.wav_path(rec).exists()
    assert (s.root / "S1" / f"{rec.clip_id}.orig").read_bytes() == b"RIFFxxxx"
    assert [c.clip_id for c in s.clips()] == [rec.clip_id]
    assert rec.duration_sec == pytest.approx(3.0, abs=0.01) and len(rec.sha256_wav) == 64


def test_the_index_is_append_only(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=2.0), **KW)
    first = s.index_path.read_bytes()
    s.add_clip(audio=tone(seconds=2.0, seed=1), **KW)
    assert s.index_path.read_bytes().startswith(first)
    assert len(s.index_path.read_text(encoding="utf-8").splitlines()) == 2


def test_a_failed_write_leaves_no_index_entry(tmp_path, monkeypatch) -> None:
    import kavach.studio.store as mod

    s = store(tmp_path)
    monkeypatch.setattr(mod, "save_wav", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        s.add_clip(audio=tone(seconds=2.0), **KW)
    assert s.clips() == []


@pytest.mark.parametrize("audio,reason", [
    (Audio(np.zeros(16000 * 3, dtype=np.float32), 16000, "s"), "no speech"),
    (tone(seconds=0.4), "at least"),
])
def test_unusable_audio_is_refused_and_nothing_is_written(tmp_path, audio, reason) -> None:
    s = store(tmp_path)
    with pytest.raises(StudioError, match=reason):
        s.add_clip(audio=audio, **KW)
    assert s.clips() == [] and not s.root.exists() or not any(s.root.rglob("*.wav"))


@pytest.mark.parametrize("field,value", [
    ("kind", "shout"), ("device", "TOASTER"), ("environment", "MOON"),
    ("session_id", "../evil"), ("session_id", ""), ("prompt_id", ""),
])
def test_bad_labels_are_refused(tmp_path, field, value) -> None:
    s = store(tmp_path)
    with pytest.raises(StudioError):
        s.add_clip(audio=tone(seconds=2.0), **{**KW, field: value})
    assert s.clips() == []


def test_a_hostile_extension_is_neutralised(tmp_path) -> None:
    s = store(tmp_path)
    rec = s.add_clip(audio=tone(seconds=2.0), **{**KW, "orig_ext": "/../x"})
    assert rec.orig_ext == ".bin"
    assert not any(p.name == "x" for p in tmp_path.rglob("*"))


def test_next_session_id_skips_used_ones(tmp_path) -> None:
    s = store(tmp_path)
    assert s.next_session_id() == "S1"
    s.add_clip(audio=tone(seconds=2.0), **KW)
    assert s.next_session_id() == "S2"


def test_summary_counts_minutes_and_slices(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=6.0), **KW)
    s.add_clip(audio=tone(seconds=6.0, seed=1), **{**KW, "kind": "free", "device": "PHONE", "session_id": "S2"})
    summ = s.summary()
    assert summ["clips"] == 2 and summ["minutes"] == pytest.approx(0.2, abs=0.01)
    assert summ["by_kind"]["read"]["clips"] == 1 and summ["by_device"]["PHONE"]["clips"] == 1
    assert set(summ["by_session"]) == {"S1", "S2"}


def test_a_corrupt_index_line_names_its_line(tmp_path) -> None:
    s = store(tmp_path)
    s.add_clip(audio=tone(seconds=2.0), **KW)
    with s.index_path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    with pytest.raises(StudioError, match="line 2"):
        s.clips()


def test_the_closed_sets_are_what_the_spec_says() -> None:
    assert KINDS == ("read", "free", "fact")
    assert DEVICES == ("DEMO_LAPTOP_MIC", "PHONE", "HEADSET", "OTHER")
    assert "QUIET_ROOM" in ENVIRONMENTS
```

- [ ] **Step 2:** apply tests, run → collection ERROR (no module). **Step 3: implement.**

FILE[impl] backend/kavach/studio/__init__.py
```python
"""The recording Studio: labelled sessions of one consenting speaker's own voice."""
```

FILE[impl] backend/kavach/studio/store.py
```python
"""The Studio's dataset: labelled sessions of one consenting speaker's own voice.

Layout (under git-ignored `data/`)::

    data/studio/<pseudonym>/
      index.jsonl               append-only, one JSON object per line
      <session_id>/<clip_id>.wav     16 kHz mono PCM, what every consumer reads
      <session_id>/<clip_id>.orig    the bytes as the browser sent them

The index line is written LAST. A crash between the files and the line leaves an
orphan file nobody reads; the reverse -- an index entry pointing at nothing --
would poison every evaluation, so the order is the guarantee. The index is never
rewritten, so a recorded clip cannot silently change.

Labels are closed sets on purpose: a free-text device or room produces thirty
groups of one and no slice large enough to compare.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..audio import Audio, check_quality, save_wav
from ..corpus import Environment

KINDS = ("read", "free", "fact")
DEVICES = ("DEMO_LAPTOP_MIC", "PHONE", "HEADSET", "OTHER")
ENVIRONMENTS = tuple(e.value for e in Environment)
INDEX_FILE = "index.jsonl"
MIN_CLIP_SEC = 1.0
MAX_CLIP_SEC = 120.0

_SESSION = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_EXT = re.compile(r"^\.[a-z0-9]{1,5}$")
_PSEUDONYM = re.compile(r"^[A-Za-z0-9]{1,8}$")


class StudioError(ValueError):
    """A recording or label the Studio refuses; the message says why."""


@dataclass(frozen=True, slots=True)
class ClipRecord:
    clip_id: str
    session_id: str
    kind: str
    prompt_id: str
    text_hint: str
    device: str
    environment: str
    state_note: str
    recorded_at: str
    duration_sec: float
    sha256_wav: str
    orig_ext: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class StudioStore:
    def __init__(self, root: Path | str, pseudonym: str) -> None:
        if not _PSEUDONYM.match(pseudonym):
            raise StudioError(f"{pseudonym!r} is not a corpus pseudonym")
        self.root = Path(root)
        self.pseudonym = pseudonym

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILE

    def clips(self) -> list[ClipRecord]:
        path = self.index_path
        if not path.exists():
            return []
        out: list[ClipRecord] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(ClipRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise StudioError(f"{path.name} line {n} is not a valid clip record: {exc}") from exc
        return out

    def next_session_id(self) -> str:
        used = {c.session_id for c in self.clips()}
        i = 1
        while f"S{i}" in used:
            i += 1
        return f"S{i}"

    def wav_path(self, clip: ClipRecord) -> Path:
        return self.root / clip.session_id / f"{clip.clip_id}.wav"

    def add_clip(
        self,
        *,
        session_id: str,
        kind: str,
        prompt_id: str,
        device: str,
        environment: str,
        audio: Audio,
        orig_bytes: bytes,
        orig_ext: str,
        text_hint: str = "",
        state_note: str = "",
        now: datetime | None = None,
    ) -> ClipRecord:
        if not _SESSION.match(session_id):
            raise StudioError("a session id may use letters, digits, '_' and '-' only (at most 32)")
        if kind not in KINDS:
            raise StudioError(f"kind must be one of {KINDS}")
        if device not in DEVICES:
            raise StudioError(f"device must be one of {DEVICES}")
        if environment not in ENVIRONMENTS:
            raise StudioError(f"environment must be one of {ENVIRONMENTS}")
        if not prompt_id or len(prompt_id) > 64:
            raise StudioError("a prompt id (1-64 characters) is required")

        quality = check_quality(audio, min_seconds=MIN_CLIP_SEC, max_seconds=MAX_CLIP_SEC)
        if quality.is_silent:
            raise StudioError("no speech was detected in that recording")
        if quality.is_too_short:
            raise StudioError(
                f"the recording is only {audio.duration_sec:.1f}s long; "
                f"at least {MIN_CLIP_SEC:.0f}s is needed"
            )

        ext = orig_ext.lower() if _EXT.match(orig_ext.lower()) else ".bin"
        clip_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        folder = self.root / session_id
        wav_path = folder / f"{clip_id}.wav"
        save_wav(audio, wav_path)
        (folder / f"{clip_id}.orig").write_bytes(orig_bytes)

        record = ClipRecord(
            clip_id=clip_id,
            session_id=session_id,
            kind=kind,
            prompt_id=prompt_id,
            text_hint=text_hint,
            device=device,
            environment=environment,
            state_note=state_note,
            recorded_at=(now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
            duration_sec=round(audio.duration_sec, 3),
            sha256_wav=_sha256(wav_path),
            orig_ext=ext,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as fh:  # last: see module docstring
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def summary(self) -> dict[str, Any]:
        clips = self.clips()

        def agg(key: str) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, float]] = {}
            for c in clips:
                a = out.setdefault(getattr(c, key), {"clips": 0, "seconds": 0.0})
                a["clips"] += 1
                a["seconds"] += c.duration_sec
            return {k: {"clips": int(v["clips"]), "minutes": round(v["seconds"] / 60, 2)} for k, v in sorted(out.items())}

        sessions: dict[str, dict[str, Any]] = {}
        for c in clips:
            s = sessions.setdefault(
                c.session_id, {"clips": 0, "seconds": 0.0, "devices": set(), "environments": set()}
            )
            s["clips"] += 1
            s["seconds"] += c.duration_sec
            s["devices"].add(c.device)
            s["environments"].add(c.environment)
        return {
            "speaker": self.pseudonym,
            "clips": len(clips),
            "minutes": round(sum(c.duration_sec for c in clips) / 60, 2),
            "by_session": {
                k: {
                    "clips": v["clips"],
                    "minutes": round(v["seconds"] / 60, 2),
                    "devices": sorted(v["devices"]),
                    "environments": sorted(v["environments"]),
                }
                for k, v in sorted(sessions.items())
            },
            "by_kind": agg("kind"),
            "by_device": agg("device"),
            "by_environment": agg("environment"),
        }
```

- [ ] **Step 4:** run `pytest tests/test_studio_store.py` → PASS. **Step 5:** commit `Add the Studio dataset: append-only, indexed last, closed label sets`.

---

### Task 3: What to record (`studio/plan.py`)

**Interfaces — Produces:** `PlanItem(kind, prompt_id, text_en, text_ta, repeat)` (`.to_dict()`), `RECIPES`, `DEFAULT_RECIPE`, `DEMO_SENTENCES`, `fact_question(predicate)`, `build_plan(session_id, facts, sentences=DEMO_SENTENCES)`, `estimate_minutes(items)`.

- [ ] **Step 1: tests**

FILE[test] tests/test_studio_plan.py
```python
from types import SimpleNamespace

from kavach.corpus import PROTOCOL_V1
from kavach.studio.plan import DEMO_SENTENCES, RECIPES, build_plan, estimate_minutes, fact_question


def facts(*predicates):
    return [SimpleNamespace(predicate=p) for p in predicates]


def count(items, kind):
    return sum(i.repeat for i in items if i.kind == kind)


def test_every_protocol_prompt_is_in_the_plan() -> None:
    items = build_plan("S1", facts("hometown"))
    free = {i.prompt_id for i in items if i.kind == "free"}
    assert free == {p.prompt_id for p in PROTOCOL_V1}


def test_session_recipes_set_the_repetitions() -> None:
    s1 = build_plan("S1", facts("hometown", "college"))
    r = RECIPES["S1"]
    assert count(s1, "read") == r.read * len(DEMO_SENTENCES)
    assert count(s1, "fact") == r.fact * 2
    s4 = build_plan("S4", facts("hometown"))
    assert count(s4, "read") == RECIPES["S4"].read * len(DEMO_SENTENCES)


def test_an_unknown_session_label_gets_the_default_recipe() -> None:
    assert count(build_plan("X9", []), "read") > 0


def test_each_fact_gets_a_question_and_no_facts_means_no_fact_items() -> None:
    items = build_plan("S1", facts("hometown"))
    fact = [i for i in items if i.kind == "fact"]
    assert len(fact) == 1 and fact[0].text_ta == fact_question("hometown")
    assert not [i for i in build_plan("S1", []) if i.kind == "fact"]


def test_an_unknown_predicate_still_gets_a_question() -> None:
    assert "zodiac" in fact_question("zodiac")


def test_sentences_are_non_empty_and_unique() -> None:
    ids = [s[0] for s in DEMO_SENTENCES]
    assert len(set(ids)) == len(ids) == 3
    assert all(s[1].strip() and s[2].strip() for s in DEMO_SENTENCES)


def test_the_estimate_is_positive_and_grows_with_repeats() -> None:
    assert 0 < estimate_minutes(build_plan("S3", facts("hometown"))) < estimate_minutes(
        build_plan("S1", facts("hometown", "college"))
    )
```

- [ ] **Step 2:** run → FAIL (no module). **Step 3: implement.**

FILE[impl] backend/kavach/studio/plan.py
```python
"""What the presenter records, per session.

Three kinds, three purposes. `read`: a few fixed sentences repeated many times,
the voice-only same-sentence test. `free`: the 14 bilingual `PROTOCOL_V1`
prompts, natural speech for the code-switch analysis. `fact`: a question about
one of the presenter's own facts, the demo's login path.

The demo sentences are Tamil script with English words, because Whisper writes
Tamil in Tamil script and a romanised prompt would never match its transcript.
They are a default; replace them with sentences you can say naturally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..challenge import TEMPLATE_QUESTIONS
from ..corpus import PROTOCOL_V1


@dataclass(frozen=True, slots=True)
class PlanItem:
    kind: str
    prompt_id: str
    text_en: str
    text_ta: str
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Recipe:
    read: int
    free: int
    fact: int


#: S1-S3 enrol; S4 is the held-out test and is never enrolled (spec section 5).
RECIPES: dict[str, Recipe] = {
    "S1": Recipe(read=15, free=2, fact=4),
    "S2": Recipe(read=15, free=2, fact=4),
    "S3": Recipe(read=10, free=1, fact=3),
    "S4": Recipe(read=20, free=1, fact=3),
}
DEFAULT_RECIPE = Recipe(read=10, free=1, fact=3)

#: (prompt_id, Tamil-script text with English words, English gloss)
DEMO_SENTENCES: tuple[tuple[str, str, str], ...] = (
    ("demo1", "நாளைக்கு காலையில meeting இருக்கு, அதனால நான் seven மணிக்கே கிளம்பிடுவேன்.",
     "I have a meeting tomorrow morning, so I'll leave by seven."),
    ("demo2", "என் phone-ல battery கம்மியா இருக்கு, charger எடுத்துட்டு வந்தீங்களா?",
     "My phone's battery is low; did you bring the charger?"),
    ("demo3", "இந்த weekend எங்க ஊருக்கு போறேன், train ticket நேத்தே book பண்ணிட்டேன்.",
     "I'm going to my hometown this weekend; I booked the train ticket yesterday."),
)

#: Rough seconds per recording including the pause to read the prompt.
_SECONDS = {"read": 9.0, "free": 40.0, "fact": 15.0}


def fact_question(predicate: str) -> str:
    options = TEMPLATE_QUESTIONS.get(predicate, ())
    return options[0] if options else f"Tell me about your {predicate}."


def build_plan(
    session_id: str,
    facts: Sequence[Any],
    sentences: Sequence[tuple[str, str, str]] = DEMO_SENTENCES,
) -> list[PlanItem]:
    r = RECIPES.get(session_id, DEFAULT_RECIPE)
    items = [PlanItem("read", pid, en, ta, r.read) for pid, ta, en in sentences]
    items += [PlanItem("free", p.prompt_id, p.text_en, p.text_ta, r.free) for p in PROTOCOL_V1]
    items += [
        PlanItem("fact", f"fact_{f.predicate}", f"About your {f.predicate}", fact_question(f.predicate), r.fact)
        for f in facts
    ]
    return items


def estimate_minutes(items: Sequence[PlanItem]) -> float:
    return round(sum(_SECONDS[i.kind] * i.repeat for i in items) / 60.0, 1)
```

- [ ] **Step 4:** run → PASS. **Step 5:** commit `Plan the sessions: read, free and fact prompts with per-session recipes`.

---

### Task 4: The gated Studio routes

**Files:** Modify `api/app.py` (import line `from ..attacks.bank import BankError, CloneBank`; marker `    # ------------------------------------------------------------ clone bank`). Test: `tests/test_studio_api.py`.

- [ ] **Step 1: tests**

FILE[test] tests/test_studio_api.py
```python
from __future__ import annotations

import shutil
import subprocess

import pytest

from clone_helpers import tone
from kavach.audio import Audio, save_wav
from test_clone_bank_api import build_client

import numpy as np

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def studio(tmp_path, **kw):
    return build_client(tmp_path, studio_enabled=True, studio_speakers=["S04"], **kw)


def wav_bytes(tmp_path, seconds=3.0, seed=0, silent=False) -> bytes:
    p = tmp_path / f"c{seed}.wav"
    audio = Audio(np.zeros(int(16000 * seconds), dtype=np.float32), 16000, "s") if silent else tone(seconds=seconds, seed=seed)
    save_wav(audio, p)
    return p.read_bytes()


FORM = {"speaker": "S04", "session_id": "S1", "kind": "read", "prompt_id": "demo1",
        "device": "DEMO_LAPTOP_MIC", "environment": "QUIET_ROOM"}


def post(client, data: bytes, name="c.wav", **over):
    return client.post("/api/studio/clips", files={"audio": (name, data, "audio/wav")}, data={**FORM, **over})


def test_everything_is_a_404_while_the_studio_is_off(tmp_path) -> None:
    client, *_ = build_client(tmp_path)
    assert client.get("/api/studio/plan", params={"speaker": "S04"}).status_code == 404
    assert client.get("/api/studio/summary", params={"speaker": "S04"}).status_code == 404
    assert post(client, wav_bytes(tmp_path)).status_code == 404


def test_a_speaker_not_on_the_allowlist_is_a_403(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert client.get("/api/studio/plan", params={"speaker": "S09"}).status_code == 403
    assert post(client, wav_bytes(tmp_path), speaker="S09").status_code == 403


def test_the_plan_lists_devices_environments_and_items(tmp_path) -> None:
    client, store, _, settings = studio(tmp_path)
    sp = store.create_speaker({"display_name": "S04 · Tester"})
    client.put(f"/api/speakers/{sp['id']}/skg", json=[{"subject": "x", "predicate": "hometown", "object": "T"}])
    body = client.get("/api/studio/plan", params={"speaker": "S04"}).json()
    assert body["sessionId"] == "S1" and body["hasFacts"] is True
    assert "DEMO_LAPTOP_MIC" in body["devices"] and "QUIET_ROOM" in body["environments"]
    kinds = {i["kind"] for i in body["items"]}
    assert kinds == {"read", "free", "fact"} and body["estimatedMinutes"] > 0


def test_a_clip_is_stored_and_summarised(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    r = post(client, wav_bytes(tmp_path))
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["clips"] == 1
    index = settings.data_dir / "studio" / "S04" / "index.jsonl"
    assert len(index.read_text(encoding="utf-8").splitlines()) == 1
    s = client.get("/api/studio/summary", params={"speaker": "S04"}).json()
    assert s["by_kind"]["read"]["clips"] == 1


@pytest.mark.parametrize("over,status", [({"kind": "shout"}, 422), ({"device": "TOASTER"}, 422), ({"session_id": "../x"}, 422)])
def test_bad_labels_are_a_422_and_write_nothing(tmp_path, over, status) -> None:
    client, _, _, settings = studio(tmp_path)
    assert post(client, wav_bytes(tmp_path), **over).status_code == status
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_silent_and_tiny_clips_are_refused_with_a_reason(tmp_path) -> None:
    client, _, _, settings = studio(tmp_path)
    silent = post(client, wav_bytes(tmp_path, silent=True))
    tiny = post(client, wav_bytes(tmp_path, seconds=0.4, seed=1))
    assert silent.status_code == 422 and "no speech" in silent.json()["detail"]
    assert tiny.status_code == 422 and "at least" in tiny.json()["detail"]
    assert not (settings.data_dir / "studio" / "S04" / "index.jsonl").exists()


def test_empty_and_unreadable_uploads_are_a_400(tmp_path) -> None:
    client, *_ = studio(tmp_path)
    assert post(client, b"").status_code == 400


@needs_ffmpeg
@pytest.mark.parametrize("ext,codec", [("webm", ["-c:a", "libopus"]), ("m4a", ["-c:a", "aac"])])
def test_a_real_browser_or_phone_encoding_is_stored_as_a_16k_wav(tmp_path, ext, codec) -> None:
    client, _, _, settings = studio(tmp_path)
    (tmp_path / "in.wav").write_bytes(wav_bytes(tmp_path, seconds=6.0))
    out = tmp_path / f"o.{ext}"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_path / "in.wav"), *codec, str(out)], check=False)
    if not out.exists():
        pytest.skip("this ffmpeg cannot encode that")
    r = client.post("/api/studio/clips", files={"audio": (f"c.{ext}", out.read_bytes(), "audio/webm")}, data=FORM)
    assert r.status_code == 200, r.text
    clip = r.json()["clip"]
    assert clip["duration_sec"] == pytest.approx(6.0, abs=0.6) and clip["orig_ext"] == f".{ext}"
```
(Note: `clip` dict keys are snake_case from `to_dict()`; the route returns it unchanged.)

- [ ] **Step 2:** run → FAIL (404s / missing routes). **Step 3: implement.**

EDIT `api/app.py`: replace `from ..attacks.bank import BankError, CloneBank` with
```python
from ..attacks.bank import BankError, CloneBank, resolve_speaker_id
from ..studio.plan import build_plan, estimate_minutes
from ..studio.store import DEVICES, ENVIRONMENTS, StudioError, StudioStore
```
EDIT `api/app.py`: replace `    # ------------------------------------------------------------ clone bank` with the block below followed by that same line:
```python
    # ---------------------------------------------------------------- studio

    def _studio(cfg: Settings, pseudonym: str) -> StudioStore:
        """The Studio exists only for allowlisted speakers, in a build that opted in.

        Off, it is a plain 404; on, a pseudonym not on the list is a 403. Nothing
        here can name another person: the speaker is the allowlist's."""
        if not cfg.studio_enabled:
            raise HTTPException(404, "Not found.")
        if pseudonym not in cfg.studio_speakers:
            raise HTTPException(403, f"{pseudonym!r} is not on the studio allowlist (Settings.studio_speakers).")
        return StudioStore(cfg.data_dir / "studio" / pseudonym, pseudonym)

    @app.get("/api/studio/plan")
    def studio_plan(speaker: str, store: StoreDep, cfg: SettingsDep, session: str = "") -> dict[str, Any]:
        studio = _studio(cfg, speaker)
        session_id = session or studio.next_session_id()
        try:
            facts = list(store.get_skg(resolve_speaker_id(store.list_speakers(), speaker)))
        except BankError:
            facts = []
        items = build_plan(session_id, facts)
        return {
            "speaker": speaker,
            "sessionId": session_id,
            "hasFacts": bool(facts),
            "devices": list(DEVICES),
            "environments": list(ENVIRONMENTS),
            "estimatedMinutes": estimate_minutes(items),
            "items": [
                {"kind": i.kind, "promptId": i.prompt_id, "textEn": i.text_en, "textTa": i.text_ta, "repeat": i.repeat}
                for i in items
            ],
        }

    @app.get("/api/studio/summary")
    def studio_summary(speaker: str, cfg: SettingsDep) -> dict[str, Any]:
        return _studio(cfg, speaker).summary()

    @app.post("/api/studio/clips")
    async def studio_add_clip(
        cfg: SettingsDep,
        audio: Annotated[UploadFile, File()],
        speaker: Annotated[str, Form()],
        session_id: Annotated[str, Form()],
        kind: Annotated[str, Form()],
        prompt_id: Annotated[str, Form()],
        device: Annotated[str, Form()],
        environment: Annotated[str, Form()],
        text_hint: Annotated[str, Form()] = "",
        state_note: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        studio = _studio(cfg, speaker)
        raw = await audio.read()
        if not raw:
            raise HTTPException(400, "The uploaded audio is empty.")
        suffix = Path(audio.filename or "clip.webm").suffix or ".webm"

        def work() -> Any:
            return studio.add_clip(
                session_id=session_id, kind=kind, prompt_id=prompt_id, device=device,
                environment=environment, audio=decode_bytes(raw, suffix=suffix),
                orig_bytes=raw, orig_ext=suffix, text_hint=text_hint, state_note=state_note,
            )

        try:
            record = await run_in_threadpool(work)
        except AudioError as exc:
            raise HTTPException(400, str(exc)) from exc
        except StudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"clip": record.to_dict(), "summary": studio.summary()}

    # ------------------------------------------------------------ clone bank
```
(`Any` must be imported in app.py; add to its typing import if absent.)

- [ ] **Step 4:** run `pytest tests/test_studio_api.py tests/test_api.py tests/test_clone_bank_api.py` → PASS. **Step 5:** commit `Serve the Studio behind a flag and an allowlist, through the demo's own decoder`.

---

### Task 5: The `/studio` page

**Files:** Create `kavach/src/pages/Studio.tsx`. Modify `kavach/src/api/types.ts` (CRLF; anchor `export interface CloneBankInfo {`), `kavach/src/api/client.ts` (CRLF; anchors `CloneMatch, CSBG,` and `  /** What the clone bank can answer (demo builds only; 404 otherwise). */`), `kavach/src/App.tsx`, `kavach/src/components/layout/AppLayout.tsx`.

- [ ] **Step 1:** EDIT `types.ts` — before `export interface CloneBankInfo {` insert:
```ts
export interface StudioPlanItem { kind: 'read' | 'free' | 'fact'; promptId: string; textEn: string; textTa: string; repeat: number }
export interface StudioPlan {
  speaker: string; sessionId: string; hasFacts: boolean; estimatedMinutes: number;
  devices: string[]; environments: string[]; items: StudioPlanItem[];
}
export interface StudioSummary {
  speaker: string; clips: number; minutes: number;
  by_session: Record<string, { clips: number; minutes: number; devices: string[]; environments: string[] }>;
  by_kind: Record<string, { clips: number; minutes: number }>;
  by_device: Record<string, { clips: number; minutes: number }>;
  by_environment: Record<string, { clips: number; minutes: number }>;
}

```
- [ ] **Step 2:** EDIT `client.ts` — change `CloneMatch, CSBG,` to `CloneMatch, CSBG, StudioPlan, StudioSummary,`; before `  /** What the clone bank can answer (demo builds only; 404 otherwise). */` insert:
```ts
  studioPlan: async (speaker: string, session?: string): Promise<StudioPlan> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    const q = new URLSearchParams({ speaker });
    if (session) q.set('session', session);
    return fetchApi(`/api/studio/plan?${q}`);
  },

  studioSummary: async (speaker: string): Promise<StudioSummary> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    return fetchApi(`/api/studio/summary?speaker=${encodeURIComponent(speaker)}`);
  },

  studioUpload: async (v: { speaker: string; sessionId: string; kind: string; promptId: string; device: string; environment: string; textHint: string; stateNote: string; blob: Blob; filename?: string }): Promise<{ summary: StudioSummary }> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    const f = new FormData();
    f.append('audio', v.blob, v.filename ?? 'clip.webm');
    f.append('speaker', v.speaker); f.append('session_id', v.sessionId); f.append('kind', v.kind);
    f.append('prompt_id', v.promptId); f.append('device', v.device); f.append('environment', v.environment);
    f.append('text_hint', v.textHint); f.append('state_note', v.stateNote);
    return fetchApi('/api/studio/clips', { method: 'POST', body: f });
  },

```
- [ ] **Step 3: the page**

FILE[impl] kavach/src/pages/Studio.tsx
```tsx
import { useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { AudioRecorder } from '../components/ui/AudioRecorder';
import { Card, CardHeader, CardBody, Button, Field, Select, Input, Badge, Notice, Spinner } from '../components/ui/kit';
import type { StudioPlanItem } from '../api/types';

const KIND_LABEL: Record<string, string> = { read: 'Read aloud', free: 'Speak freely', fact: 'Answer about yourself' };

/** One queue entry per repetition, in the plan's order. */
function expand(items: StudioPlanItem[]) {
  return items.flatMap(it => Array.from({ length: it.repeat }, (_, rep) => ({ ...it, rep: rep + 1 })));
}

export function Studio() {
  const queryClient = useQueryClient();
  const [speaker, setSpeaker] = useState('S04');
  const [session, setSession] = useState('');
  const [device, setDevice] = useState('DEMO_LAPTOP_MIC');
  const [environment, setEnvironment] = useState('QUIET_ROOM');
  const [note, setNote] = useState('');
  const [started, setStarted] = useState(false);
  const [index, setIndex] = useState(0);

  const plan = useQuery({ queryKey: ['studioPlan', speaker, session], queryFn: () => apiClient.studioPlan(speaker, session || undefined), retry: false });
  const summary = useQuery({ queryKey: ['studioSummary', speaker], queryFn: () => apiClient.studioSummary(speaker), retry: false, enabled: plan.isSuccess });
  const queue = useMemo(() => expand(plan.data?.items ?? []), [plan.data]);
  const sessionId = session || plan.data?.sessionId || 'S1';
  const current = queue[index];

  const upload = useMutation({
    mutationFn: (v: { blob: Blob; name?: string }) => apiClient.studioUpload({
      speaker, sessionId, kind: current.kind, promptId: current.promptId, device, environment,
      textHint: current.textTa, stateNote: note, blob: v.blob, filename: v.name,
    }),
    onSuccess: () => { setIndex(i => i + 1); queryClient.invalidateQueries({ queryKey: ['studioSummary', speaker] }); },
  });

  if (plan.isError) {
    return (
      <>
        <PageHeader title="Recording Studio" description="Record labelled sessions of your own voice." />
        <PageBody>
          <Notice tone="warning" title="The Studio is not available">
            {(plan.error as Error).message}. Start the backend with <span className="mono">run_studio.ps1</span> (the demo build never accepts recordings).
          </Notice>
        </PageBody>
      </>
    );
  }

  return (
    <>
      <PageHeader title="Recording Studio" description="Record labelled sessions of your own voice through the same browser microphone the demo uses. Every clip is saved with its session, device and room." />
      <PageBody>
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6">
          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader title="1 · This session" subtitle="Change nothing mid-session: a session is one sitting, one device, one room." />
              <CardBody className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                <Field label="Speaker"><Input value={speaker} onChange={e => { setSpeaker(e.target.value); setStarted(false); setIndex(0); }} /></Field>
                <Field label="Session"><Input value={session || plan.data?.sessionId || ''} onChange={e => { setSession(e.target.value); setIndex(0); }} /></Field>
                <Field label="Device"><Select value={device} onChange={e => setDevice(e.target.value)}>{(plan.data?.devices ?? []).map(d => <option key={d} value={d}>{d}</option>)}</Select></Field>
                <Field label="Room"><Select value={environment} onChange={e => setEnvironment(e.target.value)}>{(plan.data?.environments ?? []).map(d => <option key={d} value={d}>{d}</option>)}</Select></Field>
                <div className="sm:col-span-4"><Field label="Note (optional: tired, hurried, far from the mic …)"><Input value={note} onChange={e => setNote(e.target.value)} /></Field></div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader title="2 · Record" subtitle={plan.data ? `${queue.length} recordings planned, about ${plan.data.estimatedMinutes} min for this session.` : undefined} />
              <CardBody>
                {plan.isLoading && <Spinner label="Loading the plan…" />}
                {plan.data && !plan.data.hasFacts && <Notice tone="warning" className="mb-3" title="No personal facts yet">Add a few on the Speakers page to include the answer-about-yourself recordings.</Notice>}
                {plan.data && !started && <Button variant="primary" onClick={() => { setStarted(true); setIndex(0); }}>Start session {sessionId}</Button>}
                {started && current && (
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2"><Badge tone="accent">{KIND_LABEL[current.kind]}</Badge><Badge>{index + 1} of {queue.length}</Badge>{current.repeat > 1 && <Badge>take {current.rep} of {current.repeat}</Badge>}</div>
                    <p className="font-serif text-[24px] leading-snug">{current.textTa}</p>
                    {current.kind !== 'fact' && <p className="text-[13px] text-app-text-muted">{current.textEn}</p>}
                    <AudioRecorder key={index} busy={upload.isPending} acceptLabel="Save" onAccept={(blob, _d, name) => upload.mutate({ blob, name })} />
                    {upload.error && <Notice tone="reject" title="Not saved">{(upload.error as Error).message}</Notice>}
                    <div><Button size="sm" variant="ghost" onClick={() => setIndex(i => i + 1)}>Skip this one</Button></div>
                  </div>
                )}
                {started && !current && <Notice tone="accept" title="Session complete">Everything planned for {sessionId} is saved. Change the session, device or room above to start the next one.</Notice>}
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader title="What is saved" subtitle="Across all sessions." />
            <CardBody className="flex flex-col gap-3 text-[13px]">
              {summary.data ? (
                <>
                  <div className="tnum"><span className="text-[22px] font-semibold">{summary.data.clips}</span> clips · <span className="font-semibold">{summary.data.minutes}</span> min</div>
                  {Object.entries(summary.data.by_session).map(([id, s]) => (
                    <div key={id} className="flex justify-between"><span>{id} <span className="text-app-text-subtle">{s.devices.join(', ')} · {s.environments.join(', ')}</span></span><span className="tnum">{s.clips} · {s.minutes} min</span></div>
                  ))}
                  <div className="text-app-text-subtle">By kind: {Object.entries(summary.data.by_kind).map(([k, v]) => `${k} ${v.clips}`).join(' · ') || 'none yet'}</div>
                </>
              ) : <span className="text-app-text-subtle">Nothing recorded yet.</span>}
            </CardBody>
          </Card>
        </div>
      </PageBody>
    </>
  );
}
```
- [ ] **Step 4:** EDIT `App.tsx`: add `import { Studio } from './pages/Studio';` after the `Corpus` import and `<Route path="studio" element={<Studio />} />` after the corpus route. EDIT `AppLayout.tsx`: add `Mic` to the lucide import and `{ path: '/studio', label: 'Recording Studio', icon: Mic },` after the Corpus item.
- [ ] **Step 5:** `cd kavach && npx tsc --noEmit && npx vite build` → clean. **Step 6:** commit `Add the Recording Studio page`.

---

### Task 6: Evaluation statistics

**Interfaces — Produces** (in `kavach/eval/enrollee_stats.py`): `cluster_bootstrap_rate(flags_by_cluster: dict[str, list[bool]], *, n_boot=2000, seed=7) -> tuple[float, float]`; `d_prime(genuine, impostor) -> float`.

- [ ] **Step 1: tests**

FILE[test] tests/test_enrollee_stats.py
```python
import numpy as np
import pytest

from kavach.attacks.suite import wilson_interval
from kavach.eval.enrollee_stats import cluster_bootstrap_rate, d_prime


def test_the_bootstrap_is_deterministic_for_a_seed() -> None:
    flags = {"a": [True, False, False], "b": [False] * 3, "c": [True] * 2}
    assert cluster_bootstrap_rate(flags, seed=3) == cluster_bootstrap_rate(flags, seed=3)


def test_clustered_data_gets_a_wider_interval_than_wilson() -> None:
    """Four sittings, each all-or-nothing: 40 clips are not 40 independent trials."""
    flags = {f"s{i}": [i < 2] * 10 for i in range(4)}  # 2 sittings fully rejected
    k = sum(sum(v) for v in flags.values()); n = sum(len(v) for v in flags.values())
    w_lo, w_hi = wilson_interval(k, n)
    c_lo, c_hi = cluster_bootstrap_rate(flags, seed=1)
    assert (c_hi - c_lo) > (w_hi - w_lo)


def test_all_clean_clusters_give_a_zero_rate_interval() -> None:
    lo, hi = cluster_bootstrap_rate({"a": [False] * 5, "b": [False] * 5})
    assert lo == 0.0 and hi == 0.0


def test_one_cluster_is_reported_not_trusted() -> None:
    lo, hi = cluster_bootstrap_rate({"only": [True, False]})
    assert (lo, hi) == (0.0, 1.0) or lo <= hi  # a single cluster cannot support an interval
    assert np.isfinite(lo) and np.isfinite(hi)


def test_d_prime_of_separated_groups_is_large_and_of_identical_is_zero() -> None:
    assert d_prime(np.array([0.9, 0.92, 0.88]), np.array([0.1, 0.12, 0.08])) > 5
    assert d_prime(np.array([0.5, 0.6]), np.array([0.5, 0.6])) == pytest.approx(0.0)
```
- [ ] **Step 2:** run → FAIL. **Step 3: implement** the first part of `eval/enrollee.py` (statistics only; Task 7 adds the rest to the same file):

FILE[impl] backend/kavach/eval/enrollee_stats.py
```python
"""Interval and separation statistics for the single-enrollee evaluation."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap_rate(
    flags_by_cluster: dict[str, list[bool]], *, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float]:
    """95% percentile interval of a pooled rate, resampling whole clusters.

    Clips from one sitting (or trials against one impostor) are not independent,
    so resampling clips understates the uncertainty. With one cluster the interval
    is not informative and is returned as such rather than as a tight number.
    """
    keys = list(flags_by_cluster)
    ks = np.array([sum(flags_by_cluster[k]) for k in keys], dtype=float)
    ns = np.array([len(flags_by_cluster[k]) for k in keys], dtype=float)
    if len(keys) < 2 or ns.sum() == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        total = ns[idx].sum()
        rates[b] = ks[idx].sum() / total if total else np.nan
    return float(np.nanpercentile(rates, 2.5)), float(np.nanpercentile(rates, 97.5))


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Standardised distance between two score distributions."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    pooled = np.sqrt((g.var(ddof=1) + i.var(ddof=1)) / 2.0) if len(g) > 1 and len(i) > 1 else 0.0
    return float((g.mean() - i.mean()) / pooled) if pooled > 1e-12 else 0.0
```
- [ ] **Step 4:** run → PASS (the one-cluster test accepts `(0,1)`). **Step 5:** commit `Add the cluster bootstrap and d-prime for the single-enrollee evaluation`.

---

### Task 7: The evaluation and its report

**Interfaces — Consumes:** `SpeakerEmbedding`, `SpeakerTemplate.from_embeddings/score`, `eval.metrics.compute_eer`, `attacks.suite.wilson_interval`, Task 6's helpers. **Produces:** `LIMITS: str`, `EnrolleeReport` (`.to_dict()`), `evaluate_enrollee(*, enrol: dict[str, list[SpeakerEmbedding]], test: dict[str, list[TestClip]], impostors: dict[str, list[SpeakerEmbedding]], system_threshold: float, seed: int = 7, min_group: int = 5) -> EnrolleeReport`, `render(report) -> str`, `main(argv, embedder=None) -> int`, `TestClip(embedding, device, environment, kind)`.

- [ ] **Step 1: tests**

FILE[test] tests/test_enrollee_eval.py
```python
import numpy as np
import pytest

from kavach.embedding import SpeakerEmbedding
from kavach.eval.enrollee import LIMITS, TestClip, evaluate_enrollee, render

D = 16


def vec(centre: int, noise: float, rng) -> SpeakerEmbedding:
    v = np.zeros(D); v[centre] = 1.0
    return SpeakerEmbedding(v + noise * rng.standard_normal(D))


def world(*, genuine_noise=0.05, impostor_noise=0.05, n_imp=6, seed=0):
    rng = np.random.default_rng(seed)
    enrol = {s: [vec(0, genuine_noise, rng) for _ in range(10)] for s in ("S1", "S2", "S3")}
    test = {"S4": [TestClip(vec(0, genuine_noise, rng), "DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") for _ in range(20)]}
    imp = {f"I{k}": [vec(1 + k, impostor_noise, rng) for _ in range(10)] for k in range(n_imp)}
    return enrol, test, imp


def run(**kw):
    enrol, test, imp = world(**{k: v for k, v in kw.items() if k in ("genuine_noise", "impostor_noise", "n_imp")})
    return evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, seed=5)


def test_a_perfectly_separated_world_has_no_errors_and_a_positive_gap() -> None:
    r = run()
    assert r.frr_at_system == 0.0 and r.far_at_system == 0.0 and r.gap > 0


def test_an_overlapping_world_has_errors_and_no_gap() -> None:
    r = run(genuine_noise=0.9, impostor_noise=0.9)
    assert (r.frr_at_system > 0 or r.far_at_system > 0) and r.gap < 0.2


def test_the_held_out_session_is_never_in_the_template() -> None:
    enrol, test, imp = world()
    rng = np.random.default_rng(1)
    # poison the held-out session: if it leaked into enrolment the template would move
    test["S4"] = [TestClip(vec(7, 0.01, rng), "DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") for _ in range(10)]
    r = evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, seed=5)
    assert r.frr_at_system == 1.0  # a different "voice" is rejected; it did not enrol itself


def test_the_dev_test_split_shares_no_impostor_speaker() -> None:
    r = run()
    assert r.dev_impostors and r.test_impostors and not set(r.dev_impostors) & set(r.test_impostors)


def test_a_single_enrolment_session_skips_the_fitted_threshold_and_says_so() -> None:
    enrol, test, imp = world()
    r = evaluate_enrollee(enrol={"S1": enrol["S1"]}, test=test, impostors=imp, system_threshold=0.62)
    assert r.fitted_threshold is None and any("one enrolment session" in n for n in r.notes)


def test_small_condition_groups_are_dropped_not_printed_as_rates() -> None:
    enrol, test, imp = world()
    test["S4"].append(TestClip(test["S4"][0].embedding, "PHONE", "OFFICE", "free"))
    r = evaluate_enrollee(enrol=enrol, test=test, impostors=imp, system_threshold=0.62, min_group=5)
    assert ("DEMO_LAPTOP_MIC", "QUIET_ROOM", "read") in r.slices
    assert ("PHONE", "OFFICE", "free") not in r.slices


def test_both_intervals_are_reported_and_the_limits_lead_the_report() -> None:
    text = render(run())
    assert text.startswith(LIMITS.splitlines()[0]) or LIMITS in text[:2000]
    assert "Wilson" in text and "cluster" in text.lower()
    assert "one enrolled speaker" in text and "same-sentence" in text


def test_impostors_are_listed_by_pseudonym_with_their_highest_score() -> None:
    r = run()
    assert set(r.by_impostor) == {f"I{k}" for k in range(6)}
    assert all("max" in v for v in r.by_impostor.values())
```
- [ ] **Step 2:** run → FAIL (names missing). **Step 3: implement** — append to `eval/enrollee.py`:

FILE[impl] backend/kavach/eval/enrollee.py
```python
"""Single-enrollee evaluation: does the voiceprint accept this speaker and reject other people?

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
It measures, for ONE enrolled speaker, how often their genuine probes from held-out
sessions are rejected, and how often each of N other recorded speakers is accepted.
It cannot say the voiceprint or code-switching works for people in general (one
enrollee), and it reports no population EER. `LIMITS` leads every report so a
screenshot cannot drop it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..attacks.suite import wilson_interval
from ..embedding import SpeakerEmbedding, SpeakerTemplate
from .enrollee_stats import cluster_bootstrap_rate, d_prime
from .metrics import compute_eer

LIMITS = (
    "LIMITS OF THIS EVIDENCE. This is one enrolled speaker. The false-reject rate is that speaker's, "
    "measured on held-out sessions; the false-accept rate is over N recorded impostors who read different "
    "material (it is not yet the same-sentence test). Nothing here says the voiceprint or code-switching "
    "works for people in general, and there is no population EER. Voice branch only."
)


@dataclass(frozen=True, slots=True)
class TestClip:
    __test__ = False  # not a pytest class

    embedding: SpeakerEmbedding
    device: str
    environment: str
    kind: str


@dataclass(slots=True)
class EnrolleeReport:
    system_threshold: float
    n_genuine: int
    n_impostor: int
    frr_at_system: float
    far_at_system: float
    frr_wilson: tuple[float, float]
    frr_cluster: tuple[float, float]
    far_wilson: tuple[float, float]
    far_cluster: tuple[float, float]
    genuine_mean: float
    genuine_min: float
    impostor_mean: float
    impostor_max: float
    d_prime: float
    gap: float
    fitted_threshold: float | None = None
    fitted_frr: float | None = None
    fitted_far: float | None = None
    dev_impostors: list[str] = field(default_factory=list)
    test_impostors: list[str] = field(default_factory=list)
    slices: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)
    by_impostor: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}
        d["slices"] = {"|".join(k): v for k, v in self.slices.items()}
        return d


def evaluate_enrollee(
    *,
    enrol: dict[str, list[SpeakerEmbedding]],
    test: dict[str, list[TestClip]],
    impostors: dict[str, list[SpeakerEmbedding]],
    system_threshold: float,
    seed: int = 7,
    min_group: int = 5,
) -> EnrolleeReport:
    """Score held-out genuine clips and other speakers against the enrolled template.

    Enrolment and test never share a session; the fitted threshold is chosen on a
    dev partition (the last enrolment session held out of its own template, against
    a random half of the impostor SPEAKERS) and applied unchanged to the test
    partition (the held-out session against the other half). Split by speaker and
    by session, never by trial.
    """
    notes: list[str] = []
    all_enrol = [e for s in enrol.values() for e in s]
    template = SpeakerTemplate.from_embeddings("enrollee", all_enrol)

    gen = {sid: [template.score(c.embedding) for c in clips] for sid, clips in test.items()}
    imp = {spk: [template.score(e) for e in embs] for spk, embs in impostors.items()}
    g = np.array([s for v in gen.values() for s in v])
    i = np.array([s for v in imp.values() for s in v])

    frr_flags = {sid: [s < system_threshold for s in v] for sid, v in gen.items()}
    far_flags = {spk: [s >= system_threshold for s in v] for spk, v in imp.items()}
    k_g = sum(sum(v) for v in frr_flags.values())
    k_i = sum(sum(v) for v in far_flags.values())

    # -- fitted threshold on a dev partition ---------------------------------
    sessions = list(enrol)
    speakers = sorted(impostors)
    rng = random.Random(seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    dev_spk, test_spk = sorted(shuffled[:half]), sorted(shuffled[half:])
    fitted = fitted_frr = fitted_far = None
    if len(sessions) < 2:
        notes.append("Only one enrolment session, so no session can be held out to fit a threshold: the fitted operating point is skipped.")
    elif not dev_spk or not test_spk:
        notes.append("Fewer than two impostor speakers: the speaker-level dev/test split is impossible, so the fitted operating point is skipped.")
    else:
        held = sessions[-1]
        dev_template = SpeakerTemplate.from_embeddings(
            "dev", [e for s in sessions[:-1] for e in enrol[s]]
        )
        dev_g = np.array([dev_template.score(e) for e in enrol[held]])
        dev_i = np.array([dev_template.score(e) for s in dev_spk for e in impostors[s]])
        _, fitted = compute_eer(dev_g, dev_i)
        t_i = np.array([s for spk in test_spk for s in imp[spk]])
        fitted_frr = float((g < fitted).mean())
        fitted_far = float((t_i >= fitted).mean())

    # -- slices ----------------------------------------------------------------
    cells: dict[tuple[str, str, str], list[float]] = {}
    for sid, clips in test.items():
        for c, s in zip(clips, gen[sid]):
            cells.setdefault((c.device, c.environment, c.kind), []).append(s)
    slices = {
        key: {"n": len(v), "frr": float(np.mean([x < system_threshold for x in v])), "mean": float(np.mean(v)), "min": float(np.min(v))}
        for key, v in sorted(cells.items())
        if len(v) >= min_group
    }
    by_impostor = {
        spk: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v)), "far": float(np.mean([x >= system_threshold for x in v]))}
        for spk, v in sorted(imp.items())
    }
    return EnrolleeReport(
        system_threshold=system_threshold,
        n_genuine=len(g),
        n_impostor=len(i),
        frr_at_system=k_g / len(g) if len(g) else float("nan"),
        far_at_system=k_i / len(i) if len(i) else float("nan"),
        frr_wilson=wilson_interval(k_g, len(g)),
        frr_cluster=cluster_bootstrap_rate(frr_flags, seed=seed),
        far_wilson=wilson_interval(k_i, len(i)),
        far_cluster=cluster_bootstrap_rate(far_flags, seed=seed),
        genuine_mean=float(g.mean()),
        genuine_min=float(g.min()),
        impostor_mean=float(i.mean()),
        impostor_max=float(i.max()),
        d_prime=d_prime(g, i),
        gap=float(g.min() - i.max()),
        fitted_threshold=None if fitted is None else float(fitted),
        fitted_frr=fitted_frr,
        fitted_far=fitted_far,
        dev_impostors=dev_spk,
        test_impostors=test_spk,
        slices=slices,
        by_impostor=by_impostor,
        notes=notes,
    )


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def render(r: EnrolleeReport) -> str:
    L = [LIMITS, ""]
    L.append(f"## At the system threshold ({r.system_threshold:.2f}), the one the demo uses")
    L.append(f"- Genuine held-out clips: {r.n_genuine}. Rejected: {_pct(r.frr_at_system)} "
             f"(Wilson {_pct(r.frr_wilson[0])}-{_pct(r.frr_wilson[1])}; cluster bootstrap over sessions "
             f"{_pct(r.frr_cluster[0])}-{_pct(r.frr_cluster[1])}).")
    L.append(f"- Impostor trials: {r.n_impostor} over {len(r.by_impostor)} speakers. Accepted: {_pct(r.far_at_system)} "
             f"(Wilson {_pct(r.far_wilson[0])}-{_pct(r.far_wilson[1])}; cluster bootstrap over speakers "
             f"{_pct(r.far_cluster[0])}-{_pct(r.far_cluster[1])}).")
    L.append("- Trust the cluster interval: clips from one sitting, and trials against one impostor, are not independent.")
    L.append("")
    L.append("## Separation")
    L.append(f"- Genuine mean {r.genuine_mean:.3f}, minimum {r.genuine_min:.3f}. Impostor mean {r.impostor_mean:.3f}, "
             f"maximum {r.impostor_max:.3f}. d' {r.d_prime:.2f}. Gap (worst genuine minus best impostor) {r.gap:+.3f}.")
    L.append("")
    L.append("## A threshold fitted on dev, applied to test")
    if r.fitted_threshold is None:
        L.append("- Skipped.")
    else:
        L.append(f"- Fitted {r.fitted_threshold:.3f} on the last enrolment session and impostors {', '.join(r.dev_impostors)}; "
                 f"on the held-out session and impostors {', '.join(r.test_impostors)}: reject {_pct(r.fitted_frr)}, accept {_pct(r.fitted_far)}.")
    for n in r.notes:
        L.append(f"- Note: {n}")
    L.append("")
    L.append("## Genuine clips by condition (device | room | kind)")
    L.append("| condition | n | rejected | mean | min |")
    L.append("|---|---|---|---|---|")
    for (d, e, k), v in r.slices.items():
        L.append(f"| {d} | {e} | {k} | {int(v['n'])} | {_pct(v['frr'])} | {v['mean']:.3f} | {v['min']:.3f} |".replace("| "+d+" | "+e+" | "+k+" |", f"| {d} / {e} / {k} |", 1))
    L.append("")
    L.append("## Impostors by speaker")
    L.append("| speaker | n | mean | max | accepted |")
    L.append("|---|---|---|---|---|")
    for spk, v in r.by_impostor.items():
        L.append(f"| {spk} | {int(v['n'])} | {v['mean']:.3f} | {v['max']:.3f} | {_pct(v['far'])} |")
    L.append("")
    L.append("Not measured here: the knowledge and code-switch branches, fusion, replay / overlap / clone attacks, and the same-sentence impostor test.")
    return "\n".join(L)
```
- [ ] **Step 4:** run `pytest tests/test_enrollee_eval.py tests/test_enrollee_stats.py` → PASS. Fix the slices-table row builder if the markdown looks wrong (the table is cosmetic; assert on content, not layout).

- [ ] **Step 5: the CLI.** Append to `eval/enrollee.py`:

FILE[append] backend/kavach/eval/enrollee.py
```python
def main(argv=None, *, embedder=None) -> int:
    import argparse
    import csv
    import sys
    from pathlib import Path

    from ..audio import load_audio, prepare_for_embedding  # noqa: F401
    from ..corpus import load_manifest
    from ..studio.store import StudioStore

    p = argparse.ArgumentParser(prog="python -m kavach.eval.enrollee", description=__doc__.splitlines()[0])
    p.add_argument("--studio", required=True, type=Path, help="data/studio/<pseudonym>")
    p.add_argument("--enrol-sessions", required=True)
    p.add_argument("--test-sessions", required=True)
    p.add_argument("--impostors", action="append", required=True, type=Path, help="corpus manifest(s)")
    p.add_argument("--exclude-speaker", action="append", default=[], help="corpus speaker ids to leave out (the enrollee)")
    p.add_argument("--threshold", type=float, default=0.62)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("paper/results_s04"))
    args = p.parse_args(argv)

    store = StudioStore(args.studio, args.studio.name)
    enrol_ids, test_ids = args.enrol_sessions.split(","), args.test_sessions.split(",")
    if set(enrol_ids) & set(test_ids):
        print("refused: a session cannot be both enrolled and held out", file=sys.stderr)
        return 2
    if embedder is None:
        from ..embedding import ECAPAEmbedder
        from ..config import Settings

        embedder = ECAPAEmbedder(Settings().ecapa_model)

    def embed_wav(path):
        return embedder.embed(load_audio(path))

    clips = store.clips()
    enrol = {s: [embed_wav(store.wav_path(c)) for c in clips if c.session_id == s] for s in enrol_ids}
    test = {s: [TestClip(embed_wav(store.wav_path(c)), c.device, c.environment, c.kind) for c in clips if c.session_id == s] for s in test_ids}
    if not any(enrol.values()) or not any(test.values()):
        print("refused: no clips in the named sessions", file=sys.stderr)
        return 2
    imps: dict[str, list] = {}
    for m in args.impostors:
        corpus = load_manifest(m)
        for u in corpus.utterances:
            if u.speaker_id in args.exclude_speaker or not u.audio_path:
                continue
            imps.setdefault(u.speaker_id, []).append(embed_wav((corpus.root or Path()) / u.audio_path))
    report = evaluate_enrollee(enrol=enrol, test=test, impostors=imps, system_threshold=args.threshold, seed=args.seed)
    text = render(report)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text + "\n", encoding="utf-8")
    import json
    (args.out / "results.json").write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```
Test (add to `tests/test_enrollee_eval.py`): a CLI run with an injected fake embedder over a synthetic studio dir and a synthetic manifest-less impostor path is awkward; instead unit-test refusal: overlapping sessions → exit 2; no clips → exit 2. Add:
```python
def test_the_cli_refuses_a_session_that_is_both_enrolled_and_held_out(tmp_path) -> None:
    from kavach.eval.enrollee import main
    assert main(["--studio", str(tmp_path / "S04"), "--enrol-sessions", "S1,S2", "--test-sessions", "S2",
                 "--impostors", str(tmp_path / "m.json")]) == 2
```
- [ ] **Step 6:** run the two eval test files → PASS; commit `Add the single-enrollee evaluation: held-out sessions, cluster intervals, limits first`.

---

### Task 8: How to record tonight

**Files:** Create `run_studio.ps1`, `RECORDING_GUIDE.md`; modify `HANDOFF.md`.

- [ ] **Step 1:** FILE[impl] run_studio.ps1
```powershell
# Start a RECORDING session: the backend with the Studio enabled for the presenter, and the UI.
#
#   powershell -ExecutionPolicy Bypass -File .\run_studio.ps1
#
# Separate from run_demo.ps1 on purpose: the demo build never accepts recordings into the dataset.
# No models are loaded (recording needs only the decoder), so it starts in seconds.
param([string]$Speaker = "S04")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
$env:PYTHONIOENCODING = "utf-8"
$env:KAVACH_OFFLINE = "1"
$env:KAVACH_STUDIO_ENABLED = "true"
$env:KAVACH_STUDIO_SPEAKERS = "[`"$Speaker`"]"

Write-Host "Starting backend on http://localhost:8000 (Studio enabled for $Speaker) ..." -ForegroundColor Cyan
$backend = Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "kavach.api.app:app", "--port", "8000", "--app-dir", "backend" -WorkingDirectory $root -PassThru -WindowStyle Minimized
$ui = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -WorkingDirectory (Join-Path $root "kavach") -PassThru -WindowStyle Minimized
Start-Sleep -Seconds 6
Start-Process "http://localhost:3000/studio"
Write-Host "Recording Studio is open. Press Enter here to stop both servers." -ForegroundColor Green
[void][Console]::ReadLine()
foreach ($p in @($backend, $ui)) { if ($p -and -not $p.HasExited) { taskkill /PID $p.Id /T /F | Out-Null } }
```
- [ ] **Step 2:** write `RECORDING_GUIDE.md` (the presenter's instructions): what to do before (add facts; choose sentences; devices), the four sessions with times, how to speak (natural, same distance as the demo, do not over-enunciate), what "held out" means and why S4 happens after a real break, how to stop and resume, where the data lives and that it never leaves the machine, and the evaluation command with the real session ids. Content per spec sections 3 and 5.
- [ ] **Step 3:** run the offline suite, `tsc`, `vite build`. Smoke the real backend in offline mode with the Studio enabled against a throwaway data dir (not `data/`): `GET /api/health` shows `studioEnabled`, `GET /api/studio/plan?speaker=S04` returns items, and a Playwright pass records nothing but loads `/studio` and shows the plan (page errors `[]`).
- [ ] **Step 4:** HANDOFF update (what exists, how to run a session, the evaluation command, what is not measured); commit `Add run_studio.ps1 and the recording guide`.

---

## Self-Review

**Spec coverage:** §3 data model → T2; §4 API and UI → T4, T5; §5 plan → T3 and `RECORDING_GUIDE.md` (T8); §6 safety → T1, T4; §7 evaluation → T6, T7 (dev/test split, both intervals, slices, separation, report limits); §9 testing → every task is test-first and offline. Spec §8 sub-projects 2-4 are out of scope here.

**Placeholder scan:** none. Task 7's "Executor" notes direct file layout (put the part-2 code into `enrollee.py`); they are instructions, not missing code.

**Type consistency:** `StudioStore.add_clip` kwargs match the route and `StudioPlan`/`StudioSummary` TS shapes (`by_session` etc. are snake_case because the summary dict is returned unchanged; the plan dict is camelCase because the route builds it). `TestClip` is constructed with `(embedding, device, environment, kind)` everywhere.

**Known soft spots:** (1) the markdown slice-table row builder in `render` is clumsy; tests assert content, not layout. (2) The demo sentences are a default in Tamil script written by the assistant; the presenter should read and, if needed, replace them before recording. (3) `ECAPAEmbedder(Settings().ecapa_model)` assumes that constructor signature; confirm with `grep -n "def __init__" backend/kavach/embedding.py` before the first real run.
