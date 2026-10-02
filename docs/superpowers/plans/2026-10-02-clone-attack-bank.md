# Clone-Attack Bank + Demo Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Attack Lab's modelled A4 acoustic score and the demo's missing clone flow with a bank of pre-generated, measured clones of the presenter's voice, and make the live demo survive hostile input and a preflight check before anyone is watching.

**Architecture:** A clone is generated in a separate CUDA venv (kNN-VC), then annotated in the main env with the real Whisper / tagger / ECAPA / answer matcher, giving a `bank.json` of clips with *measured* scores. A gated API serves the clip that answers the issued challenge; the Attack Lab and a new demo button consume it. A `demo_check` preflight and a polite-failure pass make the demo defensible under scrutiny.

**Tech Stack:** Python 3.11 (`.venv`), FastAPI, pydantic-settings, pytest (offline), React + Vite + TypeScript, Playwright (miniconda python, for UI checks), kNN-VC (torch hub) in a new `.venv-clone`.

**Spec:** `docs/superpowers/specs/2026-10-02-clone-attack-bank-design.md` (approved 2026-10-02). Scope here is spec stages 0-1c plus the demo-robustness requirement the user added on approval: *"whatever scrutiny they do in the demo our demo should work 10000% proper."* Spec stage 2 (IndicF5) gets its own plan after its go/no-go probe.

## Global Constraints

- Work from `C:\Bavesh\Sem7\Speech-Processing\speech_proj\speech_proj\speech` (the repo root, "`speech/`"). Run Python as `.venv/Scripts/python.exe`; set `PYTHONPATH=backend` for `-m kavach...` commands; `pytest` (bare) prints the pass count because `addopts = "-q"`.
- Python 3.10+ only (`dataclass(slots=True)`). Tests are offline and load no model; real checkpoints only behind `@pytest.mark.models`. Do not add a test that downloads anything.
- `Settings.clone_victims` default is `[]` (nobody may be cloned) and `Settings.demo_attack_bank` default is `False`. Every clone-bank route returns **404** when the flag is off. `/api/health` reports `demoAttackBank`.
- The bank lives under `settings.attack_dir / "clones" / <pseudonym>` (git-ignored `data/`). `bank.json`, file names and logs carry the corpus pseudonym (`S04`) only, never a participant's name. Display names in the demo DB are `S04 · <first name>`; copy nothing from them except the leading pseudonym.
- `bank.json` stores `AttackType.value` strings (`"A4_clone_knowledge"`); the API wire uses different strings (`"A4_CLONE_KNOWLEDGE"`, `"A3_CLONE_NAIVE"`, `"A5_CLONE_ADAPTIVE"`) via `converters.ATTACK_TO_WIRE`. Never mix them.
- `AttackTable.paper_ready()` is not modified. Lab runs stay `simulated: true`. Measured trials are capped at the number of distinct clips. Yield is printed beside every measured rate.
- **Line endings:** `config.py`, `api/app.py`, `api/pipeline.py`, `kavach/src/api/client.ts`, `kavach/src/api/types.ts` and `run_demo.ps1` are CRLF. Edit them only with **single-line** `Edit` anchors (every anchor in this plan was checked unique). Do not rewrite these files whole.
- Never install into, or modify, the miniconda environments (`C:\Users\baves\miniconda3`): it is the user's training environment.
- **Ask the user before** each heavy step: any `pip install` of torch, any model download, any GPU run, starting the real backend with models, running the full suite more than once at the end. Before any GPU/CPU-heavy step run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` and look for a `python.exe` training job; if one is running, stop and ask.
- Commit locally only; **never push**. End every commit message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- The presenter (`S04`) has **no SKG facts** in the demo DB today (only S08 and S09 do). Tasks 10-11 need the user to enter facts and provide a teammate's spoken answers; stop and ask at those points.

## Review Focus

Input classes and failure modes the spec implies but a happy-path test would miss, most likely to bite first. Each has a test in the task named.

1. **Unusable audio at login** (silence, <1 s, garbage bytes, a 44.1 kHz first upload): must produce a polite rejection in well under a second, never a hallucinated transcript, never ffmpeg stderr, never a multi-second stall. — Task 1.
2. **A challenge whose fact no clone answers**, and a consumed / expired / unknown challenge on the match route: a clear message naming what the bank covers, never a 500 and never the wrong clip. — Task 6.
3. **A corrupted or tampered bank** (hash mismatch, missing file, wrong victim, not on the allowlist): refused with a reason; every other route keeps working; the lab says it fell back instead of silently substituting modelled numbers. — Tasks 3, 6, 7.
4. **An unannotated or flagged clip** (`looks_translated`, `repetition_loop`): never served, never counted in a measured row. — Tasks 3, 6, 7.
5. **The bank routes while the flag is off, even for a real clip id**: 404, and the answer never appears in any other route's body. — Task 6.

## File Structure

| File | Role | Task |
|---|---|---|
| `backend/kavach/audio.py` | polite decode error; `warm_resample()` | 1 |
| `backend/kavach/api/app.py` | warm-up wiring; three gated routes; health flag | 1, 2, 6 |
| `backend/kavach/api/pipeline.py` | early reject of unusable audio; `Pipeline.clone_bank()` | 1, 6 |
| `backend/kavach/config.py` | `clone_victims`, `demo_attack_bank` | 2 |
| `backend/kavach/attacks/bank.py` (new) | `CloneClip`, `CloneBank`, validation, yield | 3 |
| `backend/kavach/attacks/clone.py` | `VoiceConverter` Protocol | 4 |
| `backend/kavach/attacks/make_clone_bank.py` (new) | stage 1 generator + CLI | 4 |
| `backend/kavach/attacks/annotate_bank.py` (new) | stage 2 annotator + CLI | 5 |
| `backend/kavach/api/schemas.py`, `converters.py` | wire types | 2, 6, 7 |
| `backend/kavach/api/attacks.py` | measured rows in the lab | 7 |
| `kavach/src/**` | demo button, lab badge | 8 |
| `backend/kavach/demo_check.py` (new) | preflight | 9 |
| `backend/kavach/attacks/backends/knn_vc.py` (new) | kNN-VC converter | 10 |
| `tests/clone_helpers.py` (new) | shared test builders | 3 |
| `DEMO_RUNBOOK.md` (new) | rehearsed drills and honest answers | 11 |

---

### Task 1: Make the live login fail politely

**Files:**
- Modify: `backend/kavach/audio.py` (decode error text; add `warm_resample`)
- Modify: `backend/kavach/api/app.py:38` (import) and the `for warm_up in (` line (warm-up)
- Modify: `backend/kavach/api/pipeline.py` after `notes.extend(quality.warnings)` (early reject)
- Test: `tests/test_polite_failures.py` (new)

**Interfaces:**
- Produces: `kavach.audio.warm_resample() -> None`; unusable audio (silent or too short) yields `VerificationOutcome` with `fusion.decision is Decision.REJECT` and a single plain-language explanation line, with no model called.
- Consumes: `check_quality` (existing, `QualityReport.is_silent`, `.is_too_short`, `.duration_sec`), `Settings.min_audio_seconds`.

Why first: the probe on 2026-10-02 showed (a) a garbage upload surfaces raw ffmpeg stderr in the UI's "Verification failed" box, (b) the first non-16 kHz upload in a process stalls for several seconds (librosa's lazy import), and (c) silence reaches Whisper, which invents text that is then scored. All three happen in front of an audience if someone uploads the wrong file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_polite_failures.py`:

```python
"""The live login must fail politely.

Probed on 2026-10-02 with hostile and odd inputs: no server errors, but three
things an examiner would see. A garbage upload showed raw decoder output; the
first non-16 kHz upload stalled for seconds while librosa loaded; silence went
on to Whisper, which invents text for it, and the invention was scored.
"""

from __future__ import annotations

import io
import shutil
import threading
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kavach.api.app import create_app, get_pipeline, get_settings, get_store
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.audio import AudioError, decode_bytes, warm_resample
from kavach.config import Settings

SR = 16_000


def wav(seconds: float, kind: str = "tone") -> bytes:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    x = np.zeros(n) if kind == "silence" else 0.3 * np.sin(2 * np.pi * 140 * t)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
    )


@pytest.fixture()
def store(settings) -> Store:
    s = Store(settings.db_path, settings.audio_dir)
    yield s
    s.close()


@pytest.fixture()
def pipeline(store, settings) -> Pipeline:
    return Pipeline(store, settings)


@pytest.fixture()
def client(store, pipeline, settings) -> TestClient:
    app = create_app(settings)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def challenge_id(client: TestClient) -> str:
    speaker = client.post(
        "/api/speakers", json={"displayName": "Probe", "consentGiven": True}
    ).json()
    client.put(
        f"/api/speakers/{speaker['id']}/skg",
        json=[{"subject": "x", "predicate": "hometown", "object": "Thanjavur"}],
    )
    return client.post("/api/challenge", json={"speakerId": speaker["id"]}).json()["id"]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class TestUnreadableUploads:
    @needs_ffmpeg
    def test_garbage_does_not_show_decoder_output(self) -> None:
        with pytest.raises(AudioError) as exc:
            decode_bytes(b"not audio at all" * 50, suffix=".wav")
        message = str(exc.value)
        assert "ffmpeg" not in message.lower() and "[in#" not in message, message
        assert "WAV" in message, "say which formats would have worked"

    @needs_ffmpeg
    def test_the_route_returns_a_400_with_that_message(self, client: TestClient) -> None:
        response = client.post(
            "/api/authenticate",
            files={"audio": ("x.wav", b"not audio at all" * 50, "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        assert response.status_code == 400
        assert "ffmpeg" not in response.json()["detail"].lower()


class TestWarmUp:
    def test_warm_resample_runs(self) -> None:
        assert warm_resample() is None

    def test_start_up_pays_the_resampling_cost_in_the_background(
        self, store, pipeline, settings, monkeypatch
    ) -> None:
        """The first non-16 kHz upload took 3-12 s because librosa loads lazily.
        That must be paid at start-up, in the warm-up thread, not on stage."""
        import kavach.api.app as app_module

        done = threading.Event()
        monkeypatch.setattr(app_module, "_store", store)
        monkeypatch.setattr(app_module, "_pipeline", pipeline)
        monkeypatch.setattr(app_module, "warm_resample", done.set)
        cfg = settings.model_copy(update={"warm_models_on_start": True})
        with TestClient(app_module.create_app(cfg)):
            assert done.wait(15), "start-up never called warm_resample"


class TestUnusableAudio:
    def test_silence_is_rejected_without_running_a_model(
        self, client: TestClient, pipeline: Pipeline, monkeypatch
    ) -> None:
        def boom(*a, **k):
            raise AssertionError("a model ran on audio with nothing in it")

        monkeypatch.setattr(pipeline, "annotate", boom)
        response = client.post(
            "/api/authenticate",
            files={"audio": ("s.wav", wav(3.0, "silence"), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        assert response.status_code == 200
        body = response.json()
        text = " ".join(body["explanation"]).lower()
        assert body["decision"] == "REJECT"
        assert "no speech" in text and "system failure" not in text, body["explanation"]

    def test_a_clip_that_is_too_short_says_how_short(
        self, client: TestClient, pipeline: Pipeline, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            pipeline, "annotate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))
        )
        response = client.post(
            "/api/authenticate",
            files={"audio": ("s.wav", wav(0.3), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        text = " ".join(response.json()["explanation"])
        assert response.json()["decision"] == "REJECT"
        assert "0.3" in text and "at least" in text, text

    def test_ordinary_audio_is_not_caught_by_the_early_reject(self, client: TestClient) -> None:
        response = client.post(
            "/api/authenticate",
            files={"audio": ("t.wav", wav(3.0), "audio/wav")},
            data={"challengeId": challenge_id(client)},
        )
        text = " ".join(response.json()["explanation"]).lower()
        assert "no speech" not in text and "seconds long" not in text, text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_polite_failures.py -p no:cacheprovider`
Expected: collection ERROR `cannot import name 'warm_resample' from 'kavach.audio'`. That is the right failure (feature missing).

- [ ] **Step 3: Implement `audio.py` changes**

In `backend/kavach/audio.py`, replace the decode failure line

```python
        raise AudioError(f"ffmpeg failed to decode the upload: {detail}") from exc
```

with (keep the `detail` variable above it only if still used; delete it if not):

```python
        raise AudioError(
            "That file could not be read as audio. Use a WAV, MP3, M4A, OGG or WebM recording."
        ) from exc
```

and add, directly after the `resample` function:

```python
def warm_resample() -> None:
    """Pay the resampler's first-use cost now instead of on the first upload.

    librosa imports lazily and its first resample takes several seconds; the
    next one is instant. An audience watching the first non-16 kHz upload
    stall for that long reads it as a hang.
    """
    resample(Audio(np.zeros(4410, dtype=np.float32), 44_100, "warm"), TARGET_SAMPLE_RATE)
```

- [ ] **Step 4: Wire the warm-up and the early reject**

Using single-line `Edit` anchors (CRLF files):

`backend/kavach/api/app.py`: replace `from ..audio import AudioError, decode_bytes` with `from ..audio import AudioError, decode_bytes, warm_resample`; replace `                for warm_up in (` with

```python
                for warm_up in (
                    warm_resample,
```

`backend/kavach/api/pipeline.py`: replace `        notes.extend(quality.warnings)` with

```python
        notes.extend(quality.warnings)

        # A recording with nothing usable in it is rejected before any model
        # sees it. Whisper invents text for silence ("Thank you for watching"),
        # and that invention would be tagged, scored against the speaker's
        # graph and explained to the audience as if it had been said.
        if quality.is_silent or quality.is_too_short:
            if quality.is_silent:
                reason = "no speech was detected in the recording"
            else:
                reason = (
                    f"the recording is only {quality.duration_sec:.1f}s long and at "
                    f"least {self.settings.min_audio_seconds:.0f}s is needed"
                )
            result = fuse(branches, self._policy())
            result.explanation = [
                f"Rejected: {reason}. That is a problem with the recording, not "
                "evidence about the speaker -- record the answer again."
            ]
            return VerificationOutcome(
                fusion=result,
                annotation=None,
                csbg_score=None,
                speaker_id=speaker_id,
                challenge_id=challenge.id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                notes=notes,
            )
```

- [ ] **Step 5: Run the tests and the neighbours**

Run: `.venv/Scripts/python.exe -m pytest tests/test_polite_failures.py tests/test_api.py tests/test_pipeline_layers.py -p no:cacheprovider`
Expected: all pass. If an existing test asserted the old `ffmpeg failed to decode` text, update that assertion to the new message (grep: `grep -rn "ffmpeg failed" tests`).

- [ ] **Step 6: Commit**

```bash
git add backend/kavach/audio.py backend/kavach/api/app.py backend/kavach/api/pipeline.py tests/test_polite_failures.py
git commit -m "Make the live login fail politely under hostile input" -m "<what the probe found and what changed>" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Settings and the health flag

**Files:**
- Modify: `backend/kavach/config.py` (anchor `    @field_validator("data_dir", "audio_dir", "attack_dir")`, and anchor `            "demo_reveal_answers": self.demo_reveal_answers,`)
- Modify: `backend/kavach/api/schemas.py` (anchor `    """Surfaced so a build that leaks challenge answers announces itself."""`)
- Modify: `backend/kavach/api/app.py` (anchor `            demo_reveal_answers=cfg.demo_reveal_answers,`)
- Test: `tests/test_clone_bank_api.py` (new; Task 6 extends it)

**Interfaces:**
- Produces: `Settings.clone_victims: list[str]`, `Settings.demo_attack_bank: bool`, `Settings.reportable()["demo_attack_bank"]`, `/api/health` field `demoAttackBank: bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clone_bank_api.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_api.py -p no:cacheprovider`
Expected: FAIL — `KeyError: 'demo_attack_bank'` (settings) and `KeyError: 'demoAttackBank'`.

- [ ] **Step 3: Implement**

`config.py` — replace the single line `    @field_validator("data_dir", "audio_dir", "attack_dir")` with:

```python
    clone_victims: list[str] = Field(default_factory=list)
    """Corpus pseudonyms (`S04`) whose voice may be cloned. **Empty: nobody.**

    Fail closed. Cloning a person's voice needs that person's permission (the
    consent register's public-release OK does not cover it), so the generator
    and the bank loader both refuse anyone not listed here. `run_demo.ps1`
    sets it for the presenter only."""

    demo_attack_bank: bool = False
    """Serve the clone bank and let the Attack Lab use it.

    **Off, and it must stay off outside a demo.** A clip *says* the answer to a
    challenge, so a route that returns the clip for an issued challenge hands
    the knowledge factor to whoever calls it -- the same reasoning as
    `demo_reveal_answers`. `/api/health` reports the flag so a demo build
    announces itself."""

    @field_validator("data_dir", "audio_dir", "attack_dir")
```

and replace `            "demo_reveal_answers": self.demo_reveal_answers,` with the same line followed by `            "demo_attack_bank": self.demo_attack_bank,`.

`schemas.py` — after the line `    """Surfaced so a build that leaks challenge answers announces itself."""` add:

```python

    demo_attack_bank: bool = False
    """True when the clone-bank routes are live. Those routes return audio that
    says the answer to a challenge, so a build with them on must announce it."""
```

`app.py` — replace `            demo_reveal_answers=cfg.demo_reveal_answers,` with that line plus `            demo_attack_bank=cfg.demo_attack_bank,`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_api.py tests/test_api.py -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/kavach/config.py backend/kavach/api/schemas.py backend/kavach/api/app.py tests/test_clone_bank_api.py
git commit -m "Add the clone-bank settings, off and empty by default" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The bank (`attacks/bank.py`)

**Files:**
- Create: `backend/kavach/attacks/bank.py`
- Create: `tests/clone_helpers.py`
- Test: `tests/test_clone_bank.py` (new)
- Modify: `docs/superpowers/specs/2026-10-02-clone-attack-bank-design.md` (one line: the `attack` example value)

**Interfaces:**
- Produces (exact names later tasks import):
  `BANK_VERSION: str`, `BANK_FILE = "bank.json"`, `AUDIO_DIR = "audio"`, `BLOCKING_FLAGS: tuple[str, ...]`, `class BankError(ValueError)`, `sha256_file(path: Path) -> str`, `check_allowed(victim: str, allowed: Sequence[str]) -> None`, `resolve_speaker_id(speakers: Iterable[dict], pseudonym: str) -> str`,
  `CloneClip` (fields: `clip_id, attack: AttackType, backend, audio_path, sha256, duration_sec, created_at, fact_key=None, backend_version="", source: dict[str,str], transcript=None, tokens=None, annotation_source=None, ecapa_similarity=None, threshold=None, admissible=None, answer_score=None, flags: list[str]`; properties `annotated`, `usable`; `to_dict()`, `from_dict(d)`),
  `YieldSummary(generated, annotated, admissible, yield_rate, sources)`,
  `CloneBank(victim, victim_speaker_id, root, clips)` with `new(root, victim, victim_speaker_id, *, allowed_victims, overwrite=False)`, `load(root, *, allowed_victims, expected_speaker_id=None)`, `save()`, `add(clip)`, `audio_file(clip) -> Path`, `read_audio(clip) -> Audio`, `measured(attack=None) -> list[CloneClip]`, `match(fact_key) -> CloneClip | None`, `covered_facts() -> list[str]`, `yield_summary(attack=None) -> YieldSummary`.
- Consumes: `kavach.attacks.AttackType`, `kavach.audio.Audio/load_audio`.

- [ ] **Step 1: Write the shared test builders**

Create `tests/clone_helpers.py`:

```python
"""Builders shared by the clone-bank tests. Not a test module."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kavach.attacks import AttackType
from kavach.attacks.bank import AUDIO_DIR, CloneBank, CloneClip, sha256_file
from kavach.audio import Audio, save_wav
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token

SR = 16_000


def tone(seconds: float = 3.0, freq: float = 140.0, amp: float = 0.3, seed: int = 0) -> Audio:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    x = amp * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))
    return Audio(x.astype(np.float32), SR, f"tone{freq}")


def token_dicts() -> list[dict]:
    """Stored-token wire dicts, built from the real enums so values cannot drift."""
    toks = [
        Token("naan", Language.TA, SemanticClass.OTHER, 0.9, 0, 300),
        Token("hometown", Language.EN, SemanticClass.OTHER, 0.9, 300, 800),
    ]
    return [
        {
            "text": t.text,
            "language": t.language.value,
            "semanticClass": t.semantic_class.value,
            "lidConfidence": t.lid_confidence,
            "startMs": t.start_ms,
            "endMs": t.end_ms,
        }
        for t in toks
    ]


def new_bank(tmp_path: Path, victim: str = "S04", speaker_id: str = "spk_victim") -> CloneBank:
    return CloneBank.new(
        tmp_path / "clones" / victim, victim, speaker_id, allowed_victims=[victim]
    )


def add_clip(
    bank: CloneBank,
    *,
    fact_key: str | None = "hometown",
    attack: AttackType = AttackType.A4_CLONE_KNOWLEDGE,
    annotated: bool = True,
    similarity: float = 0.8,
    admissible: bool = True,
    answer_score: float | None = 0.9,
    flags: list[str] | None = None,
    speaker: str = "attacker_1",
) -> CloneClip:
    clip_id = f"clone_{len(bank.clips):08x}"
    rel = f"{AUDIO_DIR}/{clip_id}__SYNTHETIC.wav"
    path = bank.root / rel
    save_wav(tone(seed=len(bank.clips)), path)
    clip = CloneClip(
        clip_id=clip_id,
        attack=attack,
        backend="fake",
        audio_path=rel,
        sha256=sha256_file(path),
        duration_sec=3.0,
        created_at="2026-10-02T00:00:00+00:00",
        fact_key=fact_key,
        source={"kind": "teammate_speech", "file": f"{fact_key}.wav", "speaker": speaker},
    )
    if annotated:
        clip.transcript = "naan hometown Thanjavur"
        clip.tokens = token_dicts()
        clip.annotation_source = "LLM"
        clip.ecapa_similarity = similarity
        clip.threshold = 0.62
        clip.admissible = admissible
        clip.answer_score = answer_score
        clip.flags = list(flags or [])
    bank.add(clip)
    return clip
```

- [ ] **Step 2: Write the failing bank tests**

Create `tests/test_clone_bank.py`:

```python
"""The bank is the trust boundary for clone clips: what it refuses matters most."""

from __future__ import annotations

import json

import pytest

from clone_helpers import add_clip, new_bank, tone
from kavach.attacks import AttackType
from kavach.attacks.bank import (
    BANK_FILE,
    BankError,
    CloneBank,
    check_allowed,
    resolve_speaker_id,
)
from kavach.audio import save_wav


def reload(bank: CloneBank, **kw) -> CloneBank:
    return CloneBank.load(bank.root, allowed_victims=["S04"], **kw)


class TestRoundTrip:
    def test_save_and_load_keep_everything(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        loaded = reload(bank, expected_speaker_id="spk_victim")
        assert loaded.victim == "S04" and loaded.victim_speaker_id == "spk_victim"
        assert loaded.clips[0].to_dict() == bank.clips[0].to_dict()
        assert loaded.clips[0].attack is AttackType.A4_CLONE_KNOWLEDGE

    def test_save_is_atomic(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        assert not list(bank.root.glob("*.tmp"))

    def test_the_file_carries_the_pseudonym_and_marks_itself_synthetic(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        data = json.loads((bank.root / BANK_FILE).read_text(encoding="utf-8"))
        assert data["victim"] == "S04" and data["synthetic"] is True
        assert data["clips"][0]["attack"] == "A4_clone_knowledge"


class TestRefusals:
    def test_a_victim_not_on_the_allowlist_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="allowlist"):
            CloneBank.load(bank.root, allowed_victims=["S09"])

    def test_an_empty_allowlist_refuses_everyone(self, tmp_path) -> None:
        with pytest.raises(BankError, match="allowlist"):
            check_allowed("S04", [])

    def test_a_missing_audio_file_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        bank.audio_file(clip).unlink()
        with pytest.raises(BankError, match="missing"):
            reload(bank)

    def test_a_changed_audio_file_is_refused(self, tmp_path) -> None:
        """The hash is what proves a clip is the one that was screened."""
        bank = new_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        save_wav(tone(seed=99), bank.audio_file(clip))
        with pytest.raises(BankError, match="hash"):
            reload(bank)

    def test_a_bank_for_a_different_speaker_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="speaker"):
            reload(bank, expected_speaker_id="spk_someone_else")

    def test_an_unknown_bank_version_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        path = bank.root / BANK_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["bank_version"] = "99"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(BankError, match="version"):
            reload(bank)

    def test_a_path_that_escapes_the_bank_is_refused(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        path = bank.root / BANK_FILE
        data = json.loads(path.read_text(encoding="utf-8"))
        data["clips"][0]["audio_path"] = "../../outside.wav"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(BankError):
            reload(bank)

    def test_new_refuses_to_overwrite_a_bank(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank)
        bank.save()
        with pytest.raises(BankError, match="exists"):
            CloneBank.new(bank.root, "S04", "spk_victim", allowed_victims=["S04"])
        CloneBank.new(bank.root, "S04", "spk_victim", allowed_victims=["S04"], overwrite=True)


class TestSelection:
    def test_unannotated_clips_are_never_measured(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, annotated=False)
        assert bank.measured() == [] and bank.match("hometown") is None

    def test_a_flagged_clip_is_never_measured_or_matched(self, tmp_path) -> None:
        """A transcript Whisper translated is a fabricated language choice."""
        bank = new_bank(tmp_path)
        add_clip(bank, flags=["looks_translated: all English"])
        assert bank.measured() == [] and bank.match("hometown") is None

    def test_match_picks_the_admissible_clip_with_the_best_similarity(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, similarity=0.70)
        best = add_clip(bank, similarity=0.85)
        add_clip(bank, similarity=0.95, admissible=False)
        assert bank.match("hometown").clip_id == best.clip_id

    def test_match_only_returns_a4_clips_for_that_fact(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, attack=AttackType.A3_CLONE, fact_key=None)
        add_clip(bank, fact_key="college")
        assert bank.match("hometown") is None
        assert bank.match("college") is not None

    def test_covered_facts_lists_only_what_can_be_answered(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, fact_key="hometown")
        add_clip(bank, fact_key="college", admissible=False)
        add_clip(bank, fact_key="favouriteFood", annotated=False)
        assert bank.covered_facts() == ["hometown"]


class TestYield:
    def test_yield_is_admissible_over_annotated(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, admissible=True)
        add_clip(bank, admissible=True)
        add_clip(bank, admissible=False)
        add_clip(bank, annotated=False)
        s = bank.yield_summary()
        assert (s.generated, s.annotated, s.admissible) == (4, 3, 2)
        assert s.yield_rate == pytest.approx(2 / 3)

    def test_yield_is_none_when_nothing_is_annotated(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, annotated=False)
        assert bank.yield_summary().yield_rate is None

    def test_sources_counts_distinct_attackers(self, tmp_path) -> None:
        bank = new_bank(tmp_path)
        add_clip(bank, speaker="attacker_1")
        add_clip(bank, speaker="attacker_1")
        add_clip(bank, speaker="attacker_2")
        assert bank.yield_summary().sources == 2


class TestResolveSpeaker:
    SPEAKERS = [
        {"id": "spk_a", "display_name": "S04 · Someone"},
        {"id": "spk_b", "display_name": "S08 · Other"},
        {"id": "spk_c", "display_name": "S040 · Not S04"},
    ]

    def test_it_matches_the_pseudonym_prefix_only(self) -> None:
        assert resolve_speaker_id(self.SPEAKERS, "S04") == "spk_a"

    def test_it_refuses_when_there_is_no_match(self) -> None:
        with pytest.raises(BankError):
            resolve_speaker_id(self.SPEAKERS, "S11")

    def test_it_refuses_when_there_are_two(self) -> None:
        with pytest.raises(BankError):
            resolve_speaker_id(self.SPEAKERS + [{"id": "spk_d", "display_name": "S04"}], "S04")
```

- [ ] **Step 3: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank.py -p no:cacheprovider`
Expected: collection ERROR `No module named 'kavach.attacks.bank'`.

- [ ] **Step 4: Write the implementation**

Create `backend/kavach/attacks/bank.py`:

```python
"""A bank of pre-generated voice clones of one consenting victim.

Stage 1 (`make_clone_bank`) writes the clips; stage 2 (`annotate_bank`) measures
them; the API and the Attack Lab only ever read this. Nothing here imports a
model library, so the main environment and the whole test suite can load it.

WHAT THE LOADER REFUSES, AND WHY
--------------------------------
A clone clip is a recording of someone's voice that they never made, so the
bank is a trust boundary and `load` is strict. It refuses a victim who is not
on the allowlist (consent), a clip whose file is missing or whose hash no
longer matches (the clip served is not the clip that was screened), a bank
that belongs to a different speaker, an unknown version, and any path that
escapes the bank directory. A refusal names its reason; callers report it and
do not fall back silently.

WHAT COUNTS AS MEASURED
-----------------------
A clip is `usable` only once stage 2 has annotated it and no blocking flag is
set. Whisper sometimes translates Tamil into English instead of transcribing
it (HANDOFF trap 0); an untouched transcript of that kind is a fabricated
language choice, so a flagged clip is excluded from every measured row and is
never served.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..audio import Audio, load_audio
from . import AttackType

BANK_VERSION = "1"
BANK_FILE = "bank.json"
AUDIO_DIR = "audio"

#: Flag prefixes that keep a clip out of every measured row.
BLOCKING_FLAGS = ("looks_translated", "repetition_loop")


class BankError(ValueError):
    """The bank cannot be trusted; the message says why."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_allowed(victim: str, allowed: Sequence[str]) -> None:
    """Refuse anyone not on the allowlist. Called first, before any work."""
    if victim not in allowed:
        raise BankError(
            f"{victim!r} is not on the clone allowlist "
            f"(Settings.clone_victims = {list(allowed)!r}). Only a person who has "
            "agreed to be cloned may be."
        )


def resolve_speaker_id(speakers: Iterable[dict[str, Any]], pseudonym: str) -> str:
    """The DB speaker id whose display name starts with `pseudonym`.

    Display names in the demo DB are `S04 · <first name>`. Only the leading
    pseudonym is used; the name never leaves this function.
    """
    matches = [
        s["id"]
        for s in speakers
        if (s.get("display_name") or "") == pseudonym
        or (s.get("display_name") or "").startswith(pseudonym + " ")
    ]
    if len(matches) != 1:
        raise BankError(
            f"expected exactly one speaker for pseudonym {pseudonym!r}, found {len(matches)}"
        )
    return matches[0]


@dataclass(slots=True)
class CloneClip:
    """One clone, with provenance and (after stage 2) measurements."""

    clip_id: str
    attack: AttackType
    backend: str
    audio_path: str
    sha256: str
    duration_sec: float
    created_at: str
    fact_key: str | None = None
    backend_version: str = ""
    source: dict[str, str] = field(default_factory=dict)

    # Written by stage 2; absent before it.
    transcript: str | None = None
    tokens: list[dict[str, Any]] | None = None
    annotation_source: str | None = None
    ecapa_similarity: float | None = None
    threshold: float | None = None
    admissible: bool | None = None
    answer_score: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def annotated(self) -> bool:
        return self.transcript is not None and self.ecapa_similarity is not None

    @property
    def usable(self) -> bool:
        """Annotated, and free of a flag saying the transcript is untrustworthy."""
        return self.annotated and not any(
            f.split(":", 1)[0] in BLOCKING_FLAGS for f in self.flags
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "attack": self.attack.value,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "fact_key": self.fact_key,
            "source": dict(self.source),
            "audio_path": self.audio_path,
            "sha256": self.sha256,
            "duration_sec": self.duration_sec,
            "created_at": self.created_at,
            "transcript": self.transcript,
            "tokens": self.tokens,
            "annotation_source": self.annotation_source,
            "ecapa_similarity": self.ecapa_similarity,
            "threshold": self.threshold,
            "admissible": self.admissible,
            "answer_score": self.answer_score,
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CloneClip:
        return cls(
            clip_id=d["clip_id"],
            attack=AttackType(d["attack"]),
            backend=d["backend"],
            audio_path=d["audio_path"],
            sha256=d["sha256"],
            duration_sec=float(d["duration_sec"]),
            created_at=d["created_at"],
            fact_key=d.get("fact_key"),
            backend_version=d.get("backend_version", ""),
            source=dict(d.get("source") or {}),
            transcript=d.get("transcript"),
            tokens=d.get("tokens"),
            annotation_source=d.get("annotation_source"),
            ecapa_similarity=d.get("ecapa_similarity"),
            threshold=d.get("threshold"),
            admissible=d.get("admissible"),
            answer_score=d.get("answer_score"),
            flags=list(d.get("flags") or []),
        )


@dataclass(slots=True)
class YieldSummary:
    generated: int
    annotated: int
    admissible: int
    yield_rate: float | None
    sources: int


@dataclass(slots=True)
class CloneBank:
    victim: str
    victim_speaker_id: str
    root: Path
    clips: list[CloneClip] = field(default_factory=list)

    # ---------------------------------------------------------- construction

    @classmethod
    def new(
        cls,
        root: Path | str,
        victim: str,
        victim_speaker_id: str,
        *,
        allowed_victims: Sequence[str],
        overwrite: bool = False,
    ) -> CloneBank:
        check_allowed(victim, allowed_victims)
        root = Path(root)
        if (root / BANK_FILE).exists():
            if not overwrite:
                raise BankError(f"a bank already exists at {root}; pass overwrite to replace it")
            for old in (root / AUDIO_DIR).glob("*__SYNTHETIC.wav"):
                old.unlink()
        (root / AUDIO_DIR).mkdir(parents=True, exist_ok=True)
        return cls(victim=victim, victim_speaker_id=victim_speaker_id, root=root)

    @classmethod
    def load(
        cls,
        root: Path | str,
        *,
        allowed_victims: Sequence[str],
        expected_speaker_id: str | None = None,
    ) -> CloneBank:
        root = Path(root)
        path = root / BANK_FILE
        if not path.exists():
            raise BankError(f"no bank at {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BankError(f"{path} is not valid JSON: {exc}") from exc
        if data.get("bank_version") != BANK_VERSION:
            raise BankError(
                f"bank version {data.get('bank_version')!r}, this code reads {BANK_VERSION!r}"
            )
        victim = data.get("victim", "")
        check_allowed(victim, allowed_victims)
        if data.get("synthetic") is not True:
            raise BankError("the bank is not marked synthetic")
        speaker_id = data.get("victim_speaker_id", "")
        if expected_speaker_id is not None and speaker_id != expected_speaker_id:
            raise BankError(
                f"the bank was built for speaker {speaker_id!r}, not {expected_speaker_id!r}"
            )
        try:
            clips = [CloneClip.from_dict(c) for c in data.get("clips", [])]
        except (KeyError, ValueError, TypeError) as exc:
            raise BankError(f"malformed clip record: {exc}") from exc

        bank = cls(victim=victim, victim_speaker_id=speaker_id, root=root, clips=clips)
        for clip in clips:
            file = bank.audio_file(clip)
            if not file.exists():
                raise BankError(f"clip {clip.clip_id}: audio file is missing ({clip.audio_path})")
            if sha256_file(file) != clip.sha256:
                raise BankError(
                    f"clip {clip.clip_id}: audio does not match its recorded hash -- "
                    "the file was changed after it was generated"
                )
        return bank

    # ------------------------------------------------------------------ I/O

    def save(self) -> Path:
        """Write `bank.json` atomically: a crash mid-write must not corrupt it."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / BANK_FILE
        tmp = path.with_suffix(".json.tmp")
        payload = {
            "bank_version": BANK_VERSION,
            "victim": self.victim,
            "victim_speaker_id": self.victim_speaker_id,
            "synthetic": True,
            "clips": [c.to_dict() for c in self.clips],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def add(self, clip: CloneClip) -> None:
        self.clips.append(clip)

    def audio_file(self, clip: CloneClip) -> Path:
        root = self.root.resolve()
        path = (root / clip.audio_path).resolve()
        if root != path and root not in path.parents:
            raise BankError(f"clip {clip.clip_id}: audio path escapes the bank directory")
        return path

    def read_audio(self, clip: CloneClip) -> Audio:
        return load_audio(self.audio_file(clip))

    # -------------------------------------------------------------- queries

    def measured(self, attack: AttackType | None = None) -> list[CloneClip]:
        return [
            c for c in self.clips if c.usable and (attack is None or c.attack is attack)
        ]

    def match(self, fact_key: str) -> CloneClip | None:
        """The A4 clip answering `fact_key`: admissible, best voiceprint match."""
        candidates = [
            c
            for c in self.measured(AttackType.A4_CLONE_KNOWLEDGE)
            if c.fact_key == fact_key and c.admissible
        ]
        return max(candidates, key=lambda c: c.ecapa_similarity or 0.0, default=None)

    def covered_facts(self) -> list[str]:
        return sorted(
            {
                c.fact_key
                for c in self.measured(AttackType.A4_CLONE_KNOWLEDGE)
                if c.fact_key and c.admissible
            }
        )

    def yield_summary(self, attack: AttackType | None = None) -> YieldSummary:
        pool = [c for c in self.clips if attack is None or c.attack is attack]
        annotated = [c for c in pool if c.annotated]
        admissible = [c for c in annotated if c.admissible]
        return YieldSummary(
            generated=len(pool),
            annotated=len(annotated),
            admissible=len(admissible),
            yield_rate=(len(admissible) / len(annotated)) if annotated else None,
            sources=len({c.source.get("speaker", "unknown") for c in pool}),
        )


__all__ = [
    "AUDIO_DIR",
    "BANK_FILE",
    "BANK_VERSION",
    "BLOCKING_FLAGS",
    "BankError",
    "CloneBank",
    "CloneClip",
    "YieldSummary",
    "check_allowed",
    "resolve_speaker_id",
    "sha256_file",
]
```

- [ ] **Step 5: Correct the spec's example value**

In `docs/superpowers/specs/2026-10-02-clone-attack-bank-design.md` replace the line containing `"attack": "A4_CLONE_KNOWLEDGE",      // A3_CLONE | A4_CLONE_KNOWLEDGE | A5_STYLE_ADAPTIVE` with `"attack": "A4_clone_knowledge",      // AttackType.value: A3_clone | A4_clone_knowledge | A5_style_adaptive`.

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank.py -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add backend/kavach/attacks/bank.py tests/clone_helpers.py tests/test_clone_bank.py docs/superpowers/specs/2026-10-02-clone-attack-bank-design.md
git commit -m "Add the clone bank: provenance, a strict loader, and measured-only selection" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Stage 1 generator (`make_clone_bank.py`) and the `VoiceConverter` protocol

**Files:**
- Modify: `backend/kavach/attacks/clone.py` (insert before `class XTTSCloner:`; extend `__all__`)
- Create: `backend/kavach/attacks/make_clone_bank.py`
- Test: `tests/test_make_clone_bank.py` (new)

**Interfaces:**
- Consumes: Task 3's `CloneBank`, `CloneClip`, `BankError`, `check_allowed`, `resolve_speaker_id`, `sha256_file`, `AUDIO_DIR`; `clone.MIN_REFERENCE_SEC` (6.0).
- Produces: `VoiceConverter` Protocol; `GenerationReport(bank, uncovered_facts, warnings)`; `generate_bank(*, victim, victim_speaker_id, converter, target_reference, sources, facts, out_root, allowed_victims, attacker_label="attacker_1", overwrite=False, now=None) -> GenerationReport`; `build_converter(name) -> VoiceConverter`; `main(argv) -> int` (exit 0 ok, 2 refused).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_make_clone_bank.py`:

```python
from __future__ import annotations

import json

import pytest

from clone_helpers import tone
from kavach.attacks import AttackType
from kavach.attacks.bank import BANK_FILE, BankError, CloneBank
from kavach.attacks.clone import VoiceConverter
from kavach.attacks.make_clone_bank import generate_bank, main
from kavach.audio import Audio


class FakeConverter:
    """Stands in for kNN-VC: same words, 'new voice' = a different pitch tone."""

    def name(self) -> str:
        return "fake"

    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio:
        return tone(seconds=source.duration_sec, freq=210.0)


def sources(*facts: str) -> dict[str, Audio]:
    return {f: tone(seconds=3.0, seed=i) for i, f in enumerate(facts)}


def generate(tmp_path, **kw):
    args = dict(
        victim="S04",
        victim_speaker_id="spk_victim",
        converter=FakeConverter(),
        target_reference=[tone(seconds=8.0)],
        sources=sources("hometown", "college"),
        facts=["hometown", "college", "favouriteFood"],
        out_root=tmp_path / "clones" / "S04",
        allowed_victims=["S04"],
    )
    args.update(kw)
    return generate_bank(**args)


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeConverter(), VoiceConverter)


def test_one_a4_clip_per_covered_fact(tmp_path) -> None:
    report = generate(tmp_path)
    bank = CloneBank.load(report.bank.root, allowed_victims=["S04"])
    assert sorted(c.fact_key for c in bank.clips) == ["college", "hometown"]
    assert all(c.attack is AttackType.A4_CLONE_KNOWLEDGE for c in bank.clips)
    assert report.uncovered_facts == ["favouriteFood"]


def test_clips_are_marked_synthetic_in_the_file_name(tmp_path) -> None:
    report = generate(tmp_path)
    assert all("SYNTHETIC" in c.audio_path for c in report.bank.clips)


def test_nothing_in_the_bank_carries_a_name(tmp_path) -> None:
    report = generate(tmp_path)
    text = (report.bank.root / BANK_FILE).read_text(encoding="utf-8")
    assert json.loads(text)["victim"] == "S04"
    assert "·" not in text, "a display name leaked into the bank"


def test_a_victim_not_on_the_allowlist_is_refused_before_any_work(tmp_path) -> None:
    class Exploding(FakeConverter):
        def convert(self, *a, **k):
            raise AssertionError("converted audio for someone who has not agreed")

    with pytest.raises(BankError, match="allowlist"):
        generate(tmp_path, converter=Exploding(), allowed_victims=[])


def test_too_little_target_audio_is_refused(tmp_path) -> None:
    with pytest.raises(BankError, match="reference"):
        generate(tmp_path, target_reference=[tone(seconds=2.0)])


def test_short_but_usable_target_audio_warns(tmp_path) -> None:
    report = generate(tmp_path)  # 8 s: usable, far under the recommended amount
    assert any("minutes" in w for w in report.warnings)


def test_a_second_run_does_not_silently_replace_the_bank(tmp_path) -> None:
    generate(tmp_path)
    with pytest.raises(BankError, match="exists"):
        generate(tmp_path)
    generate(tmp_path, overwrite=True)


def test_the_cli_refuses_a_victim_who_is_not_allowed_without_touching_a_model(
    tmp_path, monkeypatch, capsys
) -> None:
    import kavach.attacks.make_clone_bank as mod

    monkeypatch.setattr(mod, "build_converter", lambda name: (_ for _ in ()).throw(AssertionError("loaded a model")))
    monkeypatch.setenv("KAVACH_DATA_DIR", str(tmp_path))
    code = main(["--victim", "S04", "--sources", str(tmp_path), "--backend", "knn_vc"])
    assert code == 2
    assert "allowlist" in capsys.readouterr().err
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_make_clone_bank.py -p no:cacheprovider`
Expected: collection ERROR `cannot import name 'VoiceConverter'`.

- [ ] **Step 3: Add the protocol**

In `backend/kavach/attacks/clone.py`, replace the single line `class XTTSCloner:` with:

```python
@runtime_checkable
class VoiceConverter(Protocol):
    """Converts one person's *speech* into another person's voice.

    The sibling of `CloneBackend` for voice conversion (kNN-VC): the input is
    audio, not text, so it cannot satisfy `SynthesisRequest`. The attacker
    supplies the words and the style; the converter supplies only the voice,
    which is exactly the A4 story -- voice stolen, answer known, code-switching
    habits the attacker's own.
    """

    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio: ...

    def name(self) -> str: ...


class XTTSCloner:
```

and in `__all__` add `    "VoiceConverter",` after `    "SynthesisRequest",`.

- [ ] **Step 4: Write the generator**

Create `backend/kavach/attacks/make_clone_bank.py`:

```python
"""Stage 1: convert a teammate's spoken answers into the victim's voice.

    PYTHONPATH=backend .venv-clone/Scripts/python.exe -m kavach.attacks.make_clone_bank \\
        --victim S04 --sources data/clone_sources --backend knn_vc

Runs in the clone environment (CUDA torch). Writes clips and provenance only;
`annotate_bank` (stage 2, main environment) measures them.

CONSENT COMES FIRST
-------------------
The allowlist check is the first thing both `generate_bank` and `main` do,
before the store is opened, before any audio is read and before a model is
imported. A person who has not agreed to be cloned is refused with no side
effect.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from ..audio import Audio, load_audio, save_wav
from . import AttackType
from .bank import (
    AUDIO_DIR,
    BankError,
    CloneBank,
    CloneClip,
    check_allowed,
    resolve_speaker_id,
    sha256_file,
)
from .clone import MIN_REFERENCE_SEC, VoiceConverter

#: kNN-VC matches each source frame to the nearest frames of the target's
#: speech; with much less than this the pool is thin and quality drops.
RECOMMENDED_TARGET_SEC = 300.0


@dataclass(slots=True)
class GenerationReport:
    bank: CloneBank
    uncovered_facts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def generate_bank(
    *,
    victim: str,
    victim_speaker_id: str,
    converter: VoiceConverter,
    target_reference: list[Audio],
    sources: dict[str, Audio],
    facts: Sequence[str],
    out_root: Path,
    allowed_victims: Sequence[str],
    attacker_label: str = "attacker_1",
    overwrite: bool = False,
    now: Callable[[], str] | None = None,
) -> GenerationReport:
    check_allowed(victim, allowed_victims)  # first: before any work

    target_sec = sum(a.duration_sec for a in target_reference)
    if target_sec < MIN_REFERENCE_SEC:
        raise BankError(
            f"only {target_sec:.1f}s of reference audio for the victim; "
            f"{MIN_REFERENCE_SEC:.0f}s is the floor and a clone built from less is not a test of the defence"
        )
    warnings: list[str] = []
    if target_sec < RECOMMENDED_TARGET_SEC:
        warnings.append(
            f"{target_sec / 60:.1f} minutes of reference audio; kNN-VC works best with "
            f"{RECOMMENDED_TARGET_SEC / 60:.0f}+ minutes. A low yield may reflect this, "
            "not the defence."
        )

    bank = CloneBank.new(
        out_root, victim, victim_speaker_id, allowed_victims=allowed_victims, overwrite=overwrite
    )
    stamp = now or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for fact in facts:
        source = sources.get(fact)
        if source is None:
            continue
        converted = converter.convert(source, target_reference)
        clip_id = f"clone_{uuid.uuid4().hex[:8]}"
        rel = f"{AUDIO_DIR}/{clip_id}__SYNTHETIC.wav"
        path = bank.root / rel
        save_wav(converted, path)
        bank.add(
            CloneClip(
                clip_id=clip_id,
                attack=AttackType.A4_CLONE_KNOWLEDGE,
                backend=converter.name(),
                audio_path=rel,
                sha256=sha256_file(path),
                duration_sec=converted.duration_sec,
                created_at=stamp(),
                fact_key=fact,
                source={"kind": "teammate_speech", "file": f"{fact}.wav", "speaker": attacker_label},
            )
        )
    bank.save()
    return GenerationReport(
        bank=bank,
        uncovered_facts=[f for f in facts if f not in sources],
        warnings=warnings,
    )


def build_converter(name: str) -> VoiceConverter:
    if name == "knn_vc":
        from .backends.knn_vc import KnnVcConverter  # torch is imported inside

        return KnnVcConverter()
    raise BankError(f"unknown backend {name!r}; stage 1 has only 'knn_vc'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kavach.attacks.make_clone_bank",
        description="Convert a teammate's spoken answers into the victim's voice.",
    )
    p.add_argument("--victim", required=True, help="Corpus pseudonym, e.g. S04.")
    p.add_argument("--sources", required=True, type=Path, help="Folder of <fact>.wav answers.")
    p.add_argument("--backend", default="knn_vc")
    p.add_argument("--attacker-label", default="attacker_1")
    p.add_argument("--out", type=Path, default=None, help="Default: <attack_dir>/clones/<victim>.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..config import Settings

    settings = Settings()
    try:
        check_allowed(args.victim, settings.clone_victims)  # before the store, audio or torch
    except BankError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    from ..api.store import Store

    store = Store(settings.db_path, settings.audio_dir)
    try:
        speaker_id = resolve_speaker_id(store.list_speakers(), args.victim)
        facts = [f.predicate for f in store.get_skg(speaker_id)]
        if not facts:
            print(
                f"refused: {args.victim} has no knowledge-graph facts, so no question can "
                "be asked of them and there is nothing for a clone to answer.",
                file=sys.stderr,
            )
            return 2
        reference = [load_audio(store.audio_path(r["id"])) for r in store.list_utterances(speaker_id)]
        sources = {
            f: load_audio(args.sources / f"{f}.wav")
            for f in facts
            if (args.sources / f"{f}.wav").exists()
        }
        report = generate_bank(
            victim=args.victim,
            victim_speaker_id=speaker_id,
            converter=build_converter(args.backend),
            target_reference=reference,
            sources=sources,
            facts=facts,
            out_root=args.out or settings.attack_dir / "clones" / args.victim,
            allowed_victims=settings.clone_victims,
            attacker_label=args.attacker_label,
            overwrite=args.overwrite,
        )
    except BankError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()

    print(f"wrote {len(report.bank.clips)} clip(s) to {report.bank.root}")
    for w in report.warnings:
        print(f"warning: {w}")
    if report.uncovered_facts:
        print("no spoken answer yet for: " + ", ".join(report.uncovered_facts))
    print("next: annotate them in the main environment (python -m kavach.attacks.annotate_bank)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_make_clone_bank.py tests/test_attacks.py -p no:cacheprovider`
Expected: pass. (`test_the_cli_refuses...` sets `KAVACH_DATA_DIR`; `clone_victims` default `[]` makes it refuse.)

- [ ] **Step 6: Commit**

```bash
git add backend/kavach/attacks/clone.py backend/kavach/attacks/make_clone_bank.py tests/test_make_clone_bank.py
git commit -m "Add the stage-1 clone generator, consent-first, behind a VoiceConverter protocol" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Stage 2 annotator (`annotate_bank.py`)

**Files:**
- Create: `backend/kavach/attacks/annotate_bank.py`
- Test: `tests/test_annotate_bank.py` (new)

**Interfaces:**
- Consumes: `CloneBank`; `clone.screen_clone(clone, template, embedder, *, threshold) -> CloneQualityReport(similarity, threshold, admissible, ...)`; `asr.FOREIGN_SCRIPT`; duck-typed `asr.transcribe(audio, fast=False) -> Transcript`, `lid.tag_utterance(text, *, utterance_id, speaker_id, timings) -> UtteranceTokens` (+ optional `lid.last_llm_error`), `embedder.embed(Audio) -> SpeakerEmbedding`, `template: SpeakerTemplate`, `matcher.match(answer, expected) -> MatchResult(.score)`, `skg.get(predicate) -> Fact | None` (`.value`).
- Produces: `AnnotationReport(annotated, skipped, flagged)`, `annotate_bank(bank, *, speaker_id, asr, lid, embedder, template, matcher, skg, threshold, redo=False) -> AnnotationReport`, `main(argv) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_annotate_bank.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from clone_helpers import add_clip, new_bank, tone
from kavach.asr import Transcript, Word
from kavach.attacks.annotate_bank import annotate_bank
from kavach.attacks.bank import CloneBank
from kavach.csbg.ontology import Language, SemanticClass
from kavach.csbg.tokens import Token, UtteranceTokens
from kavach.embedding import SpeakerEmbedding, SpeakerTemplate
from kavach.skg import SpeakerKG


class FakeASR:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio, fast=False):
        self.calls += 1
        words = [Word(w, i * 300, i * 300 + 250) for i, w in enumerate(self.text.split())]
        return Transcript(text=self.text, words=words)


class FakeLID:
    last_llm_error = None

    def tag_utterance(self, text, *, utterance_id, speaker_id=None, timings=None):
        return UtteranceTokens(
            utterance_id=utterance_id,
            tokens=[Token(w, Language.EN, SemanticClass.OTHER, 0.9) for w in text.split()],
            speaker_id=speaker_id,
            transcript=text,
        )


class FakeEmbedder:
    """Every clip embeds to `vector`; the template decides admissibility."""

    def __init__(self, vector) -> None:
        self.vector = np.asarray(vector, dtype=float)

    def embed(self, audio):
        return SpeakerEmbedding(self.vector)


class FakeMatcher:
    def match(self, answer, expected):
        class R:
            score = 1.0 if expected.lower() in answer.lower() else 0.1

        return R()


def template(vector):
    return SpeakerTemplate.from_embeddings("spk_victim", [SpeakerEmbedding(np.asarray(vector, float))])


def skg_with(**facts) -> SpeakerKG:
    kg = SpeakerKG("spk_victim")
    for pred, value in facts.items():
        kg.add_fact(pred, value)
    return kg


def run(bank, *, text="my hometown is Thanjavur", embed=(1, 0, 0), tmpl=(1, 0, 0), asr=None, **kw):
    return annotate_bank(
        bank,
        speaker_id="spk_victim",
        asr=asr or FakeASR(text),
        lid=FakeLID(),
        embedder=FakeEmbedder(embed),
        template=template(tmpl),
        matcher=FakeMatcher(),
        skg=skg_with(hometown="Thanjavur"),
        threshold=0.62,
        **kw,
    )


def test_a_clip_gets_measured_scores(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    report = run(bank)
    clip = CloneBank.load(bank.root, allowed_victims=["S04"]).clips[0]
    assert report.annotated == 1
    assert clip.ecapa_similarity == pytest.approx(1.0) and clip.admissible is True
    assert clip.answer_score == 1.0 and clip.transcript == "my hometown is Thanjavur"
    assert clip.tokens and clip.tokens[0]["language"] == Language.EN.value


def test_a_clone_the_voiceprint_stops_is_inadmissible_but_still_recorded(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, embed=(0, 1, 0))  # orthogonal to the template
    clip = bank.clips[0]
    assert clip.admissible is False and clip.ecapa_similarity == pytest.approx(0.0)
    assert bank.yield_summary().yield_rate == 0.0


def test_a_wrong_answer_scores_low(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, text="no idea honestly")
    assert bank.clips[0].answer_score == pytest.approx(0.1)


def test_a_translated_transcript_is_flagged_and_excluded(tmp_path) -> None:
    """Whisper writing fluent English for Tamil speech is a fabricated choice."""
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)

    # Transcript has slots, so override the method on a subclass.
    class T(Transcript):
        __slots__ = ()

        def looks_translated(self):
            return "no Tamil share"

    class Translated(FakeASR):
        def transcribe(self, audio, fast=False):
            base = super().transcribe(audio, fast)
            return T(text=base.text, words=base.words)

    report = run(bank, asr=Translated("my hometown is Thanjavur"))
    assert report.flagged == [bank.clips[0].clip_id]
    assert bank.clips[0].flags and bank.measured() == []


def test_foreign_script_tokens_are_dropped(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=False)
    run(bank, text="my 안녕 hometown")
    assert all("안녕" != t["text"] for t in bank.clips[0].tokens)


def test_already_annotated_clips_are_skipped_unless_redone(tmp_path) -> None:
    bank = new_bank(tmp_path)
    add_clip(bank, annotated=True)
    asr = FakeASR("x")
    assert run(bank, asr=asr).skipped == 1 and asr.calls == 0
    assert run(bank, asr=asr, redo=True).annotated == 1 and asr.calls == 1


def test_progress_is_saved_after_every_clip(tmp_path) -> None:
    """A crash on clip 3 must not lose clips 1 and 2."""
    bank = new_bank(tmp_path)
    for _ in range(3):
        add_clip(bank, annotated=False)

    class DiesOnThird(FakeASR):
        def transcribe(self, audio, fast=False):
            if self.calls == 2:
                raise RuntimeError("boom")
            return super().transcribe(audio, fast)

    with pytest.raises(RuntimeError):
        run(bank, asr=DiesOnThird("my hometown is Thanjavur"))
    reloaded = CloneBank.load(bank.root, allowed_victims=["S04"])
    assert sum(c.annotated for c in reloaded.clips) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotate_bank.py -p no:cacheprovider`
Expected: collection ERROR `No module named 'kavach.attacks.annotate_bank'`.

- [ ] **Step 3: Implement**

Create `backend/kavach/attacks/annotate_bank.py`:

```python
"""Stage 2: transcribe, tag and score every clip in a bank.

    PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.attacks.annotate_bank --victim S04

Runs in the main environment, with the real Whisper, tagger, ECAPA template and
answer matcher -- the same components a live login uses, so a clip's numbers
are what the system would have said about it.

Progress is saved after every clip: a crash costs one clip, the same convention
as `kavach.annotate`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..asr import FOREIGN_SCRIPT
from .bank import BankError, CloneBank, check_allowed, resolve_speaker_id
from .clone import screen_clone


@dataclass(slots=True)
class AnnotationReport:
    annotated: int = 0
    skipped: int = 0
    flagged: list[str] = field(default_factory=list)


def _token_dict(t: Any) -> dict[str, Any]:
    """The stored-token wire shape (`schemas.Token`), without importing the API."""
    return {
        "text": t.text,
        "language": t.language.value,
        "semanticClass": t.semantic_class.value,
        "lidConfidence": t.lid_confidence,
        "startMs": t.start_ms,
        "endMs": t.end_ms,
    }


def annotate_bank(
    bank: CloneBank,
    *,
    speaker_id: str,
    asr: Any,
    lid: Any,
    embedder: Any,
    template: Any,
    matcher: Any,
    skg: Any,
    threshold: float,
    redo: bool = False,
) -> AnnotationReport:
    report = AnnotationReport()
    for clip in bank.clips:
        if clip.annotated and not redo:
            report.skipped += 1
            continue
        audio = bank.read_audio(clip)
        transcript = asr.transcribe(audio, fast=False)
        tokens = lid.tag_utterance(
            transcript.text,
            utterance_id=clip.clip_id,
            speaker_id=speaker_id,
            timings=transcript.timings,
        )
        # Hallucinated-script fragments are not words anyone said (HANDOFF).
        kept = [t for t in tokens.tokens if not FOREIGN_SCRIPT.search(t.text)]

        flags: list[str] = []
        translated = transcript.looks_translated()
        if translated:
            flags.append(f"looks_translated: {translated}")
        loop = transcript.repetition_loop()
        if loop:
            flags.append(f"repetition_loop: {loop[0]!r} x{loop[1]}")

        quality = screen_clone(audio, template, embedder, threshold=threshold)

        answer = None
        if clip.fact_key:
            fact = skg.get(clip.fact_key)
            if fact is not None:
                answer = float(matcher.match(transcript.text, fact.value).score)

        clip.transcript = transcript.text
        clip.tokens = [_token_dict(t) for t in kept]
        clip.annotation_source = "LLM" if getattr(lid, "last_llm_error", None) is None else "LEXICON"
        clip.ecapa_similarity = float(quality.similarity)
        clip.threshold = float(quality.threshold)
        clip.admissible = bool(quality.admissible)
        clip.answer_score = answer
        clip.flags = flags
        bank.save()  # after every clip: a crash costs one clip

        report.annotated += 1
        if flags:
            report.flagged.append(clip.clip_id)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m kavach.attacks.annotate_bank",
        description="Measure every clip in a clone bank with the real pipeline.",
    )
    p.add_argument("--victim", required=True)
    p.add_argument("--redo", action="store_true", help="Re-measure clips already annotated.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..api.pipeline import Pipeline
    from ..api.store import Store
    from ..config import Settings

    settings = Settings()
    try:
        check_allowed(args.victim, settings.clone_victims)
    except BankError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    store = Store(settings.db_path, settings.audio_dir)
    try:
        speaker_id = resolve_speaker_id(store.list_speakers(), args.victim)
        bank = CloneBank.load(
            settings.attack_dir / "clones" / args.victim,
            allowed_victims=settings.clone_victims,
            expected_speaker_id=speaker_id,
        )
        pipeline = Pipeline(store, settings)
        template = pipeline.load_template(speaker_id)
        if template is None or pipeline.embedder is None or pipeline.asr is None:
            print(
                "refused: the victim needs an enrolled voice template and the ECAPA and ASR "
                "models loaded (python -m kavach.prefetch, then enrol).",
                file=sys.stderr,
            )
            return 2
        report = annotate_bank(
            bank,
            speaker_id=speaker_id,
            asr=pipeline.asr,
            lid=pipeline.lid,
            embedder=pipeline.embedder,
            template=template,
            matcher=pipeline.matcher,
            skg=store.get_skg(speaker_id),
            threshold=settings.speaker_threshold,
            redo=args.redo,
        )
    except BankError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()

    summary = bank.yield_summary()
    print(f"annotated {report.annotated}, skipped {report.skipped}, flagged {len(report.flagged)}")
    if summary.yield_rate is not None:
        print(
            f"attack yield: {summary.admissible}/{summary.annotated} clones fooled the voiceprint "
            f"({summary.yield_rate:.0%}), from {summary.sources} source speaker(s)"
        )
    for cid in report.flagged:
        print(f"flagged (excluded from measured rows): {cid}")
    print("covered facts: " + (", ".join(bank.covered_facts()) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotate_bank.py -p no:cacheprovider`
Expected: pass. (`SpeakerKG.add_fact(predicate, value, *, raw_answer=..., ...)` was checked against `skg.py`.)

- [ ] **Step 5: Commit**

```bash
git add backend/kavach/attacks/annotate_bank.py tests/test_annotate_bank.py
git commit -m "Add the stage-2 annotator: measure every clone with the real pipeline" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The three gated routes and `Pipeline.clone_bank()`

**Files:**
- Modify: `backend/kavach/api/pipeline.py` (anchors: `from ..audio import Audio, AudioError, check_quality, decode_bytes, load_audio`; `        self.ledger = ChallengeLedger(`; `    def issue_challenge(self, speaker_id: str) -> Challenge:`)
- Modify: `backend/kavach/api/schemas.py` (insert before `class Health(Model):`; extend `__all__`)
- Modify: `backend/kavach/api/app.py` (import; insert routes before `    # ------------------------------------------------- challenge and auth`)
- Test: `tests/test_clone_bank_api.py` (extend)

**Interfaces:**
- Consumes: `CloneBank`, `BankError`, `resolve_speaker_id`, `BANK_FILE`; `Pipeline.ledger.get(id) -> Challenge | None` (`.consumed`, `.is_expired`, `.speaker_id`, `.expected_predicate`).
- Produces: `Pipeline.clone_bank() -> CloneBank | None` (raises `BankError`); routes `GET /api/clone-bank` → `CloneBankInfo`, `POST /api/clone-bank/match {challengeId}` → `CloneMatch`, `GET /api/clone-bank/{clip_id}/audio`; schemas `CloneClipInfo`, `CloneBankInfo`, `CloneMatchRequest`, `CloneMatch`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_clone_bank_api.py`:

```python
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


class TestGate:
    def test_every_route_is_a_404_while_the_flag_is_off_even_for_a_real_clip(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        clip = add_clip(bank)
        bank.save()
        off = settings.model_copy(update={"demo_attack_bank": False})
        pipeline.settings = off
        client.app.dependency_overrides[get_settings] = lambda: off
        cid = issue(client, speaker["id"])
        assert client.get("/api/clone-bank").status_code == 404
        assert client.post("/api/clone-bank/match", json={"challengeId": cid}).status_code == 404
        assert client.get(f"/api/clone-bank/{clip.clip_id}/audio").status_code == 404

    def test_no_other_route_serves_the_clone_text_while_off(self, tmp_path) -> None:
        client, store, pipeline, settings, speaker, bank = with_bank(tmp_path)
        add_clip(bank)
        bank.save()
        off = settings.model_copy(update={"demo_attack_bank": False})
        pipeline.settings = off
        client.app.dependency_overrides[get_settings] = lambda: off
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
        dest = bank.root
        save_wav(tone(seed=7), dest / clip.audio_path)  # changed after generation
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_api.py -p no:cacheprovider`
Expected: the new tests FAIL with 404s / missing routes.

- [ ] **Step 3: Add the schemas**

In `backend/kavach/api/schemas.py`, replace `class Health(Model):` with (note: the replaced line is re-added at the end of the inserted block):

```python
class CloneClipInfo(Model):
    """One clone, described by its measured scores. Never its words."""

    id: str
    attack_type: AttackTypeStr
    backend: str
    fact_key: str | None = None
    similarity: float | None = None
    admissible: bool | None = None
    duration_sec: float


class CloneBankInfo(Model):
    enabled: bool
    victim_speaker_id: str = ""
    clips: list[CloneClipInfo] = Field(default_factory=list)
    covered_facts: list[str] = Field(default_factory=list)
    n_generated: int = 0
    n_annotated: int = 0
    n_admissible: int = 0
    yield_rate: float | None = None
    n_sources: int = 0
    problems: list[str] = Field(default_factory=list)


class CloneMatchRequest(Model):
    challenge_id: str


class CloneMatch(Model):
    clip_id: str
    audio_url: str
    attack_type: AttackTypeStr
    backend: str
    similarity: float


class Health(Model):
```

and add `"CloneBankInfo", "CloneClipInfo", "CloneMatch", "CloneMatchRequest",` to `__all__` in alphabetical position.

- [ ] **Step 4: Add `Pipeline.clone_bank()`**

`pipeline.py` (single-line anchors): replace `from ..audio import Audio, AudioError, check_quality, decode_bytes, load_audio` with the same line plus

```python
from ..attacks.bank import BANK_FILE, BankError, CloneBank, resolve_speaker_id
```

replace `        self.ledger = ChallengeLedger(` ... careful: that line continues on the same statement; instead anchor on the single token line and add after the statement. Because the statement may span lines, use this instead: add the cache at the start of the class docstring-free `__init__` by replacing the line

`    def issue_challenge(self, speaker_id: str) -> Challenge:` with

```python
    def clone_bank(self) -> CloneBank | None:
        """The pre-generated clone bank, or None when the feature is off or no bank exists.

        Raises:
            BankError: A bank exists but cannot be trusted. Callers report it;
                they do not fall back silently, because a measured row quietly
                replaced by a modelled one looks exactly the same on screen.
        """
        if not self.settings.demo_attack_bank:
            return None
        cache = self.__dict__.setdefault("_bank_cache", {})
        for victim in self.settings.clone_victims:
            root = self.settings.attack_dir / "clones" / victim
            path = root / BANK_FILE
            if not path.exists():
                continue
            mtime = path.stat().st_mtime
            cached = cache.get(victim)
            if cached and cached[0] == mtime:
                return cached[1]
            bank = CloneBank.load(
                root,
                allowed_victims=self.settings.clone_victims,
                expected_speaker_id=resolve_speaker_id(self.store.list_speakers(), victim),
            )
            cache[victim] = (mtime, bank)
            return bank
        return None

    def issue_challenge(self, speaker_id: str) -> Challenge:
```

(`self.__dict__.setdefault` avoids editing the multi-line `__init__`; `Pipeline` is a plain class, not slotted — confirm with `grep -n "^class Pipeline" -A 3 backend/kavach/api/pipeline.py`.)

- [ ] **Step 5: Add the routes**

`app.py`: add to the imports (single-line anchor `from ..audio import AudioError, decode_bytes, warm_resample`) the line `from ..attacks.bank import BankError, CloneBank` before it. Then replace `    # ------------------------------------------------- challenge and auth` with:

```python
    # ------------------------------------------------------------ clone bank

    def _require_bank_enabled(cfg: Settings) -> None:
        """A clip *says* the answer, so these routes exist only in a demo build
        that announces itself (`/api/health` -> demoAttackBank). Off, they are a
        plain 404, indistinguishable from routes that do not exist."""
        if not cfg.demo_attack_bank:
            raise HTTPException(404, "Not found.")

    def _load_bank(pipeline: Pipeline) -> CloneBank | None:
        try:
            return pipeline.clone_bank()
        except BankError as exc:
            raise HTTPException(503, f"The clone bank cannot be used: {exc}") from exc

    @app.get("/api/clone-bank", response_model=schemas.CloneBankInfo)
    def clone_bank_info(pipeline: PipelineDep, cfg: SettingsDep) -> schemas.CloneBankInfo:
        _require_bank_enabled(cfg)
        bank = _load_bank(pipeline)
        if bank is None:
            return schemas.CloneBankInfo(
                enabled=True, problems=["No clone bank has been generated yet."]
            )
        summary = bank.yield_summary()
        problems: list[str] = []
        if summary.annotated < summary.generated:
            problems.append(
                f"{summary.generated - summary.annotated} clip(s) are not annotated yet and "
                "are not served; run kavach.attacks.annotate_bank."
            )
        return schemas.CloneBankInfo(
            enabled=True,
            victim_speaker_id=bank.victim_speaker_id,
            clips=[
                schemas.CloneClipInfo(
                    id=c.clip_id,
                    attack_type=conv.ATTACK_TO_WIRE[c.attack],
                    backend=c.backend,
                    fact_key=c.fact_key,
                    similarity=c.ecapa_similarity,
                    admissible=c.admissible,
                    duration_sec=c.duration_sec,
                )
                for c in bank.clips
                if c.usable
            ],
            covered_facts=bank.covered_facts(),
            n_generated=summary.generated,
            n_annotated=summary.annotated,
            n_admissible=summary.admissible,
            yield_rate=summary.yield_rate,
            n_sources=summary.sources,
            problems=problems,
        )

    @app.post("/api/clone-bank/match", response_model=schemas.CloneMatch)
    def clone_bank_match(
        payload: schemas.CloneMatchRequest, pipeline: PipelineDep, cfg: SettingsDep
    ) -> schemas.CloneMatch:
        """The attacker's pre-cloned answer to the challenge just issued.

        Realistic by construction: an attacker who knows every fact can
        pre-clone an answer to every possible question, so the bank is looked
        up *after* the random challenge is issued. The challenge is not
        consumed -- the clip is then submitted to /api/authenticate as usual.
        """
        _require_bank_enabled(cfg)
        challenge = pipeline.ledger.get(payload.challenge_id)
        if challenge is None:
            raise HTTPException(404, "Unknown challenge.")
        if challenge.consumed or challenge.is_expired:
            raise HTTPException(409, "That challenge is no longer valid; issue a new one.")
        bank = _load_bank(pipeline)
        if bank is None or bank.victim_speaker_id != challenge.speaker_id:
            raise HTTPException(404, "No clone bank exists for this speaker.")
        clip = bank.match(challenge.expected_predicate)
        if clip is None:
            covered = ", ".join(bank.covered_facts()) or "none"
            raise HTTPException(
                404,
                f"No cloned answer for this question ({challenge.expected_predicate}). "
                f"The bank covers: {covered}. Issue a new challenge.",
            )
        return schemas.CloneMatch(
            clip_id=clip.clip_id,
            audio_url=f"/api/clone-bank/{clip.clip_id}/audio",
            attack_type=conv.ATTACK_TO_WIRE[clip.attack],
            backend=clip.backend,
            similarity=float(clip.ecapa_similarity or 0.0),
        )

    @app.get("/api/clone-bank/{clip_id}/audio")
    def clone_bank_audio(clip_id: str, pipeline: PipelineDep, cfg: SettingsDep) -> FileResponse:
        _require_bank_enabled(cfg)
        bank = _load_bank(pipeline)
        clip = next((c for c in bank.clips if c.clip_id == clip_id and c.usable), None) if bank else None
        if clip is None:
            raise HTTPException(404, "Not found.")
        return FileResponse(bank.audio_file(clip), media_type="audio/wav")

    # ------------------------------------------------- challenge and auth
```

- [ ] **Step 6: Run to verify**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_api.py tests/test_api.py -p no:cacheprovider`
Expected: pass. If a gate test fails because `pipeline.settings = off` does not affect the already-built `Pipeline.settings` reference elsewhere, set both `pipeline.settings` and the dependency override as the tests do.

- [ ] **Step 7: Commit**

```bash
git add backend/kavach/api/pipeline.py backend/kavach/api/schemas.py backend/kavach/api/app.py tests/test_clone_bank_api.py
git commit -m "Serve the clone bank behind a flag that announces itself" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Measured rows in the Attack Lab

**Files:**
- Modify: `backend/kavach/api/attacks.py` (anchors below)
- Modify: `backend/kavach/api/schemas.py` (anchor `    the project's section 5.1.3."""`)
- Modify: `backend/kavach/api/converters.py` (`attack_run_to_wire`)
- Test: `tests/test_clone_bank_lab.py` (new)

**Interfaces:**
- Consumes: `Pipeline.clone_bank()`, `CloneBank.measured/yield_summary`, `CloneClip.tokens/ecapa_similarity/answer_score`, `converters.utterance_tokens_from_wire`.
- Produces: `AttackRun.acoustic_source: Literal["measured","modelled"]` (wire `acousticSource`); `attack_run_to_wire(..., acoustic_source="modelled")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_clone_bank_lab.py`:

```python
from __future__ import annotations

import pytest

from clone_helpers import add_clip, new_bank, token_dicts, tone
from kavach.api.attacks import run_attack
from kavach.api.pipeline import Pipeline
from kavach.api.store import Store
from kavach.attacks import AttackType
from kavach.audio import save_wav
from kavach.config import Settings
from test_api import ENGLISH_NUMBERS, TAMIL_NUMBERS, enrol_tokens, speaker_utterances

VICTIM = "S04"


def lab(tmp_path, *, bank_flag=True, clips=5, annotated=True, **clip_kw):
    settings = Settings(
        data_dir=tmp_path,
        audio_dir=tmp_path / "raw",
        attack_dir=tmp_path / "attacks",
        db_path=tmp_path / "kavach.db",
        demo_attack_bank=bank_flag,
        clone_victims=[VICTIM],
    )
    store = Store(settings.db_path, settings.audio_dir)
    victim = store.create_speaker({"display_name": f"{VICTIM} · Victim"})["id"]
    enrol_tokens(store, victim, speaker_utterances(TAMIL_NUMBERS, n=10))
    for i in range(3):
        other = store.create_speaker({"display_name": f"Other{i}"})["id"]
        enrol_tokens(store, other, speaker_utterances(ENGLISH_NUMBERS, n=10))
    pipeline = Pipeline(store, settings)
    bank = bank_at(settings, victim)
    for _ in range(clips):
        add_clip(bank, annotated=annotated, similarity=0.9, **clip_kw)
    bank.save()
    return store, pipeline, settings, victim, bank


def bank_at(settings, victim):
    from kavach.attacks.bank import CloneBank

    return CloneBank.new(
        settings.attack_dir / "clones" / VICTIM, VICTIM, victim, allowed_victims=[VICTIM]
    )


def a4(store, pipeline, victim, trials=40):
    return run_attack(
        attack=AttackType.A4_CLONE_KNOWLEDGE,
        speaker_id=victim,
        trials=trials,
        store=store,
        pipeline=pipeline,
    )


def test_a4_uses_the_bank_and_says_so(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert run.acoustic_source == "measured"
    assert any("Measured, not modelled" in n for n in run.notes)


def test_trials_are_capped_at_the_number_of_distinct_clips(tmp_path) -> None:
    """Resampling five clips to forty trials would give an interval narrower
    than the evidence."""
    store, pipeline, _, victim, _ = lab(tmp_path, clips=5)
    run = a4(store, pipeline, victim, trials=40)
    assert run.trials <= 5
    assert any("capped" in n.lower() or "distinct clips" in n for n in run.notes)


def test_yield_is_printed_with_the_row(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert any("Attack yield" in n for n in run.notes)
    assert run.yield_rate is not None


def test_the_run_is_still_simulated_and_not_paper_ready(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = a4(store, pipeline, victim)
    assert run.simulated is True
    assert any("paper" in n.lower() for n in run.notes)


def test_without_the_flag_the_run_is_modelled(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path, bank_flag=False)
    assert a4(store, pipeline, victim).acoustic_source == "modelled"


def test_a3_stays_modelled_because_the_bank_has_only_a4_clips(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path)
    run = run_attack(
        attack=AttackType.A3_CLONE, speaker_id=victim, trials=20, store=store, pipeline=pipeline
    )
    assert run.acoustic_source == "modelled"


def test_unannotated_clips_are_not_used(tmp_path) -> None:
    store, pipeline, _, victim, _ = lab(tmp_path, annotated=False)
    assert a4(store, pipeline, victim).acoustic_source == "modelled"


def test_a_tampered_bank_is_reported_not_silently_replaced(tmp_path) -> None:
    store, pipeline, settings, victim, bank = lab(tmp_path)
    save_wav(tone(seed=42), bank.audio_file(bank.clips[0]))
    run = a4(store, pipeline, victim)
    assert run.acoustic_source == "modelled"
    assert any("could not be used" in n for n in run.notes), run.notes


def test_a_bank_for_another_speaker_is_not_used(tmp_path) -> None:
    store, pipeline, settings, victim, bank = lab(tmp_path)
    other = store.create_speaker({"display_name": "S09 · Someone"})["id"]
    enrol_tokens(store, other, speaker_utterances(TAMIL_NUMBERS, n=10))
    run = run_attack(
        attack=AttackType.A4_CLONE_KNOWLEDGE, speaker_id=other, trials=20, store=store, pipeline=pipeline
    )
    assert run.acoustic_source == "modelled"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_lab.py -p no:cacheprovider`
Expected: FAIL — `AttackRun` has no `acoustic_source`.

- [ ] **Step 3: Wire schema and converter**

`schemas.py`: replace `    the project's section 5.1.3."""` with that line plus

```python

    acoustic_source: Literal["measured", "modelled"] = "modelled"
    """Where the acoustic column came from: 'measured' when pre-generated clones
    were scored by the real ECAPA model, 'modelled' when drawn from a documented
    distribution. Shown beside every run so a modelled row is never read as a
    measurement."""
```

`converters.py` `attack_run_to_wire`: add parameter `acoustic_source: str = "modelled",` after `simulated: bool = True,` and pass `acoustic_source=acoustic_source,` into `schemas.AttackRun(...)` after `simulated=simulated,`.

- [ ] **Step 4: Wire the lab**

`backend/kavach/api/attacks.py` (LF; use these exact edits):

1. Replace `from ..attacks.clone import CloneBatchStats` with
   `from ..attacks.bank import BankError, CloneClip` + newline + the original line.
2. Replace `    built: list[AttackTrial] = []` with:

```python
    bank_clips = _bank_clips(pipeline, speaker_id, attack, notes)
    acoustic_source = "measured" if bank_clips else "modelled"
    if bank_clips:
        trials = min(max(1, trials), len(bank_clips))
        notes.extend(_bank_notes(pipeline, attack, bank_clips))

    built: list[AttackTrial] = []
```

3. Replace `        probe = _probe_tokens(attack, victim_utterances, attacker_pool, style, rng)` with:

```python
        clip = bank_clips[i % len(bank_clips)] if bank_clips else None
        probe = (
            _bank_probe(clip, speaker_id)
            if clip is not None
            else _probe_tokens(attack, victim_utterances, attacker_pool, style, rng)
        )
```

4. Replace `        acoustic = model.draw(rng)` with `        acoustic = float(clip.ecapa_similarity) if clip is not None else model.draw(rng)`.
5. Replace `            knowledge_score=1.0 if _knows_answer(attack) else 0.0,` with:

```python
            knowledge_score=(
                clip.answer_score
                if clip is not None and clip.answer_score is not None
                else (1.0 if _knows_answer(attack) else 0.0)
            ),
```

6. Replace `            text_generator="template",` with `            text_generator="asr" if clip is not None else "template",`.
7. Replace `                "acoustic_source": "modelled",` with `                "acoustic_source": acoustic_source,`.
8. Replace `        run_id=new_id(f"atk_{attack.value}"),` with that line plus `        acoustic_source=acoustic_source,`.
9. Insert, directly before `def _victim_audio(`:

```python
def _bank_clips(
    pipeline: Pipeline, speaker_id: str, attack: AttackType, notes: list[str]
) -> list[CloneClip]:
    """Measured clones of this victim for this attack, or [] with a stated reason.

    A broken bank is reported in the run's notes, never swallowed: a measured
    row quietly replaced by a modelled one looks exactly the same on screen.
    """
    try:
        bank = pipeline.clone_bank()
    except BankError as exc:
        notes.append(f"The clone bank could not be used, so this run is modelled: {exc}")
        return []
    if bank is None or bank.victim_speaker_id != speaker_id:
        return []
    return bank.measured(attack)


def _bank_notes(pipeline: Pipeline, attack: AttackType, clips: list[CloneClip]) -> list[str]:
    bank = pipeline.clone_bank()
    summary = bank.yield_summary(attack) if bank is not None else None
    lines = [
        f"Measured, not modelled: this run used {len(clips)} pre-generated clone(s) of the "
        "victim's voice. The acoustic score is each clip's real ECAPA similarity to the "
        "enrolled template, the CSBG score is the real scorer over the real transcript of "
        "that clip, and the knowledge score is the real answer matcher."
    ]
    if summary is not None and summary.yield_rate is not None:
        lines.append(
            f"Attack yield: {summary.admissible}/{summary.annotated} clones fooled the "
            f"voiceprint ({summary.yield_rate:.0%}), from {summary.sources} source "
            "speaker(s). Clones the voiceprint stopped are excluded from the rates below, "
            "not counted as defended."
        )
    lines.append(
        "Trials are capped at the number of distinct clips: resampling a handful of clips "
        "to a larger count would give an interval narrower than the evidence. The run is "
        "still simulated -- one session per speaker, a same-sitting template and the demo's "
        "thresholds -- so `paper_ready()` refuses it."
    )
    return lines


def _bank_probe(clip: CloneClip, speaker_id: str) -> UtteranceTokens:
    """The clip's real, stored tokens as the scorer's input."""
    return conv.utterance_tokens_from_wire(
        clip.clip_id,
        [schemas.Token.model_validate(t) for t in (clip.tokens or [])],
        speaker_id=speaker_id,
        transcript=clip.transcript or "",
    )


def _victim_audio(
```

(Finish the last replacement so the original `def _victim_audio(` signature continues unchanged.)

- [ ] **Step 5: Run to verify**

Run: `.venv/Scripts/python.exe -m pytest tests/test_clone_bank_lab.py tests/test_api.py tests/test_attacks.py -p no:cacheprovider`
Expected: pass. A trial's token count may be below `min_scored_tokens` for the helper's two-token clips; `csbg_reliable` then marks the CSBG branch unavailable (existing behavior) and `run.trials` is unchanged. If `test_trials_are_capped` fails because `run.trials` counts only admissible cells, that is the intended accounting: assert `<= 5`.

- [ ] **Step 6: Commit**

```bash
git add backend/kavach/api/attacks.py backend/kavach/api/schemas.py backend/kavach/api/converters.py tests/test_clone_bank_lab.py
git commit -m "Let the Attack Lab score real clones, capped at the clips it has" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The demo button and the lab badge (frontend)

**Files:**
- Modify: `kavach/src/api/types.ts` (anchors `  yieldRate?: number | null;`, `export interface SpeakerIapmr {`)
- Modify: `kavach/src/api/client.ts` (line-1 import; `  health: async (): Promise<{ status: string, models: string[], device: string }> => {`; `  getSpeakers: async (): Promise<Speaker[]> => {`)
- Modify: `kavach/src/pages/Authenticate.tsx`, `kavach/src/pages/AttackLab.tsx` (LF)
- Verify: `npx tsc --noEmit`, `npx vite build`

**Interfaces:**
- Consumes: `/api/health` → `demoAttackBank`; `POST /api/clone-bank/match`; `AttackRun.acousticSource`.
- Produces: `CloneMatch` TS type; `apiClient.matchCloneClip(challengeId)`; a fourth demo button shown only when the bank is on.

There is no frontend test runner in this project; the checks are `tsc` and `vite build`, plus the Playwright check in Task 11.

- [ ] **Step 1: Types and client**

`types.ts`: replace `  yieldRate?: number | null;` with that line plus `  acousticSource?: 'measured' | 'modelled';`; replace `export interface SpeakerIapmr {` with

```ts
export interface CloneMatch {
  clipId: string;
  audioUrl: string;
  attackType: string;
  backend: string;
  similarity: number;
}

export interface SpeakerIapmr {
```

`client.ts`: replace the first line's import list to add `CloneMatch` (`import { AuthResult, Challenge, CSBG, EvalMetrics, Speaker, Utterance, Triple, AttackRun, AttackType, PerSpeakerIapmr, OfflineRun } from './types';` becomes the same with `, CloneMatch` before `}`); replace the `health:` signature line with `  health: async (): Promise<{ status: string, models: string[], device: string, demoAttackBank?: boolean }> => {`; replace `  getSpeakers: async (): Promise<Speaker[]> => {` with:

```ts
  /** The attacker's cloned answer to an issued challenge (demo builds only; 404 otherwise). */
  matchCloneClip: async (challengeId: string): Promise<CloneMatch> => {
    if (USE_MOCK) throw new Error('The clone bank needs the real backend.');
    return fetchApi('/api/clone-bank/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challengeId }),
    });
  },

  getSpeakers: async (): Promise<Speaker[]> => {
```

- [ ] **Step 2: `Authenticate.tsx`**

Replace `  const issue = useMutation({` with:

```tsx
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: apiClient.health });

  const issue = useMutation({
```

Replace the line
```
                  <DemoClips claimedId={speakerId} speakers={speakers ?? []} busy={auth.isPending}
```
with
```
                  <DemoClips claimedId={speakerId} speakers={speakers ?? []} busy={auth.isPending}
                    challengeId={challenge.id} cloneEnabled={!!health?.demoAttackBank}
```
In the `DemoClips` props type add `challengeId: string; cloneEnabled: boolean;` and destructure them: replace `function DemoClips({ claimedId, speakers, busy, onSubmit }: {` with `function DemoClips({ claimedId, speakers, busy, onSubmit, challengeId, cloneEnabled }: {` and replace `  claimedId: string;\n  speakers: import('../api/types').Speaker[];` with that plus `\n  challengeId: string;\n  cloneEnabled: boolean;`.

Replace `  const run = async (kind: 'replay' | 'impostor' | 'standin') => {` with `  const run = async (kind: 'replay' | 'impostor' | 'standin' | 'clone') => {`.

Replace `      const ownerId = kind === 'impostor' ? impostor : claimedId;` with:

```tsx
      if (kind === 'clone') {
        // The attacker's pre-cloned answer to the challenge just issued. A 404
        // here carries a useful reason ("the bank covers: ..."), shown below.
        const match = await apiClient.matchCloneClip(challengeId);
        onSubmit(
          await fetchExact(assetUrl(match.audioUrl)),
          `clone_${match.clipId}.wav`,
          `Clone attack · synthetic ${match.backend} clip of ${nameOf(claimedId)} (voiceprint ${match.similarity.toFixed(2)})`,
        );
        return;
      }
      const ownerId = kind === 'impostor' ? impostor : claimedId;
```

Replace

```
          Genuine stand-in
        </Button>
```
with
```
          Genuine stand-in
        </Button>
        {cloneEnabled && (
          <Button size="sm" variant="secondary" disabled={busy || !!preparing} loading={preparing === 'clone'} onClick={() => run('clone')}>
            Clone attack
          </Button>
        )}
```

and replace `className="grid grid-cols-1 sm:grid-cols-3 gap-2"` with `className={cn('grid grid-cols-1 gap-2', cloneEnabled ? 'sm:grid-cols-4' : 'sm:grid-cols-3')}`.

- [ ] **Step 2b: `AttackLab.tsx`**

Replace `{run.simulated ? ' · simulated' : ''}` with `{run.simulated ? ' · simulated' : ''}{run.acousticSource === 'measured' ? ' · acoustic measured' : ' · acoustic modelled'}`.

- [ ] **Step 3: Verify**

Run (from `speech/kavach`): `npx tsc --noEmit` — Expected: no output. `npx vite build` — Expected: `built in`. Then `git status --short` should list only the five frontend files; `kavach/dist` is already ignored/untracked per earlier builds (confirm it does not appear).

- [ ] **Step 4: Commit**

```bash
git add kavach/src
git commit -m "Add the clone-attack demo button, hidden unless the bank is on" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: The demo preflight (`demo_check`)

**Files:**
- Create: `backend/kavach/demo_check.py`
- Test: `tests/test_demo_check.py` (new)

**Interfaces:**
- Produces: `Check(name, level, detail, fix="")` with `level in {"PASS","WARN","FAIL"}`; `class TransportError(Exception)` (`.status`, `.detail`); `Transport` Protocol (`get(path)`, `post_json(path, body)`, `post_audio(challenge_id, wav, filename)`, `get_bytes(path)`); `HttpTransport(base)`; `run_checks(t, *, presenter, flows=False) -> list[Check]`; `render(checks) -> str`; `main(argv) -> int` (0 only when no FAIL).
- Consumes: the running backend's `/api/health`, `/api/speakers`, `/api/speakers/{id}/skg`, `/api/speakers/{id}/csbg`, `/api/speakers/{id}/utterances`, `/api/challenge`, `/api/authenticate`, `/api/clone-bank*`; `kavach.audio.decode_bytes/save_wav`.

Purpose: one command the presenter runs before going on stage. It asserts what the demo needs and names the fix for anything missing, so a problem is found at T-30 minutes, not in front of the audience.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_demo_check.py`:

```python
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
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
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
    "models": ["faster-whisper/small", "speechbrain/spkrec-ecapa-voxceleb",
               "gemini/gemini-3.1-flash-lite (tagging, challenges)", "sentence-transformers/LaBSE"],
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
    names = {c.name for c in run_checks(good({"/api/health": h}), presenter="S04") if c.level == WARN}
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
        ("AUTH", "impostor"): auth("REJECT", [b("signal_integrity", True), b("speaker_embedding", False, 0.1)]),
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


def test_a_genuine_stand_in_that_is_rejected_fails() -> None:
    bad = auth("REJECT", [b("speaker_embedding", False)])
    got = by_name(run_checks(flows(standin=bad), presenter="S04", flows=True))
    assert got["flow: genuine stand-in is accepted"].level == FAIL


def test_a_slow_login_warns() -> None:
    slow = auth("ACCEPT", [b("signal_integrity", True)], ms=60000)
    got = run_checks(flows(standin=slow), presenter="S04", flows=True)
    assert any(c.level == WARN and "slow" in c.name for c in got)


def test_render_ends_in_a_verdict() -> None:
    ok = render(run_checks(good(), presenter="S04"))
    assert ok.strip().splitlines()[-1].startswith("READY")
    bad = render(run_checks(Fake({"/api/health": OSError("x")}), presenter="S04"))
    assert bad.strip().splitlines()[-1].startswith("NOT READY")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_check.py -p no:cacheprovider`
Expected: collection ERROR `No module named 'kavach.demo_check'`.

- [ ] **Step 3: Implement**

Create `backend/kavach/demo_check.py`:

```python
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

#: Seconds above which a login is called slow. A cold first login took 31 s on
#: this laptop; past this an audience reads it as a hang.
SLOW_LOGIN_MS = 45_000

#: Other speakers needed for the CSBG's background model (api.pipeline.MIN_COHORT).
MIN_OTHERS = 3


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


def _static_checks(t: Transport, presenter: str) -> tuple[list[Check], dict[str, Any] | None, list[str]]:
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
        out.append(Check("challenge answers are hidden", WARN, "demoRevealAnswers is ON",
                         "unset KAVACH_DEMO_REVEAL_ANSWERS: anyone can read the answer in the network tab"))
    if health.get("reportable", {}).get("integrity_check_splice"):
        out.append(Check("splice detection is off", WARN, "it is ON",
                         "it rejects genuine phone audio; leave KAVACH_INTEGRITY_CHECK_SPLICE unset"))

    speakers = t.get("/api/speakers")
    me = _find(speakers, presenter)
    if me is None:
        out.append(Check("presenter exists", FAIL, f"no speaker {presenter!r}",
                         "run_demo.ps1 -Seed rebuilds the 12-speaker demo database"))
        return out, health, []
    out.append(Check("presenter exists", PASS, presenter))

    minutes = float(me.get("totalDurationSec", 0.0)) / 60.0
    out.append(Check("presenter has enrolment audio", PASS if minutes >= 5 else WARN,
                     f"{minutes:.1f} min", "" if minutes >= 5 else "add clips with the microphone top-up in the Speakers drawer"))

    facts = t.get(f"/api/speakers/{me['id']}/skg")
    out.append(
        Check(
            "presenter has knowledge-graph facts",
            PASS if facts else FAIL,
            f"{len(facts)} fact(s)",
            "" if facts else "open Speakers, choose the presenter, add facts (hometown, college, ...): "
            "without one a login cannot be challenged",
        )
    )
    try:
        t.get(f"/api/speakers/{me['id']}/csbg")
        out.append(Check("presenter has a code-switch graph", PASS, "present"))
    except TransportError as exc:
        out.append(Check("presenter has a code-switch graph", FAIL, str(exc), "rebuild the presenter in Speakers"))

    others = [s for s in speakers if s["id"] != me["id"]]
    out.append(Check("enough other speakers for the background model", PASS if len(others) >= MIN_OTHERS else FAIL,
                     f"{len(others)} other speaker(s)", "" if len(others) >= MIN_OTHERS else "run_demo.ps1 -Seed"))

    predicates = [f["predicate"] for f in facts]
    if health.get("demoAttackBank"):
        try:
            bank = t.get("/api/clone-bank")
        except TransportError as exc:
            out.append(Check("clone bank is usable", FAIL, str(exc),
                             "regenerate or re-annotate the bank (see DEMO_RUNBOOK.md)"))
        else:
            covered = set(bank.get("coveredFacts", []))
            missing = [p for p in predicates if p not in covered]
            out.append(
                Check(
                    "clone bank covers the presenter's facts",
                    PASS if not missing else WARN,
                    "all covered" if not missing else "no cloned answer for: " + ", ".join(missing),
                    "" if not missing else "the Clone attack button will say so for those questions; "
                    "record those answers and regenerate",
                )
            )
    else:
        out.append(Check("clone bank is enabled", WARN, "the Clone attack button will not appear",
                         "set KAVACH_DEMO_ATTACK_BANK=true and KAVACH_CLONE_VICTIMS in run_demo.ps1"))
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


def _long_clip(t: Transport, speaker_id: str) -> str | None:
    clips = [u for u in t.get(f"/api/speakers/{speaker_id}/utterances") if u["durationSec"] >= 8]
    return clips[0]["audioUrl"] if clips else None


def _flow_checks(t: Transport, speakers: list[dict[str, Any]], me: dict[str, Any], health: dict[str, Any]) -> list[Check]:
    out: list[Check] = []
    own_url = _long_clip(t, me["id"])
    others = [s for s in speakers if s["id"] != me["id"]]
    other_url = next((u for u in (_long_clip(t, s["id"]) for s in others) if u), None)
    if own_url is None or other_url is None:
        return [Check("flows can be staged", FAIL, "no stored clip of 8 s or more", "enrol more audio")]
    own, other = t.get_bytes(own_url), t.get_bytes(other_url)

    def login(wav: bytes, name: str) -> dict[str, Any]:
        cid = t.post_json("/api/challenge", {"speakerId": me["id"]})["id"]
        return t.post_audio(cid, wav, name)

    results: dict[str, dict[str, Any]] = {}
    for kind, wav in (("standin", _cut_wav(own, 2.0, 20.0)), ("replay", own), ("impostor", _cut_wav(other, 2.0, 20.0))):
        results[kind] = login(wav, f"{kind}.wav")

    r = results["standin"]
    out.append(Check("flow: genuine stand-in is accepted", PASS if r["decision"] == "ACCEPT" else FAIL,
                     f"{r['decision']} at {r['fusedScore']:.3f}",
                     "" if r["decision"] == "ACCEPT" else "re-enrol or add microphone clips; check the voiceprint branch"))
    r = results["replay"]
    gate = _branch(r, "signal_integrity")
    ok = r["decision"] == "REJECT" and gate is not None and not gate["passed"]
    out.append(Check("flow: replay is rejected at the integrity gate", PASS if ok else FAIL,
                     f"{r['decision']}; integrity gate {'tripped' if gate and not gate['passed'] else 'did not trip'}",
                     "" if ok else "the duplicate detector did not recognise a stored clip"))
    r = results["impostor"]
    voice, gate = _branch(r, "speaker_embedding"), _branch(r, "signal_integrity")
    ok = r["decision"] == "REJECT" and voice is not None and not voice["passed"] and (gate is None or gate["passed"])
    out.append(Check("flow: impostor is rejected by the voiceprint", PASS if ok else FAIL,
                     f"{r['decision']}; voice {'failed' if voice and not voice['passed'] else 'ok'}; "
                     f"integrity {'ok' if gate is None or gate['passed'] else 'TRIPPED'}",
                     "" if ok else "the voiceprint must decide, not the integrity gate: check splice detection is off"))

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
            out.append(Check("flow: clone attack", WARN, "no issued challenge had a cloned answer",
                             "regenerate the bank to cover more of the presenter's facts"))
        else:
            cid, m = matched
            res = t.post_audio(cid, t.get_bytes(m["audioUrl"]), "clone.wav")
            gate = _branch(res, "signal_integrity")
            ok = gate is None or gate["passed"]
            out.append(Check("flow: clone attack reaches the voiceprint", PASS if ok else FAIL,
                             f"{res['decision']} at {res['fusedScore']:.3f} (an outcome, not a pass/fail)",
                             "" if ok else "the integrity gate rejected a clone before the voiceprint ran"))
            results["clone"] = res

    for kind, res in results.items():
        ms = res.get("latencyMs") or 0
        if ms > SLOW_LOGIN_MS:
            out.append(Check(f"slow login: {kind}", WARN, f"{ms / 1000:.0f} s",
                             "warm the models first (the first login is the slow one)"))
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
    p = argparse.ArgumentParser(prog="python -m kavach.demo_check", description=__doc__.splitlines()[0])
    p.add_argument("--presenter", default="S04")
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--flows", action="store_true", help="Also drive the demo logins end to end.")
    args = p.parse_args(argv)
    checks = run_checks(HttpTransport(args.base), presenter=args.presenter, flows=args.flows)
    print(render(checks))
    return 1 if any(c.level == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_demo_check.py -p no:cacheprovider`
Expected: pass. The fake `post_audio` maps the file name's prefix (`standin.wav` → `standin`) to the scripted result; the `Fake.get_bytes` returns a 12 s tone, long enough for `_cut_wav`.

- [ ] **Step 5: Commit**

```bash
git add backend/kavach/demo_check.py tests/test_demo_check.py
git commit -m "Add a demo preflight that names the fix for anything missing" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: kNN-VC backend and the first real bank (spec stage 1b)

**This task needs the user at several points. Stop and ask at each marked step.**

**Files:**
- Create: `backend/kavach/attacks/backends/__init__.py`, `backend/kavach/attacks/backends/knn_vc.py`
- Create: `requirements-clone.txt`
- Create (not committed): `.venv-clone/`, `data/clone_sources/<fact>.wav`, `data/attacks/clones/S04/`
- Modify: `.gitignore` (add `.venv-clone/`)

**Interfaces:**
- Consumes: `VoiceConverter` Protocol (Task 4).
- Produces: `KnnVcConverter(*, device="cuda", topk=4, prematched=True)` with `name() -> "knn_vc"` and `convert(source, target_reference) -> Audio` (16 kHz mono).

- [ ] **Step 1: Write the module** (no model is loaded at import; it is covered by the manual smoke test below, not by the offline suite)

Create `backend/kavach/attacks/backends/__init__.py` containing only `"""Clone backends. Each imports its model library lazily."""`.

Create `backend/kavach/attacks/backends/knn_vc.py`:

```python
"""kNN-VC voice conversion (bshall/knn-vc): WavLM features + a HiFi-GAN vocoder.

Converts *speech* into the target speaker's voice by replacing each source frame
with the average of its nearest frames of the target's speech. The words, the
timing and the code-switching style are the source speaker's; only the voice
changes -- which is exactly attack A4.

Runs in the clone environment (`.venv-clone`, CUDA torch). `torch` is imported
inside the methods so this module imports anywhere; the first `convert` call
downloads the WavLM and vocoder checkpoints through torch hub.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ...audio import Audio, save_wav

OUTPUT_SAMPLE_RATE = 16_000


class KnnVcConverter:
    def __init__(self, *, device: str = "cuda", topk: int = 4, prematched: bool = True) -> None:
        self._device = device
        self._topk = topk
        self._prematched = prematched
        self._model = None

    def name(self) -> str:
        return "knn_vc"

    @property
    def model(self):
        if self._model is None:
            import torch

            self._model = torch.hub.load(
                "bshall/knn-vc",
                "knn_vc",
                prematched=self._prematched,
                trust_repo=True,
                pretrained=True,
                device=self._device,
            )
        return self._model

    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio:
        model = self.model
        with tempfile.TemporaryDirectory() as d:
            src_path = save_wav(source, Path(d) / "source.wav")
            ref_paths = [
                str(save_wav(a, Path(d) / f"ref_{i}.wav")) for i, a in enumerate(target_reference)
            ]
            query = model.get_features(str(src_path))
            matching_set = model.get_matching_set(ref_paths)
            wav = model.match(query, matching_set, topk=self._topk)
        samples = wav.detach().cpu().numpy().astype(np.float32).ravel()
        return Audio(samples, OUTPUT_SAMPLE_RATE, "knn_vc")
```

Add `.venv-clone/` to `.gitignore` (single new line after `.venv/` if present, else at the end under "Python").

- [ ] **Step 2: Offline suite still green, then commit**

Run: `.venv/Scripts/python.exe -m pytest tests/test_make_clone_bank.py tests/test_clone_bank.py -p no:cacheprovider` — Expected: pass (the module is not imported by the suite).

```bash
git add backend/kavach/attacks/backends .gitignore
git commit -m "Add the kNN-VC converter, lazily importing torch" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: ASK the user, then create the clone environment**

Ask: "May I create `.venv-clone` (Python 3.11) and install CUDA torch (about 2.5 GB download) plus the project's core requirements into it? I will not touch your miniconda environment." Check `nvidia-smi` and for a training `python.exe` first. On a yes:

```bash
py -3.11 -m venv .venv-clone
.venv-clone/Scripts/python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
.venv-clone/Scripts/python.exe -m pip install -r requirements-core.txt
.venv-clone/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a CUDA build and `True`. Write the same four commands into `requirements-clone.txt` as comments so the setup is reproducible, and commit it.

- [ ] **Step 4: ASK the user, then smoke-test the kNN-VC API**

Ask: "May I download the kNN-VC checkpoints (WavLM and the vocoder, a few hundred MB via torch hub) and run a ten-second conversion on the GPU?" On a yes run:

```bash
PYTHONPATH=backend .venv-clone/Scripts/python.exe - <<'PY'
import numpy as np
from kavach.attacks.backends.knn_vc import KnnVcConverter
from kavach.audio import Audio
sr = 16000
t = np.arange(sr * 4) / sr
src = Audio((0.3 * np.sin(2 * np.pi * 140 * t)).astype(np.float32), sr, "s")
ref = [Audio((0.3 * np.sin(2 * np.pi * 210 * t)).astype(np.float32), sr, f"r{i}") for i in range(3)]
out = KnnVcConverter().convert(src, ref)
print("ok", out.sample_rate, round(len(out.samples) / out.sample_rate, 1), "s")
PY
```
Expected: `ok 16000 ...`. **If the hub entry point, argument names or `get_features` / `get_matching_set` / `match` differ from this file, read the model's README (`torch.hub` cache, `bshall_knn-vc_master/README.md`) and fix the three calls in `convert`**, then re-run this smoke test. This is the one place the plan depends on an external API that could not be inspected offline.

- [ ] **Step 5: ASK the user for the prerequisites**

State plainly what is needed and wait:
1. The presenter enters their SKG facts in Speakers (hometown, college, favouriteFood, ... at least three). They are personal details only the presenter can supply.
2. A teammate speaks one short answer per fact, in their own Tamil-English style, once. Save each as `data/clone_sources/<predicate>.wav` (16-bit WAV, or any format ffmpeg reads, converted to WAV). Check the predicates with `PYTHONPATH=backend .venv/Scripts/python.exe -c "import sqlite3;print(sorted({r[0] for r in sqlite3.connect('data/kavach.db').execute('select predicate from facts')}))"`.
3. Confirm the presenter (S04) agrees to being cloned, and set `KAVACH_CLONE_VICTIMS='["S04"]'` (in the shell for these commands; `run_demo.ps1` gets it in Task 11).

- [ ] **Step 6: Generate the bank (ASK first: GPU run)**

```bash
KAVACH_CLONE_VICTIMS='["S04"]' PYTHONPATH=backend .venv-clone/Scripts/python.exe -m kavach.attacks.make_clone_bank --victim S04 --sources data/clone_sources --backend knn_vc
```
Expected: `wrote N clip(s) to data\attacks\clones\S04`, a `warning:` that 5.7 minutes is under the recommended 5 (it is above; if it prints, believe it), and a line naming any fact with no spoken answer. Listen to two clips; if they are unintelligible, stop and report: that is a finding about kNN-VC on this data, not something to paper over.

---

### Task 11: Annotate, wire the demo, rehearse, and document (spec stage 1c)

**Files:**
- Modify: `run_demo.ps1` (CRLF; single-line anchor `$env:KAVACH_CSBG_VETO_ENABLED = "false"`)
- Create: `DEMO_RUNBOOK.md`
- Modify: `HANDOFF.md`, `README.md`
- Verify: `kavach.demo_check --flows` x3, full suite, Playwright UI check

- [ ] **Step 1: ASK, then annotate the bank (real Whisper / tagger / ECAPA)**

Ask first ("this loads Whisper small, ECAPA and calls Gemini for a few clips"); check `nvidia-smi`. Then:

```bash
KAVACH_CLONE_VICTIMS='["S04"]' KAVACH_DEMO_ATTACK_BANK=true PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.attacks.annotate_bank --victim S04
```
Expected: `annotated N`, `attack yield: a/b ...`, and `covered facts: ...`. Report the yield honestly whatever it is. A yield of 0 is a finding ("the voiceprint stops these clones"), not a failure of the work; say so and show the user.

- [ ] **Step 2: Wire the demo script**

In `run_demo.ps1`, replace the single line `$env:KAVACH_CSBG_VETO_ENABLED = "false"` with that line plus

```powershell
# The clone-attack demo: the bank holds the presenter's cloned answers and its
# routes return audio that SAYS the answer, so this build announces itself in
# /api/health (demoAttackBank). Only the presenter has agreed to be cloned.
$env:KAVACH_DEMO_ATTACK_BANK = "true"
$env:KAVACH_CLONE_VICTIMS = '["S04"]'
```

- [ ] **Step 3: ASK, then run the preflight three times**

Ask to start the real backend (models load, GPU used). Start it as `run_demo.ps1` does (backend only is enough: `KAVACH_WARM_MODELS_ON_START=true KAVACH_CSBG_VETO_ENABLED=false KAVACH_DEMO_ATTACK_BANK=true KAVACH_CLONE_VICTIMS='["S04"]' .venv/Scripts/python.exe -m uvicorn kavach.api.app:app --port 8000 --app-dir backend`, run in the background, log to a file). Wait for `/api/health`. Then:

```bash
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.demo_check --presenter S04 --flows
```
Run it **three times in a row**, and once more after restarting the backend (a cold start). Expected: `READY` each time, no `FAIL`. Any `FAIL` or any run-to-run disagreement is a bug to fix with the debugging skill before continuing, not a thing to note. Record each run's login latencies.

- [ ] **Step 4: Drive the UI once**

With the Vite dev server running (`npm run dev` in `kavach/`), run this Playwright check with `C:\Users\baves\miniconda3\python.exe` (read-only use of that interpreter; no installs). Save it outside the repo (e.g. `%TEMP%\kavach_scratch\ui_clone.py`):

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_context(viewport={"width": 1600, "height": 1000}).new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://localhost:3000/authenticate", wait_until="networkidle")
    page.locator("select").first.select_option(index=1)  # pick the presenter in the dropdown
    page.get_by_role("button", name="Issue challenge").click()
    page.wait_for_timeout(1500)
    print("clone button visible:", page.get_by_role("button", name="Clone attack").count() == 1)
    page.get_by_role("button", name="Clone attack").click()
    page.get_by_text("Clone attack · synthetic").wait_for(timeout=120000)
    page.screenshot(path="clone_attack.png")
    print("page errors:", errors)
    b.close()
```
Adjust the speaker selection to choose `S04` by its label. Expected: `clone button visible: True`, the result panel shows the "Clone attack · synthetic" source label, and `page errors: []`. Then restart the backend with `KAVACH_DEMO_ATTACK_BANK` unset and confirm the button is **absent** (`count() == 0`).

- [ ] **Step 5: Rehearse the scrutiny drills and write the runbook**

Create `DEMO_RUNBOOK.md` with these sections, filling the *Observed* column from what you actually see in this step (do not write an outcome you did not observe):

1. **T-30 minutes**: `run_demo.ps1 -Prefetch` once (only if LaBSE is missing), `run_demo.ps1`, wait for the green status, `python -m kavach.demo_check --presenter S04 --flows` must end in `READY`.
2. **Demo order** (from `DEMO_PLAN.md` section 5, with the clone step now real).
3. **Drills** — a table `Drill | How | Observed`: wrong file type (a `.txt` renamed `.wav`); a 0.3 s clip; 3 s of silence; a 44.1 kHz m4a (first non-16 kHz upload after a cold start); a 40 s clip; resubmit the same challenge twice; wait out the 60 s challenge; unplug the network and log in (Gemini down); two logins at once; refresh mid-login; backend killed mid-demo and restarted (time to recover). Run each against the live backend and record the status code, the on-screen message and the time.
4. **Honest answers**: what to say when asked whether the CSBG works (50% EER on 5 free-speech speakers, one session each, so the claim is not supported and the demo shows the system, not that claim); why the first login is slow; why Tamil ASR output is imperfect with the small Whisper model; that A3-A5 beyond the A4 clip are modelled and labelled so; that the clone bank has one attacker and one victim.
5. **Never do on stage**: enable `demo_reveal_answers`; upload audio longer than 30 s expecting it all to count; skip the preflight.

Mic drills need the presenter's real microphone (browser WebM/Opus). State clearly in the runbook that the live-microphone path was **not** exercised by this plan and must be rehearsed by the presenter once, on the demo laptop, before presenting.

- [ ] **Step 6: Documentation**

`README.md`: one line under "Where to start": `| **[DEMO_RUNBOOK.md](DEMO_RUNBOOK.md)** | Preflight, rehearsed drills, and honest answers for presenting the demo. |`, and one sentence in the "What is not in this repository" table's `data/` row noting `data/attacks/clones/` holds synthetic clips of a consenting presenter and is never to be distributed.

`HANDOFF.md`: add a section "Update 2026-10-02 (b) -- clone bank and demo preflight" recording: what was built (commands, bank layout, flags), the measured yield and covered facts from Step 1, the three preflight runs' results, the drills' observed outcomes, what remains (stage 2 IndicF5 behind its probe; the live-mic rehearsal; the stale deck screenshots), and the test count from Step 7.

- [ ] **Step 7: Full suite, verification, commit**

Run: `.venv/Scripts/python.exe -m pytest -p no:cacheprovider` — Expected: all pass; record the count in HANDOFF.md and README.md (they state a count). Run `cd kavach && npx tsc --noEmit && npx vite build`. Stop the backend and Vite servers **by the PID listening on :8000 and :3000**, never by name:

```bash
for port in 8000 3000; do pid=$(netstat -ano | grep -E ":$port .*LISTENING" | awk '{print $5}' | head -1); [ -n "$pid" ] && taskkill //PID "$pid" //T //F; done
```

Restore the demo DB if the preflight left test logins in it that you do not want in the Evaluation history (back it up first with sqlite's backup API, as in the 2026-10-02 session; compare table row counts after restoring).

```bash
git add run_demo.ps1 DEMO_RUNBOOK.md HANDOFF.md README.md requirements-clone.txt
git commit -m "Wire the clone-attack demo, rehearse the drills, and write the runbook" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (spec section → task):
- §1 outcome / yield beside every rate → Tasks 7, 6 (yield in `CloneBankInfo`), 5 (printed).
- §2 prerequisites (S04 facts, teammate, HF terms, separate env) → Global Constraints, Task 10 steps 3-5. HF terms belong to stage 2 (not in this plan).
- §3.1 `VoiceConverter`, `bank.py`, backends, both CLIs → Tasks 4, 3, 10, 4, 5.
- §3.2 bank format, loader refusals, A5 text-to-speech only → Task 3 (refusals, tests), Task 4 (A4 only).
- §3.3 settings fail closed → Task 2.
- §3.4 routes, 404 gating, leak surface, match semantics, best-similarity tie-break → Task 6.
- §3.5 lab: measured columns, trial cap, yield, `simulated` stays → Task 7.
- §3.6 UI button, lab badge → Task 8.
- §4 safety: allowlist, pseudonym only, synthetic file names, gitignored, routes off by default, `paper_ready()` untouched → Tasks 2, 3, 4, 6, 7.
- §5 staging 0-1c → Tasks 10-11; stage 2 deferred as the spec says.
- §6 testing: bank, yield, routes, leak sweep, lab, generator/annotator with fakes → Tasks 3-7; no model in the suite.
- §7 risks → surfaced in Task 4 warnings, Task 5 flags, Task 7 notes, Task 10 step 6.
- **Added by the user at approval:** demo robustness → Tasks 1, 9, 11 (probe findings, preflight, drills, runbook).

**Placeholder scan:** the only deliberate "fill in from observation" content is the runbook's *Observed* column and HANDOFF's measured figures (Task 11), which cannot be known before the run; the steps say exactly how to obtain them. `<what the probe found...>` in Task 1's commit command is a commit-body prompt for the implementer, not code.

**Type consistency:** `CloneBank.new/load/save/match/measured/covered_facts/yield_summary/read_audio/audio_file`, `CloneClip` field names, `BankError`, `check_allowed`, `resolve_speaker_id`, `AUDIO_DIR`, `BANK_FILE` are defined in Task 3 and used with those exact names in Tasks 4-7. `Pipeline.clone_bank()` (Task 6) is what Task 7 calls. `acoustic_source` is the same snake_case name in `schemas.AttackRun`, `attack_run_to_wire`, the lab, and (camelCase) `acousticSource` in TypeScript. `AttackType.value` strings in `bank.json`, `ATTACK_TO_WIRE` on the wire.

**Known soft spots, stated rather than hidden:** (1) kNN-VC's torch-hub API was recalled, not inspected offline, so Task 10 step 4 is a smoke test with an instruction to correct the three calls. (2) The live browser microphone path (WebM/Opus from the real mic) cannot be exercised by this plan; the runbook says the presenter must rehearse it once.
