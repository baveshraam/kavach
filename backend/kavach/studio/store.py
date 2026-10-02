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
