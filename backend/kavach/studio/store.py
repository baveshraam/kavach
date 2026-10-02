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
import logging
import os
import re
import threading
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
#: An upload larger than this is refused before it is decoded: 120 s of 16 kHz WAV is ~4 MB.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

logger = logging.getLogger(__name__)

#: One writer at a time: two tabs, or two worker threads, must not interleave on the index.
_WRITE_LOCK = threading.Lock()

#: `S` and a number only. `CON`, `NUL` and `COM1` are reserved device names on Windows, and
#: `s4` and `S4` are the same folder on NTFS but two different ids in the index.
_SESSION = re.compile(r"^S[0-9]{1,3}$")
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


def _fsync_path(path: Path) -> None:
    """Flush a just-written file to disk, so 'the index is written last' survives a power cut."""
    with path.open("r+b") as fh:
        fh.flush()
        os.fsync(fh.fileno())


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
        chunks = path.read_text(encoding="utf-8").split("\n")
        tail = chunks.pop()  # "" when the file ends with a newline, else a possibly partial line
        if tail.strip():
            try:
                ClipRecord(**json.loads(tail))
                chunks.append(tail)
            except (json.JSONDecodeError, TypeError):
                logger.warning("%s ends in a partial line (a write was interrupted); ignored", path.name)
        out: list[ClipRecord] = []
        for n, line in enumerate(chunks, 1):
            if not line.strip():
                continue
            try:
                out.append(ClipRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                raise StudioError(f"{path.name} line {n} is not a valid clip record: {exc}") from exc
        return out

    def _repair_tail(self) -> None:
        """Before appending, make sure the index ends on a line boundary.

        A crash mid-write leaves a partial last line. Appending after it would glue the next
        record onto the fragment and corrupt both, so the fragment is moved to
        `index.quarantine.jsonl` (kept, not discarded) and the index is cut back to its last
        complete line. Call with `_WRITE_LOCK` held.
        """
        path = self.index_path
        if not path.exists():
            return
        data = path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n") + 1
        tail = data[cut:]
        try:
            ClipRecord(**json.loads(tail.decode("utf-8")))
            with path.open("ab") as fh:  # a complete record that only lacked its newline
                fh.write(b"\n")
            return
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            pass
        with (self.root / "index.quarantine.jsonl").open("ab") as fh:
            fh.write(tail + b"\n")
        with path.open("r+b") as fh:
            fh.truncate(cut)
        logger.warning("quarantined a partial index line (%d bytes)", len(tail))

    def verified_wav_path(self, clip: ClipRecord) -> Path:
        """The clip's WAV, after proving it is the one that was indexed."""
        path = self.wav_path(clip)
        if not path.exists():
            raise StudioError(f"clip {clip.clip_id}: the WAV is missing")
        if _sha256(path) != clip.sha256_wav:
            raise StudioError(
                f"clip {clip.clip_id}: the WAV does not match its recorded hash -- it was changed after recording"
            )
        return path

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
            raise StudioError("a session id is 'S' and a number, for example S1 or S4")
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
        if quality.is_too_long:
            raise StudioError(
                f"the recording is longer than {MAX_CLIP_SEC:.0f}s ({audio.duration_sec:.0f}s); "
                "was the recorder left running? Record it again."
            )

        ext = orig_ext.lower() if _EXT.match(orig_ext.lower()) else ".bin"
        clip_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
        folder = self.root / session_id
        wav_path = folder / f"{clip_id}.wav"

        with _WRITE_LOCK:
            # One sitting is one device in one room. A page refresh used to carry the rest of a
            # sitting into the next session id; a mismatch here is the symptom, so refuse it.
            earlier = [c for c in self.clips() if c.session_id == session_id]
            if earlier and (earlier[0].device, earlier[0].environment) != (device, environment):
                raise StudioError(
                    f"session {session_id} was recorded on {earlier[0].device} in "
                    f"{earlier[0].environment}; a session is one sitting, one device, one room. "
                    "Start a new session to change either."
                )
            save_wav(audio, wav_path)
            _fsync_path(wav_path)
            orig_path = folder / f"{clip_id}.orig"
            orig_path.write_bytes(orig_bytes)
            _fsync_path(orig_path)

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
            self._repair_tail()
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
