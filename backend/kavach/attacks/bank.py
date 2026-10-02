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
            bank.verified_audio_file(clip)
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

    def verified_audio_file(self, clip: CloneClip) -> Path:
        """The clip's file, after proving it is the one that was screened.

        Called on load AND every time a clip is served: the loader's result is
        cached, so a hash checked only at load time would let a file replaced or
        deleted afterwards be served (or raise a 500).
        """
        file = self.audio_file(clip)
        if not file.exists():
            raise BankError(f"clip {clip.clip_id}: audio file is missing ({clip.audio_path})")
        if sha256_file(file) != clip.sha256:
            raise BankError(
                f"clip {clip.clip_id}: audio does not match its recorded hash -- "
                "the file was changed after it was generated"
            )
        return file

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
