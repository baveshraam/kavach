"""Persistence: SQLite for records, the filesystem for audio.

SQLite because the whole system is one researcher on one laptop with tens of
speakers, and a Postgres dependency would buy nothing but a setup step. Audio
stays on disk rather than in BLOB columns so the corpus can be inspected,
copied and archived with ordinary tools -- and so that deleting a row and
deleting a recording are two visible operations rather than one silent one.

WHAT IS STORED, AND WHAT IS DERIVED
-----------------------------------
Stored: speakers, utterances (with their annotated tokens), knowledge-graph
facts, authentication history, attack runs, and fitted CSBGs.

Derived on read: every code-mixing statistic on a `Speaker`. Those come from
the CSBG, which comes from the tokens. Caching them in the speakers table
would create a second source of truth that goes stale the moment an utterance
is deleted, and a stale CMI is worse than a recomputed one because it looks
right.

Annotated tokens ARE stored, even though they are derived from the audio. Not
for speed: re-deriving them means re-running ASR and an LLM tagging pass over
the whole corpus, which costs money and is not deterministic. The annotation
is data.

⚠️  PRIVACY
-----------
This database holds voiceprints next to hometowns, schools and family names.
`kavach.skg` sets out the constraints; the two that bind here are that the
file is not encrypted (encrypting it is the application's decision, and
`data/` is git-ignored) and that `delete_speaker` must erase everything --
audio, tokens, facts, graphs and history -- not just the speaker row. A
deletion request that leaves the recordings behind has not been honoured.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..csbg.graph import CSBG
from ..skg import SpeakerKG
from .converters import iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
    id                TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL,
    age_range         TEXT DEFAULT '',
    gender            TEXT DEFAULT '',
    dominant_language TEXT DEFAULT 'Balanced',
    other_languages   TEXT DEFAULT '[]',
    device            TEXT DEFAULT '',
    environment       TEXT DEFAULT '',
    consent_given     INTEGER DEFAULT 0,
    enrolled_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    id           TEXT PRIMARY KEY,
    speaker_id   TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    type         TEXT NOT NULL,
    filename     TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    sample_rate  INTEGER NOT NULL,
    transcript   TEXT DEFAULT '',
    tokens       TEXT DEFAULT '[]',
    annotated    INTEGER DEFAULT 0,
    recorded_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_speaker ON utterances(speaker_id);

CREATE TABLE IF NOT EXISTS facts (
    speaker_id TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    predicate  TEXT NOT NULL,
    value      TEXT NOT NULL,
    raw_answer TEXT DEFAULT '',
    confidence REAL DEFAULT 1.0,
    verified   INTEGER DEFAULT 0,
    PRIMARY KEY (speaker_id, predicate)
);

CREATE TABLE IF NOT EXISTS graphs (
    speaker_id TEXT PRIMARY KEY REFERENCES speakers(id) ON DELETE CASCADE,
    payload    TEXT NOT NULL,
    built_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS templates (
    speaker_id TEXT PRIMARY KEY REFERENCES speakers(id) ON DELETE CASCADE,
    payload    TEXT NOT NULL,
    built_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_history (
    id         TEXT PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    payload    TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_time ON auth_history(timestamp DESC);

CREATE TABLE IF NOT EXISTS attack_runs (
    id           TEXT PRIMARY KEY,
    speaker_id   TEXT NOT NULL,
    attack_type  TEXT NOT NULL,
    payload      TEXT NOT NULL,
    generated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attack_time ON attack_runs(generated_at DESC);
"""


def new_id(prefix: str) -> str:
    """Short prefixed identifier, matching the frontend's `spk_`/`utt_` shape."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class StoreError(RuntimeError):
    """A requested record does not exist or cannot be written."""


class Store:
    """Records in SQLite, audio on disk.

    Thread-safe by way of one connection guarded by a lock. FastAPI runs sync
    route handlers in a threadpool, so a bare `sqlite3` connection shared
    across them would raise `ProgrammingError` on the second concurrent
    request. A connection pool would be the scalable answer; a lock is the
    right answer for a single-researcher system, and it cannot deadlock
    because no operation here takes a second lock.
    """

    def __init__(self, db_path: str | Path, audio_dir: str | Path) -> None:
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Without this SQLite accepts the REFERENCES clauses above and then
        # ignores them, so deleting a speaker would silently orphan their
        # utterances instead of cascading. It is per-connection, not per-file.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------- plumbing

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- speakers

    def create_speaker(self, data: dict[str, Any]) -> dict[str, Any]:
        """Insert a speaker. Returns the stored row."""
        speaker_id = data.get("id") or new_id("spk")
        row = {
            "id": speaker_id,
            "display_name": data.get("display_name", ""),
            "age_range": data.get("age_range", ""),
            "gender": data.get("gender", ""),
            "dominant_language": data.get("dominant_language", "Balanced"),
            "other_languages": json.dumps(list(data.get("other_languages") or [])),
            "device": data.get("device", ""),
            "environment": data.get("environment", ""),
            "consent_given": int(bool(data.get("consent_given", False))),
            "enrolled_at": data.get("enrolled_at") or iso(),
        }
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO speakers (id, display_name, age_range, gender, "
                "dominant_language, other_languages, device, environment, "
                "consent_given, enrolled_at) VALUES (:id, :display_name, :age_range, "
                ":gender, :dominant_language, :other_languages, :device, "
                ":environment, :consent_given, :enrolled_at)",
                row,
            )
        return self.get_speaker(speaker_id)  # type: ignore[return-value]

    def get_speaker(self, speaker_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
        return self._speaker_row(row) if row else None

    def list_speakers(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM speakers ORDER BY enrolled_at DESC"
            ).fetchall()
        return [self._speaker_row(r) for r in rows]

    @staticmethod
    def _speaker_row(row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["other_languages"] = json.loads(out.get("other_languages") or "[]")
        out["consent_given"] = bool(out.get("consent_given"))
        return out

    def delete_speaker(self, speaker_id: str) -> bool:
        """Erase a speaker and everything derived from them.

        Audio files are removed before the rows, so a crash mid-delete leaves
        orphaned rows pointing at missing files -- visible and repairable --
        rather than orphaned recordings with nothing referencing them, which
        nobody would ever find.
        """
        with self._lock:
            files = self._conn.execute(
                "SELECT filename FROM utterances WHERE speaker_id = ?", (speaker_id,)
            ).fetchall()

        for f in files:
            path = self.audio_dir / f["filename"]
            path.unlink(missing_ok=True)

        with self._tx() as conn:
            conn.execute("DELETE FROM auth_history WHERE speaker_id = ?", (speaker_id,))
            conn.execute("DELETE FROM attack_runs WHERE speaker_id = ?", (speaker_id,))
            cur = conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
        return cur.rowcount > 0

    # ----------------------------------------------------------- utterances

    def add_utterance(
        self,
        *,
        speaker_id: str,
        type: str,
        audio_bytes: bytes,
        extension: str,
        duration_sec: float,
        sample_rate: int,
        transcript: str = "",
        tokens: list[dict[str, Any]] | None = None,
        annotated: bool = False,
    ) -> dict[str, Any]:
        """Write audio to disk and record the utterance.

        Raises:
            StoreError: If the speaker does not exist. Recording audio against
                a missing speaker would leave a file nothing references.
        """
        if self.get_speaker(speaker_id) is None:
            raise StoreError(f"Unknown speaker {speaker_id!r}; create it before recording.")

        utt_id = new_id("utt")
        filename = f"{speaker_id}__{utt_id}{extension}"
        (self.audio_dir / filename).write_bytes(audio_bytes)

        row = {
            "id": utt_id,
            "speaker_id": speaker_id,
            "type": type,
            "filename": filename,
            "duration_sec": float(duration_sec),
            "sample_rate": int(sample_rate),
            "transcript": transcript,
            "tokens": json.dumps(tokens or []),
            "annotated": int(bool(annotated)),
            "recorded_at": iso(),
        }
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO utterances (id, speaker_id, type, filename, duration_sec, "
                "sample_rate, transcript, tokens, annotated, recorded_at) "
                "VALUES (:id, :speaker_id, :type, :filename, :duration_sec, "
                ":sample_rate, :transcript, :tokens, :annotated, :recorded_at)",
                row,
            )
        return self.get_utterance(utt_id)  # type: ignore[return-value]

    def get_utterance(self, utterance_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM utterances WHERE id = ?", (utterance_id,)
            ).fetchone()
        return self._utterance_row(row) if row else None

    def list_utterances(self, speaker_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if speaker_id:
                rows = self._conn.execute(
                    "SELECT * FROM utterances WHERE speaker_id = ? ORDER BY recorded_at",
                    (speaker_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM utterances ORDER BY recorded_at DESC"
                ).fetchall()
        return [self._utterance_row(r) for r in rows]

    def _utterance_row(self, row: sqlite3.Row) -> dict[str, Any]:
        out = dict(row)
        out["tokens"] = json.loads(out.get("tokens") or "[]")
        out["annotated"] = bool(out.get("annotated"))
        # The frontend puts this straight into an <audio src>, so it must be a
        # URL the browser can fetch, not a filesystem path.
        out["audio_url"] = f"/api/audio/{out['id']}"
        return out

    def audio_path(self, utterance_id: str) -> Path:
        """On-disk location of an utterance's audio.

        Raises:
            StoreError: If the utterance is unknown or its file is missing.
        """
        row = self.get_utterance(utterance_id)
        if row is None:
            raise StoreError(f"Unknown utterance {utterance_id!r}.")
        path = self.audio_dir / row["filename"]
        if not path.exists():
            raise StoreError(
                f"Audio for {utterance_id!r} is recorded in the database but missing "
                f"from disk at {path}. The corpus and the database have diverged."
            )
        return path

    def update_annotation(
        self,
        utterance_id: str,
        *,
        transcript: str,
        tokens: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Attach an ASR transcript and annotated tokens to an utterance."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE utterances SET transcript = ?, tokens = ?, annotated = 1 "
                "WHERE id = ?",
                (transcript, json.dumps(tokens), utterance_id),
            )
        return self.get_utterance(utterance_id)

    def delete_utterance(self, utterance_id: str) -> bool:
        row = self.get_utterance(utterance_id)
        if row is None:
            return False
        (self.audio_dir / row["filename"]).unlink(missing_ok=True)
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM utterances WHERE id = ?", (utterance_id,))
        return cur.rowcount > 0

    # ------------------------------------------------------------------ SKG

    def get_skg(self, speaker_id: str) -> SpeakerKG:
        """A speaker's knowledge graph, empty if they have no facts yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM facts WHERE speaker_id = ? ORDER BY predicate",
                (speaker_id,),
            ).fetchall()
        kg = SpeakerKG(speaker_id)
        for r in rows:
            kg.add_fact(
                r["predicate"],
                r["value"],
                raw_answer=r["raw_answer"] or "",
                confidence=r["confidence"],
                verified=bool(r["verified"]),
            )
        return kg

    def put_skg(self, kg: SpeakerKG) -> SpeakerKG:
        """Replace a speaker's facts wholesale.

        Replace rather than merge because the editor sends the full set: a
        merge would make deleting a fact in the UI impossible, since the
        deleted row simply would not appear in the payload.
        """
        with self._tx() as conn:
            conn.execute("DELETE FROM facts WHERE speaker_id = ?", (kg.speaker_id,))
            conn.executemany(
                "INSERT INTO facts (speaker_id, predicate, value, raw_answer, "
                "confidence, verified) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        kg.speaker_id,
                        f.predicate,
                        f.value,
                        f.raw_answer,
                        f.confidence,
                        int(f.verified),
                    )
                    for f in kg.facts
                ],
            )
        return self.get_skg(kg.speaker_id)

    # ----------------------------------------------------------------- CSBG

    def save_csbg(self, graph: CSBG) -> None:
        payload = json.dumps(graph.to_dict())
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO graphs (speaker_id, payload, built_at) VALUES (?, ?, ?) "
                "ON CONFLICT(speaker_id) DO UPDATE SET payload = excluded.payload, "
                "built_at = excluded.built_at",
                (graph.speaker_id, payload, iso()),
            )

    def load_csbg(self, speaker_id: str) -> CSBG | None:
        """A stored CSBG, or None if never built.

        Returns None rather than raising when the stored graph was built under
        an older ontology: `CSBG.from_dict` refuses to load it (comparing
        graphs across ontology versions compares different class indices), and
        the caller's correct response is to rebuild from the tokens, which is
        the same response as "never built".
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM graphs WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        if row is None:
            return None
        try:
            return CSBG.from_dict(json.loads(row["payload"]))
        except (ValueError, KeyError):
            return None

    def all_csbgs(self) -> dict[str, CSBG]:
        """Every stored graph, keyed by speaker. Feeds the background model."""
        with self._lock:
            rows = self._conn.execute("SELECT speaker_id, payload FROM graphs").fetchall()
        out: dict[str, CSBG] = {}
        for r in rows:
            try:
                out[r["speaker_id"]] = CSBG.from_dict(json.loads(r["payload"]))
            except (ValueError, KeyError):
                continue
        return out

    # ------------------------------------------------------- voice template

    def save_template(self, speaker_id: str, payload: dict[str, Any]) -> None:
        """Store an ECAPA speaker template (`SpeakerTemplate.to_dict()`)."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO templates (speaker_id, payload, built_at) VALUES (?, ?, ?) "
                "ON CONFLICT(speaker_id) DO UPDATE SET payload = excluded.payload, "
                "built_at = excluded.built_at",
                (speaker_id, json.dumps(payload), iso()),
            )

    def load_template(self, speaker_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM templates WHERE speaker_id = ?", (speaker_id,)
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    # --------------------------------------------------------- auth history

    def record_auth(self, speaker_id: str, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO auth_history (id, speaker_id, payload, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (
                    payload["id"],
                    speaker_id,
                    json.dumps(payload),
                    payload.get("timestamp") or iso(),
                ),
            )

    def list_auth(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM auth_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    # ---------------------------------------------------------- attack runs

    def record_attack(self, speaker_id: str, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO attack_runs (id, speaker_id, attack_type, payload, "
                "generated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    payload["id"],
                    speaker_id,
                    payload["attackType"],
                    json.dumps(payload),
                    payload.get("generatedAt") or iso(),
                ),
            )

    def list_attacks(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM attack_runs ORDER BY generated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]


__all__ = ["SCHEMA", "Store", "StoreError", "new_id"]
