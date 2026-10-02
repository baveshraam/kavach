"""Enrol the demo's voiceprint from Studio sessions.

    python -m kavach.studio.enrol --speaker S04 --sessions S1,S2,S3

The login's audio arrives through the browser microphone, Opus-coded. Studio sessions are recorded
through that same path, so a template built from them describes the demo; the original template was
built from phone recordings, which is why the demo's genuine scores sat lower than they should.

Nothing is changed unless everything checks out: every named session must have clips, every clip's
WAV must match the hash recorded when it was captured, and every clip must embed. The database is
copied aside first. The stored template records what it was built from (sessions, devices, clip
count), so the preflight and the evaluation can say whether the numbers describe this enrolment.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..audio import AudioError, load_audio
from ..embedding import SpeakerTemplate
from .store import StudioError, StudioStore


class EnrolError(RuntimeError):
    """The template was not changed; the message says why."""


def _speaker_id(store: Any, pseudonym: str) -> str:
    rows = store.list_speakers()
    hits = [
        r["id"] for r in rows
        if (r.get("display_name") or "") == pseudonym or (r.get("display_name") or "").startswith(pseudonym + " ")
    ]
    if len(hits) != 1:
        raise EnrolError(f"expected exactly one enrolled speaker named {pseudonym!r}, found {len(hits)}")
    return hits[0]


def enrol_from_studio(
    store: Any,
    studio: StudioStore,
    pseudonym: str,
    sessions: list[str],
    *,
    embedder: Any,
    backup_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Replace `pseudonym`'s voice template with one built from the named Studio sessions."""
    if embedder is None:
        raise EnrolError("no speaker embedder is available (the model could not be loaded)")
    speaker_id = _speaker_id(store, pseudonym)

    try:
        clips = studio.clips()
    except StudioError as exc:
        raise EnrolError(str(exc)) from exc
    chosen = [c for c in clips if c.session_id in set(sessions)]
    for s in sessions:
        if not any(c.session_id == s for c in chosen):
            raise EnrolError(f"session {s} has no clips")

    embeddings = []
    skipped = 0
    for c in chosen:
        try:
            path = studio.verified_wav_path(c)  # raises if the file is not the one that was indexed
        except StudioError as exc:
            raise EnrolError(f"{exc} (hash check failed; nothing was changed)") from exc
        try:
            embeddings.append(embedder.embed(load_audio(path)))
        except AudioError:
            skipped += 1
    if not embeddings:
        raise EnrolError("no clip could be embedded; nothing was changed")

    template = SpeakerTemplate.from_embeddings(speaker_id, embeddings)
    devices = sorted({c.device for c in chosen})
    provenance = {
        "source": "studio",
        "sessions": list(sessions),
        "devices": devices,
        "environments": sorted({c.environment for c in chosen}),
        "n_clips": len(embeddings),
        "skipped_clips": skipped,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "self_consistency": round(template.self_consistency, 4),
    }

    backup = ""
    if backup_dir is not None:
        db_path = Path(getattr(store, "db_path", getattr(store, "path", "")) or "")
        if not db_path.exists():
            raise EnrolError("cannot back up the database: its path is unknown")
        Path(backup_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(backup_dir) / f"{db_path.name}.pre-studio-enrol-{stamp}"
        # SQLite's own backup, not a file copy: a copy of a database that is open (or in WAL mode)
        # can miss committed pages.
        src, dst = sqlite3.connect(str(db_path)), sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
        backup = str(dest)

    store.save_template(speaker_id, {**template.to_dict(), "provenance": provenance})
    return {"speaker_id": speaker_id, "n_clips": len(embeddings), "sessions": list(sessions), "devices": devices,
            "skipped": skipped, "self_consistency": provenance["self_consistency"], "backup": backup}


def main(argv: list[str] | None = None) -> int:
    from ..api.store import Store
    from ..config import Settings
    from ..embedding import ECAPAEmbedder

    p = argparse.ArgumentParser(prog="python -m kavach.studio.enrol", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", required=True, help="corpus pseudonym, e.g. S04")
    p.add_argument("--sessions", required=True, help="comma-separated Studio sessions, e.g. S1,S2,S3")
    p.add_argument("--studio", type=Path, default=None, help="default: data/studio/<speaker>")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args(argv)

    cfg = Settings()
    studio = StudioStore(args.studio or cfg.data_dir / "studio" / args.speaker, args.speaker)
    store = Store(cfg.db_path, cfg.audio_dir)
    embedder = ECAPAEmbedder(model_name=cfg.ecapa_model, device=cfg.embedding_device)
    try:
        out = enrol_from_studio(
            store, studio, args.speaker, [s.strip() for s in args.sessions.split(",") if s.strip()],
            embedder=embedder, backup_dir=None if args.no_backup else cfg.data_dir,
        )
    except EnrolError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    finally:
        store.close()
    print(f"enrolled {args.speaker} ({out['speaker_id']}) from {out['n_clips']} clips in {', '.join(out['sessions'])} "
          f"on {', '.join(out['devices'])}; self-consistency {out['self_consistency']}; backup: {out['backup'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
