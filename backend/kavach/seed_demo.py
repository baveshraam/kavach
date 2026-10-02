"""Build the demo database from the recorded corpora.

    python -m kavach.seed_demo                      # corpus_v2 + corpus_v3
    python -m kavach.seed_demo --no-templates       # skip ECAPA (fast)

Every consented speaker in the manifests becomes an enrolled speaker, with
their real audio, their real transcripts and the tokens the annotation pass
already produced. **Nothing is re-transcribed or re-tagged**: annotation is an
ASR pass plus an LLM call per utterance and is not deterministic, so the
manifest tags are data, and replaying them is the only way the demo shows the
same graphs the offline experiments scored.

Utterances the annotation pass excluded (Whisper translated instead of
transcribing) are skipped with their reason, for the same reason the
experiment runner skips them: their tokens are language choices nobody made.

The existing database is moved aside, never deleted. Knowledge-graph facts
typed into the old database are carried over to the same corpus speaker, since
those were entered by hand and are the one thing here that cannot be rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf

from .api.pipeline import Pipeline
from .api.store import Store
from .config import Settings, get_settings
from .skg import SpeakerKG

DEFAULT_CORPORA = ("corpus_v2", "corpus_v3")

#: Old demo rows were named "S08 (consented)"; the corpus id is the prefix.
_OLD_NAME_PREFIX_LEN = 3


def _consented(data_dir: Path) -> set[str]:
    path = data_dir / "consent_register.csv"
    if not path.exists():
        return set()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return {
        row["speaker_id"]
        for row in csv.DictReader(lines)
        if (row.get("consent_given") or "").strip().upper() == "YES"
    }


def _folders(data_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("speakers.csv", "speakers_free.csv"):
        path = data_dir / name
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    out[row["speaker_id"]] = row["folder"]
    return out


def _dominant(tokens: list[dict[str, Any]]) -> str:
    ta = sum(1 for t in tokens if t.get("language") == "TA")
    en = sum(1 for t in tokens if t.get("language") == "EN")
    if ta + en == 0:
        return "Balanced"
    share = ta / (ta + en)
    return "Tamil" if share >= 0.6 else "English" if share <= 0.3 else "Balanced"


def _old_facts(db_path: Path) -> dict[str, list[tuple[str, str, str, float, int]]]:
    """Facts from a previous demo database, keyed by corpus speaker id."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        names = dict(conn.execute("SELECT id, display_name FROM speakers"))
        out: dict[str, list[tuple[str, str, str, float, int]]] = {}
        for sid, pred, value, raw, conf, ver in conn.execute(
            "SELECT speaker_id, predicate, value, raw_answer, confidence, verified FROM facts"
        ):
            corpus_id = (names.get(sid) or "")[:_OLD_NAME_PREFIX_LEN]
            if corpus_id:
                out.setdefault(corpus_id, []).append((pred, value, raw, conf, ver))
        return out
    finally:
        conn.close()


def _move_aside(settings: Settings) -> Path | None:
    if not settings.db_path.exists():
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = settings.data_dir / f"demo_backup_{stamp}"
    backup.mkdir(parents=True)
    shutil.move(str(settings.db_path), backup / settings.db_path.name)
    if settings.audio_dir.exists():
        shutil.move(str(settings.audio_dir), backup / settings.audio_dir.name)
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    return backup


def seed(corpora: list[str], *, templates: bool = True, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    data_dir = settings.data_dir
    consented = _consented(data_dir)
    folders = _folders(data_dir)
    carried = _old_facts(settings.db_path)

    backup = _move_aside(settings)
    if backup:
        print(f"previous demo database moved to {backup}")

    store = Store(settings.db_path, settings.audio_dir)
    pipeline = Pipeline(store, settings)

    created: dict[str, str] = {}
    for corpus in corpora:
        root = data_dir / corpus
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        free = manifest.get("provenance") == "RECORDED"
        by_speaker: dict[str, list[dict[str, Any]]] = {}
        for utt in manifest["utterances"]:
            by_speaker.setdefault(utt["speaker_id"], []).append(utt)

        for sid, utts in sorted(by_speaker.items()):
            if sid not in consented:
                print(f"  {sid}: skipped, no consent on record")
                continue
            if sid in created:
                continue
            all_tokens = [t for u in utts for t in u.get("tokens") or []]
            folder = folders.get(sid, "")
            row = store.create_speaker(
                {
                    "display_name": f"{sid} · {folder}" if folder else sid,
                    "age_range": "18-25",
                    "dominant_language": _dominant(all_tokens),
                    "device": "phone",
                    "environment": "free speech" if free else "scripted reading",
                    "consent_given": True,
                }
            )
            spk = row["id"]
            created[sid] = spk

            kept = skipped = 0
            for utt in utts:
                if utt.get("excluded_reason"):
                    skipped += 1
                    continue
                path = root / utt["audio_path"]
                info = sf.info(str(path))
                store.add_utterance(
                    speaker_id=spk,
                    type="free-speech" if free else "code-mixed",
                    audio_bytes=path.read_bytes(),
                    extension=path.suffix,
                    duration_sec=float(utt.get("duration_sec") or info.duration),
                    sample_rate=int(info.samplerate),
                    transcript=utt.get("transcript", ""),
                    tokens=utt.get("tokens") or [],
                    annotated=bool(utt.get("tokens")),
                )
                kept += 1

            if sid in carried:
                kg = SpeakerKG(spk)
                for pred, value, raw, conf, _ver in carried[sid]:
                    kg.add_fact(pred, value, raw_answer=raw, confidence=conf)
                store.put_skg(kg)

            print(f"  {sid} -> {spk}: {kept} utterances" + (f", {skipped} excluded" if skipped else ""))

    # CSBGs after every speaker exists: the background model for each is the
    # other eleven, and fitting as we go would give early speakers a thinner one.
    print("building CSBGs ...")
    for sid, spk in created.items():
        pipeline.build_csbg(spk)

    if templates:
        print("building ECAPA templates (CPU, a few minutes) ...")
        for sid, spk in created.items():
            t = pipeline.build_template(spk)
            label = f"self-consistency {t.self_consistency:.2f}" if t else "unavailable"
            print(f"  {sid}: {label}")

    print(f"done: {len(created)} speakers in {settings.db_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", action="append", help="corpus folder under data/ (repeatable)")
    parser.add_argument("--no-templates", action="store_true", help="skip ECAPA templates")
    args = parser.parse_args(argv)
    seed(list(args.corpus or DEFAULT_CORPORA), templates=not args.no_templates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
