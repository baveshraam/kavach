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
