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
