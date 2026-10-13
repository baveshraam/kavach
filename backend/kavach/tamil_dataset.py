"""Index the enrolled speaker's recorded Tamil words.

    python -m kavach.tamil_dataset --speaker S04

The words are recorded in the Studio (Settings.studio_enabled, `/studio?mode=tamil`), one clip per word, through
the same browser microphone path the login hears. This turns those clips into `data/tamil_words/words.jsonl`,
one row per clip:

    word_id, text, romanised, gloss, language ("ta"), transcription (what the Tamil decode wrote),
    heard (did that decode write exactly this word), asr_model, audio_ref (path under data/), clip_id,
    session_id, sha256_wav, duration_sec, recorded_at, device, environment

The audio itself stays in the git-ignored Studio store; this file holds references and text only, and lives
under `data/`, which is git-ignored as a whole. What the rows are for:

* Enrolment and evaluation data -- does the recogniser hear THIS speaker say THIS word? Words it cannot hear are
  excluded from the live challenge (`Pipeline._mixed_pool`), because a word that fails for the genuine speaker
  only makes the genuine speaker fail.
* NOT identity. No login compares a probe with these clips, and nothing here changes the voiceprint or the
  voice threshold. A replayed word recording proves nothing, and `heard` is a statement about the recogniser.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from .audio import load_audio
from .mixed_phrase import MIXED_TA_PROMPT
from .tamil_words import BANK, by_id, normalise_tamil, tamil_same, words_index_path


def heard_exactly(expected: str, transcription: str) -> bool:
    """True if the Tamil decode wrote the word and nothing else Tamil: every Tamil token is that word (a repeat is fine)."""
    tokens = [t for t in (normalise_tamil(p) for p in transcription.replace("-", " ").split()) if t]
    return bool(tokens) and all(tamil_same(expected, t) for t in tokens)


def build_dataset(studio: Any, asr: Any, data_dir: Path | str, *, model_name: str = "") -> dict[str, Any]:
    """Transcribe every Studio clip of kind 'tamil' and write words.jsonl. Returns a summary."""
    bank = by_id()
    clips = [c for c in studio.clips() if c.kind == "tamil"]
    if not clips:
        raise ValueError("no clips of kind 'tamil' have been recorded yet (Studio: mode=tamil)")
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []
    unknown: list[str] = []
    for c in clips:
        word = bank.get(c.prompt_id)
        if word is None:
            unknown.append(c.prompt_id)
            continue
        wav = studio.verified_wav_path(c)  # raises if the file is not the one that was indexed
        text = asr.transcribe(load_audio(wav), language="ta", initial_prompt=MIXED_TA_PROMPT, fast=True).text
        try:
            ref = wav.resolve().relative_to(data_dir.resolve()).as_posix()
        except ValueError:
            ref = wav.as_posix()
        rows.append({
            **word.to_dict(), "transcription": text, "heard": heard_exactly(word.ta, text), "asr_model": model_name,
            "audio_ref": ref, "clip_id": c.clip_id, "session_id": c.session_id, "sha256_wav": c.sha256_wav,
            "duration_sec": c.duration_sec, "recorded_at": c.recorded_at, "device": c.device, "environment": c.environment,
        })
    out = words_index_path(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, out)  # a reader never sees half a file
    heard_ids = {r["word_id"] for r in rows if r["heard"]}
    recorded_ids = {r["word_id"] for r in rows}
    return {
        "clips": len(rows), "words_recorded": len(recorded_ids), "words_in_bank": len(BANK),
        "words_heard": len(heard_ids), "words_not_heard": sorted(recorded_ids - heard_ids),
        "unknown_prompt_ids": unknown, "by_session": dict(Counter(r["session_id"] for r in rows)), "index": str(out),
    }


def main(argv: list[str] | None = None) -> int:
    from .asr import WhisperASR
    from .config import Settings
    from .studio.store import StudioError, StudioStore

    p = argparse.ArgumentParser(prog="python -m kavach.tamil_dataset", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", default="S04")
    p.add_argument("--studio", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = Settings()
    model = cfg.mixed_whisper_model or cfg.whisper_model
    studio = StudioStore(args.studio or cfg.data_dir / "studio" / args.speaker, args.speaker)
    asr = WhisperASR(model_size=model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type,
                     language=cfg.whisper_language, suppress_numerals=cfg.suppress_numerals)
    try:
        s = build_dataset(studio, asr, cfg.data_dir, model_name=model)
    except (ValueError, StudioError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"{s['clips']} clips of {s['words_recorded']}/{s['words_in_bank']} words; the recogniser ({model}) heard the exact word for "
          f"{s['words_heard']} of {s['words_recorded']}. Index: {s['index']}")
    if s["words_not_heard"]:
        print("not heard (excluded from live challenges): " + ", ".join(s["words_not_heard"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
