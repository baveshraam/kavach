"""Does the configured speech recognition follow Tamil-English code-switched speech?

    python -m kavach.asr_audit --speaker S04 --models configured,large-v3

Runs real recordings with a known reference through the ASR exactly as the login would call it, under
three decode settings, and reports what survives: Tamil words, English words, and the failure modes
(translated to English, Tamil romanised, Tamil written for English, foreign-script hallucination).

The reference text is the Studio clip's `text_hint`, so this measures the sentence regime. It does NOT
measure a list of English words with two isolated Tamil words in it (the mixed challenge): that audio
does not exist until the enrolled speaker records it. Results go to data/tamil_words/audit/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .asr import CODE_MIX_PROMPT, FOREIGN_SCRIPT, TAMIL_SCRIPT

_PIECE = re.compile(r"[஀-௿]+|[A-Za-z][A-Za-z']*")
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿"), None)

#: name -> (language, initial_prompt). `login_en` is what `Pipeline._verify_phrase` uses today.
SETTINGS: dict[str, tuple[str | None, str | None]] = {
    "login_en": ("en", ""),
    "auto_mixprompt": (None, CODE_MIX_PROMPT),
    "forced_ta_mixprompt": ("ta", CODE_MIX_PROMPT),
    "forced_ta_noprompt": ("ta", ""),   # what the mixed login's Tamil decode uses (MIXED_TA_PROMPT)
}


def pieces(text: str) -> list[str]:
    """Tamil-script runs and Latin runs, NFC-normalised, Latin lower-cased. Digits and punctuation drop out."""
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE)
    return [p.lower() if p[0].isascii() else p for p in _PIECE.findall(text)]


def is_tamil(piece: str) -> bool:
    return bool(TAMIL_SCRIPT.match(piece))


def _best(piece: str, pool: list[str]) -> float:
    return max((SequenceMatcher(None, piece, p).ratio() for p in pool), default=0.0)


def score_clip(reference: str, hypothesis: str) -> dict[str, Any]:
    """Per-clip recall of the reference's Tamil and English words, and what the hypothesis did instead."""
    ref, hyp = pieces(reference), pieces(hypothesis)
    ref_ta, ref_en = [p for p in ref if is_tamil(p)], [p for p in ref if not is_tamil(p)]
    hyp_ta, hyp_en = [p for p in hyp if is_tamil(p)], [p for p in hyp if not is_tamil(p)]
    out: dict[str, Any] = {
        "ref_ta": len(ref_ta), "ref_en": len(ref_en), "hyp_ta": len(hyp_ta), "hyp_en": len(hyp_en),
        # English reference words heard as Latin words (fuzzy 0.8, the login's own tolerance).
        "en_found": sum(_best(p, hyp_en) >= 0.8 for p in ref_en),
        # Tamil reference words heard as the identical Tamil string, and as a near string (0.8).
        "ta_exact": sum(p in hyp_ta for p in ref_ta),
        "ta_near": sum(_best(p, hyp_ta) >= 0.8 for p in ref_ta),
        "foreign_script": bool(FOREIGN_SCRIPT.search(hypothesis)),
    }
    # Failure mode, in order of severity.
    if not hyp:
        mode = "empty"
    elif out["foreign_script"]:
        mode = "foreign_script_hallucination"
    elif ref_ta and not hyp_ta:
        mode = "no_tamil_script_at_all"          # translated to English or romanised Tamil
    elif ref_en and not hyp_en:
        mode = "english_written_in_tamil"        # English words transliterated into Tamil script
    elif out["en_found"] == len(ref_en) and out["ta_near"] == len(ref_ta):
        mode = "ok"
    else:
        mode = "partial"
    out["mode"] = mode
    return out


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ta, en = sum(r["ref_ta"] for r in rows), sum(r["ref_en"] for r in rows)
    return {
        "clips": n,
        "english_word_recall": round(sum(r["en_found"] for r in rows) / en, 3) if en else None,
        "tamil_word_recall_exact": round(sum(r["ta_exact"] for r in rows) / ta, 3) if ta else None,
        "tamil_word_recall_near": round(sum(r["ta_near"] for r in rows) / ta, 3) if ta else None,
        "clips_fully_ok": sum(r["mode"] == "ok" for r in rows),
        "modes": dict(Counter(r["mode"] for r in rows)),
    }


def load_reference_clips(speaker: str, data_dir: Path, kind: str, limit: int | None):
    from .studio.store import StudioStore

    studio = StudioStore(data_dir / "studio" / speaker, speaker)
    clips = [c for c in studio.clips() if c.kind == kind and c.text_hint.strip()]
    return studio, clips[:limit] if limit else clips


def audit(models: list[str], speaker: str, kind: str, limit: int | None, out_dir: Path) -> dict[str, Any]:
    from .asr import WhisperASR
    from .audio import load_audio
    from .config import Settings

    cfg = Settings()
    studio, clips = load_reference_clips(speaker, cfg.data_dir, kind, limit)
    if not clips:
        raise SystemExit(f"no '{kind}' clips with reference text for {speaker}")
    report: dict[str, Any] = {"speaker": speaker, "kind": kind, "n_clips": len(clips),
                              "configured_model": cfg.whisper_model, "per_model": {}}
    audio = {c.clip_id: load_audio(studio.verified_wav_path(c)) for c in clips}
    for name in models:
        size = cfg.whisper_model if name == "configured" else name
        asr = WhisperASR(model_size=size, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type,
                         language=cfg.whisper_language, suppress_numerals=cfg.suppress_numerals)
        per_setting: dict[str, Any] = {}
        for sname, (lang, prompt) in SETTINGS.items():
            rows = []
            for c in clips:
                t = asr.transcribe(audio[c.clip_id], language=lang, initial_prompt=prompt, fast=True)
                r = score_clip(c.text_hint, t.text)
                r.update(clip=c.clip_id, session=c.session_id, detected=t.language)
                rows.append(r)
            per_setting[sname] = {"summary": summarise(rows), "rows": rows}
            print(f"[{size:9}] {sname:20} {json.dumps(per_setting[sname]['summary'], ensure_ascii=False)}", flush=True)
        report["per_model"][size] = {"device": getattr(asr, "resolved_device", "?"), "settings": per_setting}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"asr_audit_{speaker}_{kind}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    from .config import Settings

    p = argparse.ArgumentParser(prog="python -m kavach.asr_audit", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", default="S04")
    p.add_argument("--kind", default="read", help="Studio clip kind whose text_hint is the reference (default: read)")
    p.add_argument("--models", default="configured", help="comma list; 'configured' means Settings.whisper_model")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    cfg = Settings()
    audit([m.strip() for m in args.models.split(",") if m.strip()], args.speaker, args.kind, args.limit,
          cfg.data_dir / "tamil_words" / "audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
