"""Does speech recognition hear the presenter's ten words?

    python -m kavach.studio.words_check --speaker S04

The login accepts only when the words on screen are heard, in order. Whisper `small` on a
Tamil-English speaker reading isolated English nouns is not guaranteed to hear them, and a login the
speech recognition cannot follow is rejected no matter how good the voice match is. So before demo
day the presenter's own words clips are transcribed exactly as the login transcribes them
(English, no code-mix prompt, greedy) and scored with the login's own matcher: the pass rate is the
share of attempts that would have cleared the words gate.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio import load_audio
from ..phrase import match_phrase
from .store import StudioError, StudioStore


@dataclass(slots=True)
class WordsCheck:
    n: int
    passed: int
    threshold: float
    failures: list[dict[str, Any]] = field(default_factory=list)
    word_misses: Counter = field(default_factory=Counter)
    by_session: dict[str, tuple[int, int]] = field(default_factory=dict)
    scores: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    def pass_rate_at(self, threshold: float) -> float:
        """The share that would clear the words gate at another threshold (what loosening it would buy)."""
        return sum(sc >= threshold for sc in self.scores) / self.n if self.n else 0.0

    def report(self) -> str:
        L = [f"Six-word clips: {self.n}. Cleared the words gate (at least {self.threshold:.0%} of the words, in order): "
             f"{self.passed} ({self.pass_rate:.1%})."]
        for s, (p, n) in sorted(self.by_session.items()):
            L.append(f"- {s}: {p} of {n}")
        if self.word_misses:
            L.append("Words most often missed: " + ", ".join(f"{w} ({c})" for w, c in self.word_misses.most_common(8)))
        for f in self.failures[:10]:
            L.append(f"- {f['clip']}: expected [{f['expected']}] heard {f['heard']!r}; missing {f['missing']}")
        L.append(
            f"At half the words (0.50) the pass rate would be {self.pass_rate_at(0.5):.1%}. The price, for ten shown words: "
            "a recording made for another attempt matched 4 or more of them in order 1.4e-4 of the time and 5 or more never "
            "(100,000 random pairs), and it would still need the voice to match."
        )
        if self.pass_rate < 0.95:
            L.append(
                "Fewer than 95% of the presenter's own attempts would pass the words gate. Options, in order: use a "
                "larger speech-recognition model (KAVACH_WHISPER_MODEL=medium, if the machine can run it live), "
                "lower `phrase_min_match` (6 of 10 is 0.60; 5 of 10 is 0.50, which still never matched by chance), or show "
                "fewer words (`phrase_words`, at the cost of a shorter voice sample)."
            )
        else:
            L.append("The words gate is not what will fail on demo day.")
        return "\n".join(L)


def check_words(studio: StudioStore, *, asr: Any, threshold: float) -> WordsCheck:
    clips = [c for c in studio.clips() if c.kind == "words"]
    if not clips:
        raise ValueError("no clips of kind 'words' have been recorded yet")
    out = WordsCheck(n=0, passed=0, threshold=threshold)
    for c in clips:
        try:
            path = studio.verified_wav_path(c)
        except StudioError as exc:
            raise ValueError(f"{exc} (hash check failed)") from exc
        expected = re.findall(r"[a-z]+", c.text_hint.split(":", 1)[-1].lower())
        t = asr.transcribe(load_audio(path), language="en", initial_prompt="", fast=True)
        m = match_phrase(expected, t.text, words=getattr(t, "words", None))
        out.n += 1
        out.scores.append(m.score)
        ok = m.score >= threshold
        out.passed += int(ok)
        p, n = out.by_session.get(c.session_id, (0, 0))
        out.by_session[c.session_id] = (p + int(ok), n + 1)
        out.word_misses.update(m.missing_words)
        if not ok:
            out.failures.append({"clip": c.clip_id, "expected": ", ".join(expected), "heard": t.text, "missing": ", ".join(m.missing_words)})
    return out


def main(argv: list[str] | None = None, *, asr: Any = None) -> int:
    from ..config import Settings

    cfg = Settings()
    p = argparse.ArgumentParser(prog="python -m kavach.studio.words_check", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", default="S04")
    p.add_argument("--studio", type=Path, default=None)
    args = p.parse_args(argv)
    studio = StudioStore(args.studio or cfg.data_dir / "studio" / args.speaker, args.speaker)
    if asr is None:
        from ..asr import WhisperASR

        asr = WhisperASR(model_size=cfg.whisper_model, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type,
                         language=cfg.whisper_language, suppress_numerals=cfg.suppress_numerals)
    try:
        result = check_words(studio, asr=asr, threshold=cfg.phrase_min_match)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(result.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
