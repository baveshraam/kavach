"""A phrase of random words, spoken once, that binds a recording to one challenge.

Replay is the cheapest attack on a voice login: record the real speaker saying anything,
play it back. A fixed sentence is exactly what makes it work. A phrase drawn at random for
each attempt cannot be recorded in advance, so the recording either contains the words that
were shown on this attempt or it was made for some other one.

The check reads the *ASR transcript*, not the audio, and is deliberately forgiving about
speech-recognition slips (case, a plural, one dropped word) and deliberately strict about
substance (the words must appear, in order). It is a liveness check, not an identity check:
it cannot stop someone who can synthesise the real speaker's voice saying the new words,
which is what the voiceprint and the anti-replay gates are for.

The words are plain, concrete English nouns that Tamil-English speakers read without
hesitation and that Whisper writes in Latin script when asked for English. They are chosen so
that no two are close enough for the fuzzy match to mistake one for another (a test pins it).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Sequence

#: Two tokens at least this similar count as the same word (forgives "tigers" for "tiger").
FUZZY_RATIO = 0.8

WORDS: tuple[str, ...] = (
    "anchor", "apple", "arrow", "autumn", "badge", "bamboo", "banana", "basket", "beacon",
    "bottle", "branch", "bridge", "bronze", "bucket", "butter", "cabin", "camel", "candle",
    "canyon", "carpet", "castle", "cherry", "circle", "cloud", "copper", "coral", "cotton",
    "crystal", "desert", "dragon", "eagle", "engine", "falcon", "feather", "forest",
    "fountain", "garden", "ginger", "glacier", "golden", "granite", "guitar", "hammer",
    "harbor", "helmet", "honey", "island", "jacket", "jungle", "kettle", "ladder", "lantern",
    "lemon", "lizard", "magnet", "marble", "meadow", "mirror", "monkey", "needle",
    "orange", "orchard", "otter", "paddle", "palace", "pencil", "pepper", "pillow", "planet",
    "pocket", "puzzle", "rabbit", "ribbon", "river", "silver", "spider",
    "summer", "sunset", "temple", "thunder", "ticket", "tiger", "tunnel", "turtle",
    "umbrella", "valley", "velvet", "violin", "walnut", "window", "winter",
    "yellow", "zebra", "balcony", "blanket", "buffalo", "cabbage", "compass", "diamond",
    "dolphin", "giraffe", "gravel", "hornet", "iceberg", "jewel", "lagoon",
    "mango", "nutmeg", "oyster", "parrot", "pirate", "quartz", "raven", "saffron", "tractor",
    "uniform", "volcano", "whistle", "yogurt", "cactus", "chimney", "cushion", "drawer",
    "festival", "gallery", "harvest", "orchid", "peacock", "pyramid", "rainbow",
    "scooter", "tornado", "trumpet", "village",
)


def new_phrase(n_words: int = 6, *, rng: random.Random | None = None) -> list[str]:
    """`n_words` distinct words drawn at random. Unpredictable unless a seeded `rng` is passed."""
    rng = rng or random.SystemRandom()
    return list(rng.sample(WORDS, n_words))


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    score: float
    matched: int
    total: int
    matched_words: tuple[str, ...]
    missing_words: tuple[str, ...]
    detail: str
    span_ms: tuple[int, int] | None = None
    """Where in the recording the matched words were spoken, from the ASR word timings; None when
    no timings were given or nothing matched. The voice is judged on this span, so speech before
    or after the phrase (a judge saying "let me try") cannot dilute or hijack the voiceprint."""


def _same_word(expected: str, heard: str) -> bool:
    if heard == expected:
        return True
    sm = SequenceMatcher(None, expected, heard)
    # difflib's documented cheap upper bounds first: most pairs of unrelated words fail them.
    if sm.real_quick_ratio() < FUZZY_RATIO or sm.quick_ratio() < FUZZY_RATIO:
        return False
    return sm.ratio() >= FUZZY_RATIO


def match_phrase(expected: Sequence[str], transcript: str, *, words: Sequence[Any] | None = None) -> PhraseMatch:
    """How much of the shown phrase the transcript contains, in order.

    Each expected word is looked for after the previous match, so the right words in the wrong
    order score low: a recording of those words made for another attempt is not an answer to
    this one.

    With `words` (ASR words carrying `.text`, `.start_ms`, `.end_ms`) the match runs over them and
    the result says where in the recording the matched words were spoken (`span_ms`).
    """
    wanted = [w.lower() for w in expected]
    if words:
        tokens = [
            (tok, int(w.start_ms), int(w.end_ms))
            for w in words
            for tok in re.findall(r"[a-z]+", str(w.text).lower())
        ]
    else:
        tokens = [(tok, -1, -1) for tok in re.findall(r"[a-z]+", transcript.lower())]
    heard = [t[0] for t in tokens]
    matched: list[str] = []
    missing: list[str] = []
    hits: list[int] = []
    pos = 0
    for w in wanted:
        hit = next((i for i in range(pos, len(heard)) if _same_word(w, heard[i])), None)
        if hit is None:
            missing.append(w)
        else:
            matched.append(w)
            hits.append(hit)
            pos = hit + 1
    total = len(wanted)
    score = len(matched) / total if total else 0.0
    span = None
    if hits and tokens[hits[0]][1] >= 0:
        span = (tokens[hits[0]][1], tokens[hits[-1]][2])
    if not heard:
        detail = "nothing in the transcript could be matched to the words shown"
    else:
        detail = f"matched {len(matched)} of {total}: heard {', '.join(matched) or 'none'}; missing {', '.join(missing) or 'none'}"
    return PhraseMatch(score, len(matched), total, tuple(matched), tuple(missing), detail, span)
