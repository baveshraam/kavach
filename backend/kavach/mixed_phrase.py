"""The experimental mixed-language challenge: ten English words and two Tamil words, read in order.

The English-only login (`phrase.py`) is untouched and stays the baseline. This module adds only:

* `new_mixed_phrase`  -- draw 10 English + 2 Tamil words, the Tamil ones at non-adjacent positions that vary
  per challenge, so every Tamil word sits between English words and the speaker has to switch language twice.
* `match_mixed`       -- decide, from two decodes of the same recording, whether the speaker said that phrase.

Why two decodes. Measured on the enrolled speaker's own code-switched clips (`kavach.asr_audit`), the
recogniser as the English login calls it (English forced) never writes a Tamil word in Tamil script, and
decoded as Tamil or auto it writes English words in Tamil script. No single decode hears both halves, so the
English words are read from the English decode and the Tamil words from the Tamil decode.

What is checked, and every check fails closed:

1. English words: at least `min_english` of the ten, in order (the baseline's fuzzy match, unchanged).
2. Tamil words: BOTH must be heard, as that exact Tamil word (`tamil_words.tamil_same`), in the Tamil
   decode. A Tamil word that is not heard is a failure with its own stage, never a pass on English alone.
3. Order: each Tamil word must lie in time between the English words that surround it in the phrase, and the
   Tamil words must come in phrase order. Without word timings from both decodes the order cannot be checked,
   and an unchecked order is a failure, not a pass.

None of this proves identity. It proves the recording was made for this challenge; the voice gate decides who.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .phrase import _same_word
from .tamil_words import BANK, TamilWord, heard_as, is_tamil_token, normalise_english, phonetic_same

DEFAULT_ENGLISH = 10
DEFAULT_TAMIL = 2

#: Share of the ten English words that must be heard, in order. Same as the English login.
DEFAULT_MIN_ENGLISH = 0.6

#: The two decodes are separate passes with their own timestamp jitter; a Tamil word may sit this far outside
#: the gap between its English neighbours before the order is called wrong.
ORDER_SLACK_MS = 400

FAILURE_STAGES = ("", "english_words", "tamil_words", "order", "no_timings")

#: Initial prompts for the two decodes. Empty = none, which is what the English login uses. Not tuned yet:
#: they are set from the ASR audit, and a prompt that is not measured to help stays empty.
MIXED_EN_PROMPT = ""
MIXED_TA_PROMPT = ""
#: The demo's eight Tamil words, as a prompt for the Tamil decode (only when `Settings.mixed_tamil_prompt`).
MIXED_TA_DEMO_PROMPT = "தண்ணீர், வீடு, மலை, பூனை, கடல், யானை, புத்தகம், அம்மா"


def _gloss_collides(word: TamilWord, english: Sequence[str]) -> bool:
    """True if a recogniser that translated this Tamil word would land on one of the English words drawn."""
    if any(phonetic_same(word.ta, e) for e in english):
        return True  # an English word that sounds like the Tamil word could be said in its place
    return any(_same_word(normalise_english(g), e) for g in word.gloss.split() for e in english if normalise_english(g))


def _spread(n_slots: int, n_tamil: int, rng: random.Random) -> list[int]:
    """`n_tamil` slot positions in 0..n_slots-1, none adjacent: each Tamil word is flanked by English."""
    for _ in range(1000):
        pos = sorted(rng.sample(range(n_slots), n_tamil))
        if all(b - a > 1 for a, b in zip(pos, pos[1:])):
            return pos
    raise RuntimeError("could not place the Tamil words apart")  # unreachable for 12 slots and 2 words


def new_mixed_phrase(
    n_english: int = DEFAULT_ENGLISH,
    n_tamil: int = DEFAULT_TAMIL,
    *,
    rng: random.Random | None = None,
    pool: Sequence[TamilWord] | None = None,
) -> tuple[list[str], list[int]]:
    """(tokens, tamil_positions). Unpredictable unless a seeded `rng` is passed."""
    from .phrase import new_phrase

    rng = rng or random.SystemRandom()
    english = new_phrase(n_english, rng=rng)
    candidates = [w for w in (pool if pool is not None else BANK) if not _gloss_collides(w, english)]
    if len(candidates) < n_tamil:
        raise ValueError(f"only {len(candidates)} Tamil words available for {n_tamil} slots")
    tamil = [w.ta for w in rng.sample(candidates, n_tamil)]
    positions = _spread(n_english + n_tamil, n_tamil, rng)
    tokens: list[str] = []
    e, t = iter(english), iter(tamil)
    for i in range(n_english + n_tamil):
        tokens.append(next(t) if i in positions else next(e))
    return tokens, positions


@dataclass(frozen=True, slots=True)
class MixedMatch:
    """Outcome of a mixed-phrase check. English, Tamil and order are reported separately, never merged."""

    score: float
    """0.0 unless the Tamil words and the order passed; then the share of English words heard."""
    english_matched: int
    english_total: int
    tamil_matched: int
    tamil_total: int
    order_ok: bool
    failure_stage: str
    matched_words: tuple[str, ...]
    missing_words: tuple[str, ...]
    detail: str
    span_ms: tuple[int, int] | None = None
    heard_english: str = ""
    heard_tamil: str = ""
    tamil_status: dict[str, str] = field(default_factory=dict)
    """expected Tamil word -> 'heard' | 'not_heard' | 'wrong_place'."""
    tamil_enforced: bool = True
    """False when the Tamil words were observed but did not decide the result (`Settings.mixed_check_tamil`)."""

    @property
    def english_share(self) -> float:
        return self.english_matched / self.english_total if self.english_total else 0.0

    def to_log(self) -> dict[str, Any]:
        """The challenge-matching half of a trial record; the voice score is logged next to it, never inside it."""
        return {
            "english": f"{self.english_matched}/{self.english_total}", "tamil": f"{self.tamil_matched}/{self.tamil_total}",
            "order_ok": self.order_ok, "failure_stage": self.failure_stage, "score": round(self.score, 3),
            "tamil_status": dict(self.tamil_status), "tamil_enforced": self.tamil_enforced, "missing": list(self.missing_words),
        }


def _latin_tokens(words: Sequence[Any]) -> list[tuple[str, int, int]]:
    return [(t, int(w.start_ms), int(w.end_ms)) for w in words for t in re.findall(r"[a-z]+", str(w.text).lower())]


def _tamil_tokens(words: Sequence[Any]) -> list[tuple[str, int, int]]:
    out = []
    for w in words:
        for piece in re.findall(r"[஀-௿]+", str(w.text)):
            out.append((piece, int(w.start_ms), int(w.end_ms)))
    return out


def match_mixed(
    expected: Sequence[str],
    english: Any,
    tamil: Any,
    *,
    min_english: float = DEFAULT_MIN_ENGLISH,
    check_tamil: bool = True,
) -> MixedMatch:
    """Check a recording against a mixed phrase.

    Args:
        expected: the phrase tokens in order; a token with Tamil letters is a Tamil word.
        english: Transcript of the English-forced decode (needs `.words` with timings, `.text`).
        tamil: Transcript of the Tamil-forced decode, or None if that decode did not run.
    """
    slots = [("ta" if is_tamil_token(t) else "en", t) for t in expected]
    en_expected = [(i, normalise_english(t)) for i, (k, t) in enumerate(slots) if k == "en"]
    ta_expected = [(i, t) for i, (k, t) in enumerate(slots) if k == "ta"]
    heard_en = str(getattr(english, "text", "") or "")
    heard_ta = str(getattr(tamil, "text", "") or "") if tamil is not None else ""

    en_words = getattr(english, "words", None) or []
    ta_words = (getattr(tamil, "words", None) or []) if tamil is not None else []
    en_tokens = _latin_tokens(en_words) if en_words else [(t, -1, -1) for t in re.findall(r"[a-z]+", heard_en.lower())]
    ta_tokens = _tamil_tokens(ta_words) if ta_words else []

    # 1. English words, in order (same walk as phrase.match_phrase).
    pos, en_hit, en_used = 0, {}, set()
    for i, want in en_expected:
        hit = next((j for j in range(pos, len(en_tokens)) if _same_word(want, en_tokens[j][0])), None)
        if hit is not None:
            en_hit[i] = en_tokens[hit]
            en_used.add(hit)
            pos = hit + 1

    # 2. Tamil words: the same Tamil word, or one that SOUNDS like it (the recogniser may write it in Tamil script, in
    #    Latin letters, or as a near miss). Candidates come from the Tamil decode and from English-decode tokens that were
    #    not used up as English words; they are taken in time order, so the Tamil words must come in phrase order.
    cands = [(t, a, b) for t, a, b in _tamil_tokens(ta_words) + _latin_tokens(ta_words)]
    cands += [en_tokens[j] for j in range(len(en_tokens)) if j not in en_used]
    cands.sort(key=lambda c: c[1])
    ta_hit: dict[int, tuple[str, int, int]] = {}
    taken: set[int] = set()
    floor = -1
    for i, want in ta_expected:
        hit = next((k for k, c in enumerate(cands) if k not in taken and c[1] >= floor and heard_as(want, c[0])), None)
        if hit is not None:
            ta_hit[i] = cands[hit]
            taken.add(hit)
            floor = cands[hit][1]

    # 3. Order: a Tamil word must sit between the English words around it. Needs timings from both decodes.
    status: dict[str, str] = {}
    timings_ok = bool(en_words) and all(h[1] >= 0 for h in list(en_hit.values()) + list(ta_hit.values()))
    order_ok = True
    for i, want in ta_expected:
        if i not in ta_hit:
            status[want] = "not_heard"
            continue
        status[want] = "heard"
        if not timings_ok:
            continue
        _, t0, t1 = ta_hit[i]
        before = max((en_hit[j][2] for j in en_hit if j < i), default=None)
        after = min((en_hit[j][1] for j in en_hit if j > i), default=None)
        if (before is not None and t0 < before - ORDER_SLACK_MS) or (after is not None and t1 > after + ORDER_SLACK_MS):
            status[want] = "wrong_place"
            order_ok = False

    n_en, n_ta = len(en_expected), len(ta_expected)
    en_matched, ta_matched = len(en_hit), len(ta_hit)
    english_ok = n_en > 0 and en_matched / n_en >= min_english
    tamil_ok = ta_matched == n_ta
    if ta_matched and not timings_ok:
        order_ok = False  # heard, but its place cannot be verified: fail closed

    if not english_ok:
        stage = "english_words"
    elif not check_tamil:
        stage = ""  # Tamil observed and reported above, but not enforced
    elif not tamil_ok:
        stage = "tamil_words"
    elif not timings_ok:
        stage = "no_timings"
    elif not order_ok:
        stage = "order"
    else:
        stage = ""

    matched = [t for i, (k, t) in enumerate(slots) if (i in en_hit) or (i in ta_hit)]
    missing = [t for i, (k, t) in enumerate(slots) if not ((i in en_hit) or (i in ta_hit))]
    heard_spans = [h for h in list(en_hit.values()) + list(ta_hit.values()) if h[1] >= 0]
    span = (min(h[1] for h in heard_spans), max(h[2] for h in heard_spans)) if heard_spans else None

    order_text = {"": "ok", "order": "wrong", "no_timings": "not verified (a decode had no word timings)"}.get(stage, "not assessed")
    detail = (
        f"English {en_matched} of {n_en} heard (need {min_english:.0%}); "
        f"Tamil {ta_matched} of {n_ta} heard"
        + (f" ({', '.join(f'{w}: {s}' for w, s in status.items())})" if status else "")
        + f"; order {order_text}"
        + (f". Failed at: {stage}." if stage else ".")

    )
    return MixedMatch(
        score=(en_matched / n_en) if (stage == "" and n_en) else 0.0,
        english_matched=en_matched, english_total=n_en, tamil_matched=ta_matched, tamil_total=n_ta,
        order_ok=stage not in ("order", "no_timings") and (check_tamil or order_ok),
        failure_stage=stage, matched_words=tuple(matched), missing_words=tuple(missing),
        detail=detail, span_ms=span, heard_english=heard_en, heard_tamil=heard_ta, tamil_status=status,
        tamil_enforced=check_tamil,
    )


__all__ = ["MIXED_EN_PROMPT", "MIXED_TA_PROMPT", "MIXED_TA_DEMO_PROMPT", "new_mixed_phrase", "match_mixed", "MixedMatch", "DEFAULT_ENGLISH", "DEFAULT_TAMIL", "DEFAULT_MIN_ENGLISH", "ORDER_SLACK_MS"]
