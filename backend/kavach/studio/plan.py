"""What the presenter records, per session.

Four kinds, four purposes. `read`: a few fixed sentences repeated many times,
the voice-only same-sentence test. `free`: the 14 bilingual `PROTOCOL_V1`
prompts, natural speech for the code-switch analysis. `fact`: a question about
one of the presenter's own facts, the personal-question login. `words`: six random
words, read the way the Unlock screen asks for them: the login's own task, so the
calibration measures the scores the login will actually see (a few seconds of a list of
words, not a sentence).

The demo sentences are Tamil script with English words, because Whisper writes
Tamil in Tamil script and a romanised prompt would never match its transcript.
They are a default; replace them with sentences you can say naturally.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..challenge import TEMPLATE_QUESTIONS
from ..corpus import PROTOCOL_V1
from ..phrase import new_phrase


@dataclass(frozen=True, slots=True)
class PlanItem:
    kind: str
    prompt_id: str
    text_en: str
    text_ta: str
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Recipe:
    read: int
    free: int
    fact: int
    words: int = 0


#: S1-S3 enrol; S4 and S5 are held out and never enrolled. Two, because one held-out sitting is one
#: cluster and cannot support a false-reject interval (the evaluation says so).
RECIPES: dict[str, Recipe] = {
    "S1": Recipe(read=15, free=2, fact=4, words=12),
    "S2": Recipe(read=15, free=2, fact=4, words=12),
    "S3": Recipe(read=10, free=1, fact=3, words=10),
    "S4": Recipe(read=20, free=1, fact=3, words=15),
    "S5": Recipe(read=15, free=1, fact=3, words=15),
}
DEFAULT_RECIPE = Recipe(read=10, free=1, fact=3, words=8)

#: (prompt_id, Tamil-script text with English words, English gloss)
DEMO_SENTENCES: tuple[tuple[str, str, str], ...] = (
    ("demo1", "நாளைக்கு காலையில meeting இருக்கு, அதனால நான் seven மணிக்கே கிளம்பிடுவேன்.",
     "I have a meeting tomorrow morning, so I'll leave by seven."),
    ("demo2", "என் phone-ல battery கம்மியா இருக்கு, charger எடுத்துட்டு வந்தீங்களா?",
     "My phone's battery is low; did you bring the charger?"),
    ("demo3", "இந்த weekend எங்க ஊருக்கு போறேன், train ticket நேத்தே book பண்ணிட்டேன்.",
     "I'm going to my hometown this weekend; I booked the train ticket yesterday."),
)

#: Rough seconds per recording including the pause to read the prompt.
_SECONDS = {"read": 9.0, "free": 40.0, "fact": 15.0, "words": 8.0}


def words_items(session_id: str, n: int) -> list[PlanItem]:
    """`n` prompts of six random words, each different, the same every time this is called.

    The page rebuilds the plan on every request and keeps its place by index, so the words are
    drawn from a generator seeded by the session id rather than from the system's randomness.
    """
    rng = random.Random(f"kavach-words-{session_id}")
    out = []
    for k in range(n):
        shown = ", ".join(new_phrase(6, rng=rng))
        out.append(PlanItem("words", f"words_{session_id}_{k + 1:02d}", "Read these six words clearly, then stop", f"Say: {shown}", 1))
    return out


def fact_question(predicate: str) -> str:
    options = TEMPLATE_QUESTIONS.get(predicate, ())
    return options[0] if options else f"Tell me about your {predicate}."


def build_plan(
    session_id: str,
    facts: Sequence[Any],
    sentences: Sequence[tuple[str, str, str]] = DEMO_SENTENCES,
) -> list[PlanItem]:
    r = RECIPES.get(session_id, DEFAULT_RECIPE)
    items = [PlanItem("read", pid, en, ta, r.read) for pid, ta, en in sentences]
    items += [PlanItem("free", p.prompt_id, p.text_en, p.text_ta, r.free) for p in PROTOCOL_V1]
    items += [
        PlanItem("fact", f"fact_{f.predicate}", f"About your {f.predicate}", fact_question(f.predicate), r.fact)
        for f in facts
    ]
    items += words_items(session_id, r.words)
    return items


def estimate_minutes(items: Sequence[PlanItem]) -> float:
    return round(sum(_SECONDS[i.kind] * i.repeat for i in items) / 60.0, 1)
