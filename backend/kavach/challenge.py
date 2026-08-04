"""Adaptive challenge generation -- where the SKG and the CSBG meet.

A challenge is not a random prompt. It is chosen so that answering it
produces the maximum identifying information about *this specific speaker*:

    1. Ask the CSBG: which semantic class does this speaker deviate from the
       population in most? (`csbg.discriminative_classes`)
    2. Ask the SKG: do we hold a personal fact that would elicit that class?
       (`skg.facts_for_class`)
    3. Ask an LLM: phrase it as a natural code-mixed Tamil-English question.

So a speaker whose tell is NUMBER is asked about their hostel room number; a
speaker whose tell is FOOD is asked what they ate. The probe is aimed at the
biometric. That coupling is the one genuinely novel slice of this module.

Prior art -- be honest about this in the paper
----------------------------------------------
Generating authentication challenges from a personal knowledge graph is NOT
new: US Patent 10,362,016 ("Dynamic knowledge-based authentication") builds a
graph of a user's life events and generates challenges from it. Cite it;
do not claim the mechanism.

What remains defensible is narrow and should be stated in one sentence: that
patent generates challenges to test *knowledge*; KAVACH generates them to
*elicit a measurable biometric signal* in a targeted semantic class. That
difference is real but small — one sentence in the paper, not a contribution
bullet.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from .csbg.graph import CSBG
from .csbg.ontology import ELICITABLE_CLASSES, SemanticClass
from .csbg.scoring import discriminative_classes
from .skg import FACTS_BY_CLASS, Fact, SpeakerKG

#: Template questions per fact type, used when no LLM is configured and as a
#: fallback if generation fails. Deliberately code-mixed with a Tamil matrix
#: frame and English insertions, matching how the target population actually
#: speaks.
#:
#: These are a FALLBACK, not the design. A fixed template set has a bounded
#: challenge space, so a determined attacker could enumerate it -- which is
#: precisely the weakness the LLM path exists to remove. Report which path
#: produced the challenges in any experiment.
TEMPLATE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "hometown": ("நீங்க எந்த ஊர்-ல இருந்து வர்றீங்க?", "உங்க native place எது?"),
    "school": ("நீங்க எந்த school-ல படிச்சீங்க?", "உங்க school-ஓட பேர் என்ன?"),
    "college": ("இப்ப எந்த college-ல இருக்கீங்க?", "உங்க college எது?"),
    "favouriteFood": ("உங்களுக்கு பிடிச்ச food என்ன?", "எந்த dish உங்களுக்கு ரொம்ப பிடிக்கும்?"),
    "comfortFood": ("Mood சரியில்லைனா என்ன சாப்பிடுவீங்க?",),
    "siblingName": ("உங்களுக்கு brother அல்லது sister இருக்காங்களா? பேர் என்ன?",),
    "familyRole": ("வீட்ல யார்-கிட்ட அதிகமா பேசுவீங்க?",),
    "commute": ("College-க்கு எப்படி போவீங்க?", "தினமும் எந்த vehicle-ல travel பண்றீங்க?"),
    "commuteTime": ("அந்த journey எவ்வளவு time ஆகும்?",),
    "hostelRoom": ("உங்க hostel room number என்ன?", "உங்க house number சொல்லுங்க."),
    "favouriteFestival": ("எந்த festival உங்களுக்கு ரொம்ப பிடிக்கும்?",),
    "favouriteFilm": ("எந்த படம் நீங்க பல தடவை பாத்திருக்கீங்க?",),
    "phoneBrand": ("நீங்க எந்த phone use பண்றீங்க?",),
    "firstJob": ("உங்க first job அல்லது internship எது?",),
}


@dataclass(slots=True)
class Challenge:
    """An issued challenge, bound to one speaker and one login attempt."""

    id: str
    speaker_id: str
    question_text: str
    target_class: SemanticClass
    """The CSBG class this question is designed to elicit."""

    expected_predicate: str
    expected_answer: str
    """Ground truth from the SKG. Never sent to the client."""

    issued_at: float
    expires_at: float
    generator: str = "template"
    """'llm' or 'template'. Report which was used -- template challenges have
    a bounded, enumerable space."""

    discriminability: float = 0.0
    """JSD between this speaker and the population for `target_class`. Higher
    means the question is better aimed at this speaker."""

    consumed: bool = False
    """Set once an answer is scored. A challenge is single-use: without this,
    a captured challenge-answer pair could be replayed."""

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    @property
    def is_valid(self) -> bool:
        """Usable for a verification attempt right now."""
        return not self.consumed and not self.is_expired

    def public_dict(self) -> dict[str, Any]:
        """Client-safe view.

        Deliberately omits `expected_answer` and `expected_predicate`.
        Leaking either would hand the attacker the knowledge factor, so the
        split is enforced here rather than left to each API route.
        """
        return {
            "id": self.id,
            "speaker_id": self.speaker_id,
            "question_text": self.question_text,
            "target_class": self.target_class.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "seconds_remaining": round(self.seconds_remaining, 1),
            "generator": self.generator,
        }


class ChallengeError(RuntimeError):
    """A challenge could not be generated or is not usable."""


@dataclass(slots=True)
class ChallengeLedger:
    """Records issued challenges, enforcing single use and no repeats.

    Both properties are load-bearing. Reuse would let an attacker who observed
    one login replay that answer; unbounded validity would remove the time
    limit on capturing one.
    """

    ttl_seconds: int = 60
    _issued: dict[str, Challenge] = field(default_factory=dict)
    _history: dict[str, set[str]] = field(default_factory=dict)
    """speaker_id -> set of question texts already used."""

    def record(self, challenge: Challenge) -> None:
        self._issued[challenge.id] = challenge
        self._history.setdefault(challenge.speaker_id, set()).add(challenge.question_text)

    def get(self, challenge_id: str) -> Challenge | None:
        return self._issued.get(challenge_id)

    def consume(self, challenge_id: str) -> Challenge:
        """Fetch and mark a challenge used.

        Raises:
            ChallengeError: If unknown, already used, or expired. All three
                are security-relevant and must fail loudly rather than
                degrading to an unauthenticated accept.
        """
        challenge = self._issued.get(challenge_id)
        if challenge is None:
            raise ChallengeError(f"Unknown challenge id {challenge_id!r}.")
        if challenge.consumed:
            raise ChallengeError(
                f"Challenge {challenge_id!r} has already been used. "
                "Challenges are single-use; issue a new one."
            )
        if challenge.is_expired:
            raise ChallengeError(
                f"Challenge {challenge_id!r} expired "
                f"{time.time() - challenge.expires_at:.0f}s ago."
            )
        challenge.consumed = True
        return challenge

    def previously_asked(self, speaker_id: str) -> set[str]:
        return self._history.get(speaker_id, set())

    def purge_expired(self) -> int:
        """Drop expired challenges. Returns how many were removed."""
        stale = [cid for cid, c in self._issued.items() if c.is_expired]
        for cid in stale:
            del self._issued[cid]
        return len(stale)

    def forget_speaker(self, speaker_id: str) -> None:
        """Erase a speaker's challenge history (deletion requests)."""
        self._history.pop(speaker_id, None)
        for cid in [c for c, ch in self._issued.items() if ch.speaker_id == speaker_id]:
            del self._issued[cid]


def select_target(
    csbg: CSBG | None,
    ubm: CSBG | None,
    skg: SpeakerKG,
    *,
    exclude_classes: set[SemanticClass] | None = None,
) -> tuple[Fact, SemanticClass, float]:
    """Choose which fact to ask about. THE ADAPTIVE TARGETING STEP.

    Ranks the speaker's discriminative classes and returns the first for which
    the SKG actually holds a fact. Falls back to any available fact when no
    CSBG exists yet (first login after enrolment) -- targeting is an
    optimisation, and authentication must not depend on it.

    Args:
        csbg: The speaker's graph, or None to skip targeting.
        ubm: Background model, or None to skip targeting.
        skg: The speaker's knowledge graph.
        exclude_classes: Classes to skip, e.g. recently used ones.

    Returns:
        (fact, target_class, discriminability).

    Raises:
        ChallengeError: If the SKG holds no usable fact at all.
    """
    exclude = exclude_classes or set()
    available = skg.available_classes() - exclude
    if not available:
        raise ChallengeError(
            f"Speaker {skg.speaker_id!r} has no knowledge-graph facts available "
            "(after exclusions). Complete the enrolment interview."
        )

    if csbg is not None and ubm is not None:
        for cls_, jsd in discriminative_classes(csbg, ubm, top_k=len(ELICITABLE_CLASSES)):
            if cls_ in available:
                facts = skg.facts_for_class(cls_)
                if facts:
                    return secrets.choice(facts), cls_, jsd

    # No CSBG yet, or no discriminative class overlapped the SKG.
    cls_ = secrets.choice(sorted(available, key=lambda c: c.value))
    facts = skg.facts_for_class(cls_)
    if not facts:
        raise ChallengeError(f"No facts for class {cls_.value} despite it being available.")
    return secrets.choice(facts), cls_, 0.0


def _template_question(fact: Fact, already_asked: set[str]) -> str:
    """Pick an unused template question for a fact."""
    options = TEMPLATE_QUESTIONS.get(fact.predicate, ())
    if not options:
        ft = fact.fact_type
        return ft.question if ft else f"Tell me about your {fact.predicate}."
    fresh = [q for q in options if q not in already_asked]
    return secrets.choice(fresh or list(options))


def build_generation_prompt(
    fact: Fact, target_class: SemanticClass, already_asked: set[str]
) -> str:
    """User-turn prompt for LLM challenge generation."""
    avoid = (
        "\n\nDo not reuse any of these previously-asked questions:\n"
        + "\n".join(f"- {q}" for q in sorted(already_asked)[:20])
        if already_asked
        else ""
    )
    return f"""Write ONE authentication question in code-mixed Tamil-English.

The question must ask the speaker about: {fact.predicate}
Their answer will be: {fact.value}

The answer should naturally contain words of the semantic category \
{target_class.value}, because we are measuring how this speaker handles that \
category.

Requirements:
- Natural code-mixed Tamil-English as a young bilingual Tamil speaker would \
actually say it -- Tamil grammatical frame with English words inserted where \
they naturally occur.
- Tamil in Tamil script; keep English words in Latin script.
- One short question, answerable in a few words.
- Do not include the answer or hint at it.
- Output only the question. No preamble, no translation, no quotes.{avoid}"""


#: Cached system prompt for challenge generation. Byte-stable so prompt
#: caching applies across logins.
CHALLENGE_SYSTEM_PROMPT = """You write short authentication questions in \
code-mixed Tamil-English for a speaker-verification research system.

Your questions are spoken aloud to a bilingual Tamil-English speaker, who \
answers in their own words. The questions must sound like natural speech from \
a Tamil-English bilingual, not like a translated form field.

Code-mixing conventions to follow:
- Tamil supplies the grammatical frame; English words are inserted where \
bilingual speakers naturally use them (institutions, technology, numbers, \
professional terms).
- Write Tamil in Tamil script and English in Latin script, as people type when \
mixing.
- Attach Tamil case suffixes to English stems with a hyphen where that is how \
it is said: "college-ல", "bus-ku", "hostel-ல".

Always output exactly one question and nothing else."""


class ChallengeGenerator:
    """Generates adaptive, single-use challenges.

    With `llm_client=None` it uses the template bank, which is fully offline
    and reproducible but has a bounded challenge space.
    """

    def __init__(
        self,
        *,
        ledger: ChallengeLedger | None = None,
        llm_client: Any = None,
        model: str = "claude-opus-5",
        effort: str = "medium",
        ttl_seconds: int = 60,
    ) -> None:
        self.ledger = ledger or ChallengeLedger(ttl_seconds=ttl_seconds)
        self.llm_client = llm_client
        self.model = model
        self.effort = effort
        self.ttl_seconds = ttl_seconds

    def generate(
        self,
        speaker_id: str,
        skg: SpeakerKG,
        *,
        csbg: CSBG | None = None,
        ubm: CSBG | None = None,
        exclude_classes: set[SemanticClass] | None = None,
    ) -> Challenge:
        """Issue a challenge for a login attempt.

        Raises:
            ChallengeError: If the speaker has no usable SKG facts.
        """
        fact, target_class, discriminability = select_target(
            csbg, ubm, skg, exclude_classes=exclude_classes
        )
        already_asked = self.ledger.previously_asked(speaker_id)

        question, generator = self._generate_question(fact, target_class, already_asked)

        now = time.time()
        challenge = Challenge(
            id=f"chg_{secrets.token_hex(6)}",
            speaker_id=speaker_id,
            question_text=question,
            target_class=target_class,
            expected_predicate=fact.predicate,
            expected_answer=fact.value,
            issued_at=now,
            expires_at=now + self.ttl_seconds,
            generator=generator,
            discriminability=discriminability,
        )
        self.ledger.record(challenge)
        return challenge

    def _generate_question(
        self, fact: Fact, target_class: SemanticClass, already_asked: set[str]
    ) -> tuple[str, str]:
        """Return (question_text, generator_name)."""
        if self.llm_client is None:
            return _template_question(fact, already_asked), "template"

        try:
            response = self.llm_client.messages.create(
                model=self.model,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": CHALLENGE_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"effort": self.effort},
                messages=[
                    {
                        "role": "user",
                        "content": build_generation_prompt(fact, target_class, already_asked),
                    }
                ],
            )
            if response.stop_reason == "refusal":
                return _template_question(fact, already_asked), "template_after_refusal"
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            if not text:
                return _template_question(fact, already_asked), "template_after_empty"
            return text.strip().strip('"'), "llm"
        except Exception:
            # A login must not fail because the LLM is unreachable. Falling
            # back to a template degrades challenge unpredictability, not
            # availability -- and `generator` records that it happened, so a
            # run with many fallbacks is visible rather than silent.
            return _template_question(fact, already_asked), "template_after_error"


__all__ = [
    "Challenge",
    "ChallengeError",
    "ChallengeLedger",
    "ChallengeGenerator",
    "select_target",
    "build_generation_prompt",
    "CHALLENGE_SYSTEM_PROMPT",
    "TEMPLATE_QUESTIONS",
]
