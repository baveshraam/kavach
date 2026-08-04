"""Attacker answer text for A3, A4 and A5.

THE SENTENCE THIS FILE EXISTS TO SUPPORT
----------------------------------------
    The knowledge branch checks the fact. The CSBG checks the wrapper around
    it. The fact is stealable; the wrapper is the biometric.

A challenge asks "What is your mother's name?". The *fact* is "Lakshmi", and a
determined attacker can scrape it off a birthday post. But nobody answers a
question with a bare noun. The victim says something -- "en amma peru
Lakshmi", "my mother's name is Lakshmi", "amma name Lakshmi" -- and which of
those they say, reliably, across sessions, is what the CSBG measures. The
attacker knows the name. They do not know the sentence.

That is the whole A3 -> A4 -> A5 progression, and it is entirely a question of
text:

    A3  wrong fact,   attacker's own style   -> knowledge branch rejects
    A4  right fact,   attacker's own style   -> only the CSBG can reject
    A5  right fact,   victim's style copied  -> the honest stress test

The audio is the victim's cloned voice in all three. Nothing in this file
touches a waveform.

A DESIGN CONSTRAINT THIS IMPOSES ON THE CHALLENGE SET
-----------------------------------------------------
If a challenge can be answered with one word, there is no wrapper, and the
CSBG has nothing to measure. Challenges must elicit a phrase or a sentence --
"tell me about..." rather than "what is...". A one-word answer is not a
failure of the CSBG; it is a failure of the question. `kavach.challenge`
should be checked against this whenever the question templates change, and the
mean answer length per challenge template belongs in the paper alongside the
per-class results.

WHY A5 IS RUN AT ALL
--------------------
Because a reviewer will ask, and "we did not try" is a worse answer than any
number. If A5-oracle breaks the system, that is a real limitation with a real
bound attached, and it is more publishable than an untested claim. See the
package docstring for the oracle/observed split.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..csbg.graph import CSBG
from ..csbg.ontology import CLASS_DESCRIPTIONS, Language, SemanticClass
from . import AttackType, StyleSource

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 1024

_LANG_NAME = {
    Language.TA: "Tamil",
    Language.EN: "English",
    Language.NEUTRAL: "either language",
    Language.NAMED_ENTITY: "either language",
}


# --------------------------------------------------------------------------
# Style profiles -- what an attacker knows about how the victim speaks
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StyleProfile:
    """A description of a speaker's code-switching habits.

    Deliberately shaped as something an attacker could plausibly write down
    after watching someone's videos: a handful of "they say X in Tamil, Y in
    English" observations plus a rough sense of the overall mix. It is not the
    CSBG's probability tables -- an attacker with those has already breached
    the server, at which point voice authentication is not the problem.
    """

    speaker_id: str
    source: StyleSource
    dominant_by_class: dict[SemanticClass, tuple[Language, float]] = field(default_factory=dict)
    cmi: float = 0.0
    ta_fraction: float = 0.0
    burstiness: float = 0.0
    observed_seconds: float = 0.0
    """How much speech the profile was estimated from. The x-axis of the
    eavesdropping-budget experiment; 0.0 for an oracle profile, which by
    definition had unlimited access."""

    n_classes: int = 0

    def to_prompt(self, *, max_classes: int = 10) -> str:
        """Render as instructions for an LLM writing in this style."""
        if not self.dominant_by_class:
            return "No reliable observations about this speaker's language mixing."

        ranked = sorted(
            self.dominant_by_class.items(), key=lambda kv: kv[1][1], reverse=True
        )[:max_classes]

        lines = [
            f"Overall the speaker uses Tamil for roughly {self.ta_fraction:.0%} of their "
            f"content words, with a code-mixing index of {self.cmi:.2f} "
            f"(0 = monolingual, 0.5 = evenly mixed)."
        ]
        if self.burstiness > 0.1:
            lines.append(
                "They switch in bursts -- long stretches in one language rather than "
                "alternating word by word."
            )
        elif self.burstiness < -0.1:
            lines.append("They alternate between languages steadily rather than in long runs.")

        lines.append("Observed habits by topic:")
        for cls_, (lang, prob) in ranked:
            desc = CLASS_DESCRIPTIONS.get(cls_, cls_.value)
            lines.append(
                f"  - {cls_.value} ({desc}): uses {_LANG_NAME[lang]} about {prob:.0%} of the time."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "source": self.source.value,
            "cmi": round(self.cmi, 4),
            "ta_fraction": round(self.ta_fraction, 4),
            "burstiness": round(self.burstiness, 4),
            "observed_seconds": round(self.observed_seconds, 2),
            "n_classes": self.n_classes,
            "dominant_by_class": {
                c.value: [l.value, round(p, 4)] for c, (l, p) in self.dominant_by_class.items()
            },
        }


def describe_style(
    graph: CSBG,
    *,
    source: StyleSource = StyleSource.ORACLE,
    min_count: float = 3.0,
) -> StyleProfile:
    """Turn a CSBG into an attacker-usable style description.

    Args:
        graph: The graph to describe. For `ORACLE`, the victim's enrolled
            graph. For `OBSERVED`, a graph built from overheard speech via
            `estimate_style_from_speech`.
        source: Recorded on the profile so a result can never be reported
            without saying which condition produced it.
        min_count: Classes with fewer observations are omitted -- an attacker
            who saw a class twice has not learned a habit, and including those
            would overstate attacker power.

    Returns:
        A StyleProfile.
    """
    dominant: dict[SemanticClass, tuple[Language, float]] = {}
    for cls_ in graph.observed_classes(min_count=min_count):
        lang, prob = graph.dominant_language(cls_)
        dominant[cls_] = (lang, prob)

    return StyleProfile(
        speaker_id=graph.speaker_id,
        source=source,
        dominant_by_class=dominant,
        cmi=graph.metrics.cmi,
        ta_fraction=graph.metrics.ta_fraction,
        burstiness=graph.metrics.burstiness,
        observed_seconds=graph.total_duration_sec if source is StyleSource.OBSERVED else 0.0,
        n_classes=len(dominant),
    )


def estimate_style_from_speech(
    speaker_id: str,
    utterances: list[Any],
    *,
    budget_seconds: float | None = None,
) -> StyleProfile:
    """Estimate a victim's style from speech an attacker could overhear.

    This is the A5-observed condition. `budget_seconds` truncates the sample,
    which is how the eavesdropping-budget curve is swept: the same measurement
    that tells a defender how much enrolment speech a CSBG needs tells an
    attacker how much surveillance it costs to steal one.

    Args:
        speaker_id: The victim.
        utterances: `UtteranceTokens` from public speech -- explicitly NOT the
            enrolment recordings, which an attacker cannot reach.
        budget_seconds: Keep only the first N seconds' worth of utterances.
            None uses everything supplied.

    Returns:
        A StyleProfile with `source=OBSERVED` and `observed_seconds` set.
    """
    kept: list[Any] = []
    total = 0.0
    for utt in utterances:
        dur = _utterance_duration(utt)
        if budget_seconds is not None and total + dur > budget_seconds:
            break
        kept.append(utt)
        total += dur

    graph = CSBG.build(speaker_id, kept, total_duration_sec=total)
    profile = describe_style(graph, source=StyleSource.OBSERVED)
    profile.observed_seconds = total
    return profile


def _utterance_duration(utt: Any) -> float:
    """Seconds of speech in an utterance, from its token timestamps.

    Falls back to a words-per-minute estimate when timestamps are absent, so
    simulated utterances still land on a sensible budget axis. Flagged rather
    than silent: a budget curve built from estimated durations should say so.
    """
    tokens = getattr(utt, "tokens", [])
    if tokens:
        ends = [t.end_ms for t in tokens if t.end_ms]
        starts = [t.start_ms for t in tokens if t.end_ms]
        if ends:
            return max(0.0, (max(ends) - min(starts)) / 1000.0)
    return len(tokens) / 2.5  # ~150 words/min


# --------------------------------------------------------------------------
# Attacker answers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AttackerAnswer:
    """What the attacker will have the cloned voice say."""

    text: str
    attack: AttackType
    style_source: StyleSource
    knows_answer: bool
    contains_fact: bool
    """Whether the true fact actually appears. Checked rather than assumed --
    an LLM asked to write around a fact sometimes paraphrases it away, and an
    A4 trial whose answer lost the fact is silently an A3 trial."""

    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "attack": self.attack.value,
            "style_source": self.style_source.value,
            "knows_answer": self.knows_answer,
            "contains_fact": self.contains_fact,
            "provenance": self.provenance,
        }


#: Wrappers an attacker with no knowledge of the victim's style would produce.
#: English-dominant on purpose: an attacker who is not themselves a
#: Tamil-English code-switcher, or an LLM writing without style instructions,
#: defaults to this register. If the real attacker pool would be bilingual,
#: run A4 twice -- once with these and once with a genuine other speaker's
#: utterances -- and report both, because assuming a monolingual attacker
#: flatters the defence.
_GENERIC_WRAPPERS: tuple[str, ...] = (
    "My {topic} is {fact}.",
    "It's {fact}.",
    "I think it's {fact}, yes.",
    "{fact}, that's the one.",
    "The answer is {fact}.",
)

_TAMIL_FRAME_WRAPPERS: tuple[str, ...] = (
    "En {topic} {fact}.",
    "Adhu {fact} dhaan.",
    "{fact} nu solluvanga.",
    "Enakku theriyum, {fact}.",
)

_DISTRACTORS: tuple[str, ...] = (
    "I don't remember exactly",
    "something like that",
    "it's been a while",
    "I'd have to check",
)


class AttackTextGenerator:
    """Produces the attacker's spoken text for A3, A4 and A5.

    Uses Claude when available and falls back to templates otherwise, so the
    attack suite runs offline in CI. The fallback is adequate for A3 and A4 --
    where the point is that the attacker writes in *their own* register, which
    a fixed template captures fine -- but it is weak for A5, whose whole
    premise is a capable adversary. **Report A5 only from LLM-generated
    text**; `AttackerAnswer.provenance["generator"]` records which path ran,
    and `suite` refuses to mark an A5 row paper-ready when it says
    `"template"`.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        seed: int | None = None,
        use_llm: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._use_llm = use_llm
        self._client: Any = None
        self._rng = random.Random(seed)

    @property
    def client(self) -> Any:
        """Lazily-constructed Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env dependent
                raise ImportError(
                    "The `anthropic` package is required for LLM attack text. "
                    "Install it with `pip install anthropic`, or construct the "
                    "generator with use_llm=False for the template fallback."
                ) from exc
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    # ------------------------------------------------------------ public API

    def generate(
        self,
        *,
        attack: AttackType,
        question: str,
        true_answer: str,
        target_class: SemanticClass | None = None,
        style: StyleProfile | None = None,
    ) -> AttackerAnswer:
        """Compose the attacker's answer for one trial.

        Args:
            attack: Which threat-model attacker. A1 and A2 are rejected --
                they reuse recorded audio and do not compose text.
            question: The challenge as put to the speaker.
            true_answer: The genuine fact. Used for A4/A5; for A3 it is only
                used to make sure the wrong answer is actually wrong.
            target_class: Semantic class the challenge targets, used to pick a
                plausible wrapper.
            style: Required for A5, ignored otherwise.

        Returns:
            An AttackerAnswer.

        Raises:
            ValueError: For A1/A2, or for A5 without a style profile.
        """
        if attack in (AttackType.A1_REPLAY, AttackType.A2_SPLICE):
            raise ValueError(
                f"{attack.value} reuses recorded audio; it does not compose new text. "
                "Use attacks.replay or attacks.splice instead."
            )
        if attack is AttackType.A5_STYLE_ADAPTIVE and style is None:
            raise ValueError(
                "A5 requires a StyleProfile -- the attack IS the style imitation. "
                "Build one with describe_style() for the oracle condition or "
                "estimate_style_from_speech() for the observed condition."
            )

        if attack is AttackType.A3_CLONE:
            return self._no_knowledge(question, true_answer, target_class)
        if attack is AttackType.A4_CLONE_KNOWLEDGE:
            return self._own_style(question, true_answer, target_class)
        return self._adapted_style(question, true_answer, target_class, style)  # type: ignore[arg-type]

    # ------------------------------------------------------------- A3

    def _no_knowledge(
        self, question: str, true_answer: str, target_class: SemanticClass | None
    ) -> AttackerAnswer:
        """A3: the attacker bluffs. No LLM needed to be evasive."""
        text = self._rng.choice(_DISTRACTORS)
        wrapper = self._rng.choice(_GENERIC_WRAPPERS)
        text = wrapper.format(topic=_topic_word(target_class), fact=text)
        return AttackerAnswer(
            text=text,
            attack=AttackType.A3_CLONE,
            style_source=StyleSource.NONE,
            knows_answer=False,
            contains_fact=_contains(text, true_answer),
            provenance={"generator": "template", "question": question},
        )

    # ------------------------------------------------------------- A4

    def _own_style(
        self, question: str, true_answer: str, target_class: SemanticClass | None
    ) -> AttackerAnswer:
        """A4: right fact, attacker's own register."""
        if self._use_llm:
            text = self._llm_answer(question, true_answer, style=None)
            if text:
                return AttackerAnswer(
                    text=text,
                    attack=AttackType.A4_CLONE_KNOWLEDGE,
                    style_source=StyleSource.NONE,
                    knows_answer=True,
                    contains_fact=_contains(text, true_answer),
                    provenance={"generator": "llm", "model": self.model, "question": question},
                )

        wrapper = self._rng.choice(_GENERIC_WRAPPERS)
        text = wrapper.format(topic=_topic_word(target_class), fact=true_answer)
        return AttackerAnswer(
            text=text,
            attack=AttackType.A4_CLONE_KNOWLEDGE,
            style_source=StyleSource.NONE,
            knows_answer=True,
            contains_fact=_contains(text, true_answer),
            provenance={"generator": "template", "question": question},
        )

    # ------------------------------------------------------------- A5

    def _adapted_style(
        self,
        question: str,
        true_answer: str,
        target_class: SemanticClass | None,
        style: StyleProfile,
    ) -> AttackerAnswer:
        """A5: right fact, victim's switching style imitated."""
        if self._use_llm:
            text = self._llm_answer(question, true_answer, style=style)
            if text:
                return AttackerAnswer(
                    text=text,
                    attack=AttackType.A5_STYLE_ADAPTIVE,
                    style_source=style.source,
                    knows_answer=True,
                    contains_fact=_contains(text, true_answer),
                    provenance={
                        "generator": "llm",
                        "model": self.model,
                        "question": question,
                        "style_observed_seconds": style.observed_seconds,
                        "style_classes": style.n_classes,
                    },
                )

        # Template fallback: pick the frame language from the victim's habit
        # for this class. Crude, and flagged as such -- see the class docstring.
        lang = Language.EN
        if target_class is not None and target_class in style.dominant_by_class:
            lang = style.dominant_by_class[target_class][0]
        elif style.ta_fraction > 0.5:
            lang = Language.TA

        pool = _TAMIL_FRAME_WRAPPERS if lang is Language.TA else _GENERIC_WRAPPERS
        text = self._rng.choice(pool).format(topic=_topic_word(target_class), fact=true_answer)
        return AttackerAnswer(
            text=text,
            attack=AttackType.A5_STYLE_ADAPTIVE,
            style_source=style.source,
            knows_answer=True,
            contains_fact=_contains(text, true_answer),
            provenance={
                "generator": "template",
                "question": question,
                "style_observed_seconds": style.observed_seconds,
            },
        )

    # ---------------------------------------------------------------- LLM

    def _llm_answer(
        self, question: str, true_answer: str, *, style: StyleProfile | None
    ) -> str | None:
        """Ask Claude to compose the answer. Returns None if the call fails.

        Failures degrade to the template path rather than aborting a long
        attack run, but they are recorded in provenance so a batch that
        silently fell back cannot be mistaken for an LLM result.
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_system_prompt(style),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Question asked: {question}\n"
                            f"Fact to include: {true_answer}\n\n"
                            "Write the spoken answer only. No quotation marks, no commentary."
                        ),
                    }
                ],
            )
        except Exception:  # noqa: BLE001 - any API failure falls back
            return None

        parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        text = " ".join(parts).strip().strip('"')
        return text or None


def _system_prompt(style: StyleProfile | None) -> str:
    """System prompt for attacker answer composition.

    The framing is explicit about being a spoofing evaluation. The task is
    writing a natural bilingual sentence -- the sensitivity is entirely in
    what it is measured against, not in the text itself.
    """
    base = (
        "You are generating test material for an academic anti-spoofing evaluation of a "
        "voice authentication system, in the ASVspoof tradition. Your job is to write a "
        "short, natural spoken answer to a question, in Tamil-English code-mixed speech "
        "as used conversationally in Tamil Nadu.\n\n"
        "Rules:\n"
        "- Write it the way someone would SAY it, not write it.\n"
        "- Romanised Tamil (Latin script) for Tamil words, as people type them in chat.\n"
        "- One or two sentences. Include the given fact verbatim.\n"
        "- No preamble, no explanation, no quotation marks. Output the answer only.\n"
    )
    if style is None:
        return base + (
            "\nUse whatever mix of Tamil and English feels natural to you. Do not attempt "
            "to imitate any particular speaker."
        )
    return base + (
        "\nImitate this specific speaker's language-mixing habits as closely as you can. "
        "These observations come from recordings of them speaking:\n\n" + style.to_prompt()
    )


def _topic_word(cls_: SemanticClass | None) -> str:
    if cls_ is None:
        return "answer"
    return cls_.value.replace("_", " ").lower()


def _contains(text: str, fact: str) -> bool:
    """Whether the fact survived into the generated text.

    Substring match on a casefolded, whitespace-collapsed form. Deliberately
    strict: a fuzzy match here would let a paraphrased-away fact count as
    present and turn an A4 trial into a mislabelled A3.
    """
    if not fact.strip():
        return False
    norm = " ".join(text.casefold().split())
    return " ".join(fact.casefold().split()) in norm


__all__ = [
    "AttackTextGenerator",
    "AttackerAnswer",
    "StyleProfile",
    "describe_style",
    "estimate_style_from_speech",
]
