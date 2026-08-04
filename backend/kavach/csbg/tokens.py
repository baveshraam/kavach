"""Token representation -- the atomic unit every CSBG is built from.

A Token is the output of the ASR + language-ID + semantic-tagging pipeline
(see kavach.lid). Everything in the CSBG is a statistic over sequences of
these, so this module is deliberately dependency-free and trivially
constructible in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ontology import Language, SemanticClass


@dataclass(frozen=True, slots=True)
class Token:
    """A single word with its language and semantic annotations.

    Attributes:
        text: Surface form as transcribed.
        language: TA / EN / NEUTRAL / NAMED_ENTITY.
        semantic_class: Domain from the ontology.
        lid_confidence: Confidence in `language`, in [0, 1]. Tokens below
            `Config.lid_confidence_floor` are downweighted rather than dropped
            (see UtteranceTokens.confident_tokens) because dropping them
            biases the switch-point count -- a deleted token silently merges
            its neighbours into a false adjacency.
        start_ms / end_ms: Timing from ASR, used for splice-attack detection
            and for the UI. Not used in CSBG maths.
    """

    text: str
    language: Language
    semantic_class: SemanticClass
    lid_confidence: float = 1.0
    start_ms: int = 0
    end_ms: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lid_confidence <= 1.0:
            raise ValueError(
                f"lid_confidence must be in [0, 1], got {self.lid_confidence!r} "
                f"for token {self.text!r}"
            )

    @property
    def is_language_choice(self) -> bool:
        """True if this token reflects a real TA-vs-EN decision."""
        return self.language.is_language_choice


@dataclass(slots=True)
class UtteranceTokens:
    """Tokens of one utterance, kept grouped because switch points must not
    be counted across utterance boundaries.

    Concatenating two utterances and counting transitions across the join
    would invent a switch that the speaker never made. Every transition
    statistic in the CSBG iterates within utterances, never across them.
    """

    utterance_id: str
    tokens: list[Token] = field(default_factory=list)
    speaker_id: str | None = None
    transcript: str = ""

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    @property
    def choice_tokens(self) -> list[Token]:
        """Tokens representing an actual TA/EN choice, in order.

        This is the sequence switch points are counted over: NEUTRAL and
        NAMED_ENTITY tokens are transparent, so "நான் Chennai போனேன்"
        (TA, NE, TA) contains zero switches, which is correct -- the speaker
        did not switch language to say a proper noun.
        """
        return [t for t in self.tokens if t.is_language_choice]

    def confident_tokens(self, floor: float) -> list[Token]:
        """Language-choice tokens whose LID confidence clears `floor`."""
        return [t for t in self.choice_tokens if t.lid_confidence >= floor]

    @property
    def n_language_independent(self) -> int:
        """Count of NEUTRAL + NAMED_ENTITY tokens (the `u` term in CMI)."""
        return sum(1 for t in self.tokens if not t.is_language_choice)


def count_switch_points(tokens: list[Token]) -> int:
    """Number of adjacent pairs in `tokens` that change language.

    Expects an already-filtered sequence from `choice_tokens`; passing raw
    tokens will count NEUTRAL boundaries as switches.
    """
    return sum(1 for a, b in zip(tokens, tokens[1:]) if a.language is not b.language)


def total_tokens(utterances: list[UtteranceTokens]) -> int:
    return sum(len(u) for u in utterances)


def total_choice_tokens(utterances: list[UtteranceTokens]) -> int:
    return sum(len(u.choice_tokens) for u in utterances)
