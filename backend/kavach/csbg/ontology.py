"""Semantic concept ontology for the Code-Switch Behaviour Graph.

The central hypothesis of KAVACH is that a bilingual speaker's choice of
language is conditioned on *what they are talking about* -- the semantic
domain -- and that this conditioning is stable and speaker-specific.

This module defines the concept classes that conditioning is measured over,
plus the language tag set. Every downstream CSBG computation is a
distribution over these classes, so the ontology is deliberately fixed and
versioned: changing it invalidates previously-built graphs.

Design constraints on the class set:

1.  *Small enough to estimate.* With ~5 minutes of speech per speaker we get
    roughly 600-900 content tokens. At 21 classes that is ~30-40 tokens per
    class on average, which is enough for a stable P(lang | class) but not
    for much more. Adding classes makes every estimate worse.

2.  *Chosen for switch-propensity contrast, not linguistic completeness.*
    These are not semantic primitives from any lexical resource. They are the
    domains where the code-switching literature and Tamil-English speaker
    intuition suggest language choice actually varies between individuals --
    numbers, kinship, food, technology. A class where every speaker behaves
    identically carries no biometric information no matter how principled it
    is.

3.  *Collapsible.* SUPERCLASS_OF backs off to 6 coarse groups when a speaker
    has too little data for the fine-grained estimate. See csbg.graph.
"""

from __future__ import annotations

from enum import Enum

# Bump when the class set changes; CSBGs built under a different version
# must not be compared. Enforced in csbg.graph.CSBG.
ONTOLOGY_VERSION = "1.0"


class Language(str, Enum):
    """Language tag assigned to a single token.

    NEUTRAL and NAMED_ENTITY are *language-independent* -- they are excluded
    from language-choice distributions and count as `u` in the CMI formula
    (Gambäck & Das 2014). Tagging a proper noun as "Tamil" would be an
    arbitrary decision that adds noise to every downstream estimate: a
    speaker saying "Chennai" has not made a language choice.
    """

    TA = "TA"
    EN = "EN"
    NEUTRAL = "NEUTRAL"
    NAMED_ENTITY = "NAMED_ENTITY"

    @property
    def is_language_choice(self) -> bool:
        """True if this token represents an actual TA-vs-EN decision."""
        return self in (Language.TA, Language.EN)


#: The two languages a choice is made between. Order is fixed -- probability
#: vectors are always indexed [TA, EN].
CHOICE_LANGUAGES: tuple[Language, ...] = (Language.TA, Language.EN)


class SemanticClass(str, Enum):
    """Semantic domain of a token. See module docstring for selection rationale."""

    NUMBER = "NUMBER"
    TIME_DATE = "TIME_DATE"
    KINSHIP = "KINSHIP"
    FOOD = "FOOD"
    PLACE_LOCAL = "PLACE_LOCAL"
    PLACE_GLOBAL = "PLACE_GLOBAL"
    TECH_DIGITAL = "TECH_DIGITAL"
    EDU_WORK = "EDU_WORK"
    MONEY_COMMERCE = "MONEY_COMMERCE"
    EMOTION_STATE = "EMOTION_STATE"
    BODY_HEALTH = "BODY_HEALTH"
    TRANSPORT = "TRANSPORT"
    RELIGION_FESTIVAL = "RELIGION_FESTIVAL"
    MEDIA_ENTERTAIN = "MEDIA_ENTERTAIN"
    DISCOURSE_MARKER = "DISCOURSE_MARKER"
    POLITENESS = "POLITENESS"
    QUANTITY_MEASURE = "QUANTITY_MEASURE"
    ACTION_VERB = "ACTION_VERB"
    FUNCTION_WORD = "FUNCTION_WORD"
    NAMED_ENTITY = "NAMED_ENTITY"
    OTHER = "OTHER"


#: Canonical ordering. All probability vectors and matrices are indexed by
#: this order, so it must never be reordered without bumping ONTOLOGY_VERSION.
CLASS_ORDER: tuple[SemanticClass, ...] = tuple(SemanticClass)

CLASS_INDEX: dict[SemanticClass, int] = {c: i for i, c in enumerate(CLASS_ORDER)}

N_CLASSES = len(CLASS_ORDER)


class SuperClass(str, Enum):
    """Coarse backoff groups used when a fine class is too sparse to estimate."""

    QUANTITATIVE = "QUANTITATIVE"
    SOCIAL = "SOCIAL"
    MATERIAL = "MATERIAL"
    INSTITUTIONAL = "INSTITUTIONAL"
    STRUCTURAL = "STRUCTURAL"
    RESIDUAL = "RESIDUAL"


SUPERCLASS_OF: dict[SemanticClass, SuperClass] = {
    SemanticClass.NUMBER: SuperClass.QUANTITATIVE,
    SemanticClass.QUANTITY_MEASURE: SuperClass.QUANTITATIVE,
    SemanticClass.MONEY_COMMERCE: SuperClass.QUANTITATIVE,
    SemanticClass.TIME_DATE: SuperClass.QUANTITATIVE,
    SemanticClass.KINSHIP: SuperClass.SOCIAL,
    SemanticClass.EMOTION_STATE: SuperClass.SOCIAL,
    SemanticClass.POLITENESS: SuperClass.SOCIAL,
    SemanticClass.BODY_HEALTH: SuperClass.SOCIAL,
    SemanticClass.FOOD: SuperClass.MATERIAL,
    SemanticClass.TRANSPORT: SuperClass.MATERIAL,
    SemanticClass.PLACE_LOCAL: SuperClass.MATERIAL,
    SemanticClass.PLACE_GLOBAL: SuperClass.MATERIAL,
    SemanticClass.TECH_DIGITAL: SuperClass.INSTITUTIONAL,
    SemanticClass.EDU_WORK: SuperClass.INSTITUTIONAL,
    SemanticClass.RELIGION_FESTIVAL: SuperClass.INSTITUTIONAL,
    SemanticClass.MEDIA_ENTERTAIN: SuperClass.INSTITUTIONAL,
    SemanticClass.DISCOURSE_MARKER: SuperClass.STRUCTURAL,
    SemanticClass.FUNCTION_WORD: SuperClass.STRUCTURAL,
    SemanticClass.ACTION_VERB: SuperClass.STRUCTURAL,
    SemanticClass.NAMED_ENTITY: SuperClass.RESIDUAL,
    SemanticClass.OTHER: SuperClass.RESIDUAL,
}

# Every class must have a superclass or backoff silently drops tokens.
assert set(SUPERCLASS_OF) == set(SemanticClass), "SUPERCLASS_OF is incomplete"


#: Classes carrying little or no biometric signal, excluded from scoring by
#: default.
#:
#: FUNCTION_WORD: in Tamil-English code-switching Tamil overwhelmingly
#: supplies the grammatical frame (Myers-Scotton's Matrix Language). Function
#: words are therefore near-deterministically Tamil for almost every speaker,
#: so the class has high token counts but near-zero between-speaker variance.
#: Including it dilutes the frequency-weighted divergence with a term that is
#: always ~0.
#:
#: NAMED_ENTITY / OTHER: not language choices at all.
#:
#: Excluding these is a *hypothesis*, not a fact -- run the ablation in
#: eval.ablation to confirm it helps on your data before reporting it.
LOW_SIGNAL_CLASSES: frozenset[SemanticClass] = frozenset(
    {
        SemanticClass.FUNCTION_WORD,
        SemanticClass.NAMED_ENTITY,
        SemanticClass.OTHER,
    }
)


#: Classes that make good challenge targets: a challenge question is only
#: useful if it reliably elicits tokens of the class we want to measure.
#: "What did you eat?" reliably produces FOOD tokens; nothing reliably
#: produces DISCOURSE_MARKER or FUNCTION_WORD on demand.
ELICITABLE_CLASSES: tuple[SemanticClass, ...] = (
    SemanticClass.NUMBER,
    SemanticClass.TIME_DATE,
    SemanticClass.KINSHIP,
    SemanticClass.FOOD,
    SemanticClass.PLACE_LOCAL,
    SemanticClass.PLACE_GLOBAL,
    SemanticClass.TECH_DIGITAL,
    SemanticClass.EDU_WORK,
    SemanticClass.MONEY_COMMERCE,
    SemanticClass.TRANSPORT,
    SemanticClass.RELIGION_FESTIVAL,
    SemanticClass.MEDIA_ENTERTAIN,
    SemanticClass.QUANTITY_MEASURE,
)


#: Human-readable descriptions. Used verbatim in the LLM tagging prompt, so
#: edits here change tagger behaviour -- keep them concrete and give examples
#: at the boundaries where classes compete (a "bus" is TRANSPORT not
#: MATERIAL; a "bus fare" is MONEY_COMMERCE).
CLASS_DESCRIPTIONS: dict[SemanticClass, str] = {
    SemanticClass.NUMBER: "Cardinal/ordinal numbers, digits, counts, phone numbers, room numbers.",
    SemanticClass.TIME_DATE: "Times, dates, days, months, years, durations, temporal adverbs (yesterday, morning).",
    SemanticClass.KINSHIP: "Family and relationship terms: mother, brother, uncle, friend, wife, அண்ணா, அம்மா.",
    SemanticClass.FOOD: "Food, drink, ingredients, dishes, cooking, eating.",
    SemanticClass.PLACE_LOCAL: "Places in India/Tamil Nadu: towns, districts, streets, local landmarks, home, hostel.",
    SemanticClass.PLACE_GLOBAL: "Foreign countries, cities and global locations outside India.",
    SemanticClass.TECH_DIGITAL: "Technology, devices, software, internet, apps, phones, computers.",
    SemanticClass.EDU_WORK: "Education and employment: school, college, exam, degree, office, job, salary, meeting.",
    SemanticClass.MONEY_COMMERCE: "Money, prices, payment, buying/selling, banking, shops (a bus *fare* is here, not TRANSPORT).",
    SemanticClass.EMOTION_STATE: "Emotions and mental states: happy, angry, tired, worried, bored.",
    SemanticClass.BODY_HEALTH: "Body parts, illness, medicine, hospital, physical condition.",
    SemanticClass.TRANSPORT: "Vehicles and travel: bus, train, auto, bike, driving, travelling (the vehicle itself).",
    SemanticClass.RELIGION_FESTIVAL: "Religion, temples, festivals, rituals, gods, Pongal, Deepavali.",
    SemanticClass.MEDIA_ENTERTAIN: "Films, music, TV, sport, games, books, social media content.",
    SemanticClass.DISCOURSE_MARKER: "Fillers and discourse glue: so, actually, basically, means, you know, அப்புறம், சரி.",
    SemanticClass.POLITENESS: "Greetings, thanks, apologies, honorifics, please/sorry.",
    SemanticClass.QUANTITY_MEASURE: "Non-numeric quantity and measurement: many, few, kilo, litre, big, small, half.",
    SemanticClass.ACTION_VERB: "Content verbs describing actions, where no other class fits better.",
    SemanticClass.FUNCTION_WORD: "Grammatical words: pronouns, postpositions, conjunctions, auxiliaries, case markers.",
    SemanticClass.NAMED_ENTITY: "Proper names of people, brands, organisations, and specific titled works.",
    SemanticClass.OTHER: "Anything not fitting the classes above.",
}

assert set(CLASS_DESCRIPTIONS) == set(SemanticClass), "CLASS_DESCRIPTIONS is incomplete"


def scoring_classes(include_low_signal: bool = False) -> tuple[SemanticClass, ...]:
    """Classes used when computing CSBG similarity.

    Args:
        include_low_signal: If True, return all classes. Used by the ablation
            study to measure what LOW_SIGNAL_CLASSES actually contributes.
    """
    if include_low_signal:
        return CLASS_ORDER
    return tuple(c for c in CLASS_ORDER if c not in LOW_SIGNAL_CLASSES)
