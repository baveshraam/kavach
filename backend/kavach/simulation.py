"""Synthetic speaker generation for testing and sanity-checking.

WHAT THIS IS FOR
----------------
This module samples utterances from *known* code-switching profiles so the
CSBG machinery can be validated before any human is recorded. It answers:

    "If speakers really do differ in their language-choice habits by margin M,
     does this implementation detect it, and how much speech does it need?"

That is a question about the *estimator*, and it can be answered exactly,
because here we know the ground-truth generating distribution.

WHAT THIS IS NOT FOR
--------------------
It cannot tell you whether real Tamil-English speakers differ enough for this
to work. That is an empirical question about humans, answerable only with the
week-2 pilot recordings. Synthetic speakers differ by construction; that a
model separates them proves the code is correct, NOT that the hypothesis is
true.

**Never report a number from this module as an experimental result.** It
belongs in tests and in the "implementation validation" appendix at most.
Simulated EERs will be far better than real ones, because real speakers are
inconsistent, ASR is noisy, and the LID tagger makes mistakes -- none of which
is modelled here beyond the crude `consistency` parameter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .csbg.ontology import (
    CLASS_ORDER,
    ELICITABLE_CLASSES,
    LOW_SIGNAL_CLASSES,
    Language,
    SemanticClass,
)
from .csbg.tokens import Token, UtteranceTokens


@dataclass(slots=True)
class SpeakerProfile:
    """Ground-truth code-switching habits of a synthetic speaker.

    Attributes:
        speaker_id: Identifier.
        class_ta_prob: P(Tamil | class) for each semantic class -- the
            speaker's true idiolect.
        consistency: How reliably the speaker follows their own profile, in
            [0, 1]. 1.0 samples exactly from `class_ta_prob`; lower values mix
            in coin-flip noise, modelling within-speaker variability across
            sessions, moods and topics. Real speakers are NOT 1.0, and the
            most useful thing this module does is let you ask "at what
            consistency does the system stop working?"
        neutral_rate: Share of tokens that are language-independent
            (named entities, numerals written as digits).
    """

    speaker_id: str
    class_ta_prob: dict[SemanticClass, float]
    consistency: float = 0.85
    neutral_rate: float = 0.08

    def sample_language(self, cls_: SemanticClass, rng: random.Random) -> Language:
        """Draw a language for a token of `cls_`."""
        base = self.class_ta_prob.get(cls_, 0.5)
        # Blend the speaker's tendency with a fair coin by `consistency`.
        p_ta = self.consistency * base + (1.0 - self.consistency) * 0.5
        return Language.TA if rng.random() < p_ta else Language.EN


def make_population(
    n_speakers: int,
    *,
    seed: int = 0,
    separation: float = 0.6,
    consistency: float = 0.85,
) -> list[SpeakerProfile]:
    """Generate synthetic speakers with partially-shared, partially-individual habits.

    Real populations have strong shared tendencies -- almost everyone uses
    English for TECH_DIGITAL -- with individual deviation on top. A generator
    that made every speaker uniformly random would make the task far easier
    than reality and produce a misleadingly good EER. Here each class gets a
    population mean, and speakers deviate from it by `separation`.

    Args:
        n_speakers: How many profiles to generate.
        seed: RNG seed.
        separation: Individual deviation from the population mean, in [0, 1].
            0.0 = every speaker identical (system must fail: this is the
            null-hypothesis control, and a test asserts EER ~ 0.5 there).
            1.0 = speakers uniformly random and trivially separable.
            ~0.6 is a deliberately optimistic guess at reality.
        consistency: Within-speaker reliability; see SpeakerProfile.

    Returns:
        `n_speakers` profiles.
    """
    rng = random.Random(seed)

    # Population-level tendencies, loosely reflecting reported Tamil-English
    # patterns: technology and education skew English, kinship and food skew
    # Tamil. These are plausible priors for a simulation, not measured values.
    population_mean: dict[SemanticClass, float] = {
        SemanticClass.NUMBER: 0.30,
        SemanticClass.TIME_DATE: 0.40,
        SemanticClass.KINSHIP: 0.85,
        SemanticClass.FOOD: 0.80,
        SemanticClass.PLACE_LOCAL: 0.70,
        SemanticClass.PLACE_GLOBAL: 0.25,
        SemanticClass.TECH_DIGITAL: 0.10,
        SemanticClass.EDU_WORK: 0.25,
        SemanticClass.MONEY_COMMERCE: 0.45,
        SemanticClass.EMOTION_STATE: 0.65,
        SemanticClass.BODY_HEALTH: 0.60,
        SemanticClass.TRANSPORT: 0.55,
        SemanticClass.RELIGION_FESTIVAL: 0.90,
        SemanticClass.MEDIA_ENTERTAIN: 0.35,
        SemanticClass.DISCOURSE_MARKER: 0.50,
        SemanticClass.POLITENESS: 0.55,
        SemanticClass.QUANTITY_MEASURE: 0.60,
        SemanticClass.ACTION_VERB: 0.75,
        SemanticClass.FUNCTION_WORD: 0.95,
        SemanticClass.NAMED_ENTITY: 0.50,
        SemanticClass.OTHER: 0.50,
    }

    profiles: list[SpeakerProfile] = []
    for i in range(n_speakers):
        probs: dict[SemanticClass, float] = {}
        for cls_ in CLASS_ORDER:
            mean = population_mean[cls_]
            deviation = (rng.random() - 0.5) * 2.0 * separation
            probs[cls_] = min(0.97, max(0.03, mean + deviation))
        profiles.append(
            SpeakerProfile(
                speaker_id=f"sim_{i:03d}",
                class_ta_prob=probs,
                consistency=consistency,
            )
        )
    return profiles


#: Rough share of tokens per class in natural conversation. Content classes
#: are much rarer than function words, which is why FUNCTION_WORD dominates
#: raw counts and is excluded from scoring (see ontology.LOW_SIGNAL_CLASSES).
_CLASS_FREQUENCY: dict[SemanticClass, float] = {
    SemanticClass.FUNCTION_WORD: 0.26,
    SemanticClass.ACTION_VERB: 0.13,
    SemanticClass.DISCOURSE_MARKER: 0.09,
    SemanticClass.EDU_WORK: 0.07,
    SemanticClass.TIME_DATE: 0.06,
    SemanticClass.NUMBER: 0.05,
    SemanticClass.PLACE_LOCAL: 0.05,
    SemanticClass.EMOTION_STATE: 0.05,
    SemanticClass.KINSHIP: 0.04,
    SemanticClass.FOOD: 0.04,
    SemanticClass.QUANTITY_MEASURE: 0.03,
    SemanticClass.TECH_DIGITAL: 0.03,
    SemanticClass.MONEY_COMMERCE: 0.03,
    SemanticClass.MEDIA_ENTERTAIN: 0.02,
    SemanticClass.TRANSPORT: 0.02,
    SemanticClass.BODY_HEALTH: 0.02,
    SemanticClass.NAMED_ENTITY: 0.02,
    SemanticClass.POLITENESS: 0.01,
    SemanticClass.RELIGION_FESTIVAL: 0.01,
    SemanticClass.PLACE_GLOBAL: 0.01,
    SemanticClass.OTHER: 0.01,
}

_FREQ_CLASSES = list(_CLASS_FREQUENCY)
_FREQ_WEIGHTS = [_CLASS_FREQUENCY[c] for c in _FREQ_CLASSES]


def sample_utterance(
    profile: SpeakerProfile,
    rng: random.Random,
    *,
    utterance_id: str = "utt",
    n_tokens: int = 14,
    target_class: SemanticClass | None = None,
) -> UtteranceTokens:
    """Sample one utterance from a speaker profile.

    Args:
        profile: Speaker to sample from.
        rng: Random source.
        utterance_id: Identifier for the result.
        n_tokens: Utterance length in tokens.
        target_class: If given, ~40% of tokens are forced to this class,
            simulating a challenge question that successfully elicits a
            particular semantic domain. Used to test adaptive targeting.

    Returns:
        An UtteranceTokens with synthetic surface forms. Token text is
        placeholder (`ta_FOOD_3`) -- nothing downstream of the CSBG reads it.
    """
    tokens: list[Token] = []
    for idx in range(n_tokens):
        if target_class is not None and rng.random() < 0.4:
            cls_ = target_class
        else:
            cls_ = rng.choices(_FREQ_CLASSES, weights=_FREQ_WEIGHTS, k=1)[0]

        if rng.random() < profile.neutral_rate:
            lang = Language.NEUTRAL
        else:
            lang = profile.sample_language(cls_, rng)

        tokens.append(
            Token(
                text=f"{lang.value.lower()}_{cls_.value}_{idx}",
                language=lang,
                semantic_class=cls_,
                lid_confidence=1.0,
                start_ms=idx * 300,
                end_ms=(idx + 1) * 300,
            )
        )

    return UtteranceTokens(
        utterance_id=utterance_id,
        tokens=tokens,
        speaker_id=profile.speaker_id,
        transcript=" ".join(t.text for t in tokens),
    )


def sample_session(
    profile: SpeakerProfile,
    rng: random.Random,
    *,
    n_utterances: int = 30,
    tokens_per_utterance: int = 14,
    prefix: str = "utt",
) -> list[UtteranceTokens]:
    """Sample a full enrolment session.

    Defaults give ~420 tokens, roughly 3-4 minutes of speech.
    """
    return [
        sample_utterance(
            profile,
            rng,
            utterance_id=f"{prefix}_{profile.speaker_id}_{i:03d}",
            n_tokens=tokens_per_utterance,
        )
        for i in range(n_utterances)
    ]


@dataclass(slots=True)
class SimulatedCorpus:
    """A population of synthetic speakers with enrolment and trial sessions."""

    profiles: list[SpeakerProfile]
    enrolment: dict[str, list[UtteranceTokens]] = field(default_factory=dict)
    trials: dict[str, list[UtteranceTokens]] = field(default_factory=dict)

    @property
    def speaker_ids(self) -> list[str]:
        return [p.speaker_id for p in self.profiles]


def make_corpus(
    n_speakers: int = 20,
    *,
    seed: int = 0,
    separation: float = 0.6,
    consistency: float = 0.85,
    enrolment_utterances: int = 30,
    trial_utterances: int = 10,
    trial_tokens: int = 14,
) -> SimulatedCorpus:
    """Build a full synthetic corpus: profiles, enrolment sessions, trial sessions.

    Trials are sampled independently of enrolment, so scoring a speaker's
    trials against their own enrolled graph is a genuine held-out test of the
    estimator (subject to the caveats in the module docstring).
    """
    profiles = make_population(
        n_speakers, seed=seed, separation=separation, consistency=consistency
    )
    corpus = SimulatedCorpus(profiles=profiles)

    for i, profile in enumerate(profiles):
        enrol_rng = random.Random(seed * 10_000 + i * 2 + 1)
        trial_rng = random.Random(seed * 10_000 + i * 2 + 2)
        corpus.enrolment[profile.speaker_id] = sample_session(
            profile, enrol_rng, n_utterances=enrolment_utterances, prefix="enrol"
        )
        corpus.trials[profile.speaker_id] = [
            sample_utterance(
                profile,
                trial_rng,
                utterance_id=f"trial_{profile.speaker_id}_{j:03d}",
                n_tokens=trial_tokens,
            )
            for j in range(trial_utterances)
        ]

    return corpus


def style_transfer_attack(
    victim: SpeakerProfile,
    attacker: SpeakerProfile,
    rng: random.Random,
    *,
    imitation_quality: float = 0.5,
    n_tokens: int = 14,
    utterance_id: str = "attack",
) -> UtteranceTokens:
    """Simulate attack A5: a clone that also imitates the victim's switch style.

    Models an attacker who has observed some of the victim's speech and
    prompts an LLM to generate code-mixed text in their style. Interpolates
    between the attacker's own habits and the victim's:

        imitation_quality = 0.0  -> attack A4 (correct voice, attacker's idiolect)
        imitation_quality = 1.0  -> perfect style clone (system must fail)

    Sweeping this parameter answers "how well must an attacker model the
    victim's code-switching before KAVACH breaks?", which is the honest
    bound on the contribution and belongs in the paper's limitations section.
    """
    blended = {
        cls_: (
            imitation_quality * victim.class_ta_prob.get(cls_, 0.5)
            + (1.0 - imitation_quality) * attacker.class_ta_prob.get(cls_, 0.5)
        )
        for cls_ in CLASS_ORDER
    }
    hybrid = SpeakerProfile(
        speaker_id=f"{attacker.speaker_id}_as_{victim.speaker_id}",
        class_ta_prob=blended,
        consistency=attacker.consistency,
        neutral_rate=attacker.neutral_rate,
    )
    return sample_utterance(hybrid, rng, utterance_id=utterance_id, n_tokens=n_tokens)


__all__ = [
    "SpeakerProfile",
    "SimulatedCorpus",
    "make_population",
    "make_corpus",
    "sample_utterance",
    "sample_session",
    "style_transfer_attack",
    "ELICITABLE_CLASSES",
    "LOW_SIGNAL_CLASSES",
]
