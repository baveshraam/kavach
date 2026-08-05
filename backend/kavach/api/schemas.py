"""Request and response models, mirroring the frontend's `src/api/types.ts`.

The frontend is written in camelCase and the backend in snake_case, so every
model here carries `alias_generator=to_camel` with `populate_by_name=True`:
construct with Python names, serialise with JavaScript ones. Keeping the
conversion in one place means no route ever hand-builds a dict and quietly
drifts from the TypeScript interface.

**These models are the contract.** If a field here stops matching
`types.ts`, the UI breaks silently -- TypeScript cannot check a JSON payload
at runtime. `tests/test_api.py` asserts the alias set of each model against
the field names in `types.ts`, so a drift fails the build instead of the demo.

ONE DELIBERATE DEVIATION, AND WHY
---------------------------------
`types.ts` declares `Challenge.expectedAnswerEntity: string` and the mock
fills it in with the correct answer. Sending that to the client hands the
attacker the knowledge factor: whoever is authenticating can read the
expected answer out of the network tab and say it back. That would not be a
weakness in the demo, it would be the removal of an entire branch of the
system.

The field is kept so the TypeScript type still validates, but it is **empty
by default**. `Settings.demo_reveal_answers` fills it in for offline
demonstrations, defaults to False, and is reported in
`/api/health` so a demo build cannot be mistaken for a real one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

LanguageStr = Literal["TA", "EN", "NEUTRAL", "NAMED_ENTITY"]

SemanticClassStr = Literal[
    "NUMBER", "TIME_DATE", "KINSHIP", "FOOD", "PLACE_LOCAL", "PLACE_GLOBAL",
    "TECH_DIGITAL", "EDU_WORK", "MONEY_COMMERCE", "EMOTION_STATE",
    "BODY_HEALTH", "TRANSPORT", "RELIGION_FESTIVAL", "MEDIA_ENTERTAIN",
    "DISCOURSE_MARKER", "POLITENESS", "QUANTITY_MEASURE", "ACTION_VERB",
    "FUNCTION_WORD", "NAMED_ENTITY", "OTHER",
]

UtteranceType = Literal[
    "monolingual-ta", "monolingual-en", "code-mixed", "free-speech", "auth-response"
]

DecisionStr = Literal["ACCEPT", "REJECT", "BORDERLINE"]

BranchName = Literal[
    "speaker_embedding", "csbg", "knowledge", "liveness", "signal_integrity"
]

#: The frontend's attack identifiers. They differ from
#: `kavach.attacks.AttackType`; `api.converters` maps between them rather than
#: either side bending to the other -- the UI is a given, and the backend enum
#: is what the research code and its tests are written against.
AttackTypeStr = Literal[
    "A1_REPLAY", "A2_SPLICE", "A3_CLONE_NAIVE", "A4_CLONE_KNOWLEDGE", "A5_CLONE_ADAPTIVE"
]


class Model(BaseModel):
    """Base: camelCase on the wire, snake_case in Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        ser_json_inf_nan="constants",
    )


# --------------------------------------------------------------------------
# Speakers
# --------------------------------------------------------------------------


class Speaker(Model):
    id: str
    display_name: str
    age_range: str = ""
    gender: str = ""
    dominant_language: Literal["Tamil", "English", "Balanced"] = "Balanced"
    other_languages: list[str] = Field(default_factory=list)
    device: str = ""
    environment: str = ""
    consent_given: bool = False
    enrolled_at: str = ""
    utterance_count: int = 0
    total_duration_sec: float = 0.0
    cmi: float = 0.0
    """Code-Mixing Index, 0-100. The UI shows it as a percentage, so it is
    scaled here rather than in the component -- `CodeMixingMetrics.cmi` is a
    0-1 fraction."""

    i_index: float = 0.0
    matrix_language_ratio: float = 0.0
    csbg_density: float = 0.0


class SpeakerCreate(Model):
    """Everything the client supplies; the rest is derived from recordings."""

    display_name: str
    age_range: str = ""
    gender: str = ""
    dominant_language: Literal["Tamil", "English", "Balanced"] = "Balanced"
    other_languages: list[str] = Field(default_factory=list)
    device: str = ""
    environment: str = ""
    consent_given: bool = False


class Deleted(Model):
    deleted: bool = True


# --------------------------------------------------------------------------
# Utterances
# --------------------------------------------------------------------------


class Token(Model):
    text: str
    language: LanguageStr
    semantic_class: SemanticClassStr
    lid_confidence: float = 1.0
    start_ms: int = 0
    end_ms: int = 0


class Utterance(Model):
    id: str
    speaker_id: str
    type: UtteranceType
    audio_url: str
    duration_sec: float
    sample_rate: int
    transcript: str = ""
    tokens: list[Token] = Field(default_factory=list)
    annotated: bool = False
    recorded_at: str = ""


# --------------------------------------------------------------------------
# Knowledge graph
# --------------------------------------------------------------------------


class Triple(Model):
    subject: str
    predicate: str
    object: str


# --------------------------------------------------------------------------
# Challenge and authentication
# --------------------------------------------------------------------------


class ChallengeRequest(Model):
    speaker_id: str


class Challenge(Model):
    id: str
    speaker_id: str
    question_text: str
    target_class: SemanticClassStr
    expected_answer_entity: str = ""
    """Empty unless `demo_reveal_answers` is on. See the module docstring:
    populating this by default would hand the client the knowledge factor."""

    issued_at: str
    expires_at: str


class BranchScore(Model):
    name: BranchName
    score: float
    threshold: float
    weight: float
    passed: bool


class ClassDivergence(Model):
    semantic_class: SemanticClassStr
    expected_language: LanguageStr
    expected_prob: float
    observed_language: LanguageStr
    observed_prob: float
    jsd: float
    token_count: int


class AuthResult(Model):
    id: str
    speaker_id: str
    challenge_id: str
    transcript: str
    tokens: list[Token] = Field(default_factory=list)
    branches: list[BranchScore] = Field(default_factory=list)
    fused_score: float
    fused_threshold: float
    decision: DecisionStr
    divergences: list[ClassDivergence] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)
    latency_ms: int
    timestamp: str


# --------------------------------------------------------------------------
# CSBG
# --------------------------------------------------------------------------


class CSBGNode(Model):
    id: str
    kind: Literal["class", "language"]
    label: str
    token_count: int


class CSBGEdge(Model):
    source: str
    target: str
    probability: float
    observation_count: int
    edge_type: Literal["lexical_choice", "switch_transition"]


class CSBGGraph(Model):
    speaker_id: str
    nodes: list[CSBGNode] = Field(default_factory=list)
    edges: list[CSBGEdge] = Field(default_factory=list)
    cmi: float = 0.0
    i_index: float = 0.0
    matrix_language_ratio: float = 0.0
    sparse_classes: list[SemanticClassStr] = Field(default_factory=list)


class EnrolmentResult(Model):
    csbg: CSBGGraph
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------


class AttackRequest(Model):
    attack_type: AttackTypeStr
    target_speaker_id: str
    trials: int = 10


class AttackRun(Model):
    id: str
    attack_type: AttackTypeStr
    target_speaker_id: str
    trials: int
    success_rate_by_config: dict[str, float]
    """Keyed by 'ecapa_only' | 'plus_knowledge' | 'plus_csbg' | 'full_fusion'.
    These are IAPMR values: the fraction of attacks the configuration
    accepted."""

    generated_at: str
    simulated: bool = True
    """True whenever any trial came from signal processing rather than
    recorded hardware. Not in `types.ts`; extra fields are ignored by the
    client, and omitting it would let the Attack Lab's numbers be screenshotted
    as results."""

    yield_rate: float | None = None
    """Fraction of generated clones that fooled the acoustic branch. None for
    non-clone attacks. A success rate without this is uninterpretable -- see
    the project's section 5.1.3."""

    notes: list[str] = Field(default_factory=list)


class SpeakerIapmr(Model):
    """One speaker's attack success rate, aggregated over their runs."""

    speaker_id: str
    name: str = ""
    trials: int = 0
    iapmr: float = 0.0
    """Under full fusion. The number the defence is judged on."""

    iapmr_by_config: dict[str, float] = Field(default_factory=dict)
    ci_low: float = 0.0
    ci_high: float = 1.0
    """95% Wilson interval. Not the normal approximation: these rates sit at 0
    and 1, where the normal interval runs outside [0, 1] and is simply wrong."""

    below_min_trials: bool = True
    attack_types: list[AttackTypeStr] = Field(default_factory=list)


class PerSpeakerIapmr(Model):
    """Attack success per speaker, because the mean hides the failure.

    If the full system stops every attack on 24 speakers and none on the 25th,
    that speaker is completely unprotected and the mean reads 96%. §5.1.3 asks
    for this table explicitly.
    """

    speakers: list[SpeakerIapmr] = Field(default_factory=list)
    worst_speaker_id: str = ""
    """Highest IAPMR among speakers with enough trials to compare. Empty when
    no speaker clears the bar."""

    mean_iapmr: float | None = None
    """None rather than 0.0 when nothing has been measured -- a mean over no
    trials is not a rate of zero, and 0% is the value that looks like success."""

    unmeasured_speaker_ids: list[str] = Field(default_factory=list)
    """Enrolled speakers with no attack run at all. **An unmeasured speaker is
    not a protected speaker**, and leaving them out of the response entirely
    would make a partial study look complete."""

    min_trials_per_cell: int = 30
    simulated: bool = True
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class DETPoint(Model):
    far: float
    frr: float


class EvalConfiguration(Model):
    name: str
    eer: float
    min_dcf: float
    far_at_frr1: float
    frr_at_far1: float
    det_curve: list[DETPoint] = Field(default_factory=list)


class StabilityPoint(Model):
    duration_sec: float
    eer: float
    ci_low: float
    ci_high: float


class FairnessSlice(Model):
    condition: str
    group: str
    eer: float
    sample_count: int


class ScoreDistribution(Model):
    branch: str
    genuine: list[float] = Field(default_factory=list)
    impostor: list[float] = Field(default_factory=list)


class EvalMetrics(Model):
    configurations: list[EvalConfiguration] = Field(default_factory=list)
    stability_curve: list[StabilityPoint] = Field(default_factory=list)
    fairness: list[FairnessSlice] = Field(default_factory=list)
    score_distributions: list[ScoreDistribution] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class Health(Model):
    status: str
    models: list[str] = Field(default_factory=list)
    device: str = "cpu"
    version: str = ""
    demo_reveal_answers: bool = False
    """Surfaced so a build that leaks challenge answers announces itself."""

    reportable: dict[str, Any] = Field(default_factory=dict)
    """The settings that must accompany any reported number, from
    `Settings.reportable()`."""


__all__ = [
    "AttackRequest",
    "AttackRun",
    "AttackTypeStr",
    "AuthResult",
    "BranchScore",
    "CSBGEdge",
    "CSBGGraph",
    "CSBGNode",
    "Challenge",
    "ChallengeRequest",
    "ClassDivergence",
    "DETPoint",
    "Deleted",
    "EnrolmentResult",
    "EvalConfiguration",
    "EvalMetrics",
    "FairnessSlice",
    "Health",
    "PerSpeakerIapmr",
    "ScoreDistribution",
    "Speaker",
    "SpeakerCreate",
    "SpeakerIapmr",
    "StabilityPoint",
    "Token",
    "Triple",
    "Utterance",
]
