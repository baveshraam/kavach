"""Domain objects -> wire schemas. The single place the two vocabularies meet.

Every route returns a `schemas.*` model, and every one of those is built here.
Keeping the mapping in one module rather than inline in the routes means the
frontend contract has exactly one place to drift from, which is what
`tests/test_api.py` checks.

TWO PLACES THE UI AND THE BACKEND DISAGREE, AND HOW IT IS RESOLVED
------------------------------------------------------------------
1.  **Attack identifiers.** The frontend says `A3_CLONE_NAIVE`; the research
    enum says `A3_clone`. Neither side bends: the UI is a given, and the enum
    is what `attacks/` and its 91 tests are written against. `ATTACK_TO_WIRE`
    maps between them and `tests/test_api.py` asserts the map is a bijection,
    so adding a sixth attack without deciding its wire name fails the build.

2.  **Configuration keys.** The frontend's `successRateByConfig` has four
    keys: `ecapa_only`, `plus_knowledge`, `plus_csbg`, `full_fusion`. The
    first, second and fourth are `SystemConfig`'s cumulative progression;
    `plus_csbg` is `SystemConfig.ECAPA_CSBG`, the column that isolates the
    contribution by *omitting* the knowledge branch. See `attacks.suite`.

THE CSBG IS A MATRIX HERE AND A NODE-LINK GRAPH THERE
-----------------------------------------------------
`csbg.graph.CSBG` stores probabilities as arrays because that is what the
maths wants. The Graph Explorer wants nodes and edges. `csbg_to_wire`
materialises the node-link view, and it drops two things on purpose:

* Classes with no observations. An edge at exactly the smoothing prior is not
  something the speaker did -- it is the absence of evidence, and drawing it
  would show a graph far denser than the data supports.
* The 84 transition edges by default. They swamp the 42 lexical ones visually
  and are individually far noisier. `include_transitions=True` brings them
  back for the ablation figure.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np

from ..asr import Transcript
from ..attacks import AttackType
from ..attacks.suite import AttackTable, SystemConfig
from ..challenge import Challenge as DomainChallenge
from ..csbg.graph import CSBG, LANG_INDEX
from ..csbg.metrics import CodeMixingMetrics
from ..csbg.ontology import (
    CHOICE_LANGUAGES,
    CLASS_ORDER,
    Language,
    SemanticClass,
)
from ..csbg.scoring import ClassContribution, _jensen_shannon
from ..csbg.tokens import Token as DomainToken
from ..csbg.tokens import UtteranceTokens
from ..eval.metrics import VerificationMetrics
from ..fusion import FusionResult
from ..skg import KAVACH_NS, SpeakerKG, slugify
from . import schemas

# --------------------------------------------------------------------------
# Attack identifiers
# --------------------------------------------------------------------------

#: Research enum -> frontend string. Deliberately explicit rather than derived
#: from the member name: the two vocabularies disagree (`A3_CLONE` vs
#: `A3_CLONE_NAIVE`) and a clever derivation would break silently the first
#: time they disagree again.
ATTACK_TO_WIRE: dict[AttackType, schemas.AttackTypeStr] = {
    AttackType.A1_REPLAY: "A1_REPLAY",
    AttackType.A2_SPLICE: "A2_SPLICE",
    AttackType.A3_CLONE: "A3_CLONE_NAIVE",
    AttackType.A4_CLONE_KNOWLEDGE: "A4_CLONE_KNOWLEDGE",
    AttackType.A5_STYLE_ADAPTIVE: "A5_CLONE_ADAPTIVE",
}

ATTACK_FROM_WIRE: dict[str, AttackType] = {v: k for k, v in ATTACK_TO_WIRE.items()}

#: `SystemConfig` -> the frontend's `successRateByConfig` key.
CONFIG_TO_WIRE: dict[SystemConfig, str] = {
    SystemConfig.ECAPA_ONLY: "ecapa_only",
    SystemConfig.ECAPA_KNOWLEDGE: "plus_knowledge",
    SystemConfig.ECAPA_CSBG: "plus_csbg",
    SystemConfig.FULL: "full_fusion",
}


def attack_from_wire(value: str) -> AttackType:
    """Parse a frontend attack identifier.

    Raises:
        ValueError: On an unknown identifier, naming what is accepted. A
            silent fallback to A1 would produce a plausible-looking attack run
            for the wrong threat model.
    """
    try:
        return ATTACK_FROM_WIRE[value]
    except KeyError:
        raise ValueError(
            f"Unknown attack type {value!r}. Expected one of "
            f"{', '.join(sorted(ATTACK_FROM_WIRE))}."
        ) from None


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------


def iso(ts: float | datetime | None = None) -> str:
    """UTC ISO-8601, which is what every `*At` field in `types.ts` is.

    Timestamps travel as strings rather than epoch floats because the frontend
    feeds them straight to `new Date(...)`, and a bare float there parses as
    milliseconds-since-epoch -- producing dates in 1970 rather than an error.
    """
    if ts is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat()


def _finite(x: float, default: float = 0.0) -> float:
    """Replace NaN/inf with `default`.

    JSON has no representation for either. Pydantic is configured to emit them
    as the JavaScript constants, which `JSON.parse` then rejects -- so a single
    NaN from an empty score set would fail the whole response rather than one
    field.
    """
    return float(x) if math.isfinite(x) else default


# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------


def token_to_wire(token: DomainToken) -> schemas.Token:
    return schemas.Token(
        text=token.text,
        language=token.language.value,  # type: ignore[arg-type]
        semantic_class=token.semantic_class.value,  # type: ignore[arg-type]
        lid_confidence=token.lid_confidence,
        start_ms=token.start_ms,
        end_ms=token.end_ms,
    )


def tokens_to_wire(tokens: Iterable[DomainToken]) -> list[schemas.Token]:
    return [token_to_wire(t) for t in tokens]


def token_from_wire(token: schemas.Token) -> DomainToken:
    return DomainToken(
        text=token.text,
        language=Language(token.language),
        semantic_class=SemanticClass(token.semantic_class),
        lid_confidence=token.lid_confidence,
        start_ms=token.start_ms,
        end_ms=token.end_ms,
    )


def utterance_tokens_from_wire(
    utterance_id: str, tokens: Iterable[schemas.Token], *, speaker_id: str | None = None,
    transcript: str = "",
) -> UtteranceTokens:
    """Rebuild an `UtteranceTokens` from stored wire tokens.

    Used when re-scoring a speaker's CSBG from the database without
    re-transcribing: annotation is the expensive step, so it is persisted and
    replayed rather than recomputed.
    """
    return UtteranceTokens(
        utterance_id=utterance_id,
        tokens=[token_from_wire(t) for t in tokens],
        speaker_id=speaker_id,
        transcript=transcript,
    )


# --------------------------------------------------------------------------
# CSBG
# --------------------------------------------------------------------------

#: Node ids. Prefixed so a class and a language can never collide, and stable
#: across requests so the force layout does not reshuffle on every poll.
def class_node_id(cls_: SemanticClass) -> str:
    return f"class:{cls_.value}"


def language_node_id(lang: Language) -> str:
    return f"lang:{lang.value}"


def csbg_to_wire(
    graph: CSBG,
    *,
    min_count: float = 1.0,
    include_transitions: bool = False,
    sparse_threshold: float = 5.0,
) -> schemas.CSBGGraph:
    """Materialise a CSBG as the node-link graph the explorer draws.

    Args:
        graph: The speaker's fitted CSBG.
        min_count: Classes with fewer observations than this are omitted
            entirely. At 0 observations the edge probability is the smoothing
            prior, and drawing it would present a backoff estimate as an
            observed behaviour.
        include_transitions: Add the (class, prev_lang) -> lang edges. Off by
            default: there are twice as many of them, each estimated from
            roughly a quarter of the data.
        sparse_threshold: Classes below this count are listed in
            `sparse_classes` for the enrolment warning, but are still drawn if
            they clear `min_count`.

    Returns:
        A `CSBGGraph` with `2 + n_observed_classes` nodes.
    """
    nodes: list[schemas.CSBGNode] = []
    edges: list[schemas.CSBGEdge] = []

    lang_totals = {lang: 0.0 for lang in CHOICE_LANGUAGES}

    observed: list[SemanticClass] = []
    for cls_ in CLASS_ORDER:
        count = graph.observation_count(cls_)
        if count < min_count:
            continue
        observed.append(cls_)
        nodes.append(
            schemas.CSBGNode(
                id=class_node_id(cls_),
                kind="class",
                label=cls_.value,
                token_count=int(count),
            )
        )
        probs = graph.p_lang_given_class(cls_)
        counts = graph.lexical_counts[CLASS_ORDER.index(cls_)]
        for lang in CHOICE_LANGUAGES:
            li = LANG_INDEX[lang]
            lang_totals[lang] += float(counts[li])
            edges.append(
                schemas.CSBGEdge(
                    source=class_node_id(cls_),
                    target=language_node_id(lang),
                    probability=_finite(float(probs[li])),
                    observation_count=int(counts[li]),
                    edge_type="lexical_choice",
                )
            )

    # Language nodes carry the pooled token count over the *drawn* classes, so
    # the node sizes add up to what the edges show.
    for lang in CHOICE_LANGUAGES:
        nodes.append(
            schemas.CSBGNode(
                id=language_node_id(lang),
                kind="language",
                label=lang.value,
                token_count=int(lang_totals[lang]),
            )
        )

    if include_transitions:
        for cls_ in observed:
            ci = CLASS_ORDER.index(cls_)
            for prev in CHOICE_LANGUAGES:
                pi = LANG_INDEX[prev]
                row_count = float(graph.transition_counts[ci, pi].sum())
                if row_count < min_count:
                    continue
                for nxt in CHOICE_LANGUAGES:
                    ni = LANG_INDEX[nxt]
                    edges.append(
                        schemas.CSBGEdge(
                            # Transitions leave *from* a language and land on
                            # a class-in-a-language, so the source is the
                            # previous language node. That is what makes the
                            # switch structure visible as a cycle rather than
                            # a second copy of the lexical star.
                            source=language_node_id(prev),
                            target=class_node_id(cls_),
                            probability=_finite(
                                float(graph.transition_probs[ci, pi, ni])
                            ),
                            observation_count=int(graph.transition_counts[ci, pi, ni]),
                            edge_type="switch_transition",
                        )
                    )

    return schemas.CSBGGraph(
        speaker_id=graph.speaker_id,
        nodes=nodes,
        edges=edges,
        cmi=_finite(graph.metrics.cmi),
        i_index=_finite(graph.metrics.i_index),
        matrix_language_ratio=_finite(graph.metrics.ta_fraction),
        sparse_classes=[
            c.value for c in graph.sparse_classes(sparse_threshold)  # type: ignore[misc]
        ],
    )


def empty_csbg(speaker_id: str) -> schemas.CSBGGraph:
    """The graph of a speaker who has not enrolled yet.

    Returned instead of a 404 so the explorer renders an empty state rather
    than an error: "no graph yet" is a normal condition during enrolment, not
    a failure.
    """
    return schemas.CSBGGraph(
        speaker_id=speaker_id,
        nodes=[],
        edges=[],
        sparse_classes=[c.value for c in CLASS_ORDER],  # type: ignore[misc]
    )


# --------------------------------------------------------------------------
# Divergences
# --------------------------------------------------------------------------


def contribution_to_wire(c: ClassContribution) -> schemas.ClassDivergence:
    """A per-class LLR contribution as the UI's `ClassDivergence`.

    `jsd` is the Jensen-Shannon divergence between the *enrolled* distribution
    for this class and the *observed* distribution in the probe -- "how far is
    what they just said from what this speaker usually does". It is computed
    from `observed_counts`, so a class said twice each way scores near zero
    while a class said four times in the wrong language scores near one. Both
    are bounded in [0, 1], which the raw LLR is not, and the UI renders it as
    a bar.
    """
    expected = np.asarray(
        [c.expected_prob if lang is c.expected_language else 1.0 - c.expected_prob
         for lang in CHOICE_LANGUAGES],
        dtype=np.float64,
    )
    observed = np.asarray(c.observed_distribution, dtype=np.float64)
    return schemas.ClassDivergence(
        semantic_class=c.semantic_class.value,  # type: ignore[arg-type]
        expected_language=c.expected_language.value,  # type: ignore[arg-type]
        expected_prob=_finite(c.expected_prob),
        observed_language=c.observed_language.value,  # type: ignore[arg-type]
        observed_prob=_finite(c.observed_prob),
        jsd=_finite(_jensen_shannon(expected, observed)),
        token_count=c.n_tokens,
    )


def divergences_to_wire(
    contributions: Iterable[ClassContribution], *, limit: int = 8
) -> list[schemas.ClassDivergence]:
    """Convert contributions, worst evidence first, capped at `limit`.

    Capped because the panel is an explanation, not a data dump: a login that
    touched 15 classes produces 15 rows of which 3 carry the decision, and
    showing all of them buries the reason the user was rejected.
    """
    return [contribution_to_wire(c) for c in list(contributions)[:limit]]


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def fusion_to_branches(result: FusionResult) -> list[schemas.BranchScore]:
    """Branch scores from a fusion result.

    Unavailable branches are omitted rather than sent with `passed: false`.
    The UI has no "not measured" state, and a branch that could not be scored
    rendered as a failed one would show the user a rejection reason that never
    fired -- the same conflation of missing with negative that `fuse` exists
    to avoid.
    """
    return [
        schemas.BranchScore(
            name=b.branch.value,  # type: ignore[arg-type]
            score=_finite(b.score),
            threshold=_finite(b.threshold),
            weight=_finite(b.weight),
            passed=b.passed,
        )
        for b in result.branches
        if b.available
    ]


def auth_result_to_wire(
    *,
    result_id: str,
    speaker_id: str,
    challenge_id: str,
    transcript: str,
    tokens: Iterable[DomainToken],
    fusion: FusionResult,
    contributions: Iterable[ClassContribution] = (),
    latency_ms: int,
    timestamp: float | None = None,
) -> schemas.AuthResult:
    """Assemble the full authentication response.

    `explanation` carries the branch-by-branch reasoning verbatim from
    `fusion._explain`. It is the one thing an embedding-only system cannot
    produce, so it is passed through rather than summarised.
    """
    return schemas.AuthResult(
        id=result_id,
        speaker_id=speaker_id,
        challenge_id=challenge_id,
        transcript=transcript,
        tokens=tokens_to_wire(tokens),
        branches=fusion_to_branches(fusion),
        fused_score=_finite(fusion.fused_score),
        fused_threshold=_finite(fusion.threshold),
        decision=fusion.decision.value,  # type: ignore[arg-type]
        divergences=divergences_to_wire(contributions),
        explanation=list(fusion.explanation),
        latency_ms=latency_ms,
        timestamp=iso(timestamp),
    )


# --------------------------------------------------------------------------
# Challenges
# --------------------------------------------------------------------------


def challenge_to_wire(
    challenge: DomainChallenge, *, reveal_answer: bool = False
) -> schemas.Challenge:
    """Client-safe challenge.

    `expected_answer_entity` stays empty unless `reveal_answer` is set, which
    only `Settings.demo_reveal_answers` does. `DomainChallenge.public_dict`
    already omits the answer; this keeps the same split at the point where the
    frontend's type says otherwise. See `schemas`' module docstring.
    """
    return schemas.Challenge(
        id=challenge.id,
        speaker_id=challenge.speaker_id,
        question_text=challenge.question_text,
        target_class=challenge.target_class.value,  # type: ignore[arg-type]
        expected_answer_entity=challenge.expected_answer if reveal_answer else "",
        issued_at=iso(challenge.issued_at),
        expires_at=iso(challenge.expires_at),
    )


# --------------------------------------------------------------------------
# Speaker knowledge graph
# --------------------------------------------------------------------------


def skg_to_triples(kg: SpeakerKG) -> list[schemas.Triple]:
    """The SKG as the flat (subject, predicate, object) rows the UI edits.

    The subject is the speaker's KAVACH URI, so the rows the editor shows are
    the actual triples in the store rather than a table that happens to look
    like one. The full RDF graph carries more than this -- `rdfs:label` and
    `kavach:semanticClass` on each entity -- but those are derived from the
    fact, not editable, and round-tripping them through a text field would let
    a typo desynchronise an entity from its class.
    """
    subject = f"{KAVACH_NS}speaker_{slugify(kg.speaker_id)}"
    return [
        schemas.Triple(subject=subject, predicate=f.predicate, object=f.value)
        for f in kg.facts
    ]


def skg_from_triples(speaker_id: str, triples: Iterable[schemas.Triple]) -> SpeakerKG:
    """Rebuild an SKG from edited triples.

    The subject is ignored and taken from `speaker_id`: accepting it from the
    request body would let a client rewrite another speaker's facts by editing
    a string in the form.

    Empty objects are dropped rather than stored, so clearing a field in the
    editor removes the fact instead of asserting that the speaker's hometown
    is the empty string -- which would then be "matched" by any answer the
    normaliser reduces to nothing.
    """
    kg = SpeakerKG(speaker_id)
    for t in triples:
        value = t.object.strip()
        if not value or not t.predicate.strip():
            continue
        kg.add_fact(t.predicate.strip(), value, verified=True)
    return kg


# --------------------------------------------------------------------------
# Speakers
# --------------------------------------------------------------------------


def speaker_to_wire(
    row: dict[str, Any],
    *,
    metrics: CodeMixingMetrics | None = None,
    csbg_density: float = 0.0,
    utterance_count: int = 0,
    total_duration_sec: float = 0.0,
) -> schemas.Speaker:
    """Build a `Speaker` from a stored row plus derived CSBG statistics.

    The four derived fields (`cmi`, `iIndex`, `matrixLanguageRatio`,
    `csbgDensity`) are computed from the graph rather than stored, so they can
    never go stale relative to the utterances behind them.
    """
    return schemas.Speaker(
        id=row["id"],
        display_name=row.get("display_name", ""),
        age_range=row.get("age_range", ""),
        gender=row.get("gender", ""),
        dominant_language=row.get("dominant_language", "Balanced"),
        other_languages=list(row.get("other_languages") or []),
        device=row.get("device", ""),
        environment=row.get("environment", ""),
        consent_given=bool(row.get("consent_given", False)),
        enrolled_at=row.get("enrolled_at", ""),
        utterance_count=utterance_count,
        total_duration_sec=_finite(total_duration_sec),
        cmi=_finite(metrics.cmi if metrics else 0.0),
        i_index=_finite(metrics.i_index if metrics else 0.0),
        matrix_language_ratio=_finite(metrics.ta_fraction if metrics else 0.0),
        csbg_density=_finite(csbg_density),
    )


# --------------------------------------------------------------------------
# Utterances
# --------------------------------------------------------------------------


def utterance_to_wire(row: dict[str, Any]) -> schemas.Utterance:
    return schemas.Utterance(
        id=row["id"],
        speaker_id=row["speaker_id"],
        type=row.get("type", "free-speech"),
        audio_url=row.get("audio_url", ""),
        duration_sec=_finite(row.get("duration_sec", 0.0)),
        sample_rate=int(row.get("sample_rate", 16_000)),
        transcript=row.get("transcript", ""),
        tokens=[schemas.Token(**t) for t in (row.get("tokens") or [])],
        annotated=bool(row.get("annotated", False)),
        recorded_at=row.get("recorded_at", ""),
    )


def transcript_word_timings(transcript: Transcript) -> list[tuple[int, int]]:
    """ASR word timings in the shape `LIDPipeline.tag_utterance` expects."""
    return transcript.timings


# --------------------------------------------------------------------------
# Attacks
# --------------------------------------------------------------------------


def attack_run_to_wire(
    *,
    run_id: str,
    attack: AttackType,
    target_speaker_id: str,
    table: AttackTable,
    generated_at: float | None = None,
    simulated: bool = True,
    notes: Iterable[str] = (),
) -> schemas.AttackRun:
    """One row of the attack table as an `AttackRun`.

    `trials` is taken as the maximum cell count across configurations rather
    than the number requested: inadmissible clones are excluded from the rates
    (see `attacks.suite`), so requesting 40 and reporting 40 when 17 were
    admissible would misstate what the rates are computed over.
    """
    rates: dict[str, float] = {}
    n_trials = 0
    for config, key in CONFIG_TO_WIRE.items():
        cell = table.cells.get((attack, config))
        rates[key] = _finite(cell.iapmr) if cell else 0.0
        if cell:
            n_trials = max(n_trials, cell.n_trials)

    stats = table.yields.get(attack)
    problems: list[str] = list(notes)
    ready, table_problems = table.paper_ready()
    if not ready:
        problems.extend(table_problems)

    return schemas.AttackRun(
        id=run_id,
        attack_type=ATTACK_TO_WIRE[attack],
        target_speaker_id=target_speaker_id,
        trials=n_trials,
        success_rate_by_config=rates,
        generated_at=iso(generated_at),
        simulated=simulated,
        yield_rate=(
            _finite(getattr(stats, "yield_rate", 0.0)) if stats is not None else None
        ),
        notes=problems,
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def verification_metrics_to_wire(
    name: str, m: VerificationMetrics, *, max_points: int = 120
) -> schemas.EvalConfiguration:
    """A `VerificationMetrics` as one line of the evaluation page.

    The stored DET curve is subsampled again here: the metrics object keeps up
    to 200 points for plotting fidelity, and the chart cannot resolve more
    than about 120 across its width.
    """
    curve = list(m.det_curve)
    if len(curve) > max_points:
        step = len(curve) / max_points
        curve = [curve[int(i * step)] for i in range(max_points)]

    return schemas.EvalConfiguration(
        name=name,
        eer=_finite(m.eer),
        min_dcf=_finite(m.min_dcf),
        far_at_frr1=_finite(m.far_at_frr_1pct),
        frr_at_far1=_finite(m.frr_at_far_1pct),
        det_curve=[
            schemas.DETPoint(far=_finite(p.far), frr=_finite(p.frr)) for p in curve
        ],
    )


__all__ = [
    "ATTACK_FROM_WIRE",
    "ATTACK_TO_WIRE",
    "CONFIG_TO_WIRE",
    "attack_from_wire",
    "attack_run_to_wire",
    "auth_result_to_wire",
    "challenge_to_wire",
    "class_node_id",
    "contribution_to_wire",
    "csbg_to_wire",
    "divergences_to_wire",
    "empty_csbg",
    "fusion_to_branches",
    "iso",
    "language_node_id",
    "skg_from_triples",
    "skg_to_triples",
    "speaker_to_wire",
    "token_from_wire",
    "token_to_wire",
    "tokens_to_wire",
    "transcript_word_timings",
    "utterance_to_wire",
    "utterance_tokens_from_wire",
    "verification_metrics_to_wire",
]
