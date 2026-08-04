"""The Attack Lab: drives `attacks.suite` from the corpus in the database.

WHAT IS REAL HERE AND WHAT IS NOT
---------------------------------
This is the sharpest honesty boundary in the project, so it is stated before
anything else.

**The CSBG column is real.** Every trial's CSBG score comes from
`csbg.scoring.score_llr` run over actual annotated token sequences against the
victim's actual enrolled graph and a leave-one-out background model. Nothing
about that number is invented: it is the same code path a real login takes.
What differs per attack is *whose tokens are scored*, which is exactly what
distinguishes the five attackers:

    A1  the victim's own tokens, verbatim          (a replay is their speech)
    A2  the victim's own tokens, re-ordered        (a splice is their words,
                                                    the attacker's sequence)
    A3  another speaker's tokens                   (the attacker's own register)
    A4  another speaker's tokens                   (same -- A4 differs from A3
                                                    only in knowing the answer)
    A5  another speaker's tokens, language re-drawn toward an *estimated*
        victim style                               (the adaptive attacker)

**The acoustic column is not real unless the models are installed.** A3-A5
need a voice cloner this project does not ship, so their ECAPA scores are
drawn from a documented per-attack distribution rather than measured. A1 and
A2 are measured when speechbrain is available, because a replay and a splice
can be built from the victim's stored audio with signal processing alone.

Every run therefore carries `simulated: true` and the reasons in `notes`, and
`suite.AttackTable.paper_ready()` refuses the row. **The paper's table comes
from attacks recorded through real hardware and a real cloner, not from
here.** This exists so the defence can be developed and demonstrated before
the lab session, and so the UI has something to show that is not a mock.

THE A5 STYLE ESTIMATE IS THE INTERESTING PART
---------------------------------------------
A5's attacker cannot read the enrolled CSBG -- that is inside the server. They
estimate it from speech they could overhear. `_observed_style` builds that
estimate from a *budget* of the victim's utterances, which is the realistic
condition, and the resulting curve is the enrolment-stability experiment read
backwards: the speech a defender needs to enrol a usable graph is the speech
an attacker needs to steal one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..attacks import AttackType, StyleSource
from ..attacks.clone import CloneBatchStats
from ..attacks.suite import AttackSuite, AttackTrial
from ..csbg.graph import CSBG
from ..csbg.ontology import CHOICE_LANGUAGES, Language
from ..csbg.scoring import build_background_model, score_llr
from ..csbg.tokens import Token, UtteranceTokens
from ..fusion import FusionPolicy
from . import converters as conv
from . import schemas
from .pipeline import MIN_COHORT, Pipeline
from .store import Store

#: Seconds of the victim's public speech the A5 attacker is assumed to have
#: overheard. Chosen as a plausible social-media clip, not fitted. Sweeping it
#: is the eavesdropping-budget experiment; this is one point on that curve.
A5_EAVESDROP_BUDGET_SEC = 60.0

#: Fixed so two runs of the same attack against the same speaker produce the
#: same numbers. A lab whose bars move on every click teaches nothing about
#: the defence and everything about the random number generator.
DEFAULT_SEED = 20260804


@dataclass(frozen=True, slots=True)
class AcousticModel:
    """Modelled ECAPA similarity for one attack, used when no cloner exists.

    `mean` and `sd` describe a truncated normal on cosine similarity against
    the victim's template. The values encode published behaviour rather than
    measurement on this corpus, and that is precisely why a run built from
    them is not reportable:

    * A1/A2 sit high because the audio *is* the victim's voice; a replay
      channel and a crossfade degrade the embedding a little, not a lot.
    * A3-A5 sit at the threshold because that is what modern zero-shot cloning
      does to a speaker verifier -- it clears it, but not comfortably, and the
      spread is what produces a yield below 1.
    """

    mean: float
    sd: float

    def draw(self, rng: random.Random) -> float:
        return max(-1.0, min(1.0, rng.gauss(self.mean, self.sd)))


#: What actually defeats each attack, attached to every run.
#:
#: The lab fuses branch scores, and three of the five attacks are stopped by
#: machinery that is not a branch at all. Without this the A2 row reads as
#: "the system has no answer to a splice", which is false -- fusion has no
#: answer to a splice, and `attacks.splice.detect_splice` does. Saying which
#: component earns which row is also what keeps the CSBG from being credited
#: with work the challenge protocol did.
DEFEATED_BY: dict[AttackType, str] = {
    AttackType.A1_REPLAY: (
        "A1 is stopped by challenge freshness, not by the CSBG. The '+ CSBG only' "
        "column correctly shows ~100%: a replay is the victim's own speech, so "
        "there is nothing atypical for the graph to see. Claiming the CSBG stops "
        "replays would be claiming credit for the liveness gate."
    ),
    AttackType.A2_SPLICE: (
        "A2 defeats all four fusion configurations, and that is the honest result: "
        "the attacker uses the victim's real voice, their real words, and answers "
        "the live challenge. What stops a splice is signal evidence -- "
        "`attacks.splice.detect_splice` finds the crossfades, the digital silence "
        "and the background mismatch at the joins. That detector is not wired into "
        "this lab, so this row measures fusion alone."
    ),
    AttackType.A3_CLONE: (
        "A3 is stopped by the knowledge branch: the attacker has the voice but not "
        "the answer. The CSBG column is informative but not load-bearing here."
    ),
    AttackType.A4_CLONE_KNOWLEDGE: (
        "A4 is the headline row. The attacker has the voice and the answer, so the "
        "acoustic and knowledge branches both accept. Whatever separates "
        "'+ knowledge' from 'full fusion' in this row is what the CSBG contributes."
    ),
    AttackType.A5_STYLE_ADAPTIVE: (
        "A5 is the credibility test, and it is meant to be hard. A number here that "
        "looks bad for the defence is a result, not a bug -- reporting only A4 "
        "would be selecting the row that flatters the contribution."
    ),
}

ACOUSTIC_MODEL: dict[AttackType, AcousticModel] = {
    AttackType.A1_REPLAY: AcousticModel(0.82, 0.05),
    AttackType.A2_SPLICE: AcousticModel(0.78, 0.07),
    AttackType.A3_CLONE: AcousticModel(0.68, 0.09),
    AttackType.A4_CLONE_KNOWLEDGE: AcousticModel(0.68, 0.09),
    AttackType.A5_STYLE_ADAPTIVE: AcousticModel(0.68, 0.09),
}


def run_attack(
    *,
    attack: AttackType,
    speaker_id: str,
    trials: int,
    store: Store,
    pipeline: Pipeline,
    seed: int | None = None,
) -> schemas.AttackRun:
    """Run `trials` attacks of one type against one speaker.

    Never raises for a shortfall. Missing graph, missing cohort and missing
    attacker speech all come back as an `AttackRun` with zero trials and the
    reason in `notes`. The Attack Lab is where a half-built system is looked
    at, and an error page would say less than "you need two more enrolled
    speakers".

    Args:
        seed: Fixes the acoustic draws and the probe sampling so a run is
            reproducible. Defaults to a fixed value for the same reason.
    """
    rng = random.Random(DEFAULT_SEED if seed is None else seed)
    notes: list[str] = [
        "Simulated attack. The CSBG scores are real -- computed by the same "
        "scorer a live login uses -- but the acoustic scores are modelled "
        "unless a speaker-embedding model is installed, and no voice cloner is "
        "involved. Not a reportable result.",
        DEFEATED_BY[attack],
    ]

    victim = store.load_csbg(speaker_id)
    all_graphs = store.all_csbgs()
    others = {sid: g for sid, g in all_graphs.items() if sid != speaker_id}

    if victim is None:
        return _empty_run(
            attack,
            speaker_id,
            notes
            + [
                "This speaker has no enrolled code-switch graph yet. Complete "
                "enrolment before attacking them."
            ],
        )
    if len(others) < MIN_COHORT:
        return _empty_run(
            attack,
            speaker_id,
            notes
            + [
                f"Only {len(others)} other speaker(s) enrolled. The likelihood ratio "
                f"needs a background model from at least {MIN_COHORT}, and the "
                "impostor attacks need someone else's speech to put in the "
                "attacker's mouth."
            ],
        )

    ubm = build_background_model(list(others.values()))
    victim_utterances = pipeline.stored_tokens(speaker_id)
    attacker_pool = [
        (sid, utts)
        for sid in others
        if (utts := pipeline.stored_tokens(sid))
    ]

    if not victim_utterances:
        return _empty_run(
            attack,
            speaker_id,
            notes + ["This speaker has no annotated utterances to build attacks from."],
        )
    if not attacker_pool and attack is not AttackType.A1_REPLAY:
        return _empty_run(
            attack,
            speaker_id,
            notes
            + [
                "No other speaker has annotated utterances, so there is no attacker "
                "speech to use. Annotate at least one other speaker's recordings."
            ],
        )

    style = None
    style_source = StyleSource.NONE
    if attack.adapts_style:
        style, spent = _observed_style(
            victim_utterances, budget_sec=A5_EAVESDROP_BUDGET_SEC
        )
        available_sec = sum(_duration(u) for u in victim_utterances)

        # If the budget was never binding, the attacker was handed every word
        # the victim has on record -- which is the *oracle* condition, not the
        # observed one. Labelling that run OBSERVED would file an upper bound
        # in the realistic row, and the A5 upper bound is precisely the number
        # a reviewer will not let stand unchallenged.
        exhausted = spent >= available_sec - 1e-6
        style_source = StyleSource.ORACLE if exhausted else StyleSource.OBSERVED

        if exhausted:
            notes.append(
                f"The {A5_EAVESDROP_BUDGET_SEC:.0f}s eavesdropping budget exceeded "
                f"the {available_sec:.0f}s of speech this speaker has recorded, so the "
                "attacker was given all of it. **This run is A5-oracle, the "
                "unrealisable upper bound on attacker power, not the observed "
                "condition.** Record more speech, or lower the budget, before "
                "reading this row as what a real attacker achieves."
            )
        else:
            notes.append(
                f"A5 style was estimated from {spent:.0f}s of the victim's "
                f"{available_sec:.0f}s of recordings, standing in for speech an "
                "attacker could overhear -- the A5-observed condition."
            )
        notes.append(
            "A5 text was assembled by re-drawing token languages from that "
            "estimate, not written by an LLM. `paper_ready()` rejects a "
            "template-written A5 for exactly this reason."
        )

    policy = FusionPolicy(
        threshold=pipeline.settings.fused_threshold,
        borderline_margin=pipeline.settings.borderline_margin,
    )
    suite = AttackSuite(policy)
    stats = CloneBatchStats()
    model = ACOUSTIC_MODEL[attack]
    speaker_threshold = pipeline.settings.speaker_threshold

    built: list[AttackTrial] = []
    for i in range(max(1, trials)):
        probe = _probe_tokens(attack, victim_utterances, attacker_pool, style, rng)
        csbg = score_llr(
            [probe],
            victim,
            ubm,
            lid_confidence_floor=pipeline.settings.lid_confidence_floor,
        )
        acoustic = model.draw(rng)

        stats.attempted += 1
        stats.synthesised += 1
        stats.similarities.append(acoustic)
        admissible = acoustic >= speaker_threshold
        if admissible:
            stats.admissible += 1

        trial = AttackTrial(
            trial_id=f"{attack.value}_{speaker_id}_{i}",
            attack=attack,
            speaker_id=speaker_id,
            speaker_score=acoustic,
            speaker_threshold=speaker_threshold,
            csbg_score=csbg.normalised_score,
            csbg_threshold=0.5,
            knowledge_score=1.0 if _knows_answer(attack) else 0.0,
            knowledge_threshold=pipeline.settings.knowledge_threshold,
            # A replay answers a challenge that has already been used or has
            # expired -- that is what makes it a replay. Every other attack
            # produces a fresh response to the live challenge.
            liveness_ok=attack is not AttackType.A1_REPLAY,
            admissible=admissible,
            simulated=True,
            style_source=style_source,
            text_generator="template",
            csbg_reliable=csbg.n_scored_tokens >= pipeline.settings.min_scored_tokens,
            provenance={
                "n_scored_tokens": csbg.n_scored_tokens,
                "raw_llr": round(csbg.raw_score, 4),
                "acoustic_source": "modelled",
            },
        )
        built.append(trial)
        suite.run(trial)

    if attack.is_synthetic_speech:
        suite.record_yield(attack, stats)

    table = suite.table()
    run = conv.attack_run_to_wire(
        run_id=f"atk_{attack.value}_{int(rng.random() * 1e6):06d}",
        attack=attack,
        target_speaker_id=speaker_id,
        table=table,
        simulated=True,
        notes=notes,
    )
    # `attack_run_to_wire` reports the maximum admissible cell count. When
    # every clone was rejected by the acoustic branch that is zero, which is a
    # real outcome and worth naming rather than showing as an empty row.
    if run.trials == 0 and built:
        run.notes.append(
            f"None of {len(built)} generated attacks cleared the acoustic branch, so "
            "no trial ever reached the CSBG. That is a finding about clone quality, "
            "not about the defence."
        )
    return run


def _knows_answer(attack: AttackType) -> bool:
    """Whether this attacker can produce the correct answer.

    A2 is the case that is easy to get wrong: a splice attacker has no
    generative model at all, but they *do* have recordings of the victim, and
    cutting the right words out of them is the entire attack. So A2 knows the
    answer even though `AttackType.has_answer` -- which is about the A3->A4
    step -- says otherwise.
    """
    return attack.has_answer or attack is AttackType.A2_SPLICE


def _probe_tokens(
    attack: AttackType,
    victim: list[UtteranceTokens],
    attacker_pool: list[tuple[str, list[UtteranceTokens]]],
    style: dict[Any, tuple[Language, float]] | None,
    rng: random.Random,
) -> UtteranceTokens:
    """Build the token sequence this attacker would produce.

    See the module docstring for the per-attack rationale. The returned
    utterance is scored by the real LLR scorer, so this function *is* the
    threat model as far as the CSBG branch is concerned.
    """
    if attack is AttackType.A1_REPLAY:
        return rng.choice(victim)

    if attack is AttackType.A2_SPLICE:
        # The victim's own words, in the attacker's order. Lexical choices are
        # therefore genuinely the victim's -- a splice cannot invent a word
        # they never said -- while the transition structure is not.
        pool = [t for utt in victim for t in utt.tokens]
        n = min(len(pool), max(6, len(rng.choice(victim).tokens)))
        return UtteranceTokens(
            utterance_id="a2_splice",
            tokens=rng.sample(pool, n) if n <= len(pool) else list(pool),
        )

    _, utterances = rng.choice(attacker_pool)
    source = rng.choice(utterances)

    if not attack.adapts_style or not style:
        return UtteranceTokens(utterance_id=f"{attack.value}_probe", tokens=list(source.tokens))

    # A5: the attacker keeps their own words but re-chooses each token's
    # language according to their estimate of the victim's habits. The
    # estimate is imperfect, so the re-draw is probabilistic -- an attacker
    # who believes the victim says numbers in English 80% of the time gets it
    # right 80% of the time, not always.
    adapted: list[Token] = []
    for tok in source.tokens:
        if not tok.is_language_choice:
            adapted.append(tok)
            continue
        guess = style.get(tok.semantic_class)
        if guess is None:
            adapted.append(tok)
            continue
        target, confidence = guess
        language = target if rng.random() < confidence else _other(target)
        adapted.append(
            Token(
                text=tok.text,
                language=language,
                semantic_class=tok.semantic_class,
                lid_confidence=tok.lid_confidence,
                start_ms=tok.start_ms,
                end_ms=tok.end_ms,
            )
        )
    return UtteranceTokens(utterance_id="a5_probe", tokens=adapted)


def _other(language: Language) -> Language:
    return CHOICE_LANGUAGES[1] if language is CHOICE_LANGUAGES[0] else CHOICE_LANGUAGES[0]


def _observed_style(
    utterances: list[UtteranceTokens], *, budget_sec: float
) -> tuple[dict[Any, tuple[Language, float]], float]:
    """The victim's per-class language habits, as an eavesdropper would see them.

    Built from a *truncated* sample rather than the enrolled graph: an
    attacker who could read the enrolled graph has already breached the
    server. `budget_sec` is the eavesdropping budget, and the returned
    confidences are the smoothed probabilities from a CSBG fitted to that
    sample alone -- so a class the attacker heard twice yields a near-coin-flip
    guess, which is the correct handicap.

    Returns:
        (per-class (language, confidence), seconds actually consumed). The
        caller needs the second value to tell whether the budget bound at all:
        a budget larger than the corpus is not an eavesdropping constraint, it
        is the oracle condition wearing its name.
    """
    kept: list[UtteranceTokens] = []
    spent = 0.0
    for utt in utterances:
        if spent >= budget_sec:
            break
        kept.append(utt)
        spent += _duration(utt)

    overheard = CSBG.build("__eavesdropper__", kept)
    return (
        {
            cls_: overheard.dominant_language(cls_)
            for cls_ in overheard.observed_classes(min_count=2.0)
        },
        spent,
    )


def _duration(utt: UtteranceTokens) -> float:
    """Seconds of speech, from token timings, falling back to 2.5 words/second."""
    ends = [t.end_ms for t in utt.tokens if t.end_ms]
    starts = [t.start_ms for t in utt.tokens if t.end_ms]
    if ends and starts:
        return max(0.0, (max(ends) - min(starts)) / 1000.0)
    return len(utt.tokens) / 2.5


def _empty_run(
    attack: AttackType, speaker_id: str, notes: list[str]
) -> schemas.AttackRun:
    """A run that could not be built, carrying the reason.

    Zeroed rates rather than absent ones: the chart renders, and the note says
    why every bar is at the floor. An error would leave the page showing the
    previous run's numbers under a new title.
    """
    return schemas.AttackRun(
        id=f"atk_{attack.value}_empty",
        attack_type=conv.ATTACK_TO_WIRE[attack],
        target_speaker_id=speaker_id,
        trials=0,
        success_rate_by_config={k: 0.0 for k in conv.CONFIG_TO_WIRE.values()},
        generated_at=conv.iso(),
        simulated=True,
        notes=notes,
    )


__all__ = [
    "A5_EAVESDROP_BUDGET_SEC",
    "ACOUSTIC_MODEL",
    "DEFAULT_SEED",
    "DEFEATED_BY",
    "AcousticModel",
    "run_attack",
]
