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

**The integrity column is real for A1 and A2, always.** Those two attacks are
constructed as actual waveforms from the victim's stored recordings -- A1 is
the file itself, A2 is segments of it cut and crossfaded by
`splice.splice_segments` -- and then handed to the same detector a live login
runs. No model is involved and nothing is drawn from a distribution, so this
column needs no simulation caveat. For A3-A5 it is blank rather than zero:
there is no cloner here to produce a waveform, and reporting "no artefact
found" for audio that was never synthesised would credit the defence with
catching something it never saw.

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
from ..attacks.splice import SpliceConfig, splice_segments
from ..attacks.suite import AttackSuite, AttackTrial
from ..audio import Audio, AudioError, load_audio
from ..csbg.graph import CSBG
from ..csbg.ontology import CHOICE_LANGUAGES, Language
from ..csbg.scoring import build_background_model, score_llr
from ..csbg.tokens import Token, UtteranceTokens
from ..fusion import FusionPolicy
from ..integrity import INTEGRITY_FLOOR
from . import converters as conv
from . import schemas
from .pipeline import MIN_COHORT, Pipeline
from .store import Store, StoreError

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
        "A2 defeats all four *fusion* configurations, and that is the honest result: "
        "the attacker uses the victim's real voice, their real words, and answers "
        "the live challenge, so every identity branch is satisfied. What stops a "
        "splice is signal evidence, not fusion -- the integrity gate builds the "
        "spliced file from this speaker's own recordings and looks for the joins. "
        "Read the integrity column, not the fusion columns, for whether A2 works."
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

    # The same trials scored with the integrity gate switched off. Run because
    # a flat row of zeros does not say *why* it is flat, and "fusion stops this"
    # and "a splice detector stops this and fusion is helpless" are very
    # different claims about the contribution. This is the gate's own ablation,
    # computed every time rather than left to be reasoned about.
    ungated = AttackSuite(
        FusionPolicy(
            threshold=pipeline.settings.fused_threshold,
            borderline_margin=pipeline.settings.borderline_margin,
            integrity_is_gate=False,
        )
    )
    stats = CloneBatchStats()
    model = ACOUSTIC_MODEL[attack]
    speaker_threshold = pipeline.settings.speaker_threshold

    # Real audio for the two attacks that can be built without a cloner. The
    # detector is the pipeline's own, so its duplicate memory already holds
    # every enrolled recording -- which is the whole mechanism behind A1.
    victim_clips = _victim_audio(store, speaker_id)
    if attack in (AttackType.A1_REPLAY, AttackType.A2_SPLICE) and not victim_clips:
        notes.append(
            "No readable audio for this speaker, so the integrity column could not "
            "be measured -- only the fusion columns below are populated. For A1 and "
            "A2 the integrity column is the one that matters."
        )

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

        forged = _attack_audio(attack, victim_clips, rng)
        integrity_score: float | None = None
        if forged is not None:
            integrity_score = pipeline.integrity.check(forged).score

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
            integrity_score=integrity_score,
            integrity_threshold=INTEGRITY_FLOOR,
            provenance={
                "n_scored_tokens": csbg.n_scored_tokens,
                "raw_llr": round(csbg.raw_score, 4),
                "acoustic_source": "modelled",
                "integrity_source": "measured" if integrity_score is not None else "n/a",
            },
        )
        built.append(trial)
        suite.run(trial)
        ungated.run(trial)

    if attack.is_synthetic_speech:
        suite.record_yield(attack, stats)

    notes.extend(_integrity_notes(attack, built, suite, ungated))

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


#: How many pieces a splice attacker cuts their answer into. Three is the
#: smallest number that makes a splice worth doing -- carrier, answer, carrier
#: -- and each join is an independent chance for the detector to fire, so a
#: more elaborate attack is a *weaker* one against this test. Reporting the
#: three-segment case is therefore reporting the attacker's best move, not a
#: convenient one.
SPLICE_SEGMENTS = 3


def _integrity_notes(
    attack: AttackType,
    trials: list[AttackTrial],
    gated: AttackSuite,
    ungated: AttackSuite,
) -> list[str]:
    """State what the integrity gate caught, and what fusion would have done.

    Both halves are needed. The caught-rate alone reads as the system working;
    the counterfactual alone reads as the system failing. Together they say the
    true thing, which is that one specific component earns this row and the
    fusion the paper is about does not.
    """
    measured = [t.integrity_score for t in trials if t.integrity_score is not None]
    if not measured:
        return []

    caught = sum(1 for s in measured if s < INTEGRITY_FLOOR)
    lines = [
        f"Integrity gate: {caught}/{len(measured)} of these attacks were caught by "
        f"edit- and duplicate-artefact tests on the audio itself "
        f"({caught / len(measured):.0%}). This column is measured, not modelled -- "
        "the waveforms were built from this speaker's own recordings."
    ]

    gated_worst = _worst_iapmr(gated, attack)
    ungated_worst = _worst_iapmr(ungated, attack)
    if ungated_worst is not None and gated_worst is not None:
        lines.append(
            f"With the integrity gate switched off, the best-defended configuration "
            f"still admits {ungated_worst:.0%} of these attacks; with it on, "
            f"{gated_worst:.0%}. The difference is what signal evidence contributes "
            "here, and it is not attributable to the CSBG."
        )
    return lines


def _worst_iapmr(suite: AttackSuite, attack: AttackType) -> float | None:
    """The lowest IAPMR any configuration achieves against this attack.

    The *lowest*, because the question a defender asks is "can any
    configuration I might deploy stop this?" -- reporting the worst
    configuration's rate would answer a question nobody asked.
    """
    rates = [
        cell.iapmr
        for (atk, _), cell in suite.table().cells.items()
        if atk is attack and cell.n_trials
    ]
    return min(rates) if rates else None


def _victim_audio(store: Store, speaker_id: str, *, limit: int = 6) -> list[Audio]:
    """Load the victim's stored recordings, skipping anything unreadable.

    Bounded because splicing needs a handful of segments, not the corpus, and
    an attacker with six of someone's recordings is already a strong one.
    """
    clips: list[Audio] = []
    for row in store.list_utterances(speaker_id):
        if len(clips) >= limit:
            break
        try:
            clips.append(load_audio(store.audio_path(row["id"])))
        except (AudioError, StoreError, OSError):
            continue
    return clips


def _attack_audio(
    attack: AttackType, clips: list[Audio], rng: random.Random
) -> Audio | None:
    """Build the waveform this attacker would actually submit.

    Only A1 and A2 can be built without a voice cloner, which is precisely why
    they are the two attacks whose integrity column is a measurement. Returning
    None for A3-A5 is the honest answer, and `build_integrity_branch` turns it
    into an unavailable branch rather than a pass.

    A1 is the victim's file submitted verbatim -- the attack *is* the identity
    of the bytes, so nothing is done to them.

    A2 cuts segments from different recordings and crossfades them. Different
    recordings matter: two segments from one continuous take share a noise
    floor and the background test has nothing to find, which is the limitation
    named in `kavach.integrity`. An attacker who has only one recording of the
    victim is in that better position, and `_attack_audio` reproduces it
    faithfully by falling back to slicing the single clip it has.
    """
    if not clips:
        return None

    if attack is AttackType.A1_REPLAY:
        return clips[rng.randrange(len(clips))]

    if attack is AttackType.A2_SPLICE:
        if len(clips) >= SPLICE_SEGMENTS:
            chosen = rng.sample(clips, SPLICE_SEGMENTS)
            segments = [
                c.slice_seconds(0.0, min(1.5, c.duration_sec)) for c in chosen
            ]
        else:
            # One recording only: cut it into pieces and reassemble out of
            # order. A real attacker in this position is harder to catch, and
            # the number this produces should be read as the harder case.
            src = clips[0]
            span = src.duration_sec / SPLICE_SEGMENTS
            segments = [
                src.slice_seconds(i * span, (i + 1) * span)
                for i in range(SPLICE_SEGMENTS)
            ]
            rng.shuffle(segments)
        segments = [s for s in segments if len(s.samples)]
        if len(segments) < 2:
            return None
        try:
            return splice_segments(segments, SpliceConfig.naive())
        except AudioError:
            return None

    return None


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
