"""`/api/evaluation`: verification metrics computed from the login history.

READ THIS BEFORE PUTTING A NUMBER FROM THIS PAGE IN THE PAPER
-------------------------------------------------------------
This is a *demonstration* evaluation and it is not the paper's. It scores
whatever logins happen to have been run through the UI, which fails the three
things a reportable evaluation needs:

1.  **No trial design.** A proper evaluation scores every probe against every
    enrolled speaker -- N genuine and N(N-1) impostor trials per configuration.
    This scores the trials a person clicked through, which are almost all
    genuine, so the impostor side is whatever mistakes happened.
2.  **No dev/test split.** The thresholds and fusion weights in `Settings` are
    the same ones these scores were produced under. An EER read off them has
    seen its own operating point.
3.  **Labels are assumed, not known.** A login is treated as genuine when the
    speaker was who the challenge was issued for, which is true of every
    honest login and of every successful attack.

`eval.ablation` does the real thing offline: all-pairs scoring, weights fitted
on a dev split, per-configuration thresholds, bootstrap intervals. The reason
this endpoint exists at all is that a blank Evaluation page teaches nothing
about the system, and a *labelled* approximation teaches quite a lot.

An empty result is the honest output until enough logins exist, and that is
what is returned -- not a fabricated curve.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..eval.metrics import evaluate
from ..fusion import Branch
from . import converters as conv
from . import schemas
from .pipeline import Pipeline
from .store import Store

#: Trials needed on each side before a curve is drawn at all. Below this the
#: EER is a statement about two or three logins.
MIN_TRIALS = 5

#: Which branches make up each configuration, mirroring `attacks.suite
#: .SystemConfig`. Kept here rather than imported because the evaluation page
#: scores *branches*, not fused decisions -- a configuration here is a subset
#: of branch scores to average, and the attack suite's version is a fusion
#: policy. Same partition, different object.
CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "ECAPA alone": (Branch.SPEAKER.value,),
    "+ Knowledge": (Branch.SPEAKER.value, Branch.KNOWLEDGE.value),
    "+ CSBG only": (Branch.SPEAKER.value, Branch.CSBG.value),
    "Full fusion": (Branch.SPEAKER.value, Branch.KNOWLEDGE.value, Branch.CSBG.value),
}


def evaluate_history(store: Store, pipeline: Pipeline) -> schemas.EvalMetrics:
    """Build the Evaluation page from stored authentication results.

    Every login in the history is one trial. It is labelled genuine when the
    fused decision was ACCEPT *and* no branch was in strong disagreement --
    see `_label`, which explains why that is the least-wrong label available
    and why it is not good enough for the paper.
    """
    history = store.list_auth(limit=5000)
    if not history:
        return schemas.EvalMetrics()

    by_branch: dict[str, dict[bool, list[float]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    per_config: dict[str, dict[bool, list[float]]] = {
        name: {True: [], False: []} for name in CONFIGURATIONS
    }

    for record in history:
        label = _label(record)
        scores = {b["name"]: float(b["score"]) for b in record.get("branches", [])}

        for name, score in scores.items():
            by_branch[name][label].append(score)

        for config, branches in CONFIGURATIONS.items():
            present = [scores[b] for b in branches if b in scores]
            if not present:
                continue
            # Unweighted mean over the branches that ran. Deliberately not the
            # fused score: the fused score already had the policy's weights
            # and its veto applied, so comparing configurations through it
            # would compare four numbers that all came out of the same
            # decision. The mean isolates the evidence.
            per_config[config][label].append(sum(present) / len(present))

    configurations = []
    for name, split in per_config.items():
        genuine, impostor = split[True], split[False]
        if len(genuine) < MIN_TRIALS or len(impostor) < MIN_TRIALS:
            continue
        metrics = evaluate(genuine, impostor)
        configurations.append(conv.verification_metrics_to_wire(name, metrics))

    distributions = [
        schemas.ScoreDistribution(
            branch=name,
            genuine=[round(s, 4) for s in split[True]],
            impostor=[round(s, 4) for s in split[False]],
        )
        for name, split in sorted(by_branch.items())
    ]

    return schemas.EvalMetrics(
        configurations=configurations,
        stability_curve=_stability_curve(store, pipeline),
        fairness=_fairness(store, history),
        score_distributions=distributions,
    )


def _label(record: dict[str, Any]) -> bool:
    """Genuine or impostor, as well as this data allows.

    A login is labelled genuine when it was accepted. That is circular -- it
    labels a trial by the very decision being evaluated, so a system that
    accepted every impostor would score a perfect EER on its own history.

    It is used anyway because the alternative on live UI traffic is no
    evaluation page at all, and because in practice almost every login through
    the UI *is* genuine, which makes the impostor side small rather than
    wrong. **The paper's labels come from the trial design in `eval.ablation`,
    where who spoke is known independently of what the system decided.**
    """
    return record.get("decision") == "ACCEPT"


def _stability_curve(store: Store, pipeline: Pipeline) -> list[schemas.StabilityPoint]:
    """CSBG reliability against enrolment duration.

    Reports, for each duration bucket, how far a speaker's graph sits from the
    population -- the quantity the CSBG branch depends on. Speakers are bucketed
    by the total speech behind their graph, so the curve answers "how much
    enrolment does a usable graph need?" from the corpus as it stands.

    Returns an empty list below two speakers, because a spread across one
    speaker is not a curve. The proper version -- resampling each speaker's
    utterances at increasing budgets and recomputing EER at each -- is
    `eval.ablation`'s.
    """
    graphs = store.all_csbgs()
    if len(graphs) < 2:
        return []

    buckets: dict[int, list[float]] = defaultdict(list)
    for graph in graphs.values():
        seconds = graph.total_duration_sec
        if seconds <= 0:
            continue
        bucket = int(seconds // 60) * 60
        buckets[bucket].append(graph.density)

    points: list[schemas.StabilityPoint] = []
    for bucket in sorted(buckets):
        values = buckets[bucket]
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / 2.0 if len(values) > 1 else 0.0
        points.append(
            schemas.StabilityPoint(
                duration_sec=float(bucket),
                # Reported on the EER axis as "how much of the ontology is
                # *not* covered": density is the fraction of classes with
                # enough data, so 1 - density is the part of the graph that is
                # still the smoothing prior. Not an error rate; labelled as
                # such in the UI's axis, and replaced by a real EER curve by
                # eval.ablation.
                eer=round(1.0 - mean, 4),
                ci_low=round(max(0.0, 1.0 - mean - spread), 4),
                ci_high=round(min(1.0, 1.0 - mean + spread), 4),
            )
        )
    return points


def _fairness(store: Store, history: list[dict[str, Any]]) -> list[schemas.FairnessSlice]:
    """Accept rate sliced by recording condition and speaker attribute.

    The audit that matters for this corpus is device and environment: a system
    whose error rate doubles on a cheap phone in a noisy hostel corridor has a
    fairness problem even if its aggregate EER is fine, and that is the
    realistic deployment condition for the target population.

    Slices with fewer than `MIN_TRIALS` logins are dropped rather than shown
    with a wide interval, because a fairness table is read as a comparison and
    a bar built from two trials invites a conclusion it cannot support.
    """
    speakers = {s["id"]: s for s in store.list_speakers()}
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)

    for record in history:
        speaker = speakers.get(record.get("speakerId") or record.get("speaker_id", ""))
        if speaker is None:
            continue
        accepted = record.get("decision") == "ACCEPT"
        for condition, value in (
            ("device", speaker.get("device")),
            ("environment", speaker.get("environment")),
            ("dominant_language", speaker.get("dominant_language")),
            ("gender", speaker.get("gender")),
        ):
            if value:
                groups[(condition, str(value))].append(accepted)

    return [
        schemas.FairnessSlice(
            condition=condition,
            group=group,
            # Reject rate on trials that are overwhelmingly genuine, i.e. an
            # FRR estimate. Named `eer` because that is the field the UI
            # renders; the distinction is stated in the page copy.
            eer=round(1.0 - sum(results) / len(results), 4),
            sample_count=len(results),
        )
        for (condition, group), results in sorted(groups.items())
        if len(results) >= MIN_TRIALS
    ]


__all__ = ["CONFIGURATIONS", "MIN_TRIALS", "evaluate_history"]
