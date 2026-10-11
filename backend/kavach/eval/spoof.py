"""Synthetic-voice attacks scored against the single-enrollee templates, with the thresholds the study fitted.

THE RULE THAT KEEPS THIS HONEST
-------------------------------
The attacker's reference recording is the sitting held out of that fold's template (`split_reference`), never the sittings the
template was built from, and the threshold is the one fitted for that fold without the held-out sitting. An attack is therefore
scored exactly like a held-out owner probe: same template, same variant, same threshold.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ..attacks.suite import wilson_interval
from .poc import VARIANTS, PocResult, Sitting

LIMITS = (
    "LIMITS OF THIS EVIDENCE. Two synthetic-voice attacks, neither state of the art: kNN voice conversion using the owner's own "
    "recordings of one held-out sitting as the reference, and text-to-speech voices that are not clones. A strong zero-shot cloner, a "
    "replay of the owner's recording and a human impersonator were not tested. The rates are for the voice embedding alone; the live "
    "system also checks the spoken words and a fresh challenge."
)


def split_reference(sittings: list[str]) -> dict[str, str]:
    """fold -> the sitting whose clips the attacker holds. It is the fold's own held-out sitting, which its template excludes."""
    return {k: k for k in sittings}


def select_sources(items: list[tuple[str, str]], *, n_speakers: int, seed: int) -> list[tuple[str, str]]:
    """Every utterance of `n_speakers` randomly chosen speakers, in a fixed order. Items are (speaker, utterance)."""
    speakers = sorted({s for s, _ in items})
    if len(speakers) < n_speakers:
        raise ValueError(f"asked for {n_speakers} source speakers but only {len(speakers)} exist")
    chosen = set(random.Random(seed).sample(speakers, n_speakers))
    return sorted((s, u) for s, u in items if s in chosen)


def attack_report(
    sittings: list[Sitting], cohort: Any, result: PocResult, attacks: dict[str, dict[str, np.ndarray]],
    raw: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, dict[str, Any]]:
    """{attack: {variant: {score_mean, score_max, n_probes, rules: {rule: {accepted, n, rate, interval}}, per_fold}}}.

    `attacks[name][fold]` are the embeddings of the attack probes aimed at that fold's template.
    """
    order = [s.name for s in sittings]
    spk = np.array(cohort.speaker)
    norm = cohort.vec[np.isin(spk, result.protocol["split"]["norm"])]
    whole = {s.name: s.vecs for s in sittings}
    out: dict[str, dict[str, Any]] = {}
    for name, per_fold in attacks.items():
        missing = [k for k in order if k not in per_fold]
        if missing:
            raise ValueError(f"attack {name!r} has no probes for fold(s) {', '.join(missing)}: every fold needs its own")
        for k in order:
            a = np.asarray(per_fold[k])
            if a.ndim != 2 or len(a) == 0 or not np.isfinite(a).all():
                raise ValueError(f"non-finite or empty attack embedding for {name!r} fold {k}: refusing to score it")
        out[name] = {}
        for v, vres in result.variants.items():
            build = VARIANTS[v].build
            scores = {k: build(np.concatenate([whole[o] for o in order if o != k]), norm)(np.asarray(per_fold[k])) for k in order}
            allv = np.concatenate(list(scores.values()))
            if raw is not None:
                raw.setdefault(name, {})[v] = allv
            rules = {}
            for rule in next(iter(vres["folds"].values()))["thresholds"]:
                acc = int(sum((scores[k] >= vres["folds"][k]["thresholds"][rule]).sum() for k in order))
                n = int(len(allv))
                lo, hi = wilson_interval(acc, n)
                rules[rule] = {"accepted": acc, "n": n, "rate": acc / n, "interval": [lo, hi]}
            out[name][v] = {"score_mean": float(allv.mean()), "score_max": float(allv.max()), "n_probes": int(len(allv)),
                            "rules": rules, "per_fold": {k: float(scores[k].mean()) for k in order}}
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def render(report: dict[str, dict[str, Any]], result: PocResult) -> str:
    sel = result.selected
    L = [LIMITS, "", f"## Attack acceptance for `{sel}` (thresholds fitted on dev, per fold)",
         "| attack | probes | mean score | max score | " + " | ".join(f"accepted at `{r}`" for r in result.variants[sel]["operating_points"]) + " |",
         "|---|---|---|---|" + "---|" * len(result.variants[sel]["operating_points"])]
    for name, by in report.items():
        d = by[sel]
        cells = [f"{c['accepted']}/{c['n']} = {_pct(c['rate'])} ({_pct(c['interval'][0])}-{_pct(c['interval'][1])})" for c in d["rules"].values()]
        L.append(f"| {name} | {d['n_probes']} | {d['score_mean']:.3f} | {d['score_max']:.3f} | " + " | ".join(cells) + " |")
    own = result.variants[sel]["operating_points"]
    L += ["", "For scale, the owner's own held-out probes were accepted at "
          + ", ".join(f"`{r}`: {_pct(1 - o['frr'])}" for r, o in own.items()) + ".",
          "Probes cut from one utterance are not independent, so the intervals understate the uncertainty."]
    return "\n".join(L)
