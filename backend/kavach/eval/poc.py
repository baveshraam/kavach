"""Proof-of-concept evaluation of the voice gate for ONE enrolled speaker, as a pure function over embeddings.

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
One enrollee, a handful of recorded sittings, impostors who are recorded voices (not people at the microphone).
It says how well the voice embedding separates this speaker from those voices, and what a threshold fitted without
the test trials costs. It is not a population result, not a spoofing result and not a code-switch result.

THE PROTOCOL (no number is computed on data that chose it)
-----------------------------------------------------------
* Genuine: leave-one-sitting-out. Held-out sitting k is scored against a template built from the other sittings.
* Impostor speakers are split once, seeded and stratified by corpus, into three disjoint parts:
  ``norm`` (reference cohort for score normalisation and mean compensation), ``fit`` (fits the threshold) and
  ``test`` (the only impostors any reported rate is computed on).
* The threshold of fold k is fitted on an inner loop that never sees sitting k or the test speakers: the other
  sittings are scored leave-one-out against the ``fit`` speakers.
* Variants are chosen by their DEV equal-error rate alone; the test table is reported for all of them.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..calibrate_voice import _unit
from ..attacks.suite import wilson_interval
from .enrollee_stats import d_prime, reported_interval
from .metrics import compute_eer, evaluate

LIMITS = (
    "LIMITS OF THIS EVIDENCE. One enrolled speaker, a few recorded sittings, and impostors who are recorded voices "
    "(pilot speakers on phones, public read corpora), not people speaking into the same microphone. The voice "
    "embedding is the only branch measured. Nothing here is a population result, a spoofing-resistance result "
    "or a code-switch result."
)

DECISION_RULE = (
    "A variant is reported as better than the baseline only if its test EER and its test minDCF are both lower; it "
    "is reported as established only if its EER interval lies entirely below the baseline's. The headline variant is "
    "the one with the lowest dev EER, chosen before any test score is read. Nothing is added after the fact."
)

#: Parts of the impostor speakers; test gets the rest.
_DEV_SHARE = 0.25
#: A source with fewer speakers than this is too small to split and goes wholly to test.
_MIN_SOURCE = 4
#: Impostor trials needed to see this many events at a target false-accept rate (calibrate_voice uses 5 too).
_MIN_EVENTS = 5


# --------------------------------------------------------------------------- data


@dataclass(frozen=True, slots=True)
class Sitting:
    name: str
    vecs: np.ndarray  # (clips, dim): one embedding per whole clip; what a template is built from
    probes: np.ndarray | None = None  # (probes, dim): what is scored as genuine; the clips cut to the cohort's length. None = vecs
    probe_clip: np.ndarray | None = None  # (probes,): index into `vecs` of the clip each probe was cut from

    @property
    def p(self) -> np.ndarray:
        return self.vecs if self.probes is None else self.probes

    @property
    def clip_of(self) -> np.ndarray:
        if self.probes is None:
            return np.arange(len(self.vecs))
        if self.probe_clip is None or len(self.probe_clip) != len(self.probes):
            raise ValueError(f"sitting {self.name}: probe_clip must say which clip each probe was cut from")
        return np.asarray(self.probe_clip)


@dataclass(frozen=True, slots=True)
class Cohort:
    vec: np.ndarray  # (chunks, dim)
    speaker: list[str]
    source: list[str]


def split_speakers(cohort: Cohort, seed: int) -> dict[str, set[str]]:
    """Seeded, source-stratified split of impostor SPEAKERS into norm / fit / test."""
    by_source: dict[str, list[str]] = {}
    for s, src in zip(cohort.speaker, cohort.source):
        by_source.setdefault(src, [])
        if s not in by_source[src]:
            by_source[src].append(s)
    rng = random.Random(seed)
    parts: dict[str, set[str]] = {"norm": set(), "fit": set(), "test": set()}
    for src in sorted(by_source):
        spk = sorted(by_source[src])
        if len(spk) < _MIN_SOURCE:
            parts["test"].update(spk)
            continue
        rng.shuffle(spk)
        n = max(1, round(len(spk) * _DEV_SHARE))
        parts["norm"].update(spk[:n])
        parts["fit"].update(spk[n:2 * n])
        parts["test"].update(spk[2 * n:])
    return parts


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    doc: str
    build: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def _make(meansub: bool, asnorm: bool, trim: float = 0.0, k: int = 100):
    def build(enrol: np.ndarray, norm: np.ndarray):
        mu = _unit(norm).mean(axis=0) if meansub else 0.0

        def T(x):
            return _unit(_unit(x) - mu)

        e = T(enrol)
        c = e.mean(axis=0)
        c /= np.linalg.norm(c)
        if trim:
            own = e @ c
            c = e[own >= np.quantile(own, trim)].mean(axis=0)
            c /= np.linalg.norm(c)
        cohort = T(norm)
        if not asnorm:
            return lambda p: T(p) @ c
        kk = max(2, min(k, len(cohort) // 2))
        t_top = np.sort(cohort @ c)[-kk:]
        mu_t, sd_t = t_top.mean(), t_top.std() + 1e-6

        def score(p):
            P = T(p)
            p_top = np.sort(P @ cohort.T, axis=1)[:, -kk:]
            mu_p, sd_p = p_top.mean(axis=1), p_top.std(axis=1) + 1e-6
            return 0.5 * ((P @ c - mu_t) / sd_t + (P @ c - mu_p) / sd_p)

        return score

    return build


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in (
        Variant("baseline", "Cosine to the centroid of the enrolment embeddings (what the live system does).", _make(False, False)),
        Variant("meansub", "Subtract the norm cohort's mean embedding before cosine (global channel compensation).", _make(True, False)),
        Variant("asnorm", "Adaptive symmetric score normalisation against the top-100 nearest norm-cohort embeddings.", _make(False, True)),
        Variant("meansub_asnorm", "Mean compensation, then adaptive symmetric score normalisation.", _make(True, True)),
        Variant("trimmed", "Centroid after dropping the 20% of enrolment embeddings farthest from it.", _make(False, False, trim=0.2)),
    )
}


# --------------------------------------------------------------------------- statistics


def det_rates(g: np.ndarray, i: np.ndarray):
    """(thresholds, FAR, FRR) at every observed score, in O(n log n): FAR = P(impostor >= t), FRR = P(genuine < t)."""
    t = np.unique(np.concatenate([g, i]))
    far = 1.0 - np.searchsorted(np.sort(i), t, side="left") / len(i)
    frr = np.searchsorted(np.sort(g), t, side="left") / len(g)
    return t, far, frr


def fast_metrics(g: np.ndarray, i: np.ndarray, *, p_target: float = 0.05) -> dict[str, float]:
    """EER (nearest point), minDCF (c_miss = c_fa = 1), AUC and d' without the O(thresholds x trials) scan of `metrics`.

    For score sets of tens of thousands of trials, where `metrics.evaluate` would take minutes."""
    from .metrics import compute_auc

    _, far, frr = det_rates(g, i)
    k = int(np.argmin(np.abs(far - frr)))
    dcf = (p_target * frr + (1.0 - p_target) * far) / min(p_target, 1.0 - p_target)
    return {"eer": float((far[k] + frr[k]) / 2.0), "min_dcf": float(dcf.min()), "auc": float(compute_auc(g, i)),
            "d_prime": float(d_prime(g, i))}


def _eer_fast(g: np.ndarray, i: np.ndarray) -> float:
    """Nearest-point EER in O(n log n), for the bootstraps. The headline EER uses `metrics.compute_eer`."""
    _, far, frr = det_rates(g, i)
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2.0)


def _eer_intervals(g_by: dict[str, np.ndarray], i_by: dict[str, np.ndarray], n_boot: int, seed: int):
    """(probe-level interval, cluster interval). Probes are resampled alone, then whole sittings / speakers."""
    rng = np.random.default_rng(seed)
    G, I = np.concatenate(list(g_by.values())), np.concatenate(list(i_by.values()))
    gk, ik = list(g_by), list(i_by)
    probe, cluster = [], []
    for _ in range(n_boot):
        probe.append(_eer_fast(G[rng.integers(0, len(G), len(G))], I[rng.integers(0, len(I), len(I))]))
        gi, ii = rng.integers(0, len(gk), len(gk)), rng.integers(0, len(ik), len(ik))
        cluster.append(_eer_fast(np.concatenate([g_by[gk[j]] for j in gi]), np.concatenate([i_by[ik[j]] for j in ii])))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return q(probe), q(cluster)


def _threshold_eer(g: np.ndarray, i: np.ndarray) -> float:
    """The EER threshold, interpolated exactly as `metrics.compute_eer` does but in O(n log n)."""
    if g.min() > i.max():  # no overlap: compute_eer would return a tie-break, not evidence
        return float((g.min() + i.max()) / 2.0)
    t, far, frr = det_rates(g, i)
    t, far, frr = np.concatenate([[t[0] - 1.0], t]), np.concatenate([[1.0], far]), np.concatenate([[0.0], frr])
    diff = far - frr
    crossing = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(crossing) == 0:
        return float(t[int(np.argmin(np.abs(diff)))])
    k = int(crossing[0])
    alpha = 0.0 if diff[k + 1] == diff[k] else float(-diff[k] / (diff[k + 1] - diff[k]))
    return float(t[k] + alpha * (t[k + 1] - t[k]))


def _threshold_far(i: np.ndarray, target: float) -> float:
    """Lowest threshold accepting at most `target` of the dev impostors."""
    desc = np.sort(i)[::-1]
    k = int(np.floor(target * len(desc)))
    return float(np.nextafter(desc[min(k, len(desc) - 1)], np.inf))


# --------------------------------------------------------------------------- the run


@dataclass(slots=True)
class PocResult:
    protocol: dict[str, Any]
    variants: dict[str, dict[str, Any]]
    selected: str
    notes: list[str]
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = field(default_factory=dict)
    sitting_matrix: dict[str, Any] = field(default_factory=dict)
    budget: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"limits": LIMITS, "protocol": self.protocol, "variants": self.variants,
                "selected": self.selected, "notes": self.notes, "sitting_matrix": self.sitting_matrix, "budget": self.budget}


def sitting_matrix(sittings: list[Sitting]) -> dict[str, Any]:
    """Mean cosine of sitting k's probes against a template built from sitting j alone (rows = template, columns = probes).

    The diagonal is within-sitting and leave-one-clip-out (a probe is never scored against the clip it was cut from), so it is
    the optimistic same-sitting number; the off-diagonal cells are what a new day or room costs, and which sittings resemble
    each other (two on one day should)."""
    names = [s.name for s in sittings]
    unit = {s.name: _unit(s.vecs) for s in sittings}
    probe = {s.name: _unit(s.p) for s in sittings}
    cells: dict[str, dict[str, float | None]] = {}
    for j in sittings:
        row: dict[str, float | None] = {}
        for k in sittings:
            if j.name != k.name:
                c = unit[j.name].mean(axis=0)
                row[k.name] = float((probe[k.name] @ (c / np.linalg.norm(c))).mean())
            else:
                clip = k.clip_of
                sc = []
                for n in range(len(unit[j.name])):
                    rest = np.delete(unit[j.name], n, axis=0)
                    if len(rest) and (clip == n).any():
                        c = rest.mean(axis=0)
                        sc.append(probe[k.name][clip == n] @ (c / np.linalg.norm(c)))
                row[k.name] = float(np.concatenate(sc).mean()) if sc else None
        cells[j.name] = row
    return {"sittings": names, "mean_score": cells}


def _validate(sittings: list[Sitting], cohort: Cohort, enrollee: str, names: list[str]) -> None:
    for n in names:
        if n not in VARIANTS:
            raise ValueError(f"unknown variant {n!r}; known: {', '.join(VARIANTS)}")
    if len(sittings) < 3:
        raise ValueError("at least three sittings are needed: one is held out and the threshold is fitted on leave-one-out "
                         "scores among the other two, so with fewer there is no leak-free threshold")
    if len({s.name for s in sittings}) != len(sittings):
        raise ValueError("sitting names must be unique")
    dim = cohort.vec.shape[1]
    for s in sittings:
        for arr in (s.vecs, s.p):
            if arr.ndim != 2 or len(arr) == 0:
                raise ValueError(f"sitting {s.name} is empty: no clips to score")
            if arr.shape[1] != dim:
                raise ValueError(f"sitting {s.name} has dimension {arr.shape[1]}, the cohort {dim}")
            if not np.isfinite(arr).all() or (np.linalg.norm(arr, axis=1) < 1e-9).any():
                raise ValueError(f"non-finite or zero embedding in sitting {s.name}: refusing to score it")
    if len(cohort.vec) == 0 or not np.isfinite(cohort.vec).all() or (np.linalg.norm(cohort.vec, axis=1) < 1e-9).any():
        raise ValueError("non-finite, zero or missing cohort embedding: refusing to score it")
    if not (len(cohort.vec) == len(cohort.speaker) == len(cohort.source)):
        raise ValueError("cohort vectors, speakers and sources differ in length")
    if enrollee in set(cohort.speaker):
        raise ValueError(f"the enrollee {enrollee!r} is among the impostors: remove them before evaluating")


def make_protocol(sittings, cohort, split, rows, seed, names, rules, n_boot) -> dict[str, Any]:
    """Everything that must be fixed before any score is computed, with a hash of it (the pre-registration)."""
    protocol = {
        "seed": seed, "variants": names, "rules": rules, "n_boot": n_boot,
        "sittings": {s.name: int(len(s.vecs)) for s in sittings},
        "probes": {s.name: int(len(s.p)) for s in sittings},
        "split": {p: sorted(x) for p, x in split.items()},
        "trials": {p: int(len(r)) for p, r in rows.items()},
        "decision_rule": DECISION_RULE,
    }
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def prepare(sittings, cohort, *, seed=7, variants=None, far_targets=(0.01, 0.001), n_boot=1000, enrollee="S04"):
    """Validate and fix the protocol; returns (protocol, names, split, rows). Pure; touches no score."""
    names = list(variants) if variants else list(VARIANTS)
    _validate(sittings, cohort, enrollee, names)
    split = split_speakers(cohort, seed)
    for part, spk in split.items():
        if len(spk) < 2:
            raise ValueError(f"the {part} part has {len(spk)} impostor speakers; each part needs at least two speakers")
    spk_arr = np.array(cohort.speaker)
    rows = {p: np.where(np.isin(spk_arr, sorted(s)))[0] for p, s in split.items()}
    rules = ["eer"] + [f"far{t:g}" for t in far_targets]
    return make_protocol(sittings, cohort, split, rows, seed, names, rules, n_boot), names, split, rows


def run_poc(
    sittings: list[Sitting],
    cohort: Cohort,
    *,
    seed: int = 7,
    variants: list[str] | None = None,
    far_targets: tuple[float, ...] = (0.01, 0.001),
    n_boot: int = 1000,
    enrollee: str = "S04",
) -> PocResult:
    protocol, names, split, rows = prepare(sittings, cohort, seed=seed, variants=variants, far_targets=far_targets,
                                           n_boot=n_boot, enrollee=enrollee)
    spk_arr = np.array(cohort.speaker)
    vec = {p: cohort.vec[r] for p, r in rows.items()}
    sit = {s.name: s.vecs for s in sittings}
    prb = {s.name: s.p for s in sittings}
    order = [s.name for s in sittings]
    rules = protocol["rules"]

    notes: list[str] = []
    if len(order) < 5:
        notes.append(f"Only {len(order)} sittings: the sitting-cluster intervals are not informative (a percentile "
                     "bootstrap needs about five clusters), so every interval is reported as the wider of the "
                     "probe-level and the cluster one, and none of them supports a claim of statistical significance.")
    test_spk = sorted(split["test"])
    spk_idx = {s: np.where(spk_arr[rows["test"]] == s)[0] for s in test_spk}
    src_of = {s: cohort.source[int(np.where(spk_arr == s)[0][0])] for s in test_spk}

    out: dict[str, dict[str, Any]] = {}
    all_scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name in names:
        build = VARIANTS[name].build
        gen: dict[str, np.ndarray] = {}
        imp_fold: dict[str, np.ndarray] = {}
        thr: dict[str, dict[str, float]] = {}
        dev_g, dev_i = [], []
        for k in order:
            others = [o for o in order if o != k]
            inner_g, inner_i = [], []
            for j in others:
                tmpl = np.concatenate([sit[o] for o in others if o != j])
                f = build(tmpl, vec["norm"])
                inner_g.append(f(prb[j]))
                inner_i.append(f(vec["fit"]))
            ig, ii = np.concatenate(inner_g), np.concatenate(inner_i)
            dev_g.append(ig); dev_i.append(ii)
            thr[k] = {"eer": _threshold_eer(ig, ii)}
            for t in far_targets:
                thr[k][f"far{t:g}"] = _threshold_far(ii, t)
            f = build(np.concatenate([sit[o] for o in others]), vec["norm"])
            gen[k] = f(prb[k])
            imp_fold[k] = f(vec["test"])
        if not (all(np.isfinite(v).all() for v in gen.values()) and all(np.isfinite(v).all() for v in imp_fold.values())):
            raise ValueError("non-finite score: refusing to report it")

        imp_by_spk = {s: np.concatenate([imp_fold[k][spk_idx[s]] for k in order]) for s in test_spk}
        G, I = np.concatenate([gen[k] for k in order]), np.concatenate(list(imp_by_spk.values()))
        m = evaluate(G, I)
        probe_iv, cluster_iv = _eer_intervals(gen, imp_by_spk, n_boot, seed)
        # A bootstrap of a perfect system collapses to (0, 0); Wilson on the genuine count is the floor, as for the rates.
        kg_eer = int(round(m.eer * len(G)))
        w_lo, w_hi = wilson_interval(kg_eer, len(G))
        eer_iv = (min(probe_iv[0], cluster_iv[0], w_lo), max(probe_iv[1], cluster_iv[1], w_hi))

        ops: dict[str, Any] = {}
        for r in rules:
            g_flags = {k: [bool(s < thr[k][r]) for s in gen[k]] for k in order}
            i_flags = {s: [bool(x >= thr[k][r]) for k in order for x in imp_fold[k][spk_idx[s]]] for s in test_spk}
            kg, ki = sum(map(sum, g_flags.values())), sum(map(sum, i_flags.values()))
            ng, ni = len(G), len(I)
            frr_iv, far_iv = reported_interval(kg, ng, g_flags), reported_interval(ki, ni, i_flags)
            dev_n = sum(len(x) for x in dev_i) // len(order)
            ops[r] = {
                "frr": kg / ng, "frr_interval": [frr_iv[0], frr_iv[1]], "n_genuine": ng, "false_rejects": int(kg),
                "far": ki / ni, "far_interval": [far_iv[0], far_iv[1]], "n_impostor": ni, "false_accepts": int(ki),
                "cluster_interval_informative": bool(frr_iv[2] and far_iv[2]),
                "far_resolvable_on_dev": bool(r == "eer" or dev_n * float(r[3:]) >= _MIN_EVENTS),
            }

        per_sitting = {k: {"n": int(len(gen[k])), "mean": float(gen[k].mean()), "min": float(gen[k].min()),
                           **{f"frr_{r}": float(np.mean(gen[k] < thr[k][r])) for r in rules}} for k in order}
        per_source: dict[str, dict[str, Any]] = {}
        for src in sorted(set(src_of.values())):
            ss = [s for s in test_spk if src_of[s] == src]
            v = np.concatenate([imp_by_spk[s] for s in ss])
            per_source[src] = {"speakers": len(ss), "trials": int(len(v)), "mean": float(v.mean()), "max": float(v.max()),
                               **{f"far_{r}": float(np.mean([x >= thr[k][r] for s in ss for k in order for x in imp_fold[k][spk_idx[s]]])) for r in rules}}
        nearest = sorted(((float(v.max()), s, src_of[s]) for s, v in imp_by_spk.items()), reverse=True)[:5]
        devG, devI = np.concatenate(dev_g), np.concatenate(dev_i)

        out[name] = {
            "doc": VARIANTS[name].doc,
            "dev_eer": float(compute_eer(devG, devI)[0]),
            "test": {"eer": m.eer, "eer_interval": list(eer_iv), "eer_probe_interval": list(probe_iv),
                     "eer_cluster_interval": list(cluster_iv), "min_dcf": m.min_dcf, "auc": m.auc,
                     "d_prime": d_prime(G, I), "n_genuine": m.n_genuine, "n_impostor": m.n_impostor,
                     "genuine_min": float(G.min()), "impostor_max": float(I.max()), "gap": float(G.min() - I.max())},
            "operating_points": ops,
            "folds": {k: {"thresholds": thr[k]} for k in order},
            "per_sitting": per_sitting, "per_source": per_source,
            "nearest_strangers": [{"speaker": s, "source": src, "max": mx} for mx, s, src in nearest],
        }
        all_scores[name] = {"genuine": gen, "impostor": imp_by_spk}

    dev = {n: out[n]["dev_eer"] for n in names}
    selected = min(dev, key=lambda n: (dev[n], names.index(n)))
    tied = [n for n in names if dev[n] == dev[selected]]
    if len(tied) > 1:
        notes.append(f"{len(tied)} variants tie for the lowest dev EER ({', '.join(tied)}), so dev cannot choose between them; "
                     f"`{selected}` is the first in the registered order. The test table below shows all of them.")
    n_clips, n_probes = sum(protocol["sittings"].values()), sum(protocol["probes"].values())
    if n_probes > n_clips:
        notes.append(f"The {n_probes} owner probes are cut from {n_clips} clips and probes from one clip are not independent, "
                     "so the probe counts overstate the evidence for the owner's rates.")
    base = out.get("baseline")
    if base:
        for n, v in out.items():
            b = base["test"]
            v["vs_baseline"] = {
                "delta_eer": v["test"]["eer"] - b["eer"], "delta_min_dcf": v["test"]["min_dcf"] - b["min_dcf"],
                "better": bool(v["test"]["eer"] < b["eer"] and v["test"]["min_dcf"] < b["min_dcf"]),
                "established": bool(v["test"]["eer_interval"][1] < b["eer_interval"][0]),
            }
    matrix = sitting_matrix(sittings)
    budget: list[dict[str, Any]] = []
    if "baseline" in names:
        for b in (1, 2, 5, 10, 20, None):
            g_all, i_all = [], []
            for k in order:
                enrol = np.concatenate([sit[o] for o in order if o != k])[:b]
                f = VARIANTS["baseline"].build(enrol, vec["norm"])
                g_all.append(f(prb[k]))
                i_all.append(f(vec["test"]))
            g, i = np.concatenate(g_all), np.concatenate(i_all)
            budget.append({"clips": b if b else "all", "eer": fast_metrics(g, i)["eer"], "genuine_mean": float(g.mean())})
    return PocResult(protocol, out, selected, notes, all_scores, matrix, budget)


# --------------------------------------------------------------------------- report


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def _iv(iv) -> str:
    return f"{_pct(iv[0])} to {_pct(iv[1])}"


def render(r: PocResult) -> str:
    p = r.protocol
    L = [LIMITS, "", "## Protocol",
         f"- Genuine: leave-one-sitting-out over {len(p['sittings'])} sittings ("
         + ", ".join(f"{k} {n} clips -> {p['probes'][k]} probes" for k, n in p["sittings"].items()) + "). The template is the "
         "centroid of the other sittings' whole clips; the probes are what is scored.",
         f"- Impostor speakers split once (seed {p['seed']}, stratified by corpus): norm {len(p['split']['norm'])}, "
         f"fit {len(p['split']['fit'])}, test {len(p['split']['test'])} speakers ({p['trials']['test']} test chunks, "
         f"each scored against {len(p['sittings'])} fold templates).",
         "- A fold's threshold is fitted on the other sittings (leave-one-out) against the `fit` speakers only; "
         "it never sees the held-out sitting or the `test` speakers.",
         f"- Pre-registration hash {p['sha256'][:16]}. {DECISION_RULE}", ""]
    for n in r.notes:
        L.append(f"- Note: {n}")
    sel = r.variants[r.selected]
    L += ["", f"## Headline: `{r.selected}` (lowest dev EER {_pct(sel['dev_eer'])}, chosen before test)"]
    t = sel["test"]
    L += [f"- Test EER {_pct(t['eer'])} (interval {_iv(t['eer_interval'])}); minDCF {t['min_dcf']:.4f} (p_target 0.05); "
          f"AUC {t['auc']:.4f}; d' {t['d_prime']:.2f}.",
          f"- {t['n_genuine']} genuine probes, {t['n_impostor']} impostor trials. Worst genuine {t['genuine_min']:.3f}, best "
          f"impostor {t['impostor_max']:.3f}, gap {t['gap']:+.3f}."]
    for rule, o in sel["operating_points"].items():
        tag = "" if o["far_resolvable_on_dev"] else " [provisional: the dev impostors are too few to resolve this FAR]"
        L.append(f"- Threshold rule `{rule}` (fitted on dev): owner rejected {_pct(o['frr'])} "
                 f"({o['false_rejects']}/{o['n_genuine']}, {_iv(o['frr_interval'])}); strangers accepted {_pct(o['far'])} "
                 f"({o['false_accepts']}/{o['n_impostor']}, {_iv(o['far_interval'])}){tag}.")
    L += ["", "## Every variant on the test data (dev EER chose the headline, not this table)",
          "| variant | dev EER | test EER [interval] | minDCF | AUC | vs baseline |", "|---|---|---|---|---|---|"]
    for n, v in r.variants.items():
        d = v.get("vs_baseline")
        verdict = "baseline" if n == "baseline" else (
            ("established better" if d["established"] else "lower but not established" if d["better"] else "not better") if d else "")
        L.append(f"| {n} | {_pct(v['dev_eer'])} | {_pct(v['test']['eer'])} [{_iv(v['test']['eer_interval'])}] | "
                 f"{v['test']['min_dcf']:.4f} | {v['test']['auc']:.4f} | {verdict} |")
    L += ["", f"## Per held-out sitting (`{r.selected}`; the threshold is that fold's own)",
          "| sitting | probes | mean | min | rejected (eer rule) |", "|---|---|---|---|---|"]
    for k, s in sel["per_sitting"].items():
        L.append(f"| {k} | {s['n']} | {s['mean']:.3f} | {s['min']:.3f} | {_pct(s['frr_eer'])} |")
    L += ["", "## Strangers by corpus", "| corpus | speakers | trials | mean | max | accepted (eer rule) |", "|---|---|---|---|---|---|"]
    for src, s in sel["per_source"].items():
        L.append(f"| {src} | {s['speakers']} | {s['trials']} | {s['mean']:.3f} | {s['max']:.3f} | {_pct(s['far_eer'])} |")
    L += ["", "Nearest strangers (highest score against the owner): "
          + "; ".join(f"{x['speaker']} ({x['source']}) {x['max']:.3f}" for x in sel["nearest_strangers"]) + ".",
          "", "Not measured here: fusion with the knowledge and code-switch branches, replay / splice / clone attacks, "
          "the words check, people speaking into the owner's microphone."]
    return "\n".join(L)
