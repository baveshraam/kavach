"""Population benchmark: many enrollees, cross-session, one global threshold -- the multi-speaker companion to `poc`.

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
On public read English speech (LibriSpeech; a chapter is a session) it measures how the voice embedding behaves over MANY
enrollees, so intervals have real clusters (the enrollees). It is not code-switched speech, not the owner's laptop microphone,
and not a spoofing result.

PROTOCOL (nothing is computed on data that chose it)
----------------------------------------------------
* Speakers are split once, seeded and stratified by source, into ``norm`` (reference cohort for score normalisation and mean
  compensation), ``dev`` (fits the global threshold) and ``test`` (the only data any reported rate is computed on).
* An enrollee has at least two sessions. Each session is held out in turn: the template is built from the whole clips of the
  other sessions; the held-out session's 5 s probes are the genuine trials; the 5 s probes of every OTHER speaker of the same
  part are the impostor trials.
* One global threshold per rule is fitted on the dev part and applied unchanged to the test part.
* Intervals are a cluster bootstrap over enrollees (a cluster is an enrollee's genuine and impostor trials together) and Wilson;
  the reported one is never narrower than Wilson.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..attacks.suite import wilson_interval
from .enrollee_stats import MIN_CLUSTERS
from .poc import VARIANTS, _threshold_eer, _threshold_far, fast_metrics

LIMITS = (
    "LIMITS OF THIS EVIDENCE. Public read English speech (LibriSpeech), studio-quality, one chapter per session; strangers are "
    "other speakers of the same corpus. This is not Tamil-English code-switched speech, not a laptop microphone, and not a "
    "spoofing result. It shows how the voice embedding behaves over many enrollees under the same leak-free protocol as the "
    "single-enrollee study."
)

#: Share of each source's speakers held as the normalisation cohort; the rest is halved into dev and test.
_NORM_SHARE = 0.25
_MIN_ENROLLEES = 3


@dataclass(frozen=True, slots=True)
class PopSpeaker:
    id: str
    source: str
    whole: dict[str, np.ndarray]  # session -> (clips, dim): what a template is built from
    probes: dict[str, np.ndarray]  # session -> (probes, dim): 5 s chunks, what is scored

    @property
    def eligible(self) -> bool:
        both = [s for s in self.whole if s in self.probes and len(self.probes[s]) and len(self.whole[s])]
        return len(both) >= 2


def split_population(speakers: list[PopSpeaker], seed: int) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for s in speakers:
        by.setdefault(s.source, []).append(s.id)
    rng = random.Random(seed)
    parts: dict[str, list[str]] = {"norm": [], "dev": [], "test": []}
    for src in sorted(by):
        ids = sorted(by[src])
        rng.shuffle(ids)
        n_norm = max(1, round(len(ids) * _NORM_SHARE))
        rest = ids[n_norm:]
        n_dev = len(rest) // 2
        parts["norm"] += ids[:n_norm]
        parts["dev"] += rest[:n_dev]
        parts["test"] += rest[n_dev:]
    return {k: sorted(v) for k, v in parts.items()}


@dataclass(slots=True)
class PopResult:
    protocol: dict[str, Any]
    variants: dict[str, dict[str, Any]]
    artefact: dict[str, Any]
    budget: list[dict[str, Any]]
    notes: list[str]
    scores: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"limits": LIMITS, "protocol": self.protocol, "variants": self.variants, "artefact": self.artefact,
                "budget": self.budget, "notes": self.notes}


def _validate(speakers: list[PopSpeaker], names: list[str]) -> None:
    for n in names:
        if n not in VARIANTS:
            raise ValueError(f"unknown variant {n!r}; known: {', '.join(VARIANTS)}")
    if len({s.id for s in speakers}) != len(speakers):
        raise ValueError("speaker ids must be unique")
    dims = set()
    for s in speakers:
        for d in (s.whole, s.probes):
            for sess, a in d.items():
                if a.ndim != 2 or not np.isfinite(a).all() or (len(a) and (np.linalg.norm(a, axis=1) < 1e-9).any()):
                    raise ValueError(f"non-finite or zero embedding for {s.id} session {sess}: refusing to score it")
                dims.add(a.shape[1])
    if len(dims) != 1:
        raise ValueError(f"embeddings of different dimensions {sorted(dims)}: a cohort from another model?")


def _cat(parts: list[np.ndarray], dim: int) -> np.ndarray:
    return np.concatenate(parts) if parts else np.zeros((0, dim))


def _counts_interval(k: np.ndarray, n: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, bool]:
    """Wider of Wilson and the cluster bootstrap (clusters = enrollees) for a pooled rate; `informative` needs enough clusters."""
    K, N = int(k.sum()), int(n.sum())
    w_lo, w_hi = wilson_interval(K, N)
    idx = np.random.default_rng(seed).integers(0, len(k), (n_boot, len(k)))
    rates = k[idx].sum(axis=1) / np.maximum(n[idx].sum(axis=1), 1)
    c_lo, c_hi = float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))
    return min(w_lo, c_lo), max(w_hi, c_hi), bool(len(k) >= MIN_CLUSTERS and K > 0)


def _eer_cluster(g_by: dict[str, np.ndarray], i_by: dict[str, np.ndarray], n_boot: int, seed: int) -> tuple[float, float]:
    from .poc import _eer_fast

    keys = sorted(g_by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        out.append(_eer_fast(np.concatenate([g_by[k] for k in pick]), np.concatenate([i_by[k] for k in pick])))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _score_part(spk: dict[str, PopSpeaker], ids: list[str], build, norm: np.ndarray, dim: int, max_budget: int | None = None):
    """{enrollee: {genuine, impostor}} for every eligible enrollee of the part; impostors are the part's other speakers."""
    chunks, slices, at = [], {}, 0
    for i in ids:
        v = _cat([spk[i].probes[s] for s in sorted(spk[i].probes)], dim)
        slices[i] = (at, at + len(v))
        chunks.append(v)
        at += len(v)
    pool = _cat(chunks, dim)
    out: dict[str, dict[str, np.ndarray]] = {}
    for e in ids:
        sp = spk[e]
        if not sp.eligible:
            continue
        sessions = sorted(s for s in sp.whole if s in sp.probes and len(sp.probes[s]) and len(sp.whole[s]))
        g, imp = [], []
        for held in sessions:
            enrol = _cat([sp.whole[s] for s in sessions if s != held], dim)
            if max_budget is not None:
                enrol = enrol[:max_budget]
            f = build(enrol, norm)
            g.append(f(sp.probes[held]))
            allsc = f(pool)
            a, b = slices[e]
            imp.append(np.concatenate([allsc[:a], allsc[b:]]))
        out[e] = {"genuine": np.concatenate(g), "impostor": np.concatenate(imp)}
    return out


def run_population(
    speakers: list[PopSpeaker],
    *,
    seed: int = 7,
    variants: list[str] | None = None,
    far_targets: tuple[float, ...] = (0.01, 0.001),
    n_boot: int = 1000,
    budgets: tuple[int | None, ...] = (1, 2, 5, 10, None),
    on_protocol=None,
) -> PopResult:
    names = list(variants) if variants else list(VARIANTS)
    _validate(speakers, names)
    spk = {s.id: s for s in speakers}
    dim = next(iter(next(iter(speakers)).probes.values())).shape[1]
    parts = split_population(speakers, seed)
    for p in ("dev", "test"):
        n = sum(spk[i].eligible for i in parts[p])
        if n < _MIN_ENROLLEES:
            raise ValueError(f"the {p} part has {n} enrollees with at least two sessions; needs {_MIN_ENROLLEES}")
    if len(parts["norm"]) < 2:
        raise ValueError("the norm part has fewer than two speakers")
    norm = _cat([spk[i].probes[s] for i in parts["norm"] for s in sorted(spk[i].probes)], dim)
    rules = ["eer"] + [f"far{t:g}" for t in far_targets]
    protocol = {
        "seed": seed, "variants": names, "rules": rules, "n_boot": n_boot, "budgets": [b if b else "all" for b in budgets],
        "split": parts, "speakers": len(speakers),
        "enrollees": {p: sum(spk[i].eligible for i in parts[p]) for p in ("dev", "test")},
    }
    protocol["sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if on_protocol:
        on_protocol(protocol)  # the runner writes the pre-registration here, before any score exists

    scores: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        build = VARIANTS[name].build
        sc = {p: _score_part(spk, parts[p], build, norm, dim) for p in ("dev", "test")}
        scores[name] = sc
        gd = np.concatenate([v["genuine"] for v in sc["dev"].values()])
        idv = np.concatenate([v["impostor"] for v in sc["dev"].values()])
        thr = {"eer": _threshold_eer(gd, idv), **{f"far{t:g}": _threshold_far(idv, t) for t in far_targets}}
        G = np.concatenate([v["genuine"] for v in sc["test"].values()])
        I = np.concatenate([v["impostor"] for v in sc["test"].values()])
        if not (np.isfinite(G).all() and np.isfinite(I).all()):
            raise ValueError("non-finite score: refusing to report it")
        m = fast_metrics(G, I)
        g_by = {e: v["genuine"] for e, v in sc["test"].items()}
        i_by = {e: v["impostor"] for e, v in sc["test"].items()}
        eer_iv = _eer_cluster(g_by, i_by, n_boot, seed)
        ops = {}
        for r in rules:
            t = thr[r]
            kg = np.array([(v["genuine"] < t).sum() for v in sc["test"].values()])
            ng = np.array([len(v["genuine"]) for v in sc["test"].values()])
            ki = np.array([(v["impostor"] >= t).sum() for v in sc["test"].values()])
            ni = np.array([len(v["impostor"]) for v in sc["test"].values()])
            frr_iv, far_iv = _counts_interval(kg, ng, n_boot, seed), _counts_interval(ki, ni, n_boot, seed)
            ops[r] = {"threshold": thr[r], "frr": float(kg.sum() / ng.sum()), "frr_interval": list(frr_iv[:2]),
                      "false_rejects": int(kg.sum()), "n_genuine": int(ng.sum()),
                      "far": float(ki.sum() / ni.sum()), "far_interval": list(far_iv[:2]),
                      "false_accepts": int(ki.sum()), "n_impostor": int(ni.sum()),
                      "cluster_interval_informative": bool(frr_iv[2] or far_iv[2]),
                      "enrollees_with_a_false_reject": int((kg > 0).sum()), "far_resolvable_on_dev": bool(r == "eer" or len(idv) * float(r[3:]) >= 5)}
        out[name] = {
            "doc": VARIANTS[name].doc, "thresholds": thr,
            "dev": {**{k: float(v) for k, v in fast_metrics(gd, idv).items()}, "n_genuine": int(len(gd)), "n_impostor": int(len(idv))},
            "test": {**m, "eer_interval": list(eer_iv), "n_enrollees": len(g_by), "n_genuine": int(len(G)), "n_impostor": int(len(I)),
                     "genuine_min": float(G.min()), "impostor_max": float(I.max())},
            "operating_points": ops,
        }

    base = "baseline" if "baseline" in out else names[0]
    for n, v in out.items():
        b = out[base]["test"]
        v["vs_baseline"] = {"delta_eer": v["test"]["eer"] - b["eer"], "delta_min_dcf": v["test"]["min_dcf"] - b["min_dcf"],
                            "better": bool(v["test"]["eer"] < b["eer"] and v["test"]["min_dcf"] < b["min_dcf"]),
                            "established": bool(v["test"]["eer_interval"][1] < b["eer_interval"][0])}
    dev_eer = {n: out[n]["dev"]["eer"] for n in names}
    selected = min(dev_eer, key=lambda n: (dev_eer[n], names.index(n)))

    # E5: the probe-length artefact, baseline back-end, test part: match lengths, or mismatch them.
    build = VARIANTS[base].build
    whole_pool = {i: _cat([spk[i].whole[s] for s in sorted(spk[i].whole)], dim) for i in parts["test"]}
    arte = {"matched_probes": {"eer": out[base]["test"]["eer"], "auc": out[base]["test"]["auc"]}}
    gw, iw, gm = [], [], []
    ids = parts["test"]
    for e in ids:
        sp = spk[e]
        if not sp.eligible:
            continue
        sessions = sorted(s for s in sp.whole if s in sp.probes and len(sp.probes[s]) and len(sp.whole[s]))
        others_w = _cat([whole_pool[i] for i in ids if i != e], dim)
        others_p = _cat([spk[i].probes[s] for i in ids if i != e for s in sorted(spk[i].probes)], dim)
        for held in sessions:
            f = build(_cat([sp.whole[s] for s in sessions if s != held], dim), norm)
            gw.append(f(sp.whole[held]))
            iw.append(f(others_w))
            gm.append(f(others_p))
    Gw, Iw, Ip = np.concatenate(gw), np.concatenate(iw), np.concatenate(gm)
    arte["matched_whole"] = {k: v for k, v in fast_metrics(Gw, Iw).items() if k in ("eer", "auc")}
    arte["mismatched"] = {k: v for k, v in fast_metrics(Gw, Ip).items() if k in ("eer", "auc")}

    # Enrolment budget: a template from only the first n whole clips of the other sessions (baseline back-end, test part).
    budget = []
    for b in budgets:
        scb = _score_part(spk, parts["test"], build, norm, dim, max_budget=b)
        g = np.concatenate([v["genuine"] for v in scb.values()])
        i = np.concatenate([v["impostor"] for v in scb.values()])
        budget.append({"clips": b if b else "all", "eer": fast_metrics(g, i)["eer"], "genuine_mean": float(g.mean())})

    notes = []
    if protocol["enrollees"]["test"] < 30:
        notes.append(f"Only {protocol['enrollees']['test']} test enrollees: intervals are wide.")
    return PopResult(protocol | {"selected": selected}, out, arte, budget, notes, scores)


# --------------------------------------------------------------------------- report


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def _iv(iv) -> str:
    return f"{_pct(iv[0])} to {_pct(iv[1])}"


def render(r: PopResult) -> str:
    p = r.protocol
    sel = r.variants[p["selected"]]
    t = sel["test"]
    L = [LIMITS, "", "## Protocol",
         f"- {p['speakers']} speakers split once (seed {p['seed']}, stratified by source): norm {len(p['split']['norm'])}, "
         f"dev {len(p['split']['dev'])}, test {len(p['split']['test'])}; enrollees (two or more sessions): dev {p['enrollees']['dev']}, "
         f"test {p['enrollees']['test']}.",
         "- Each session of an enrollee is held out in turn; the template is the centroid of the other sessions' whole clips; probes are 5 s chunks; "
         "strangers are the other speakers of the same part. One global threshold per rule, fitted on dev, applied to test.",
         f"- Pre-registration hash {p['sha256'][:16]}. Intervals: the wider of Wilson and a cluster bootstrap over enrollees "
         f"(informative: {len(p['split']['test']) >= 5 or p['enrollees']['test'] >= 5}).", ""]
    for n in r.notes:
        L.append(f"- Note: {n}")
    L += ["", f"## Headline: `{p['selected']}` (lowest dev EER {_pct(sel['dev']['eer'])})",
          f"- Test EER {_pct(t['eer'])} (cluster interval {_iv(t['eer_interval'])}); minDCF {t['min_dcf']:.4f}; AUC {t['auc']:.4f}; d' {t['d_prime']:.2f}; "
          f"{t['n_enrollees']} enrollees, {t['n_genuine']} genuine probes, {t['n_impostor']} impostor trials."]
    for rule, o in sel["operating_points"].items():
        L.append(f"- Rule `{rule}` (threshold {o['threshold']:.3f}): owners rejected {_pct(o['frr'])} ({o['false_rejects']}/{o['n_genuine']}, "
                 f"{_iv(o['frr_interval'])}); strangers accepted {_pct(o['far'])} ({o['false_accepts']}/{o['n_impostor']}, {_iv(o['far_interval'])}); "
                 f"{o['enrollees_with_a_false_reject']} of {t['n_enrollees']} enrollees had a false reject.")
    L += ["", "## Every variant on the test part", "| variant | dev EER | test EER [interval] | minDCF | AUC |", "|---|---|---|---|---|"]
    for n, v in r.variants.items():
        L.append(f"| {n} | {_pct(v['dev']['eer'])} | {_pct(v['test']['eer'])} [{_iv(v['test']['eer_interval'])}] | {v['test']['min_dcf']:.4f} | {v['test']['auc']:.4f} |")
    L += ["", "## Probe length (baseline back-end, test part)", "| condition | EER | AUC |", "|---|---|---|"]
    for k, v in r.artefact.items():
        L.append(f"| {k.replace('_', ' ')} | {_pct(v['eer'])} | {v['auc']:.4f} |")
    L += ["", "## Enrolment budget (first n whole clips of the other sessions)", "| clips | EER | mean genuine score |", "|---|---|---|"]
    for b in r.budget:
        L.append(f"| {b['clips']} | {_pct(b['eer'])} | {b['genuine_mean']:.3f} |")
    return "\n".join(L)
