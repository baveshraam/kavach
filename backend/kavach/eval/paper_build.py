"""Turn the results files into everything the paper includes: macros for every number, tables, figures.

    python -m kavach.eval.paper_build

Reads `paper/results_poc/<embedder>/`, `paper/results_poc_whole/ecapa/`, `paper/results_population/<embedder>/` and
`paper/results_spoof/<embedder>/`; writes `paper/ieee/generated/` (numbers.tex, tab_*.tex, fig_*.pdf). `paper/ieee/main.tex` contains no
typed result: every number is a macro defined here, so a number that changes is a number that changes everywhere, and a macro that
does not exist is a LaTeX error. Generated files are never edited by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDERS = ("ecapa", "resnet", "wavlm")
LABEL = {"ecapa": "ECAPA-TDNN", "resnet": "ResNet-TDNN", "wavlm": "WavLM-SV"}
CAP = {"ecapa": "Ecapa", "resnet": "Resnet", "wavlm": "Wavlm"}
#: macro-name fragments for rules and the sitting names (LaTeX command names are letters only)
RULE = {"eer": "Eer", "far0.01": "FarOne", "far0.001": "FarTenth"}


class BuildError(RuntimeError):
    """A results file the paper needs is missing: the message says which and how to make it."""


_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def word(s: str) -> str:
    """A name with its digits spelled out, because LaTeX command names are letters only (S1 -> SOne, 10 -> OneZero)."""
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in str(s)).replace("all", "All")


def pct(x: float, nd: int = 2) -> str:
    return f"{100 * x:.{nd}f}"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace("-", "$-$")  # a real minus sign in the typeset text


def _load(path: Path, how: str) -> dict[str, Any]:
    if not path.exists():
        raise BuildError(f"{path} is missing; {how}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_all(paper: Path) -> dict[str, Any]:
    d: dict[str, Any] = {"poc": {}, "pop": {}, "spoof": {}}
    for e in EMBEDDERS:
        d["poc"][e] = _load(paper / "results_poc" / e / "results.json", f"run `run_eval.ps1` (embedder {e})")
        d["pop"][e] = _load(paper / "results_population" / e / "results.json", f"run `kavach.eval.population_run --embedder {e}`")
        d["spoof"][e] = _load(paper / "results_spoof" / e / "results.json", f"run `kavach.eval.spoof_run score --embedder {e}`")
    d["whole"] = _load(paper / "results_poc_whole" / "ecapa" / "results.json", "run `kavach.eval.poc_run --probe-seconds 0 --out paper/results_poc_whole/ecapa`")
    # the earlier code-switch-graph evaluation on the pilot corpus (tracked in the repository; exploratory, never the headline)
    d["csbg"] = _load(paper / "results" / "results.json", "it is the tracked output of `kavach.experiments`")
    return d


# ------------------------------------------------------------------------------ macros


def macros(d: dict[str, Any]) -> dict[str, str]:
    m: dict[str, str] = {}
    e0 = d["poc"]["ecapa"]
    p = e0["protocol"]
    m["nSittings"] = str(len(p["sittings"]))
    m["nClips"] = str(sum(p["sittings"].values()))
    m["nProbes"] = str(sum(p["probes"].values()))
    m["nStrangerSpeakers"] = str(sum(len(v) for v in p["split"].values()))
    m["nTestSpeakers"] = str(len(p["split"]["test"]))
    m["nNormSpeakers"] = str(len(p["split"]["norm"]))
    m["nFitSpeakers"] = str(len(p["split"]["fit"]))
    m["nTestTrials"] = str(e0["variants"]["baseline"]["test"]["n_impostor"])
    m["probeSeconds"] = f"{e0['manifest']['probe_seconds_requested']:g}"
    m["ownerProbeMedian"] = f"{e0['manifest']['genuine_probe_seconds']['median']:.1f}"
    m["wholeProbeMedian"] = f"{d['whole']['manifest']['genuine_probe_seconds']['median']:.1f}"
    m["nPilotStrangers"] = str(len({s for s in p["split"]["norm"] + p["split"]["fit"] + p["split"]["test"] if s.startswith("S")}))
    for k, n in p["sittings"].items():
        m[f"clips{word(k)}"] = str(n)
        m[f"probes{word(k)}"] = str(p["probes"][k])
    m["nBoot"] = str(p["n_boot"])
    m["seed"] = str(p["seed"])

    for e in EMBEDDERS:
        E, r = CAP[e], d["poc"][e]
        b = r["variants"]["baseline"]
        sel = r["variants"][r["selected"]]
        t = b["test"]
        pre = f"poc{E}"
        m[f"{pre}BaseEer"] = pct(t["eer"]) + "\\%"
        m[f"{pre}BaseEerLo"] = pct(t["eer_interval"][0]) + "\\%"
        m[f"{pre}BaseEerHi"] = pct(t["eer_interval"][1]) + "\\%"
        m[f"{pre}BaseDcf"] = num(t["min_dcf"], 4)
        m[f"{pre}BaseAuc"] = num(t["auc"], 4)
        m[f"{pre}BaseDprime"] = num(t["d_prime"], 2)
        m[f"{pre}BaseWorst"] = num(t["genuine_min"])
        m[f"{pre}BaseBest"] = num(t["impostor_max"])
        m[f"{pre}BaseGap"] = f"{t['gap']:+.3f}".replace("-", "$-$")
        for rule, tag in RULE.items():
            o = b["operating_points"][rule]
            m[f"{pre}BaseFrr{tag}"] = pct(o["frr"]) + "\\%"
            m[f"{pre}BaseFar{tag}"] = pct(o["far"]) + "\\%"
            m[f"{pre}BaseFrr{tag}K"] = f"{o['false_rejects']}/{o['n_genuine']}"
            m[f"{pre}BaseFar{tag}K"] = f"{o['false_accepts']}/{o['n_impostor']}"
            m[f"{pre}BaseFrr{tag}Lo"] = pct(o["frr_interval"][0]) + "\\%"
            m[f"{pre}BaseFrr{tag}Hi"] = pct(o["frr_interval"][1]) + "\\%"
            m[f"{pre}BaseFar{tag}Lo"] = pct(o["far_interval"][0]) + "\\%"
            m[f"{pre}BaseFar{tag}Hi"] = pct(o["far_interval"][1]) + "\\%"
        m[f"{pre}SelName"] = r["selected"].replace("_", "+")
        m[f"{pre}SelEer"] = pct(sel["test"]["eer"]) + "\\%"
        m[f"{pre}SelDevEer"] = pct(sel["dev_eer"]) + "\\%"
        sw = r["seed_sweep"]
        for key, tag in (("baseline_eer_min_median_max", "SweepEer"), ("baseline_frr_min_median_max", "SweepFrr"),
                         ("baseline_far_min_median_max", "SweepFar")):
            for v, name in zip(sw[key], ("Min", "Med", "Max")):
                m[f"{pre}{tag}{name}"] = pct(v) + "\\%"
        m[f"{pre}SweepN"] = str(len(sw["splits"]))
        sel_names = sorted({s["selected"] for s in sw["splits"]})
        m[f"{pre}SweepSel"] = ", ".join(x.replace("_", "+") for x in sel_names)

        q = d["pop"][e]
        pb = q["variants"]["baseline"]
        pre = f"pop{E}"
        m[f"{pre}BaseEer"] = pct(pb["test"]["eer"]) + "\\%"
        m[f"{pre}BaseEerLo"] = pct(pb["test"]["eer_interval"][0]) + "\\%"
        m[f"{pre}BaseEerHi"] = pct(pb["test"]["eer_interval"][1]) + "\\%"
        m[f"{pre}BaseDcf"] = num(pb["test"]["min_dcf"], 4)
        m[f"{pre}BaseAuc"] = num(pb["test"]["auc"], 4)
        m[f"{pre}Enrollees"] = str(pb["test"]["n_enrollees"])
        m[f"{pre}Genuine"] = str(pb["test"]["n_genuine"])
        m[f"{pre}Impostor"] = str(pb["test"]["n_impostor"])
        for rule, tag in RULE.items():
            if rule in pb["operating_points"]:
                o = pb["operating_points"][rule]
                m[f"{pre}Frr{tag}"] = pct(o["frr"]) + "\\%"
                m[f"{pre}Far{tag}"] = pct(o["far"]) + "\\%"
                m[f"{pre}Rej{tag}"] = f"{o['enrollees_with_a_false_reject']}"
        m[f"{pre}SelName"] = q["protocol"]["selected"].replace("_", "+")
        m[f"{pre}SelEer"] = pct(q["variants"][q["protocol"]["selected"]]["test"]["eer"]) + "\\%"
        for k, tag in (("matched_probes", "ArtProbes"), ("matched_whole", "ArtWhole"), ("mismatched", "ArtMismatch")):
            m[f"{pre}{tag}"] = pct(q["artefact"][k]["eer"]) + "\\%"
        if e == "ecapa":
            m["popSpeakers"] = str(q["protocol"]["speakers"])
            m["popNorm"] = str(len(q["protocol"]["split"]["norm"]))
            m["popDev"] = str(len(q["protocol"]["split"]["dev"]))
            m["popTest"] = str(len(q["protocol"]["split"]["test"]))
            m["popEnrolleesDev"] = str(q["protocol"]["enrollees"]["dev"])
            m["popEnrolleesTest"] = str(q["protocol"]["enrollees"]["test"])
            for bud in q["budget"]:
                tag = word(bud["clips"])
                m[f"popBudget{tag}"] = pct(bud["eer"]) + "\\%"
            for bud in r["budget"]:
                tag = word(bud["clips"])
                m[f"pocBudget{tag}"] = pct(bud["eer"]) + "\\%"
            sm = r["sitting_matrix"]["mean_score"]
            names = list(sm)
            within = [sm[k][k] for k in names if sm[k][k] is not None]
            cross = [sm[j][k] for j in names for k in names if j != k]
            m["sitWithin"] = num(float(np.mean(within)), 2)
            m["sitCross"] = num(float(np.mean(cross)), 2)
            m["sitCrossMin"] = num(float(np.min(cross)), 2)
            m["sitCrossMax"] = num(float(np.max(cross)), 2)
            for k, ps in b["per_sitting"].items():
                m[f"sitMean{word(k)}"] = num(ps["mean"], 3)
                m[f"sitMin{word(k)}"] = num(ps["min"], 3)
                m[f"sitFrr{word(k)}"] = pct(ps["frr_eer"]) + "\\%"
            m["nearestStranger"] = num(b["nearest_strangers"][0]["max"], 3)
            for k, f in b["folds"].items():
                m[f"pocEcapaThr{word(k)}"] = num(f["thresholds"]["eer"], 3)

        s = d["spoof"][e]
        pre = f"spoof{E}"
        for atk, tag in (("knn_vc", "Knn"), ("knn_vc_10", "KnnTen"), ("knn_vc_3", "KnnThree"), ("knn_vc_1", "KnnOne"),
                         ("tts_david", "David"), ("tts_zira", "Zira")):
            if atk not in s["report"]:
                continue
            for var, vt in (("baseline", "Base"), (s["selected"], "Sel")):
                if var not in s["report"][atk]:
                    continue
                rep = s["report"][atk][var]
                m[f"{pre}{tag}{vt}Mean"] = num(rep["score_mean"], 3)
                m[f"{pre}{tag}{vt}Max"] = num(rep["score_max"], 3)
                for rule, rt in RULE.items():
                    c = rep["rules"][rule]
                    m[f"{pre}{tag}{vt}Acc{rt}"] = pct(c["rate"], 1) + "\\%"
                    m[f"{pre}{tag}{vt}Acc{rt}K"] = f"{c['accepted']}/{c['n']}"
                vo = s["vs_owner"].get(atk, {}).get(var)
                if vo:
                    m[f"{pre}{tag}{vt}AtkEer"] = pct(vo["eer"], 1) + "\\%"
                    m[f"{pre}{tag}{vt}BlockFive"] = pct(vo["owner_frr_at_attack_far_0.05"], 1) + "\\%"
                    m[f"{pre}{tag}{vt}BlockOne"] = pct(vo["owner_frr_at_attack_far_0.01"], 1) + "\\%"
        m[f"{pre}Sel"] = s["selected"].replace("_", "+")

    w = d["whole"]["variants"]["baseline"]["test"]
    cs = d["csbg"]
    cfg = {c["name"]: c for c in cs["configurations"]}
    m["csbgSpeakers"] = str(cs["corpus"]["n_speakers"])
    m["csbgUtterances"] = str(cs["corpus"]["n_utterances"])
    m["csbgTokens"] = str(cs["coverage"]["total_tokens"])
    m["csbgChoiceTokens"] = str(cs["coverage"]["total_choice_tokens"])
    m["csbgTestSpeakers"] = str(len(cs["split"]["test_speakers"]))
    m["csbgDevSpeakers"] = str(len(cs["split"]["dev_speakers"]))
    m["csbgNGen"] = str(cfg["CSBG alone"]["n_genuine"])
    m["csbgNImp"] = str(cfg["CSBG alone"]["n_impostor"])
    m["csbgVetoed"] = str(cfg["CSBG alone"]["n_vetoed"])
    m["csbgEer"] = pct(cfg["CSBG alone"]["eer"]) + "\\%"
    m["csbgEerLo"] = pct(cfg["CSBG alone"]["eer_ci"][0]) + "\\%"
    m["csbgEerHi"] = pct(cfg["CSBG alone"]["eer_ci"][1]) + "\\%"
    m["csbgAuc"] = num(cfg["CSBG alone"]["auc"], 3)
    m["csbgVoiceEer"] = pct(cfg["ECAPA alone"]["eer"]) + "\\%"
    m["csbgFusedEer"] = pct(cfg["+ CSBG only"]["eer"]) + "\\%"
    m["wholeBaseEer"] = pct(w["eer"]) + "\\%"
    m["wholeBaseEerHi"] = pct(w["eer_interval"][1]) + "\\%"
    m["wholeGap"] = f"{w['gap']:+.3f}".replace("-", "$-$")
    m["wholeWorst"] = num(w["genuine_min"])
    m["wholeBest"] = num(w["impostor_max"])
    return m


def write_macros(m: dict[str, str], out: Path) -> None:
    lines = ["% GENERATED by kavach.eval.paper_build; do not edit.", ""]
    for k in sorted(m):
        if not k.isalpha():
            raise BuildError(f"macro name {k!r} is not letters only")
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------ tables


def _tab(head: list[str], rows: list[list[str]], align: str) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule",
                      *[" & ".join(r) + " \\\\" for r in rows], "\\bottomrule", "\\end{tabular}", ""])


def _esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def write_tables(d: dict[str, Any], out: Path) -> None:
    # models on the single enrollee
    rows = []
    for e in EMBEDDERS:
        r = d["poc"][e]
        t = r["variants"]["baseline"]["test"]
        o = r["variants"]["baseline"]["operating_points"]["eer"]
        rows.append([LABEL[e], f"{pct(t['eer'])} [{pct(t['eer_interval'][0])}--{pct(t['eer_interval'][1])}]", num(t["min_dcf"], 4),
                     f"{num(t['genuine_min'], 2)} / {num(t['impostor_max'], 2)}", pct(o["frr"]), pct(o["far"])])
    (out / "tab_models_poc.tex").write_text(_tab(
        ["Model", "EER [95\\% int.]", "minDCF", "Worst / best", "FRR", "FAR"], rows, "lcccrr"), encoding="utf-8")

    # back-end variants, ECAPA, single enrollee and population
    rows = []
    for v, x in d["poc"]["ecapa"]["variants"].items():
        pv = d["pop"]["ecapa"]["variants"][v]
        rows.append([_esc(v), pct(x["dev_eer"]), pct(x["test"]["eer"]), num(x["test"]["min_dcf"], 4),
                     pct(pv["dev"]["eer"]), pct(pv["test"]["eer"]), num(pv["test"]["min_dcf"], 4)])
    (out / "tab_variants.tex").write_text(_tab(
        ["Variant", "Dev EER", "Test EER", "minDCF", "Dev EER", "Test EER", "minDCF"], rows, "lrrrrrr"), encoding="utf-8")

    # sitting matrix, ECAPA
    sm = d["poc"]["ecapa"]["sitting_matrix"]["mean_score"]
    names = list(sm)
    rows = [[f"{j}"] + [num(sm[j][k], 3) for k in names] for j in names]
    (out / "tab_sitting.tex").write_text(_tab(["Template $\\backslash$ probes"] + names, rows, "l" + "c" * len(names)), encoding="utf-8")

    # population, models
    rows = []
    for e in EMBEDDERS:
        pb = d["pop"][e]["variants"]["baseline"]
        t = pb["test"]
        o = pb["operating_points"]["eer"]
        rows.append([LABEL[e], f"{pct(t['eer'])} [{pct(t['eer_interval'][0])}--{pct(t['eer_interval'][1])}]", num(t["min_dcf"], 4),
                     pct(o["frr"]), pct(o["far"]), str(t["n_enrollees"])])
    (out / "tab_population.tex").write_text(_tab(
        ["Model", "EER [95\\% int.]", "minDCF", "FRR", "FAR", "Enrollees"], rows, "lcccrr"), encoding="utf-8")

    # probe-length artefact
    w = d["whole"]["variants"]["baseline"]["test"]
    c = d["poc"]["ecapa"]["variants"]["baseline"]["test"]
    a = d["pop"]["ecapa"]["artefact"]
    rows = [["Owner (S04), 5 s probes vs 5 s strangers", pct(c["eer"]), num(c["gap"], 3)],
            ["Owner (S04), whole clips vs 5 s strangers", pct(w["eer"]), num(w["gap"], 3)],
            ["LibriSpeech, 5 s vs 5 s", pct(a["matched_probes"]["eer"]), "--"],
            ["LibriSpeech, whole vs whole", pct(a["matched_whole"]["eer"]), "--"],
            ["LibriSpeech, whole vs 5 s", pct(a["mismatched"]["eer"]), "--"]]
    (out / "tab_artefact.tex").write_text(_tab(["Condition", "EER (\\%)", "Gap"], rows, "lrr"), encoding="utf-8")

    # synthetic-voice attacks (baseline back-end)
    rows = []
    for e in EMBEDDERS:
        s = d["spoof"][e]
        for atk, label in (("knn_vc", "kNN-VC"), ("tts_david", "TTS-D"), ("tts_zira", "TTS-Z")):
            if atk not in s["report"]:
                continue
            rep = s["report"][atk]["baseline"]
            vo = s["vs_owner"][atk]["baseline"]
            rows.append([LABEL[e], label, num(rep["score_mean"], 3), pct(rep["rules"]["eer"]["rate"], 1),
                         pct(vo["eer"], 1), pct(vo["owner_frr_at_attack_far_0.05"], 1)])
    (out / "tab_spoof.tex").write_text(_tab(
        ["Model", "Attack", "Mean", "Accepted", "Atk. EER", "Owner rej."],
        rows, "llcccc"), encoding="utf-8")

    # attacker's budget, ECAPA and ResNet baseline
    rows = []
    for e in EMBEDDERS:
        s = d["spoof"][e]
        cells = []
        for atk in ("knn_vc_1", "knn_vc_3", "knn_vc_10", "knn_vc"):
            vo = s["vs_owner"].get(atk, {}).get("baseline")
            cells.append(pct(vo["eer"], 1) if vo else "--")
        rows.append([LABEL[e]] + cells)
    (out / "tab_attack_budget.tex").write_text(_tab(
        ["Model", "1 clip", "3 clips", "10 clips", "all clips"], rows, "lcccc"), encoding="utf-8")

    # enrolment budget
    rows = []
    pb, sb = d["pop"]["ecapa"]["budget"], d["poc"]["ecapa"]["budget"]
    for x, y in zip(sb, pb):
        rows.append([str(x["clips"]), pct(x["eer"]), num(x["genuine_mean"], 3), pct(y["eer"]), num(y["genuine_mean"], 3)])
    (out / "tab_budget.tex").write_text(_tab(
        ["Clips", "S04 EER", "S04 mean", "Libri EER", "Libri mean"], rows, "lcccc"), encoding="utf-8")


# ------------------------------------------------------------------------------ figures


def _curve(G: np.ndarray, I: np.ndarray):
    from .poc import det_rates

    _, far, frr = det_rates(G, I)
    fl = 0.5 / max(len(G), len(I))
    return np.maximum(far, fl) * 100, np.maximum(frr, fl) * 100, fl * 100


def write_figures(d: dict[str, Any], paper: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from .figures import INK, LINE_STYLES, STYLE, _new, close, save

    # 1. score distributions: owner, strangers, attacks (ECAPA baseline)
    z = np.load(paper / "results_poc" / "ecapa" / "scores.npz")
    sp = np.load(paper / "results_spoof" / "ecapa" / "scores.npz")
    G, I = z["baseline__genuine"], z["baseline__impostor"]
    fig, ax = _new((3.4, 2.35))
    lo = min(G.min(), I.min(), sp["knn_vc__baseline"].min())
    hi = max(G.max(), I.max(), sp["knn_vc__baseline"].max())
    bins = np.linspace(lo, hi, 45)
    ax.hist(I, bins=bins, density=True, color="#cfcfcf", edgecolor="white", linewidth=0.3, label=f"strangers ($n$={len(I)})", zorder=2)
    ax.hist(sp["knn_vc__baseline"], bins=bins, density=True, histtype="step", color=INK[2], linestyle="--", linewidth=1.2,
            label=f"kNN-VC clone ($n$={len(sp['knn_vc__baseline'])})", zorder=3)
    ax.hist(G, bins=bins, density=True, histtype="step", color=INK[0], linewidth=1.3, label=f"owner ($n$={len(G)})", zorder=4)
    for i, t in enumerate(d["poc"]["ecapa"]["variants"]["baseline"]["folds"].values()):
        ax.axvline(t["thresholds"]["eer"], color="#555555", linewidth=0.6, linestyle=":", zorder=1, label="fold thresholds" if i == 0 else None)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax.set_xlabel("Cosine score against the owner's template")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout(); save(fig, out / "fig_scores"); close(fig)

    # 2. DET, two panels: owner vs strangers, owner vs the kNN-VC clone; one line per model.
    import matplotlib.pyplot as plt

    from .figures import apply_style

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    styles = ("-", "--", ":")
    shades = (INK[0], INK[2], INK[1])
    floor = 100.0
    curves = []
    for k, (title, key) in enumerate((("(a) strangers", "baseline__impostor"), ("(b) kNN-VC clone of the owner", None))):
        for i, e in enumerate(EMBEDDERS):
            zz = np.load(paper / "results_poc" / e / "scores.npz")
            bad = zz[key] if key else np.load(paper / "results_spoof" / e / "scores.npz")["knn_vc__baseline"]
            far, frr, fl = _curve(zz["baseline__genuine"], bad)
            floor = min(floor, fl)
            curves.append((k, i, far, frr))
    floor = max(floor, 0.02)
    for k, i, far, frr in curves:
        axes[k].plot(np.maximum(far, floor), np.maximum(frr, floor), color=shades[i], linestyle=styles[i], linewidth=1.3, label=LABEL[EMBEDDERS[i]])
    for k, ax in enumerate(axes):
        ax.plot([floor, 100], [floor, 100], color="#cccccc", linewidth=0.6, zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(floor, 100); ax.set_ylim(floor, 100)
        ax.set_xlabel("False acceptance rate (%)")
        ax.set_title(("(a) strangers", "(b) kNN-VC clone of the owner")[k], loc="left")
    axes[0].set_ylabel("False rejection rate (%)")
    axes[0].legend(loc="upper right", framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout(); save(fig, out / "fig_det"); close(fig)


def write_pipeline_figure(out: Path) -> None:
    """The login as built: a row of gates, each able to refuse. Drawn here so the diagram can never disagree with a stale picture."""
    import matplotlib

    matplotlib.use("Agg")
    from .figures import INK, _new, close, save

    fig, ax = _new((7.0, 2.1))
    ax.set_axis_off()
    ax.grid(False)
    steps = [("1  Challenge", "ten random words\nsingle use, 60 s"), ("2  Liveness", "challenge known,\nunused, unexpired"),
             ("3  Integrity", "not a copy of a\nstored recording"), ("4  Words", "speech recognition:\n6 of 10, in order"),
             ("5  Voice", "ECAPA cosine vs\nenrolled template"), ("6  Decision", "accept / borderline\n/ refuse")]
    w, gap = 1.22, 0.14
    for i, (head, body) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 0.9), w, 0.95, boxstyle="round,pad=0.02", fc="#f2f2f2", ec=INK[1], lw=0.8))
        ax.text(x + w / 2, 1.58, head, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x + w / 2, 1.2, body, ha="center", va="center", fontsize=5.4)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, 1.37), xytext=(x + w, 1.37), arrowprops=dict(arrowstyle="->", color=INK[1], lw=0.9))
        if 1 <= i <= 4:
            ax.annotate("", xy=(x + w / 2, 0.42), xytext=(x + w / 2, 0.9), arrowprops=dict(arrowstyle="->", color="#8a4b2f", lw=0.8))
    total = len(steps) * (w + gap) - gap
    ax.add_patch(matplotlib.patches.FancyBboxPatch((1 * (w + gap), 0.05), 4 * w + 3 * gap, 0.37, boxstyle="round,pad=0.02", fc="#ffffff", ec="#8a4b2f", lw=0.8))
    ax.text(1 * (w + gap) + (4 * w + 3 * gap) / 2, 0.235, "any failed gate: refuse; the next attempts at that account wait 5, 10, 20, then 30 s", ha="center", va="center", fontsize=5.8, color="#8a4b2f")
    ax.set_xlim(-0.05, total + 0.05)
    ax.set_ylim(0, 2.0)
    save(fig, out / "fig_pipeline")
    close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m kavach.eval.paper_build", description=__doc__.splitlines()[0])
    p.add_argument("--paper", type=Path, default=Path("paper"))
    args = p.parse_args(argv)
    try:
        d = load_all(args.paper)
        out = args.paper / "ieee" / "generated"
        out.mkdir(parents=True, exist_ok=True)
        write_macros(macros(d), out / "numbers.tex")
        write_tables(d, out)
        write_figures(d, args.paper, out)
        write_pipeline_figure(out)
    except BuildError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
