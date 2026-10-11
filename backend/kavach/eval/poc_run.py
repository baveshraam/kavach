"""Run the proof-of-concept evaluation end to end from the recordings.

    python -m kavach.eval.poc_run --sessions S1,S2,S3

Embeds the presenter's Studio clips (cached by the clip's own hash, so a re-run embeds nothing), loads the cached
cohort embeddings, writes the pre-registration BEFORE any score is computed, runs `kavach.eval.poc.run_poc`, and
writes the report, JSON, LaTeX tables and figures under `paper/results_poc/`. Those files are generated: never edit them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .embedders import EMBEDDERS, cohort_dir, make_embedder
from .poc import LIMITS, Cohort, Sitting, det_rates, prepare, render, run_poc

#: The login's own task and how enrolment is built (`kavach.studio.enrol --kind words`).
DEFAULT_KIND = "words"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def embed_sittings(store: Any, sessions: list[str], kind: str, embedder: Any, cache: Path, probe_seconds: float = 5.0):
    """(whole-clip vectors, probe vectors, the clip index of each probe, probe durations, clips skipped), each per session.

    Probes are the clips cut into `probe_seconds` pieces (0 = the whole clip), the same cutting the calibration and the
    cached cohorts use, so owner and strangers are scored at one length. Cached by wav hash, model and length.
    """
    from ..audio import AudioError, load_audio
    from ..calibrate_voice import _probe_chunks
    from ..studio.store import StudioError

    model = str(getattr(embedder, "model_name", "unknown"))
    known: dict[str, Any] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    clips = [c for c in store.clips() if c.session_id in sessions and c.kind == kind]
    whole: dict[str, list] = {s: [] for s in sessions}
    probes: dict[str, list] = {s: [] for s in sessions}
    durs: dict[str, list[float]] = {s: [] for s in sessions}
    clip_of: dict[str, list[int]] = {s: [] for s in sessions}
    skipped = 0
    for c in clips:
        key = f"{c.sha256_wav}|{model}|{probe_seconds:g}"
        if key not in known:
            try:
                audio = load_audio(store.verified_wav_path(c))
            except StudioError as exc:
                raise ValueError(f"{exc} (hash check failed; nothing was evaluated)") from exc
            try:
                entry = {"whole": embedder.embed(audio).vector.tolist(), "probes": [], "durs": []}
                for ch in ([audio] if probe_seconds <= 0 else _probe_chunks(audio, probe_seconds)):
                    try:
                        entry["probes"].append(embedder.embed(ch).vector.tolist())
                        entry["durs"].append(ch.duration_sec)
                    except AudioError:
                        skipped += 1
            except AudioError:
                skipped += 1
                continue
            known[key] = entry
        clip_of[c.session_id] += [len(whole[c.session_id])] * len(known[key]["probes"])
        whole[c.session_id].append(known[key]["whole"])
        probes[c.session_id] += known[key]["probes"]
        durs[c.session_id] += known[key]["durs"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known), encoding="utf-8")
    return ({s: np.array(v) for s, v in whole.items()}, {s: np.array(v) for s, v in probes.items()},
            {s: np.array(v, dtype=int) for s, v in clip_of.items()}, durs, skipped)


def default_cohort_files(data_dir: Path, embedder: str) -> list[Path]:
    """This embedder's cached 5 s cohorts: the original corpora plus LibriSpeech test-clean when it has been embedded."""
    import glob

    d = cohort_dir(data_dir, embedder)
    files = sorted(Path(f) for f in glob.glob(str(d / f"*__{embedder}@5s.npz")))
    extra = Path(data_dir) / "cohort" / f"emb_test_{embedder}" / f"libri-test__{embedder}@5s.npz"
    return files + ([extra] if extra.exists() else [])


def load_cohort(files: list[Path], exclude: set[str]) -> tuple[Cohort, np.ndarray]:
    vec, spk, src, dur = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        keep = np.array([str(s) not in exclude for s in z["speaker"]])
        vec.append(z["vec"][keep]); dur.append(z["dur"][keep])
        spk += [str(s) for s in z["speaker"][keep]]
        src += [Path(f).name.split("__")[0]] * int(keep.sum())
    if not vec or sum(len(v) for v in vec) == 0:
        raise ValueError("no impostor embeddings: the cohort files are empty (or hold only the enrollee)")
    return Cohort(np.concatenate(vec), spk, src), np.concatenate(dur)


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20,
                              cwd=Path(__file__).resolve().parents[3]).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _tex(s: Any) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def seed_sweep(sittings: list[Sitting], cohort: Cohort, n: int, enrollee: str) -> dict[str, Any]:
    """The study repeated over `n` other random speaker splits: how much do the headline numbers depend on which strangers fell where?"""
    rows = []
    for seed in range(1, n + 1):
        d = run_poc(sittings, cohort, seed=seed, n_boot=50, enrollee=enrollee).to_dict()
        b, sel = d["variants"]["baseline"], d["variants"][d["selected"]]
        rows.append({"seed": seed, "selected": d["selected"], "baseline_eer": b["test"]["eer"], "selected_eer": sel["test"]["eer"],
                     "baseline_frr_at_eer_rule": b["operating_points"]["eer"]["frr"], "baseline_far_at_eer_rule": b["operating_points"]["eer"]["far"],
                     "baseline_gap": b["test"]["gap"]})
    q = lambda k: [float(np.min([r[k] for r in rows])), float(np.median([r[k] for r in rows])), float(np.max([r[k] for r in rows]))]  # noqa: E731
    return {"splits": rows, "baseline_eer_min_median_max": q("baseline_eer"), "selected_eer_min_median_max": q("selected_eer"),
            "baseline_frr_min_median_max": q("baseline_frr_at_eer_rule"), "baseline_far_min_median_max": q("baseline_far_at_eer_rule")}


def render_sweep(sw: dict[str, Any]) -> str:
    f = lambda k: " / ".join(f"{100 * v:.2f}%" for v in sw[k])  # noqa: E731
    sel = sorted({r["selected"] for r in sw["splits"]})
    return ("## Sensitivity to the random split of strangers\n"
            f"- {len(sw['splits'])} other seeds (the strangers are re-divided into norm / fit / test each time; the sittings and the owner do not change).\n"
            f"- Baseline test EER, min / median / max: {f('baseline_eer_min_median_max')}. Owner rejected at the dev-fitted EER threshold: {f('baseline_frr_min_median_max')}. "
            f"Strangers accepted: {f('baseline_far_min_median_max')}.\n"
            f"- Headline variant chosen on dev across seeds: {', '.join(sel)}. The splits share the same strangers, so they are not independent evidence.\n")


def write_tables(res: dict[str, Any], out: Path) -> None:
    sel = res["variants"][res["selected"]]
    t = out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    rows = ["\\begin{tabular}{lrrrr}", "\\toprule", "Threshold rule & Owner rejected (\\%) & 95\\% interval & Strangers accepted (\\%) & 95\\% interval \\\\", "\\midrule"]
    for rule, o in sel["operating_points"].items():
        rows.append(f"{_tex(rule)} & {_pct(o['frr'])} ({o['false_rejects']}/{o['n_genuine']}) & {_pct(o['frr_interval'][0])}--{_pct(o['frr_interval'][1])} "
                    f"& {_pct(o['far'])} ({o['false_accepts']}/{o['n_impostor']}) & {_pct(o['far_interval'][0])}--{_pct(o['far_interval'][1])} \\\\")
    (t / "headline.tex").write_text("\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""]), encoding="utf-8")

    rows = ["\\begin{tabular}{lrrrr}", "\\toprule", "Variant & Dev EER (\\%) & Test EER (\\%) [95\\%] & minDCF & AUC \\\\", "\\midrule"]
    for n, v in res["variants"].items():
        iv = v["test"]["eer_interval"]
        rows.append(f"{_tex(n)}{'*' if n == res['selected'] else ''} & {_pct(v['dev_eer'])} & {_pct(v['test']['eer'])} [{_pct(iv[0])}--{_pct(iv[1])}] "
                    f"& {v['test']['min_dcf']:.4f} & {v['test']['auc']:.4f} \\\\")
    (t / "variants.tex").write_text("\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""]), encoding="utf-8")

    rows = ["\\begin{tabular}{lrrrr}", "\\toprule", "Held-out sitting & Probes & Mean score & Min score & Rejected (\\%) \\\\", "\\midrule"]
    for k, s in sel["per_sitting"].items():
        rows.append(f"{_tex(k)} & {s['n']} & {s['mean']:.3f} & {s['min']:.3f} & {_pct(s['frr_eer'])} \\\\")
    (t / "per_sitting.tex").write_text("\n".join(rows + ["\\bottomrule", "\\end{tabular}", ""]), encoding="utf-8")


def write_figures(result: Any, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # files only: never open a window (Tk is unreliable on this machine)
    from .figures import INK, LINE_STYLES, STYLE, _new, close, save

    f = out / "figures"
    sc = result.scores

    def pooled(n: str):
        return (np.concatenate([sc[n]["genuine"][k] for k in sc[n]["genuine"]]),
                np.concatenate(list(sc[n]["impostor"].values())))

    # DET. Zero observed errors cannot sit on a log axis: they are drawn at half the smallest resolvable rate.
    fig, ax = _new(STYLE.single)
    floor = 100.0
    for i, n in enumerate(dict.fromkeys(["baseline", result.selected])):
        if n not in sc:
            continue
        G, I = pooled(n)
        _, far, frr = det_rates(G, I)
        fl = 0.5 / max(len(G), len(I))
        floor = min(floor, fl * 100)
        ax.plot(np.maximum(far, fl) * 100, np.maximum(frr, fl) * 100, color=INK[i % len(INK)],
                linestyle=LINE_STYLES[i % len(LINE_STYLES)], label=n, zorder=3)
    ax.plot([floor, 100], [floor, 100], color="#cccccc", linewidth=0.6, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(floor, 100); ax.set_ylim(floor, 100)
    ax.set_xlabel("False acceptance rate (%)"); ax.set_ylabel("False rejection rate (%)")
    ax.set_title("DET: held-out sittings vs unseen strangers", loc="left")
    ax.text(0.98, 0.04, "lower-left corner = zero observed errors", transform=ax.transAxes, ha="right", fontsize=STYLE.base_font - 3)
    ax.legend(loc="upper right", framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout(); save(fig, f / "det"); close(fig)

    # Score histograms, as densities: 134 owner probes would vanish next to 5,000 stranger trials as counts.
    n = result.selected
    G, I = pooled(n)
    thr = float(np.mean([v["thresholds"]["eer"] for v in result.variants[n]["folds"].values()]))
    fig, ax = _new(STYLE.single)
    bins = np.linspace(min(G.min(), I.min()), max(G.max(), I.max()), 40)
    ax.hist(I, bins=bins, density=True, color="#c4c4c4", edgecolor="white", linewidth=0.4, label=f"strangers (n={len(I)})", zorder=2)
    ax.hist(G, bins=bins, density=True, histtype="step", color=INK[0], linewidth=1.3, label=f"owner probes (n={len(G)})", zorder=3)
    ax.axvline(thr, color="#4a4a4a", linewidth=0.9, linestyle="--", zorder=4)
    ax.text(thr, ax.get_ylim()[1] * 0.97, " mean fold threshold", rotation=90, va="top", ha="left", fontsize=STYLE.base_font - 2, color="#4a4a4a")
    ax.set_xlabel("Score"); ax.set_ylabel("Density")
    ax.set_title(f"{n}: owner vs strangers", loc="left")
    ax.legend(loc="upper left", framealpha=0.9, facecolor="white", edgecolor="none", frameon=True)
    fig.tight_layout(); save(fig, f / "scores"); close(fig)


def main(argv: list[str] | None = None, *, embedder: Any = None) -> int:
    import argparse

    from ..config import Settings
    from ..studio.store import StudioStore

    cfg = Settings()
    p = argparse.ArgumentParser(prog="python -m kavach.eval.poc_run", description=__doc__.splitlines()[0])
    p.add_argument("--studio", type=Path, default=cfg.data_dir / "studio" / "S04")
    p.add_argument("--sessions", required=True, help="Studio sittings, e.g. S1,S2,S3 (at least three)")
    p.add_argument("--kind", default=DEFAULT_KIND, help="clip kind used as template and probe (default: words)")
    p.add_argument("--embedder", default="ecapa", choices=list(EMBEDDERS), help="which pretrained speaker model (default ecapa)")
    p.add_argument("--device", default=None, help="cpu or cuda; default Settings.embedding_device")
    p.add_argument("--cohort", action="append", type=Path, help="cohort embedding file(s); default data/cohort/emb/*@5s.npz")
    p.add_argument("--exclude-speaker", action="append", default=None, help="cohort speaker ids to leave out; default: the enrollee")
    p.add_argument("--probe-seconds", type=float, default=5.0,
                   help="cut owner clips into probes this long, as the cached cohorts are (default 5; 0 = whole clips)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--sweep", type=int, default=10, help="also repeat the study over this many other random splits (0 = skip)")
    p.add_argument("--variants", default=None, help="comma-separated subset; default all")
    p.add_argument("--out", type=Path, default=None, help="default paper/results_poc/<embedder>")
    p.add_argument("--cache", type=Path, default=cfg.data_dir / "eval_cache")
    args = p.parse_args(argv)
    args.out = args.out or Path("paper/results_poc") / args.embedder

    def refuse(msg: str) -> int:
        print(f"refused: {msg}", file=sys.stderr)
        return 2

    pseudonym = args.studio.name
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    files = args.cohort or default_cohort_files(cfg.data_dir, args.embedder)
    if not files:
        return refuse("no cohort embedding files found (build them with the cohort embedding step first)")
    store = StudioStore(args.studio, pseudonym)
    try:
        have = {c.session_id for c in store.clips() if c.kind == args.kind}
    except ValueError as exc:
        return refuse(str(exc))
    missing = [s for s in sessions if s not in have]
    if missing:
        return refuse(f"session(s) {', '.join(missing)} have no {args.kind!r} clips (recorded: {', '.join(sorted(have)) or 'nothing'})")
    if embedder is None:
        embedder = make_embedder(args.embedder, device=args.device or cfg.embedding_device)

    try:
        vecs, probes, clip_of, durs, skipped = embed_sittings(
            store, sessions, args.kind, embedder, args.cache / f"{pseudonym}__{args.kind}__{args.embedder}.json", args.probe_seconds)
        cohort, cohort_dur = load_cohort(files, set(args.exclude_speaker or [pseudonym]))
        sittings = [Sitting(s, vecs[s], probes[s], clip_of[s]) for s in sessions]
        names = [x.strip() for x in args.variants.split(",")] if args.variants else None
        protocol, *_ = prepare(sittings, cohort, seed=args.seed, variants=names, n_boot=args.n_boot, enrollee=pseudonym)
        args.out.mkdir(parents=True, exist_ok=True)
        # Written before a single score exists: the variants, the split and the decision rule are now fixed.
        (args.out / "preregistration.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        result = run_poc(sittings, cohort, seed=args.seed, variants=names, n_boot=args.n_boot, enrollee=pseudonym)
    except ValueError as exc:
        return refuse(str(exc))
    assert result.protocol["sha256"] == protocol["sha256"]

    allg = np.concatenate([np.array(d) for d in durs.values()])
    res = result.to_dict()
    res["manifest"] = {
        "commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
        "seed": args.seed, "embedder": args.embedder, "embedder_checkpoint": str(getattr(embedder, "model_name", "unknown")), "kind": args.kind,
        "clips_per_sitting": {s: int(len(vecs[s])) for s in sessions}, "probes_per_sitting": {s: int(len(probes[s])) for s in sessions},
        "probe_seconds_requested": args.probe_seconds, "clips_skipped_too_short": skipped,
        "cohort_files": {Path(f).name: _sha(Path(f)) for f in files},
        "cohort_chunks": int(len(cohort.vec)), "cohort_speakers": len(set(cohort.speaker)),
        "genuine_probe_seconds": {"median": float(np.median(allg)), "min": float(allg.min()), "max": float(allg.max())},
        "cohort_probe_seconds": {"median": float(np.median(cohort_dur)), "min": float(cohort_dur.min()), "max": float(cohort_dur.max())},
        "command": "python -m kavach.eval.poc_run " + " ".join(argv if argv is not None else sys.argv[1:]),
    }
    m = res["manifest"]
    text = render(result) + (
        f"\n\n## Reproducibility\n- Commit {m['commit'][:12] or 'unknown'}{' (uncommitted changes present)' if m['dirty'] else ''}; seed {m['seed']}; "
        f"embedder {m['embedder']}; clip kind {m['kind']}; {m['clips_skipped_too_short']} clip(s) skipped as too short.\n"
        f"- Owner probes last {m['genuine_probe_seconds']['median']:.1f} s (median; {m['genuine_probe_seconds']['min']:.1f}-"
        f"{m['genuine_probe_seconds']['max']:.1f}); cohort probes {m['cohort_probe_seconds']['median']:.1f} s (median). "
        "A large difference would flatter the separation, so the headline uses --probe-seconds 5 (the default).\n"
        f"- Command: `{m['command']}`\n")
    if args.sweep:
        res["seed_sweep"] = seed_sweep(sittings, cohort, args.sweep, pseudonym)
        text += "\n" + render_sweep(res["seed_sweep"])
    (args.out / "report.md").write_text(text + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_tables(res, args.out)
    write_figures(result, args.out)
    # raw scores of the baseline and the headline variant, so the paper's figures are drawn from data, not from summaries
    np.savez_compressed(args.out / "scores.npz", **{
        f"{n}__{side}": np.concatenate([result.scores[n][side][k] for k in sorted(result.scores[n][side])])
        for n in dict.fromkeys(["baseline", result.selected]) if n in result.scores for side in ("genuine", "impostor")})
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
