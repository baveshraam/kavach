"""Generate synthetic-voice attacks on the presenter and score them against the single-enrollee study.

    set KAVACH_CLONE_VICTIMS=["S04"]            (consent gate: only a person who agreed may be cloned; here, the presenter)
    python -m kavach.eval.spoof_run generate    (GPU: kNN voice conversion + Windows text-to-speech; run in .venv-clone)
    python -m kavach.eval.spoof_run score --embedder ecapa

`generate` writes audio under `data/spoof/` (the presenter's synthetic voice: never committed, never printed). `score` embeds it
with the chosen model and scores it against each fold's template with the thresholds already fitted in `paper/results_poc/<embedder>/`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .poc_run import _git, default_cohort_files, embed_sittings, load_cohort
from .spoof import attack_report, render, select_sources, split_reference

TTS_VOICES = ("Microsoft David Desktop", "Microsoft Zira Desktop")


def _slug(voice: str) -> str:
    return voice.lower().replace("microsoft ", "").replace(" desktop", "").replace(" ", "_")


def embed_wavs(paths: list[Path], embedder: Any, cache: Path, probe_seconds: float) -> list[np.ndarray]:
    """One (probes, dim) array per wav, cut to `probe_seconds` like every other probe. Cached by file hash and model."""
    from ..audio import AudioError, load_audio
    from ..calibrate_voice import _probe_chunks

    model = str(getattr(embedder, "model_name", "unknown"))
    known: dict[str, list] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out = []
    for p in paths:
        key = f"{hashlib.sha256(p.read_bytes()).hexdigest()}|{model}|{probe_seconds:g}"
        if key not in known:
            audio = load_audio(p)
            vecs = []
            for ch in ([audio] if probe_seconds <= 0 else _probe_chunks(audio, probe_seconds)):
                try:
                    vecs.append(embedder.embed(ch).vector.tolist())
                except AudioError:
                    pass
            known[key] = vecs
        out.append(np.array(known[key]).reshape(-1, len(known[key][0])) if known[key] else np.zeros((0, 0)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(known), encoding="utf-8")
    return out


def _transcript(path: Path) -> str:
    parts = path.stem.split("-")
    trans = path.parent / f"{parts[0]}-{parts[1]}.trans.txt"
    for line in trans.read_text(encoding="utf-8").splitlines():
        uid, _, text = line.partition(" ")
        if uid == path.stem:
            return text.lower()
    raise ValueError(f"no transcript for {path.name}")


def generate(args) -> int:
    from ..attacks.bank import BankError, check_allowed
    from ..audio import load_audio, save_wav
    from ..cohort import enumerate_libri
    from ..config import Settings
    from ..studio.store import StudioStore
    from ..synthetic import speak

    cfg = Settings()
    try:
        check_allowed(args.victim, cfg.clone_victims)  # before any audio or torch
    except BankError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    store = StudioStore(args.studio or cfg.data_dir / "studio" / args.victim, args.victim)
    items = enumerate_libri(args.sources, "test-clean", per_speaker=3, min_seconds=4.0, max_seconds=10.0, seed=args.seed)
    chosen = select_sources([(i.speaker, i.path) for i in items], n_speakers=args.n_speakers, seed=args.seed)
    root = cfg.data_dir / "spoof"
    refs = split_reference(sessions)
    print(f"{len(chosen)} source utterances from {args.n_speakers} LibriSpeech test-clean speakers; reference = each fold's held-out sitting")

    from ..attacks.backends.knn_vc import KnnVcConverter

    conv = KnnVcConverter(device=args.device)
    sizes = [0] + [int(x) for x in args.ref_sizes.split(",") if x.strip()]  # 0 = every clip of the sitting (directory `knn_vc`)
    for fold in sessions:
        ref_clips = [c for c in store.clips() if c.session_id == refs[fold] and c.kind == "words"]
        if not ref_clips:
            print(f"refused: sitting {refs[fold]} has no words clips to use as the attacker's reference", file=sys.stderr)
            return 2
        for size in sizes:
            reference = [load_audio(store.verified_wav_path(c)) for c in (ref_clips[:size] if size else ref_clips)]
            outdir = root / (f"knn_vc_{size}" if size else "knn_vc") / fold
            outdir.mkdir(parents=True, exist_ok=True)
            for n, (_, path) in enumerate(chosen):
                target = outdir / f"{n:03d}.wav"
                if target.exists():
                    continue
                save_wav(conv.convert(load_audio(path), reference), target)
            print(f"fold {fold}: {len(chosen)} conversions from {len(reference)} reference clip(s) in {outdir}")
    for voice in TTS_VOICES:
        outdir = root / "tts" / _slug(voice)
        outdir.mkdir(parents=True, exist_ok=True)
        made = 0
        for n, (_, path) in enumerate(chosen):
            target = outdir / f"{n:03d}.wav"
            if target.exists():
                made += 1
                continue
            wav = speak(_transcript(Path(path)), voice)
            if wav is None:
                print(f"note: no synthetic voice {voice!r} on this machine; skipped", file=sys.stderr)
                break
            target.write_bytes(wav)
            made += 1
        print(f"{voice}: {made} clips")
    (root / "manifest.json").write_text(json.dumps({"sources": [u for _, u in chosen], "folds": sessions, "seed": args.seed}), encoding="utf-8")
    return 0


def attack_vs_owner(poc_scores: Path, raw: dict[str, dict[str, np.ndarray]], selected: str) -> dict[str, Any]:
    """Owner probes against attack probes as if the attacks were the strangers: the spoofing EER, and what blocking them costs the owner.

    `owner_frr_at_attack_far_*` is the share of the owner's own held-out probes rejected by the threshold that lets only that share of
    the attack through. Pooled over folds, so a threshold-free summary (the per-fold thresholds are in the main table)."""
    from .poc import fast_metrics

    if not poc_scores.exists():
        return {}
    z = np.load(poc_scores)
    out: dict[str, Any] = {}
    for name, by in raw.items():
        for v, a in by.items():
            key = f"{v}__genuine"
            if key not in z.files:
                continue
            G = z[key]
            m = fast_metrics(G, a)
            desc = np.sort(a)[::-1]
            cost = {}
            for far in (0.05, 0.01):
                thr = float(np.nextafter(desc[min(int(np.floor(far * len(desc))), len(desc) - 1)], np.inf))
                cost[f"owner_frr_at_attack_far_{far:g}"] = float(np.mean(G < thr))
                cost[f"threshold_at_attack_far_{far:g}"] = thr
            out.setdefault(name, {})[v] = {"eer": m["eer"], "auc": m["auc"], "n_genuine": int(len(G)), "n_attack": int(len(a)), **cost}
    return out


def render_vs_owner(d: dict[str, Any]) -> str:
    if not d:
        return ""
    L = ["", "## Owner against the attacks (the attacks as the strangers; pooled over folds)",
         "| attack | variant | EER | AUC | owner rejected to let 5% of the attack in | to let 1% in |", "|---|---|---|---|---|---|"]
    for name, by in d.items():
        for v, m in by.items():
            L.append(f"| {name} | {v} | {100 * m['eer']:.2f}% | {m['auc']:.4f} | {100 * m['owner_frr_at_attack_far_0.05']:.1f}% | {100 * m['owner_frr_at_attack_far_0.01']:.1f}% |")
    return "\n".join(L)


def score(args) -> int:
    from ..config import Settings
    from ..studio.store import StudioStore
    from .embedders import make_embedder
    from .poc import Sitting

    cfg = Settings()
    poc_dir = Path(args.poc) if args.poc else Path("paper/results_poc") / args.embedder
    if not (poc_dir / "results.json").exists():
        print(f"refused: {poc_dir / 'results.json'} is missing; run the single-enrollee study for this embedder first", file=sys.stderr)
        return 2
    res = json.loads((poc_dir / "results.json").read_text(encoding="utf-8"))
    sessions = list(res["protocol"]["sittings"])
    files = default_cohort_files(cfg.data_dir, args.embedder)
    if {f.name for f in files} != set(res["manifest"]["cohort_files"]):
        print("refused: the cohort files differ from the ones the single-enrollee study used; re-run it first", file=sys.stderr)
        return 2
    root = cfg.data_dir / "spoof"
    sets: dict[str, dict[str, list[Path]]] = {}
    for kdir in sorted(root.glob("knn_vc*")):
        for fold in sessions:
            d = kdir / fold
            if d.exists():
                sets.setdefault(kdir.name, {})[fold] = sorted(d.glob("*.wav"))
    for d in sorted((root / "tts").glob("*")) if (root / "tts").exists() else []:
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            sets[f"tts_{d.name}"] = {fold: wavs for fold in sessions}
    if not sets:
        print("refused: no attack audio under data/spoof (run `generate` first)", file=sys.stderr)
        return 2

    embedder = make_embedder(args.embedder, device=args.device or cfg.embedding_device)
    store = StudioStore(args.studio or cfg.data_dir / "studio" / args.victim, args.victim)
    whole, _, _, _, _ = embed_sittings(store, sessions, res["manifest"]["kind"], embedder,
                                      cfg.data_dir / "eval_cache" / f"{args.victim}__{res['manifest']['kind']}__{args.embedder}.json",
                                      res["manifest"]["probe_seconds_requested"])
    sittings = [Sitting(s, whole[s]) for s in sessions]
    cohort, _ = load_cohort(files, {args.victim})
    cache = cfg.data_dir / "spoof" / f"emb_{args.embedder}.json"
    attacks: dict[str, dict[str, np.ndarray]] = {}
    for name, by_fold in sets.items():
        attacks[name] = {}
        for fold, wavs in by_fold.items():
            arrs = [a for a in embed_wavs(wavs, embedder, cache, res["manifest"]["probe_seconds_requested"]) if a.size]
            attacks[name][fold] = np.concatenate(arrs)
    shim = SimpleNamespace(protocol=res["protocol"], variants=res["variants"], selected=res["selected"])
    try:
        raw: dict[str, dict[str, np.ndarray]] = {}
        report = attack_report(sittings, cohort, shim, attacks, raw)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    vs_owner = attack_vs_owner(poc_dir / "scores.npz", raw, res["selected"])
    out = Path(args.out) if args.out else Path("paper/results_spoof") / args.embedder
    out.mkdir(parents=True, exist_ok=True)
    text = render(report, shim) + render_vs_owner(vs_owner)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    np.savez_compressed(out / "scores.npz", **{f"{n}__{v}": a for n, by in raw.items() for v, a in by.items() if v in ("baseline", res["selected"])})
    (out / "results.json").write_text(json.dumps({"report": report, "selected": res["selected"], "manifest": {
        "commit": _git(["rev-parse", "HEAD"]), "embedder": args.embedder, "poc_protocol_sha256": res["protocol"]["sha256"],
        "attacks": {n: {k: len(v) for k, v in f.items()} for n, f in sets.items()}}, "vs_owner": vs_owner}, indent=2, sort_keys=True), encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m kavach.eval.spoof_run", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="make the attack audio (GPU)")
    g.add_argument("--victim", default="S04")
    g.add_argument("--studio", type=Path, default=None)
    g.add_argument("--sessions", default="S1,S2,S3")
    g.add_argument("--sources", type=Path, default=Path("data/cohort/librispeech_test"))
    g.add_argument("--n-speakers", type=int, default=10)
    g.add_argument("--ref-sizes", default="1,3,10", help="also convert with only this many reference clips (the full sitting is always done)")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", default="cuda")
    s = sub.add_parser("score", help="embed and score the attack audio")
    s.add_argument("--victim", default="S04")
    s.add_argument("--studio", type=Path, default=None)
    s.add_argument("--embedder", default="ecapa")
    s.add_argument("--device", default=None)
    s.add_argument("--poc", default=None)
    s.add_argument("--out", default=None)
    args = p.parse_args(argv)
    return generate(args) if args.cmd == "generate" else score(args)


if __name__ == "__main__":
    sys.exit(main())
