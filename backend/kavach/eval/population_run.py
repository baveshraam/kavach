"""Run the population benchmark from the cached cohort embeddings.

    python -m kavach.eval.population_run --embedder ecapa

Reads LibriSpeech dev-clean and test-clean (whole clips for templates, 5 s chunks for probes; a chapter is a session), writes the
pre-registration before any score exists, then `paper/results_population/<embedder>/` (report, JSON with a manifest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .poc_run import _git, _sha
from .population import PopSpeaker, render, run_population

#: corpus -> directory of its cached embeddings, as a function of the embedder name.
CORPORA = ("libri-dev", "libri-test")


def corpus_files(data_dir: Path, embedder: str, corpus: str) -> tuple[Path, Path]:
    """(whole-clip file, 5 s chunk file) of one corpus for one embedder."""
    from .embedders import cohort_dir

    d = cohort_dir(data_dir, embedder) if corpus == "libri-dev" else Path(data_dir) / "cohort" / f"emb_test_{embedder}"
    return d / f"{corpus}__{embedder}.npz", d / f"{corpus}__{embedder}@5s.npz"


def load_speakers(pairs: list[tuple[str, Path, Path]]) -> list[PopSpeaker]:
    whole: dict[str, dict[str, list]] = {}
    probes: dict[str, dict[str, list]] = {}
    source: dict[str, str] = {}
    for corpus, wf, pf in pairs:
        for f, tgt in ((wf, whole), (pf, probes)):
            z = np.load(f, allow_pickle=False)
            for v, s, se in zip(z["vec"], z["speaker"], z["session"]):
                tgt.setdefault(str(s), {}).setdefault(str(se), []).append(v)
                source.setdefault(str(s), corpus)
    return [PopSpeaker(s, source[s], {k: np.array(v) for k, v in whole.get(s, {}).items()},
                       {k: np.array(v) for k, v in probes.get(s, {}).items()}) for s in sorted(source)]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..config import Settings
    from .embedders import EMBEDDERS

    cfg = Settings()
    p = argparse.ArgumentParser(prog="python -m kavach.eval.population_run", description=__doc__.splitlines()[0])
    p.add_argument("--embedder", default="ecapa", choices=list(EMBEDDERS))
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None, help="default paper/results_population/<embedder>")
    args = p.parse_args(argv)
    out = args.out or Path("paper/results_population") / args.embedder

    pairs, files = [], {}
    for c in CORPORA:
        wf, pf = corpus_files(cfg.data_dir, args.embedder, c)
        for f in (wf, pf):
            if not f.exists():
                print(f"refused: {f} is missing (build it with `python -m kavach.cohort --corpus {c} --embedder {args.embedder}`)", file=sys.stderr)
                return 2
            files[f.name] = _sha(f)
        pairs.append((c, wf, pf))
    try:
        speakers = load_speakers(pairs)
        out.mkdir(parents=True, exist_ok=True)

        def prereg(protocol):
            (out / "preregistration.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

        result = run_population(speakers, seed=args.seed, n_boot=args.n_boot, on_protocol=prereg)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    res = result.to_dict()
    res["manifest"] = {"commit": _git(["rev-parse", "HEAD"]), "dirty": bool(_git(["status", "--porcelain", "--untracked-files=no"])),
                       "seed": args.seed, "embedder": args.embedder, "files": files,
                       "speakers": len(speakers), "command": "python -m kavach.eval.population_run " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = render(result) + f"\n\n## Reproducibility\n- Commit {res['manifest']['commit'][:12] or 'unknown'}; seed {args.seed}; embedder {args.embedder}; command `{res['manifest']['command']}`.\n"
    np.savez_compressed(out / "scores.npz", **{
        f"{n}__{side}": np.concatenate([result.scores[n]["test"][e][side] for e in sorted(result.scores[n]["test"])])
        for n in dict.fromkeys(["baseline", result.protocol["selected"]]) if n in result.scores for side in ("genuine", "impostor")})
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
