"""Impostor cohorts: public speech corpora, embedded once and cached.

    python -m kavach.cohort --corpus libri-dev --root data/cohort/librispeech --chunk 5
    python -m kavach.cohort --corpus tamil-m   --root data/cohort/tamil       --chunk 5

The false-accept numbers in the voice calibration come from scoring recorded strangers against the
presenter's template. These are the strangers: LibriSpeech (CC BY 4.0, https://www.openslr.org/12) and
Google's Tamil multi-speaker set (SLR65, CC BY-SA 4.0, https://www.openslr.org/65). Nobody was recorded
for this; they are public corpora, and their studio quality makes the resulting false-accept rate
optimistic for a judge speaking into a laptop (the calibration report says so).

`--chunk 5` cuts each clip into 5 s probes, the length a login's voice sample is judged on, so the
stranger tail is measured at the same length. The cache file records speaker, session (a LibriSpeech
chapter; SLR65 has none) and the source path of every vector.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import Audio, load_audio, prepare_for_embedding

SPLITS = {"libri-dev": "dev-clean", "libri-test": "test-clean"}


@dataclass(frozen=True, slots=True)
class Item:
    speaker: str
    session: str
    path: str


@dataclass(slots=True)
class Cohort:
    vec: np.ndarray
    speaker: np.ndarray
    session: np.ndarray
    path: np.ndarray
    dur: np.ndarray


def _duration(path: Path) -> float:
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def enumerate_libri(
    root: Path | str, split: str, *, per_speaker: int = 30, min_seconds: float = 4.0, max_seconds: float = 18.0, seed: int = 1
) -> list[Item]:
    """Clips of `<root>/LibriSpeech/<split>/<speaker>/<chapter>/*.flac`; the chapter is the session."""
    base = Path(root) / "LibriSpeech" / split
    rng = random.Random(seed)
    out: list[Item] = []
    for spk in sorted(p.name for p in base.iterdir() if p.is_dir()):
        cand = [
            Item(f"L{spk}", f"{spk}-{ch.name}", str(f))
            for ch in sorted((base / spk).iterdir()) if ch.is_dir()
            for f in sorted(ch.glob("*.flac"))
            if min_seconds <= _duration(f) <= max_seconds
        ]
        rng.shuffle(cand)
        out += cand[:per_speaker]
    return out


def enumerate_slr65(
    root: Path | str, sub: str, *, per_speaker: int = 30, min_seconds: float = 3.0, max_seconds: float = 18.0, seed: int = 1
) -> list[Item]:
    """Clips of `<root>/<male|female>/<prefix>_<speaker>_<utt>.wav`; the speaker is in the file name."""
    base = Path(root) / sub
    by: dict[str, list[Path]] = {}
    for f in sorted(base.glob("*.wav")):
        parts = f.name.split("_")
        by.setdefault(parts[0] + "_" + parts[1], []).append(f)
    rng = random.Random(seed)
    out: list[Item] = []
    for spk, files in sorted(by.items()):
        rng.shuffle(files)
        kept = [f for f in files if min_seconds <= _duration(f) <= max_seconds][:per_speaker]
        out += [Item(f"T{spk}", "one", str(f)) for f in kept]
    return out


def embed_cohort(items: list[Item], embedder: Any, *, chunk_seconds: float = 0.0) -> Cohort:
    """Embed every clip (or every `chunk_seconds` piece of it; a tail under 60% of a chunk is dropped)."""
    if not items:
        raise ValueError("no clips to embed")
    vecs: list[np.ndarray] = []
    spk: list[str] = []
    ses: list[str] = []
    paths: list[str] = []
    dur: list[float] = []
    for it in items:
        prep = prepare_for_embedding(load_audio(it.path), max_seconds=30.0)
        if prep.duration_sec < 1.0:
            continue
        if chunk_seconds:
            n = int(chunk_seconds * prep.sample_rate)
            for k, off in enumerate(range(0, len(prep.samples), n)):
                seg = prep.samples[off:off + n]
                if len(seg) < 0.6 * n:
                    continue
                vecs.append(np.asarray(embedder.embed(Audio(seg, prep.sample_rate, "chunk")).vector, dtype=np.float32))
                spk.append(it.speaker); ses.append(it.session); paths.append(f"{it.path}@{k}"); dur.append(len(seg) / prep.sample_rate)
        else:
            vecs.append(np.asarray(embedder.embed(prep).vector, dtype=np.float32))
            spk.append(it.speaker); ses.append(it.session); paths.append(it.path); dur.append(prep.duration_sec)
    return Cohort(np.stack(vecs), np.array(spk), np.array(ses), np.array(paths), np.array(dur))


def save_cohort(c: Cohort, path: Path | str) -> Path:
    out = Path(path)
    if out.exists():
        raise FileExistsError(f"{out} exists; delete it to rebuild (a silent overwrite would change numbers already reported)")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, vec=c.vec, speaker=c.speaker, session=c.session, path=c.path, dur=c.dur)
    return out


def load_cohort(path: Path | str) -> Cohort:
    z = np.load(path, allow_pickle=False)
    return Cohort(z["vec"], z["speaker"], z["session"], z["path"], z["dur"])


def main(argv: list[str] | None = None, *, embedder: Any = None) -> int:
    from .config import Settings

    cfg = Settings()
    p = argparse.ArgumentParser(prog="python -m kavach.cohort", description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, choices=[*SPLITS, "tamil-m", "tamil-f"])
    p.add_argument("--root", type=Path, required=True, help="where the corpus was extracted")
    p.add_argument("--chunk", type=float, default=0.0, help="cut clips into pieces of this many seconds (5 = login length)")
    p.add_argument("--per-speaker", type=int, default=30)
    p.add_argument("--out", type=Path, default=cfg.data_dir / "cohort" / "emb")
    args = p.parse_args(argv)

    if args.corpus in SPLITS:
        items = enumerate_libri(args.root, SPLITS[args.corpus], per_speaker=args.per_speaker)
    else:
        items = enumerate_slr65(args.root, "male" if args.corpus == "tamil-m" else "female", per_speaker=args.per_speaker)
    if embedder is None:
        from .embedding import ECAPAEmbedder

        embedder = ECAPAEmbedder(model_name=cfg.ecapa_model, device=cfg.embedding_device)
    tag = f"ecapa@{args.chunk:g}s" if args.chunk else "ecapa"
    target = args.out / f"{args.corpus}__{tag}.npz"
    if target.exists():
        print(f"{target} exists; delete it to rebuild", file=sys.stderr)
        return 2
    print(f"{args.corpus}: {len(items)} clips, {len({i.speaker for i in items})} speakers")
    try:
        c = embed_cohort(items, embedder, chunk_seconds=args.chunk)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {save_cohort(c, target)} ({len(c.vec)} vectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
