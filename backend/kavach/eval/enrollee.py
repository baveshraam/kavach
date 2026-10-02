"""Single-enrollee evaluation: does the voiceprint accept this speaker and reject other people?

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
It measures, for ONE enrolled speaker, how often their genuine probes from held-out
sessions are rejected, and how often each of N other recorded speakers is accepted.
It cannot say the voiceprint or code-switching works for people in general (one
enrollee), and it reports no population EER. `LIMITS` leads every report so a
screenshot cannot drop it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..attacks.suite import wilson_interval
from ..embedding import SpeakerEmbedding, SpeakerTemplate
from .enrollee_stats import cluster_bootstrap_rate, d_prime
from .metrics import compute_eer

LIMITS = (
    "LIMITS OF THIS EVIDENCE. This is one enrolled speaker. The false-reject rate is that speaker's, "
    "measured on held-out sessions; the false-accept rate is over N recorded impostors who read different "
    "material (it is not yet the same-sentence test). Nothing here says the voiceprint or code-switching "
    "works for people in general, and there is no population EER. Voice branch only."
)


@dataclass(frozen=True, slots=True)
class TestClip:
    __test__ = False  # not a pytest class

    embedding: SpeakerEmbedding
    device: str
    environment: str
    kind: str


@dataclass(slots=True)
class EnrolleeReport:
    system_threshold: float
    n_genuine: int
    n_impostor: int
    frr_at_system: float
    far_at_system: float
    frr_wilson: tuple[float, float]
    frr_cluster: tuple[float, float]
    far_wilson: tuple[float, float]
    far_cluster: tuple[float, float]
    genuine_mean: float
    genuine_min: float
    impostor_mean: float
    impostor_max: float
    d_prime: float
    gap: float
    fitted_threshold: float | None = None
    fitted_frr: float | None = None
    fitted_far: float | None = None
    dev_impostors: list[str] = field(default_factory=list)
    test_impostors: list[str] = field(default_factory=list)
    slices: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)
    by_impostor: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}
        d["slices"] = {"|".join(k): v for k, v in self.slices.items()}
        return d


def evaluate_enrollee(
    *,
    enrol: dict[str, list[SpeakerEmbedding]],
    test: dict[str, list[TestClip]],
    impostors: dict[str, list[SpeakerEmbedding]],
    system_threshold: float,
    seed: int = 7,
    min_group: int = 5,
) -> EnrolleeReport:
    """Score held-out genuine clips and other speakers against the enrolled template.

    Enrolment and test never share a session; the fitted threshold is chosen on a
    dev partition (the last enrolment session held out of its own template, against
    a random half of the impostor SPEAKERS) and applied unchanged to the test
    partition (the held-out session against the other half). Split by speaker and
    by session, never by trial.
    """
    notes: list[str] = []
    all_enrol = [e for s in enrol.values() for e in s]
    template = SpeakerTemplate.from_embeddings("enrollee", all_enrol)

    gen = {sid: [template.score(c.embedding) for c in clips] for sid, clips in test.items()}
    imp = {spk: [template.score(e) for e in embs] for spk, embs in impostors.items()}
    g = np.array([s for v in gen.values() for s in v])
    i = np.array([s for v in imp.values() for s in v])

    frr_flags = {sid: [s < system_threshold for s in v] for sid, v in gen.items()}
    far_flags = {spk: [s >= system_threshold for s in v] for spk, v in imp.items()}
    k_g = sum(sum(v) for v in frr_flags.values())
    k_i = sum(sum(v) for v in far_flags.values())

    if len(test) < 2:
        notes.append(
            "Only one held-out session: the session-cluster interval for the false-reject rate is "
            "uninformative (about 0-100%) because one sitting is one cluster. Record a second held-out "
            "session (another device or room, after a break) to get a meaningful one."
        )

    # -- fitted threshold on a dev partition ---------------------------------
    sessions = list(enrol)
    speakers = sorted(impostors)
    rng = random.Random(seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    dev_spk, test_spk = sorted(shuffled[:half]), sorted(shuffled[half:])
    fitted = fitted_frr = fitted_far = None
    if len(sessions) < 2:
        notes.append("Only one enrolment session, so no session can be held out to fit a threshold: the fitted operating point is skipped.")
    elif not dev_spk or not test_spk:
        notes.append("Fewer than two impostor speakers: the speaker-level dev/test split is impossible, so the fitted operating point is skipped.")
    else:
        held = sessions[-1]
        dev_template = SpeakerTemplate.from_embeddings(
            "dev", [e for s in sessions[:-1] for e in enrol[s]]
        )
        dev_g = np.array([dev_template.score(e) for e in enrol[held]])
        dev_i = np.array([dev_template.score(e) for s in dev_spk for e in impostors[s]])
        _, fitted = compute_eer(dev_g, dev_i)
        t_i = np.array([s for spk in test_spk for s in imp[spk]])
        fitted_frr = float((g < fitted).mean())
        fitted_far = float((t_i >= fitted).mean())

    # -- slices ----------------------------------------------------------------
    cells: dict[tuple[str, str, str], list[float]] = {}
    for sid, clips in test.items():
        for c, s in zip(clips, gen[sid]):
            cells.setdefault((c.device, c.environment, c.kind), []).append(s)
    slices = {
        key: {"n": len(v), "frr": float(np.mean([x < system_threshold for x in v])), "mean": float(np.mean(v)), "min": float(np.min(v))}
        for key, v in sorted(cells.items())
        if len(v) >= min_group
    }
    by_impostor = {
        spk: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v)), "far": float(np.mean([x >= system_threshold for x in v]))}
        for spk, v in sorted(imp.items())
    }
    return EnrolleeReport(
        system_threshold=system_threshold,
        n_genuine=len(g),
        n_impostor=len(i),
        frr_at_system=k_g / len(g) if len(g) else float("nan"),
        far_at_system=k_i / len(i) if len(i) else float("nan"),
        frr_wilson=wilson_interval(k_g, len(g)),
        frr_cluster=cluster_bootstrap_rate(frr_flags, seed=seed),
        far_wilson=wilson_interval(k_i, len(i)),
        far_cluster=cluster_bootstrap_rate(far_flags, seed=seed),
        genuine_mean=float(g.mean()),
        genuine_min=float(g.min()),
        impostor_mean=float(i.mean()),
        impostor_max=float(i.max()),
        d_prime=d_prime(g, i),
        gap=float(g.min() - i.max()),
        fitted_threshold=None if fitted is None else float(fitted),
        fitted_frr=fitted_frr,
        fitted_far=fitted_far,
        dev_impostors=dev_spk,
        test_impostors=test_spk,
        slices=slices,
        by_impostor=by_impostor,
        notes=notes,
    )


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def render(r: EnrolleeReport) -> str:
    L = [LIMITS, ""]
    L.append(f"## At the system threshold ({r.system_threshold:.2f}), the one the demo uses")
    L.append(f"- Genuine held-out clips: {r.n_genuine}. Rejected: {_pct(r.frr_at_system)} "
             f"(Wilson {_pct(r.frr_wilson[0])}-{_pct(r.frr_wilson[1])}; cluster bootstrap over sessions "
             f"{_pct(r.frr_cluster[0])}-{_pct(r.frr_cluster[1])}).")
    L.append(f"- Impostor trials: {r.n_impostor} over {len(r.by_impostor)} speakers. Accepted: {_pct(r.far_at_system)} "
             f"(Wilson {_pct(r.far_wilson[0])}-{_pct(r.far_wilson[1])}; cluster bootstrap over speakers "
             f"{_pct(r.far_cluster[0])}-{_pct(r.far_cluster[1])}).")
    L.append("- Trust the cluster interval: clips from one sitting, and trials against one impostor, are not independent.")
    L.append("")
    L.append("## Separation")
    L.append(f"- Genuine mean {r.genuine_mean:.3f}, minimum {r.genuine_min:.3f}. Impostor mean {r.impostor_mean:.3f}, "
             f"maximum {r.impostor_max:.3f}. d' {r.d_prime:.2f}. Gap (worst genuine minus best impostor) {r.gap:+.3f}.")
    L.append("")
    L.append("## A threshold fitted on dev, applied to test")
    if r.fitted_threshold is None:
        L.append("- Skipped.")
    else:
        L.append(f"- Fitted {r.fitted_threshold:.3f} on the last enrolment session and impostors {', '.join(r.dev_impostors)}; "
                 f"on the held-out session and impostors {', '.join(r.test_impostors)}: reject {_pct(r.fitted_frr)}, accept {_pct(r.fitted_far)}.")
    for n in r.notes:
        L.append(f"- Note: {n}")
    L.append("")
    L.append("## Genuine clips by condition (device | room | kind)")
    L.append("| condition | n | rejected | mean | min |")
    L.append("|---|---|---|---|---|")
    for (d, e, k), v in r.slices.items():
        L.append(f"| {d} | {e} | {k} | {int(v['n'])} | {_pct(v['frr'])} | {v['mean']:.3f} | {v['min']:.3f} |".replace("| "+d+" | "+e+" | "+k+" |", f"| {d} / {e} / {k} |", 1))
    L.append("")
    L.append("## Impostors by speaker")
    L.append("| speaker | n | mean | max | accepted |")
    L.append("|---|---|---|---|---|")
    for spk, v in r.by_impostor.items():
        L.append(f"| {spk} | {int(v['n'])} | {v['mean']:.3f} | {v['max']:.3f} | {_pct(v['far'])} |")
    L.append("")
    L.append("Not measured here: the knowledge and code-switch branches, fusion, replay / overlap / clone attacks, and the same-sentence impostor test.")
    return "\n".join(L)


def main(argv=None, *, embedder=None) -> int:
    import argparse
    import csv
    import sys
    from pathlib import Path

    from ..audio import load_audio, prepare_for_embedding  # noqa: F401
    from ..corpus import load_manifest
    from ..studio.store import StudioStore

    p = argparse.ArgumentParser(prog="python -m kavach.eval.enrollee", description=__doc__.splitlines()[0])
    p.add_argument("--studio", required=True, type=Path, help="data/studio/<pseudonym>")
    p.add_argument("--enrol-sessions", required=True)
    p.add_argument("--test-sessions", required=True)
    p.add_argument("--impostors", action="append", required=True, type=Path, help="corpus manifest(s)")
    p.add_argument("--exclude-speaker", action="append", default=[], help="corpus speaker ids to leave out (the enrollee)")
    p.add_argument("--threshold", type=float, default=0.62)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("paper/results_s04"))
    args = p.parse_args(argv)

    store = StudioStore(args.studio, args.studio.name)
    enrol_ids, test_ids = args.enrol_sessions.split(","), args.test_sessions.split(",")
    if set(enrol_ids) & set(test_ids):
        print("refused: a session cannot be both enrolled and held out", file=sys.stderr)
        return 2
    if embedder is None:
        from ..embedding import ECAPAEmbedder
        from ..config import Settings

        _s = Settings()
        embedder = ECAPAEmbedder(model_name=_s.ecapa_model, device=_s.embedding_device)

    def embed_wav(path):
        return embedder.embed(load_audio(path))

    clips = store.clips()
    enrol = {s: [embed_wav(store.wav_path(c)) for c in clips if c.session_id == s] for s in enrol_ids}
    test = {s: [TestClip(embed_wav(store.wav_path(c)), c.device, c.environment, c.kind) for c in clips if c.session_id == s] for s in test_ids}
    if not any(enrol.values()) or not any(test.values()):
        print("refused: no clips in the named sessions", file=sys.stderr)
        return 2
    imps: dict[str, list] = {}
    for m in args.impostors:
        corpus = load_manifest(m)
        for u in corpus.utterances:
            if u.speaker_id in args.exclude_speaker or not u.audio_path:
                continue
            imps.setdefault(u.speaker_id, []).append(embed_wav((corpus.root or Path()) / u.audio_path))
    report = evaluate_enrollee(enrol=enrol, test=test, impostors=imps, system_threshold=args.threshold, seed=args.seed)
    text = render(report)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text + "\n", encoding="utf-8")
    import json
    (args.out / "results.json").write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
