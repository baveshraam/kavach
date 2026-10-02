"""Single-enrollee evaluation: does the voiceprint accept this speaker and reject other people?

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
It measures, for ONE enrolled speaker, how often their genuine probes from held-out
sessions are rejected, and how often each of N other recorded speakers is accepted.
It cannot say the voiceprint or code-switching works for people in general (one
enrollee), and it reports no population EER. `LIMITS` leads every report so a
screenshot cannot drop it, and it travels in `results.json` too.

THE TEMPLATE IS NOT THE DEMO'S
------------------------------
The template here is the centroid of the enrolment sessions named on the command line. The
demo's live template is whatever was enrolled in the demo database (13 phone clips from one
sitting). These numbers describe the demo only once it is re-enrolled from the same sessions;
the report says so.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..embedding import SpeakerEmbedding, SpeakerTemplate
from .enrollee_stats import d_prime, reported_interval
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
    frr_reported: tuple[float, float]
    frr_informative: bool
    far_wilson: tuple[float, float]
    far_cluster: tuple[float, float]
    far_reported: tuple[float, float]
    far_informative: bool
    genuine_mean: float
    genuine_min: float
    impostor_mean: float
    impostor_max: float
    d_prime: float
    gap: float
    enrol_sessions: dict[str, int]
    test_sessions: dict[str, int]
    n_enrol_clips: int
    min_group: int
    fitted_threshold: float | None = None
    fitted_frr: float | None = None
    fitted_far: float | None = None
    fitted_n_genuine: int = 0
    fitted_n_impostor: int = 0
    fitted_frr_interval: tuple[float, float] = (0.0, 1.0)
    fitted_far_interval: tuple[float, float] = (0.0, 1.0)
    dev_impostors: list[str] = field(default_factory=list)
    test_impostors: list[str] = field(default_factory=list)
    slices: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict)
    by_impostor: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__ if k != "trials"}
        d["slices"] = {"|".join(k): v for k, v in self.slices.items()}
        d["limits"] = LIMITS
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

    Enrolment and test never share a session; the fitted threshold is chosen on a dev
    partition (the last enrolment session held out of its own template, against a random
    half of the impostor SPEAKERS) and applied unchanged to the test partition (the held-out
    sessions against the other half). Split by speaker and by session, never by trial. The
    threshold is fitted on scores from the dev template (all enrolment sessions but the last)
    and applied to scores from the full template; the test side stays disjoint in both
    sessions and speakers, which is what keeps the test rates honest.

    Raises:
        ValueError: with no enrolment clips, no genuine trials, no impostor trials, or any
            non-finite score. A NaN would otherwise count as an accepted genuine and a
            rejected impostor, flattering the system both ways.
    """
    all_enrol = [e for s in enrol.values() for e in s]
    if not all_enrol:
        raise ValueError("no enrolment clips: the enrolment sessions are empty")
    if not any(test.values()):
        raise ValueError("no genuine trials: the held-out sessions have no clips")
    if not any(impostors.values()):
        raise ValueError("no impostor trials: there are no other speakers' clips to score")

    notes: list[str] = []
    template = SpeakerTemplate.from_embeddings("enrollee", all_enrol)

    gen = {sid: [template.score(c.embedding) for c in clips] for sid, clips in test.items()}
    imp = {spk: [template.score(e) for e in embs] for spk, embs in impostors.items()}
    g = np.array([s for v in gen.values() for s in v])
    i = np.array([s for v in imp.values() for s in v])
    if not (np.isfinite(g).all() and np.isfinite(i).all()):
        raise ValueError("non-finite score: an embedding contained NaN or infinity; refusing to count it")

    frr_flags = {sid: [s < system_threshold for s in v] for sid, v in gen.items() if v}
    far_flags = {spk: [s >= system_threshold for s in v] for spk, v in imp.items() if v}
    k_g = sum(sum(v) for v in frr_flags.values())
    k_i = sum(sum(v) for v in far_flags.values())
    frr_rep = reported_interval(k_g, len(g), frr_flags, seed=seed)
    far_rep = reported_interval(k_i, len(i), far_flags, seed=seed)

    from .enrollee_stats import cluster_bootstrap_rate
    from ..attacks.suite import wilson_interval

    if len(frr_flags) < 2:
        notes.append(
            "Only one held-out session: the session-cluster interval for the false-reject rate is "
            "uninformative (about 0-100%) because one sitting is one cluster. Record a second held-out "
            "session (another device or room, after a break) to get a meaningful one."
        )
    if not frr_rep[2]:
        notes.append(
            f"False-reject cluster interval is not informative ({len(frr_flags)} held-out session(s), "
            f"{k_g} rejection(s)): the reported interval is Wilson's."
        )
    if not far_rep[2]:
        notes.append(
            f"False-accept cluster interval is not informative ({len(far_flags)} impostor speaker(s), "
            f"{k_i} acceptance(s)): the reported interval is Wilson's."
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
    fitted_n_g = fitted_n_i = 0
    fitted_frr_iv = fitted_far_iv = (0.0, 1.0)
    if len(sessions) < 2:
        notes.append("Only one enrolment session, so no session can be held out to fit a threshold: the fitted operating point is skipped.")
    elif not dev_spk or not test_spk:
        notes.append("Fewer than two impostor speakers: the speaker-level dev/test split is impossible, so the fitted operating point is skipped.")
    else:
        held = sessions[-1]
        dev_template = SpeakerTemplate.from_embeddings("dev", [e for s in sessions[:-1] for e in enrol[s]])
        dev_g = np.array([dev_template.score(e) for e in enrol[held]])
        dev_i = np.array([dev_template.score(e) for s in dev_spk for e in impostors[s]])
        if dev_g.min() > dev_i.max():
            # compute_eer would return the LOWEST dev genuine score here: a tie-break, not evidence.
            fitted = float((dev_g.min() + dev_i.max()) / 2.0)
            notes.append("The dev classes did not overlap, so the fitted threshold is the midpoint of the gap between them.")
        else:
            _, fitted = compute_eer(dev_g, dev_i)
        t_flags_i = {spk: [s >= fitted for s in imp[spk]] for spk in test_spk}
        t_flags_g = {sid: [s < fitted for s in v] for sid, v in gen.items() if v}
        fitted_n_g, fitted_n_i = len(g), sum(len(v) for v in t_flags_i.values())
        kg = sum(sum(v) for v in t_flags_g.values())
        ki = sum(sum(v) for v in t_flags_i.values())
        fitted_frr, fitted_far = kg / fitted_n_g, ki / fitted_n_i
        fitted_frr_iv = reported_interval(kg, fitted_n_g, t_flags_g, seed=seed)[:2]
        fitted_far_iv = reported_interval(ki, fitted_n_i, t_flags_i, seed=seed)[:2]

    # -- slices, impostors and the audit trail ---------------------------------
    cells: dict[tuple[str, str, str], list[float]] = {}
    trials: list[dict[str, Any]] = []
    for sid, clips in test.items():
        for c, s in zip(clips, gen[sid]):
            cells.setdefault((c.device, c.environment, c.kind), []).append(s)
            trials.append({"label": "genuine", "group": sid, "device": c.device, "environment": c.environment, "kind": c.kind, "score": float(s)})
    for spk, scores in imp.items():
        trials.extend({"label": "impostor", "group": spk, "device": "", "environment": "", "kind": "", "score": float(s)} for s in scores)
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
        frr_at_system=k_g / len(g),
        far_at_system=k_i / len(i),
        frr_wilson=wilson_interval(k_g, len(g)),
        frr_cluster=cluster_bootstrap_rate(frr_flags, seed=seed),
        frr_reported=frr_rep[:2],
        frr_informative=frr_rep[2],
        far_wilson=wilson_interval(k_i, len(i)),
        far_cluster=cluster_bootstrap_rate(far_flags, seed=seed),
        far_reported=far_rep[:2],
        far_informative=far_rep[2],
        genuine_mean=float(g.mean()),
        genuine_min=float(g.min()),
        impostor_mean=float(i.mean()),
        impostor_max=float(i.max()),
        d_prime=d_prime(g, i),
        gap=float(g.min() - i.max()),
        enrol_sessions={s: len(v) for s, v in enrol.items()},
        test_sessions={s: len(v) for s, v in test.items()},
        n_enrol_clips=len(all_enrol),
        min_group=min_group,
        fitted_threshold=None if fitted is None else float(fitted),
        fitted_frr=fitted_frr,
        fitted_far=fitted_far,
        fitted_n_genuine=fitted_n_g,
        fitted_n_impostor=fitted_n_i,
        fitted_frr_interval=fitted_frr_iv,
        fitted_far_interval=fitted_far_iv,
        dev_impostors=dev_spk,
        test_impostors=test_spk,
        slices=slices,
        by_impostor=by_impostor,
        notes=notes,
        trials=trials,
    )


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _iv(iv: tuple[float, float]) -> str:
    return f"{_pct(iv[0])}-{_pct(iv[1])}"


def render(r: EnrolleeReport) -> str:
    L = [LIMITS, ""]
    L.append("## What was enrolled and what was tested")
    L.append(f"- Template: the centroid of {r.n_enrol_clips} clips from enrolment sessions "
             + ", ".join(f"{s} ({n})" for s, n in r.enrol_sessions.items()) + ".")
    L.append("- This is NOT the demo's live template (13 phone clips from one sitting): the demo must be re-enrolled "
             "from these sessions for these numbers to apply to it.")
    L.append("- Held out, never enrolled: " + ", ".join(f"{s} ({n} clips)" for s, n in r.test_sessions.items()) + ".")
    L.append(f"- Impostors: {len(r.by_impostor)} other recorded speakers ({', '.join(r.by_impostor)}).")
    L.append("")
    L.append(f"## At threshold {r.system_threshold:.2f}")
    L.append(f"- Genuine held-out clips: {r.n_genuine}. Rejected: {_pct(r.frr_at_system)}. "
             f"Reported interval {_iv(r.frr_reported)} (the wider of Wilson {_iv(r.frr_wilson)} and the cluster "
             f"bootstrap over sessions {_iv(r.frr_cluster)}"
             + ("" if r.frr_informative else "; the cluster interval is not informative here, so this is Wilson's") + ").")
    L.append(f"- Impostor trials: {r.n_impostor} over {len(r.by_impostor)} speakers. Accepted: {_pct(r.far_at_system)}. "
             f"Reported interval {_iv(r.far_reported)} (the wider of Wilson {_iv(r.far_wilson)} and the cluster "
             f"bootstrap over speakers {_iv(r.far_cluster)}"
             + ("" if r.far_informative else "; the cluster interval is not informative here, so this is Wilson's") + ").")
    L.append("- Clips from one sitting, and trials against one impostor, are not independent, so the cluster interval "
             "matters; but it is never reported narrower than Wilson.")
    L.append("")
    L.append("## Separation")
    L.append(f"- Genuine mean {r.genuine_mean:.3f}, minimum {r.genuine_min:.3f}. Impostor mean {r.impostor_mean:.3f}, "
             f"maximum {r.impostor_max:.3f}. d' {r.d_prime:.2f}. Gap (worst genuine minus best impostor) {r.gap:+.3f}.")
    L.append("")
    L.append("## A threshold fitted on dev, applied to test")
    if r.fitted_threshold is None:
        L.append("- Skipped.")
    else:
        L.append(f"- Fitted {r.fitted_threshold:.3f} on the dev template (enrolment sessions without the last) against impostors "
                 f"{', '.join(r.dev_impostors)}, then applied to the held-out sessions scored against the full template and "
                 f"impostors {', '.join(r.test_impostors)}.")
        L.append(f"- Rejected {_pct(r.fitted_frr)} of {r.fitted_n_genuine} genuine clips ({_iv(r.fitted_frr_interval)}); "
                 f"accepted {_pct(r.fitted_far)} of {r.fitted_n_impostor} impostor trials ({_iv(r.fitted_far_interval)}).")
    for n in r.notes:
        L.append(f"- Note: {n}")
    L.append("")
    L.append("## Genuine clips by condition (device / room / kind)")
    if not r.slices:
        L.append(f"(no condition has at least {r.min_group} clips)")
    else:
        L.append("| condition | n | rejected | mean | min |")
        L.append("|---|---|---|---|---|")
        for (d, e, k), v in r.slices.items():
            L.append(f"| {d} / {e} / {k} | {int(v['n'])} | {_pct(v['frr'])} | {v['mean']:.3f} | {v['min']:.3f} |")
    L.append("")
    L.append("## Impostors by speaker")
    L.append("| speaker | n | mean | max | accepted |")
    L.append("|---|---|---|---|---|")
    for spk, v in r.by_impostor.items():
        L.append(f"| {spk} | {int(v['n'])} | {v['mean']:.3f} | {v['max']:.3f} | {_pct(v['far'])} |")
    L.append("")
    L.append("Not measured here: the knowledge and code-switch branches, fusion, replay / overlap / clone attacks, and the same-sentence impostor test.")
    return "\n".join(L)


def collect_impostors(corpus: Any, *, exclude: set[str], embed: Callable[[Path], Any]) -> dict[str, list]:
    """Embed every clip of every speaker in a corpus except `exclude` (always includes the enrollee)."""
    root = corpus.root or Path()
    out: dict[str, list] = {}
    for u in corpus.utterances:
        if u.speaker_id in exclude or not u.audio_path:
            continue
        out.setdefault(u.speaker_id, []).append(embed(root / u.audio_path))
    return out


def main(argv=None, *, embedder=None) -> int:
    import argparse
    import csv
    import json
    import sys

    from ..audio import load_audio
    from ..corpus import load_manifest
    from ..studio.store import StudioError, StudioStore

    p = argparse.ArgumentParser(prog="python -m kavach.eval.enrollee", description=__doc__.splitlines()[0])
    p.add_argument("--studio", required=True, type=Path, help="data/studio/<pseudonym>")
    p.add_argument("--enrol-sessions", required=True)
    p.add_argument("--test-sessions", required=True)
    p.add_argument("--impostors", action="append", required=True, type=Path, help="corpus manifest(s)")
    p.add_argument("--exclude-speaker", action="append", default=[], help="extra corpus speaker ids to leave out")
    p.add_argument("--threshold", type=float, default=None, help="default: Settings.speaker_threshold (the demo's)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("paper/results_s04"))
    args = p.parse_args(argv)

    def refuse(msg: str) -> int:
        print(f"refused: {msg}", file=sys.stderr)
        return 2

    pseudonym = args.studio.name
    enrol_ids = [x.strip() for x in args.enrol_sessions.split(",") if x.strip()]
    test_ids = [x.strip() for x in args.test_sessions.split(",") if x.strip()]
    if set(enrol_ids) & set(test_ids):
        return refuse("a session cannot be both enrolled and held out")

    store = StudioStore(args.studio, pseudonym)
    try:
        clips = store.clips()
    except StudioError as exc:
        return refuse(str(exc))
    counts = Counter(c.session_id for c in clips)
    missing = [s for s in enrol_ids + test_ids if counts[s] == 0]
    if missing or not enrol_ids or not test_ids:
        return refuse(f"no clips recorded in session(s) {', '.join(missing) or '(none named)'} (recorded: {', '.join(sorted(counts)) or 'nothing'})")

    from ..config import Settings

    settings = Settings()
    threshold = args.threshold if args.threshold is not None else settings.speaker_threshold
    if embedder is None:
        from ..embedding import ECAPAEmbedder

        embedder = ECAPAEmbedder(model_name=settings.ecapa_model, device=settings.embedding_device)

    def embed_path(path: Path):
        return embedder.embed(load_audio(path))

    try:
        enrol = {s: [embed_path(store.verified_wav_path(c)) for c in clips if c.session_id == s] for s in enrol_ids}
        test = {
            s: [TestClip(embed_path(store.verified_wav_path(c)), c.device, c.environment, c.kind) for c in clips if c.session_id == s]
            for s in test_ids
        }
    except StudioError as exc:
        return refuse(str(exc))

    exclude = set(args.exclude_speaker) | {pseudonym}  # the enrollee is never their own impostor
    imps: dict[str, list] = {}
    try:
        for m in args.impostors:
            for spk, embs in collect_impostors(load_manifest(m), exclude=exclude, embed=embed_path).items():
                imps.setdefault(spk, []).extend(embs)
    except (FileNotFoundError, ValueError) as exc:
        return refuse(f"cannot read the impostor corpus: {exc}")

    print("enrolment: " + ", ".join(f"{s} {len(v)} clips" for s, v in enrol.items()))
    print("held out : " + ", ".join(f"{s} {len(v)} clips" for s, v in test.items()))
    print(f"impostors: {len(imps)} speakers (the enrollee {pseudonym} is excluded)")
    try:
        report = evaluate_enrollee(enrol=enrol, test=test, impostors=imps, system_threshold=threshold, seed=args.seed)
    except ValueError as exc:
        return refuse(str(exc))

    text = render(report)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    audit = args.studio / "eval"
    audit.mkdir(parents=True, exist_ok=True)
    with (audit / "trials.csv").open("w", newline="", encoding="utf-8") as fh:  # per-trial scores: git-ignored, for audit
        w = csv.DictWriter(fh, fieldnames=["label", "group", "device", "environment", "kind", "score"])
        w.writeheader()
        w.writerows(report.trials)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
