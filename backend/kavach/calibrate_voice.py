"""Choose the voice threshold and the inconclusive band from measured scores.

Two edges, one from each set of scores:

* **The strangers' edge `U`.** The `1 - far_target` quantile of the impostor scores (the worst
  impostor seen, when there are too few trials, fewer than about 5 expected events, to resolve it).
* **The owner's edge `G`.** The quantile of the owner's own held-out scores that `genuine_cover` of
  them reach: nearly every genuine attempt is at or above it.

When the classes are separated (`G` clears `U` by at least twice `MIN_BAND`) the accept threshold is the
**midpoint**: the same margin to the strangers' tail as to the owner's, so an unseen judge who scores a
little above the strangers measured here is still well below it, and the owner on a worse day is still
well above it. The inconclusive band reaches down halfway from the threshold to `U`. When the classes
overlap, nothing can be balanced: the false-accept rate has priority, the threshold is `U`, and the
owner's false-reject rate is reported plainly with the remedy (more enrolment audio, longer phrases).

The threshold never goes below `MIN_THRESHOLD`.

Both rates are reported with intervals, and the false-accept rate at the floor is reported too:
those are the people who get a second chance, and a second chance is only worth what the strict
second sample is worth.

This is a pure function over score arrays, so the same code serves the cosine scores the demo
uses today and any normalised scores a later change may adopt.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .attacks.suite import wilson_interval

#: Never set the threshold below this, however low the impostors score: a data set with no
#: voice that sounds like the owner says nothing about the next voice that does.
MIN_THRESHOLD = 0.50
#: The band is never narrower than this: scores are noisy to about this much.
MIN_BAND = 0.02
#: Fewer genuine scores than this and the owner's spread is a guess.
MIN_GENUINE = 30
#: Impostor trials needed to see at least this many events at the target rate.
MIN_EXPECTED_EVENTS = 5


@dataclass(slots=True)
class OperatingPoint:
    threshold: float
    floor: float
    far_target: float
    genuine_cover: float
    far_at_threshold: float
    far_at_floor: float
    frr_at_threshold: float
    frr_at_floor: float
    far_interval: tuple[float, float]
    frr_interval: tuple[float, float]
    n_genuine: int
    n_impostor: int
    provisional: bool
    ready: bool
    """True when the owner is not routinely asked twice or rejected: false-reject at the threshold
    within `max_frr` and the result is not provisional."""
    notes: list[str] = field(default_factory=list)

    @property
    def grey_margin(self) -> float:
        return round(self.threshold - self.floor, 4)

    def to_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__slots__}
        d["grey_margin"] = self.grey_margin
        return d


@dataclass(slots=True)
class LosoScores:
    genuine: np.ndarray
    impostor: np.ndarray
    by_session: dict[str, np.ndarray]
    """Genuine scores per held-out session: the unit of independence for the owner."""
    impostor_by_fold: np.ndarray | None = None
    """(folds, cohort) matrix behind `impostor`, so a report can slice it per cohort."""


def _unit(rows) -> np.ndarray:
    m = np.asarray(rows, dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("non-finite embedding: refusing to calibrate on it")
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def loso_scores(sessions: dict[str, Any], cohort, *, probes: dict[str, Any] | None = None) -> LosoScores:
    """Leave-one-session-out scores: each session against a template built from the others.

    The template is the centroid of the other sessions' embeddings (what the demo would be enrolled
    with), so a session never contributes to the template it is scored against. The cohort (other
    people's embeddings) is scored against every fold's template, so impostor and genuine scores
    come from the same templates. Needs at least two sessions.

    `probes` (session -> embeddings) are what gets scored as genuine when it differs from what the
    template is built from: the template uses whole recordings, the live probe is a few seconds of
    one phrase, and the threshold has to be chosen for the second.
    """
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out needs at least two sessions")
    units = {s: _unit(v) for s, v in sessions.items()}
    probe_units = {s: _unit(v) for s, v in (probes or sessions).items()}
    imp_pool = _unit(cohort)
    genuine: list[np.ndarray] = []
    impostor: list[np.ndarray] = []
    by_session: dict[str, np.ndarray] = {}
    for held in units:
        rest = np.concatenate([v for s, v in units.items() if s != held])
        centroid = rest.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        scores = probe_units[held] @ centroid
        by_session[held] = scores
        genuine.append(scores)
        impostor.append(imp_pool @ centroid)
    return LosoScores(np.concatenate(genuine), np.concatenate(impostor), by_session, np.stack(impostor))


def choose_operating_point(
    genuine,
    impostor,
    *,
    far_target: float = 0.001,
    genuine_cover: float = 0.99,
    max_frr: float = 0.05,
) -> OperatingPoint:
    g = np.asarray(genuine, dtype=float)
    i = np.asarray(impostor, dtype=float)
    if g.size == 0 or i.size == 0:
        raise ValueError("need both genuine and impostor scores")
    if not (np.isfinite(g).all() and np.isfinite(i).all()):
        raise ValueError("non-finite score: refusing to calibrate on it")

    notes: list[str] = []
    provisional = False

    resolvable = i.size * far_target >= MIN_EXPECTED_EVENTS
    quantile = float(np.quantile(i, 1.0 - far_target))
    if resolvable:
        stranger_edge = quantile
    else:
        provisional = True
        stranger_edge = max(quantile, float(i.max()))
        notes.append(
            f"Only {i.size} impostor trials: too few to resolve a {far_target:.2%} false-accept rate "
            f"(about {MIN_EXPECTED_EVENTS / far_target:.0f} are needed), so the strangers' edge is the "
            "worst impostor seen. Add impostor voices before trusting it."
        )

    if g.size < MIN_GENUINE:
        provisional = True
        notes.append(
            f"Only {g.size} genuine scores (under {MIN_GENUINE}): the owner's spread, and so the band "
            "under the threshold, is a guess. Record more held-out sessions."
        )
    owner_edge = float(np.quantile(g, 1.0 - genuine_cover))

    if owner_edge - stranger_edge >= 2 * MIN_BAND:
        threshold = stranger_edge + (owner_edge - stranger_edge) / 2.0
        floor = threshold - max(MIN_BAND, (threshold - stranger_edge) / 2.0)
    else:
        threshold = stranger_edge
        floor = min(owner_edge, threshold - MIN_BAND)
        notes.append(
            f"The owner's low end ({owner_edge:.3f}) is not clear of the strangers' tail ({stranger_edge:.3f}), so "
            "the margin cannot be balanced: the threshold sits at the strangers' tail (false accepts have priority)."
        )
    threshold = float(max(threshold, MIN_THRESHOLD))
    floor = float(max(min(floor, threshold - MIN_BAND), 0.0))

    frr_t = float((g < threshold).mean())
    frr_f = float((g < floor).mean())
    far_t = float((i >= threshold).mean())
    far_f = float((i >= floor).mean())
    kg = int((g < threshold).sum())
    ki = int((i >= threshold).sum())
    frr_iv = wilson_interval(kg, int(g.size))
    far_iv = wilson_interval(ki, int(i.size))

    ready = frr_t <= max_frr and not provisional
    if frr_t > max_frr:
        notes.append(
            f"The false-reject rate at the threshold is {frr_t:.1%} (more than {max_frr:.0%}): the owner "
            "would often be asked again or turned away. More enrolment audio across more sessions and "
            "devices, or longer phrases, is the remedy; lowering the threshold would admit strangers."
        )

    return OperatingPoint(
        threshold=threshold,
        floor=floor,
        far_target=far_target,
        genuine_cover=genuine_cover,
        far_at_threshold=far_t,
        far_at_floor=far_f,
        frr_at_threshold=frr_t,
        frr_at_floor=frr_f,
        far_interval=far_iv,
        frr_interval=frr_iv,
        n_genuine=int(g.size),
        n_impostor=int(i.size),
        provisional=provisional,
        ready=ready,
        notes=notes,
    )


# --------------------------------------------------------------------------
# The policy file: written by the calibration tool, read by the live pipeline
# --------------------------------------------------------------------------

#: Highest threshold the live system will accept from a file: above this nobody could pass.
MAX_THRESHOLD = 0.95
MAX_MARGIN = 0.30
POLICY_FILE = "voice_policy.json"


class PolicyError(ValueError):
    """A voice policy file that must not be trusted; the message says what is wrong with it."""


@dataclass(slots=True)
class VoicePolicy:
    threshold: float
    grey_margin: float
    provisional: bool = True
    ready: bool = False
    built_at: str = ""
    sessions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def write_voice_policy(path: Path | str, op: OperatingPoint, **provenance: Any) -> Path:
    """Write the operating point, with what it was measured on, as the live policy."""
    payload = {
        **op.to_dict(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def load_voice_policy(path: Path | str) -> VoicePolicy | None:
    """The calibrated policy, or None when there is no file.

    Anything that would make the login weaker than intended or unusable is refused: a threshold
    under `MIN_THRESHOLD` would admit strangers, one over `MAX_THRESHOLD` would admit nobody, and a
    margin outside [0, MAX_MARGIN] is not a band. The caller falls back to the defaults and says so.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyError(f"{p.name} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{p.name} must hold a JSON object")
    try:
        threshold, margin = float(raw["threshold"]), float(raw["grey_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyError(f"{p.name} needs numeric 'threshold' and 'grey_margin'") from exc
    if not (math.isfinite(threshold) and math.isfinite(margin)):
        raise PolicyError(f"{p.name} holds a non-finite number")
    if not (MIN_THRESHOLD <= threshold <= MAX_THRESHOLD):
        raise PolicyError(f"threshold {threshold} is outside [{MIN_THRESHOLD}, {MAX_THRESHOLD}]")
    if not (0.0 <= margin <= MAX_MARGIN):
        raise PolicyError(f"grey_margin {margin} is outside [0, {MAX_MARGIN}]")
    return VoicePolicy(
        threshold=threshold,
        grey_margin=margin,
        provisional=bool(raw.get("provisional", True)),
        ready=bool(raw.get("ready", False)),
        built_at=str(raw.get("built_at", "")),
        sessions=[str(x) for x in raw.get("sessions", [])],
        notes=[str(x) for x in raw.get("notes", [])],
    )


# --------------------------------------------------------------------------
# From the presenter's Studio sessions and the cached cohorts to a policy
# --------------------------------------------------------------------------

#: Probes shorter than this share of `probe_seconds` are dropped: a short tail is a different,
#: noisier probe length than the one the threshold is chosen for.
MIN_TAIL = 0.6

LIMITS = (
    "LIMITS OF THIS CALIBRATION. One enrolled speaker. The false-reject rate is that speaker's, "
    "from leave-one-session-out probes of phrase length. The false-accept rate is over recorded "
    "voices who are not the enrollee: public studio recordings and the corpus speakers, "
    "cut to the same probe length. Those voices were not recorded through this laptop's microphone, "
    "so the false-accept rate is optimistic for a judge who speaks into it. Nothing here says the "
    "system works for people in general."
)


@dataclass(slots=True)
class Calibration:
    op: OperatingPoint
    loso: LosoScores
    report: str
    policy_path: Path | None = None


def _probe_chunks(audio, seconds: float):
    from .audio import Audio, prepare_for_embedding

    prep = prepare_for_embedding(audio, max_seconds=30.0)
    n = int(seconds * prep.sample_rate)
    out = []
    for off in range(0, len(prep.samples), n):
        seg = prep.samples[off:off + n]
        if len(seg) >= MIN_TAIL * n:
            out.append(Audio(seg, prep.sample_rate, "probe"))
    if not out and prep.duration_sec >= 2.0:  # a short clip is one probe
        out.append(prep)
    return out


def _load_cohort(files, exclude: set[str]):
    vecs, names, sources = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        keep = np.array([str(s) not in exclude for s in z["speaker"]])
        vecs.append(z["vec"][keep])
        names += [str(s) for s in z["speaker"][keep]]
        sources += [Path(f).name.split("__")[0]] * int(keep.sum())
    if not vecs or sum(len(v) for v in vecs) == 0:
        raise ValueError("no impostor embeddings: the cohort files are empty (or hold only the enrollee)")
    return np.concatenate(vecs), names, sources


def calibrate_from_studio(
    studio,
    sessions: list[str],
    *,
    embedder,
    cohort_files,
    probe_seconds: float = 5.0,
    probe_kind: str | None = None,
    exclude_speakers=frozenset(),
    far_target: float = 0.001,
    out: Path | str | None = None,
) -> Calibration:
    from .audio import load_audio
    from .eval.enrollee_stats import reported_interval
    from .studio.store import StudioError

    clips = studio.clips()
    for s in sessions:
        if not any(c.session_id == s for c in clips):
            raise ValueError(f"session {s} has no clips")
        if probe_kind and not any(c.session_id == s and c.kind == probe_kind for c in clips):
            raise ValueError(f"session {s} has no clips of kind {probe_kind!r} to use as probes")

    whole: dict[str, list] = {s: [] for s in sessions}
    probes: dict[str, list] = {s: [] for s in sessions}
    meta: dict[str, list[tuple[str, str]]] = {s: [] for s in sessions}
    for c in clips:
        if c.session_id not in whole:
            continue
        try:
            path = studio.verified_wav_path(c)
        except StudioError as exc:
            raise ValueError(f"{exc} (hash check failed; nothing was calibrated)") from exc
        audio = load_audio(path)
        vec = embedder.embed(audio).vector
        whole[c.session_id].append(vec)
        if probe_kind:
            # The login's own task: a clip of this kind IS one probe, scored whole.
            if c.kind == probe_kind:
                probes[c.session_id].append(vec)
                meta[c.session_id].append((c.device, c.kind))
            continue
        for chunk in _probe_chunks(audio, probe_seconds):
            probes[c.session_id].append(embedder.embed(chunk).vector)
            meta[c.session_id].append((c.device, c.kind))

    cohort, names, sources = _load_cohort(cohort_files, set(exclude_speakers))
    loso = loso_scores(whole, cohort, probes=probes)
    op = choose_operating_point(loso.genuine, loso.impostor, far_target=far_target)

    # ---- the report --------------------------------------------------------------
    T = op.threshold
    how = (f"each {probe_kind!r} clip scored whole (the login's own task)" if probe_kind
           else f"every clip cut into {probe_seconds:g} s chunks")
    L = [LIMITS, "", "## Operating point", f"- Owner probes: {how}.",
         f"- Accept threshold {T:.3f}; inconclusive band down to {op.floor:.3f} (margin {op.grey_margin:.3f}).",
         f"- Owner: {op.n_genuine} phrase-length probes from {len(sessions)} sessions. Rejected at the threshold: "
         f"{op.frr_at_threshold:.1%} (95% interval {op.frr_interval[0]:.1%}-{op.frr_interval[1]:.1%}); "
         f"asked for a second sample or rejected below it: {op.frr_at_floor:.1%} under the floor.",
         f"- Others: {op.n_impostor} trials. Accepted at the threshold: {op.far_at_threshold:.3%} "
         f"({op.far_interval[0]:.3%}-{op.far_interval[1]:.3%}); at or above the floor (would be asked once more): {op.far_at_floor:.3%}.",
         f"- Provisional: {'yes' if op.provisional else 'no'}. Ready: {'yes' if op.ready else 'no'}."]
    for n in op.notes:
        L.append(f"- Note: {n}")

    L += ["", "## Owner, by held-out session (template = the other sessions)",
          "| session | device | probes | mean | min | rejected at threshold |", "|---|---|---|---|---|---|"]
    flags: dict[str, list[bool]] = {}
    for s in sessions:
        sc = loso.by_session[s]
        devs = sorted({d for d, _ in meta[s]})
        flags[s] = [bool(x < T) for x in sc]
        L.append(f"| {s} | {', '.join(devs)} | {len(sc)} | {sc.mean():.3f} | {sc.min():.3f} | {float((sc < T).mean()):.1%} |")
    k = sum(sum(v) for v in flags.values())
    lo, hi, informative = reported_interval(k, int(op.n_genuine), flags)
    L.append(f"\nBy-session interval for the rejection rate (wider of Wilson and cluster bootstrap): {lo:.1%}-{hi:.1%}"
             + ("" if informative else " (not informative: few sessions or no rejections, so this is Wilson's)"))

    L += ["", "## Others, by cohort (against every fold's template)",
          "| cohort | speakers | trials | mean | p99 | p99.9 | max | accepted at threshold |", "|---|---|---|---|---|---|---|---|"]
    mat = loso.impostor_by_fold
    for src in sorted(set(sources)):
        idx = np.array([i for i, x in enumerate(sources) if x == src])
        v = mat[:, idx].ravel()
        L.append(f"| {src} | {len(set(names[i] for i in idx))} | {v.size} | {v.mean():.3f} | {np.percentile(v, 99):.3f} | "
                 f"{np.percentile(v, 99.9):.3f} | {v.max():.3f} | {float((v >= T).mean()):.3%} |")

    report = "\n".join(L)
    policy_path = None
    if out is not None:
        policy_path = write_voice_policy(out, op, sessions=list(sessions), probe_seconds=probe_seconds,
                                         cohorts=sorted(set(sources)), limits=LIMITS)
    return Calibration(op, loso, report, policy_path)


def main(argv: list[str] | None = None, *, embedder=None) -> int:
    """python -m kavach.calibrate_voice --sessions S1,S2,S3,S4,S5

    Leave-one-session-out calibration of the live voice policy from the presenter's Studio
    sessions and the cached cohort embeddings (`data/cohort/emb/*@5s.npz`, built once).
    Writes `data/voice_policy.json` (the live pipeline loads it at start) and a report.
    """
    import argparse
    import glob
    import sys

    from .config import Settings
    from .studio.store import StudioStore

    cfg = Settings()
    p = argparse.ArgumentParser(prog="python -m kavach.calibrate_voice", description=main.__doc__.splitlines()[0])
    p.add_argument("--studio", type=Path, default=cfg.data_dir / "studio" / "S04")
    p.add_argument("--sessions", required=True, help="Studio sessions to cross-validate, e.g. S1,S2,S3,S4,S5")
    p.add_argument("--cohort", action="append", type=Path, help="cohort embedding file(s); default data/cohort/emb/*@5s.npz")
    p.add_argument("--exclude-speaker", action="append", default=None, help="cohort speaker ids to leave out; default: the enrollee")
    p.add_argument("--probe-seconds", type=float, default=5.0)
    p.add_argument("--probe-kind", default="auto",
                   help="'words' scores the six-words clips whole; 'none' cuts every clip into chunks; "
                        "'auto' (default) uses words clips when every session has them")
    p.add_argument("--far-target", type=float, default=0.001)
    p.add_argument("--out", type=Path, default=cfg.data_dir / POLICY_FILE)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="measure and report; do not write the policy")
    args = p.parse_args(argv)

    pseudonym = args.studio.name
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    probe_kind: str | None = None if args.probe_kind == "none" else args.probe_kind
    if probe_kind == "auto":
        try:
            have = {c.session_id for c in StudioStore(args.studio, pseudonym).clips() if c.kind == "words"}
        except Exception:  # noqa: BLE001 -- surfaced properly by calibrate_from_studio below
            have = set()
        probe_kind = "words" if sessions and all(s in have for s in sessions) else None
        print(f"probes: {'six-words clips scored whole' if probe_kind else 'every clip cut into chunks (not every session has six-words clips)'}")
    cohort_files = args.cohort or [Path(f) for f in sorted(glob.glob(str(cfg.data_dir / "cohort" / "emb" / "*@5s.npz")))]
    if not cohort_files:
        print("refused: no cohort embedding files found (build them with the cohort embedding step first)", file=sys.stderr)
        return 2
    if embedder is None:
        from .embedding import ECAPAEmbedder

        embedder = ECAPAEmbedder(model_name=cfg.ecapa_model, device=cfg.embedding_device)
    try:
        result = calibrate_from_studio(
            StudioStore(args.studio, pseudonym), sessions, embedder=embedder, cohort_files=cohort_files,
            probe_seconds=args.probe_seconds, probe_kind=probe_kind, exclude_speakers=set(args.exclude_speaker or [pseudonym]),
            far_target=args.far_target, out=None if args.dry_run else args.out,
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(result.report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result.report + "\n", encoding="utf-8")
    if result.policy_path:
        print(f"\nwrote {result.policy_path}: threshold {result.op.threshold:.3f}, band {result.op.grey_margin:.3f}"
              f" ({'provisional' if result.op.provisional else 'final'}). Restart the backend to apply it.")
    else:
        print("\n(dry run: no policy written)")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
