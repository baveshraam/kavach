"""Choose the voice threshold and the inconclusive band from measured scores.

Two different questions, answered from two different sets of scores:

* **How often may a stranger get through?** The threshold. Set so that at most `far_target` of
  the impostor trials score at or above it: the `1 - far_target` quantile of the impostor scores,
  never below `MIN_THRESHOLD`. With too few impostor trials to resolve that quantile (fewer than
  about 5 expected events) it stays above the worst impostor seen, and the result says it is
  provisional.
* **How often may the owner be asked twice?** The floor. The inconclusive band sits under the
  threshold; its lower edge is the quantile of the owner's own held-out scores that covers
  `genuine_cover` of them, so nearly every genuine attempt lands at or above the floor (and is
  asked for one more sample at worst) rather than being flatly rejected.

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


def _unit(rows) -> np.ndarray:
    m = np.asarray(rows, dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("non-finite embedding: refusing to calibrate on it")
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def loso_scores(sessions: dict[str, Any], cohort) -> LosoScores:
    """Leave-one-session-out scores: each session against a template built from the others.

    The template is the centroid of the other sessions' embeddings (what the demo would be enrolled
    with), so a session never contributes to the template it is scored against. The cohort (other
    people's embeddings) is scored against every fold's template, so impostor and genuine scores
    come from the same templates. Needs at least two sessions.
    """
    if len(sessions) < 2:
        raise ValueError("leave-one-session-out needs at least two sessions")
    units = {s: _unit(v) for s, v in sessions.items()}
    imp_pool = _unit(cohort)
    genuine: list[np.ndarray] = []
    impostor: list[np.ndarray] = []
    by_session: dict[str, np.ndarray] = {}
    for held, probes in units.items():
        rest = np.concatenate([v for s, v in units.items() if s != held])
        centroid = rest.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        scores = probes @ centroid
        by_session[held] = scores
        genuine.append(scores)
        impostor.append(imp_pool @ centroid)
    return LosoScores(np.concatenate(genuine), np.concatenate(impostor), by_session)


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
        threshold = quantile
    else:
        provisional = True
        threshold = max(quantile, float(i.max()))
        notes.append(
            f"Only {i.size} impostor trials: too few to resolve a {far_target:.2%} false-accept rate "
            f"(about {MIN_EXPECTED_EVENTS / far_target:.0f} are needed), so the threshold is held above "
            "the worst impostor seen. Add impostor voices before trusting it."
        )
    if threshold < MIN_THRESHOLD:
        threshold = MIN_THRESHOLD
    threshold = float(threshold)

    if g.size < MIN_GENUINE:
        provisional = True
        notes.append(
            f"Only {g.size} genuine scores (under {MIN_GENUINE}): the owner's spread, and so the band "
            "under the threshold, is a guess. Record more held-out sessions."
        )

    lower = float(np.quantile(g, 1.0 - genuine_cover))
    floor = min(lower, threshold - MIN_BAND)
    floor = float(max(floor, 0.0))

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
