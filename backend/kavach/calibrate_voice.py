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

import math
from dataclasses import dataclass, field
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
