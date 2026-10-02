"""Slow down repeated failed attempts at one identity.

A voice login with unlimited free retries is a lottery with unlimited tickets: someone who
sounds a little like the enrolled speaker gets as many draws as they like, and every draw is
independent. A few failures cost nothing (a noisy room, a cough), then each further failure
makes the next attempt wait longer, up to a cap. A success clears the slate.

It is deliberately mild. This is a demo and the owner must never be locked out for long: the
longest wait is `max_delay` seconds, failures are forgotten after `forget_after` seconds, and
`reset()` clears everything. What it removes is the cheap brute-force, not the owner's second
try.

State is in memory and per identity (the claimed speaker). One process, one lock.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable


class AttemptThrottle:
    def __init__(
        self,
        *,
        free_attempts: int = 3,
        base_delay: float = 5.0,
        max_delay: float = 30.0,
        forget_after: float = 300.0,
        enabled: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.free_attempts = free_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.forget_after = forget_after
        self.enabled = enabled
        self._clock = clock
        self._failures: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _live(self, key: str) -> list[float]:
        cutoff = self._clock() - self.forget_after
        kept = [t for t in self._failures[key] if t > cutoff]
        self._failures[key] = kept
        return kept

    def wait_seconds(self, key: str) -> float:
        """How long `key` must wait before its next attempt; 0 means go."""
        if not self.enabled:
            return 0.0
        with self._lock:
            failures = self._live(key)
            over = len(failures) - self.free_attempts
            if over <= 0:
                return 0.0
            delay = min(self.max_delay, self.base_delay * 2 ** (over - 1))
            return max(0.0, failures[-1] + delay - self._clock())

    def record(self, key: str, *, accepted: bool) -> None:
        """Note the outcome of an attempt: a success forgets the failures, a failure adds one."""
        with self._lock:
            if accepted:
                self._failures.pop(key, None)
            else:
                self._live(key)
                self._failures[key].append(self._clock())

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._failures.clear()
            else:
                self._failures.pop(key, None)
