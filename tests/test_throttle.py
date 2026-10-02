"""Attempts at one identity slow down as they fail, and a success clears the slate.

A voice login with unlimited free retries is a lottery with unlimited tickets: a judge who
sounds a little like the presenter gets as many draws as they like. The delay is short and
capped (it is a demo, and the owner must never be locked out for long), but it is real: each
failed attempt makes the next one wait.
"""

from __future__ import annotations

import pytest

from kavach.throttle import AttemptThrottle


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture()
def clock() -> Clock:
    return Clock()


@pytest.fixture()
def throttle(clock: Clock) -> AttemptThrottle:
    return AttemptThrottle(free_attempts=3, base_delay=5.0, max_delay=30.0, forget_after=300.0, clock=clock)


def fail(t: AttemptThrottle, key: str = "a", n: int = 1) -> None:
    for _ in range(n):
        t.record(key, accepted=False)


def test_a_fresh_identity_can_attempt_immediately(throttle):
    assert throttle.wait_seconds("a") == 0.0


def test_the_first_few_failures_cost_nothing(throttle):
    fail(throttle, n=3)
    assert throttle.wait_seconds("a") == 0.0


def test_after_that_each_failure_doubles_the_wait_up_to_the_cap(throttle, clock):
    waits = []
    for _ in range(6):
        fail(throttle)
        waits.append(throttle.wait_seconds("a"))
        clock.advance(throttle.wait_seconds("a"))  # serve the wait; the failures remain on record
    assert waits == [0.0, 0.0, 0.0, 5.0, 10.0, 20.0]


def test_the_wait_is_capped(throttle):
    fail(throttle, n=20)
    assert throttle.wait_seconds("a") == 30.0


def test_the_wait_counts_down_as_time_passes(throttle, clock):
    fail(throttle, n=4)
    assert throttle.wait_seconds("a") == pytest.approx(5.0)
    clock.advance(2.0)
    assert throttle.wait_seconds("a") == pytest.approx(3.0)
    clock.advance(10.0)
    assert throttle.wait_seconds("a") == 0.0


def test_a_success_clears_the_slate(throttle):
    fail(throttle, n=6)
    throttle.record("a", accepted=True)
    assert throttle.wait_seconds("a") == 0.0
    fail(throttle, n=3)
    assert throttle.wait_seconds("a") == 0.0  # back to three free attempts


def test_identities_are_independent(throttle):
    fail(throttle, "a", n=10)
    assert throttle.wait_seconds("b") == 0.0


def test_old_failures_are_forgotten(throttle, clock):
    fail(throttle, n=10)
    clock.advance(301.0)
    assert throttle.wait_seconds("a") == 0.0


def test_a_disabled_throttle_never_waits(clock):
    t = AttemptThrottle(enabled=False, clock=clock)
    fail(t, n=50)
    assert t.wait_seconds("a") == 0.0


def test_reset_clears_one_identity_or_all(throttle):
    fail(throttle, "a", n=10)
    fail(throttle, "b", n=10)
    throttle.reset("a")
    assert throttle.wait_seconds("a") == 0.0 and throttle.wait_seconds("b") > 0
    throttle.reset()
    assert throttle.wait_seconds("b") == 0.0
