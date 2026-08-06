"""Test-wide configuration.

WHY THIS FILE EXISTS
--------------------
For most of this project's life the test suite ran in about fifty seconds and
never touched the network, and it did so *by accident*: speechbrain and
faster-whisper were not installed, so every model load failed at the import and
the pipeline fell into degraded mode. Nothing in the tests asked for that. The
day the models were installed, the same suite began downloading a 3 GB Whisper
checkpoint and stopped finishing at all.

A suite whose behaviour flips when an unrelated `pip install` happens is not
testing a configuration, it is testing the machine. `KAVACH_OFFLINE` makes the
degraded configuration an explicit choice, and `offline_by_default` applies it
everywhere so the fast path stays fast no matter what is installed.

Tests that genuinely need a real model opt in with `@pytest.mark.models`. They
are skipped unless the packages are present, because a red suite on a laptop
without a GPU teaches nothing -- but when they do run, they run against the
real checkpoint, which is the only way the acoustic branch is ever exercised.

    pytest                  # fast, offline, no downloads
    pytest -m models        # only the tests that load real checkpoints
    pytest -m "not models"  # explicit form of the default
"""

from __future__ import annotations

import importlib.util
import os

import pytest

#: Packages whose absence skips `@pytest.mark.models` tests.
MODEL_PACKAGES = ("speechbrain", "torch")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "models: needs a real model checkpoint; loads from disk or downloads it.",
    )
    config.addinivalue_line(
        "markers",
        "llm: may reach a real LLM provider; needs a key and spends quota.",
    )


@pytest.fixture(autouse=True)
def offline_by_default(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force degraded mode unless the test asked for real models.

    Set on the environment rather than on a `Settings` instance because tests
    construct their own `Settings` from fixtures, and a field default that a
    fixture then overwrites is not a default anyone can rely on. The env var is
    read by pydantic-settings wherever `Settings` is built, including inside
    the app factory.
    """
    if request.node.get_closest_marker("models"):
        monkeypatch.delenv("KAVACH_OFFLINE", raising=False)
        return
    monkeypatch.setenv("KAVACH_OFFLINE", "1")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip model tests when the packages that back them are missing."""
    missing = [p for p in MODEL_PACKAGES if importlib.util.find_spec(p) is None]
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"needs {', '.join(missing)}")
    for item in items:
        if item.get_closest_marker("models"):
            item.add_marker(skip)


#: Every environment variable that can make a component reach for a network LLM.
#: Kept as one list because the point is to leave *no* provider reachable --
#: missing one would mean the suite still calls out, just less often.
PROVIDER_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def no_live_llm_calls(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every LLM path resolve to "no provider", unless a test opts in.

    Same argument as `offline_by_default`, one layer up. Components that fall
    back to templates decide whether to fall back by asking whether a key
    exists, so the suite's behaviour would otherwise depend on whether the
    developer running it happens to have a key in `.env` -- passing on a fresh
    clone and, on a configured laptop, quietly spending free-tier quota to
    reach the same assertion.

    Clearing the variables is not enough on its own: `load_api_keys` merges
    `.env` back into the environment on first use. It is disarmed through its
    own one-shot guard rather than by stubbing the function, because callers
    import it by value (`from .lid.llm import load_api_keys`) and a stub would
    only cover the modules someone remembered to patch. The flag is read inside
    the real function, so every caller is covered by construction -- including
    ones written later.

    Tests of the loader itself set the flag back to False, which is all the
    opt-in they need.
    """
    if request.node.get_closest_marker("llm"):
        return
    for var in PROVIDER_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    from kavach.lid import llm as llm_mod

    monkeypatch.setattr(llm_mod, "_env_file_loaded", True)


@pytest.fixture(scope="session", autouse=True)
def deterministic_hashing() -> None:
    """Keep `PYTHONHASHSEED` visible in failures.

    Several tests iterate over sets of semantic classes and would reorder
    between runs. They do not depend on that order, and if one ever starts to,
    the seed printed in the failure is what makes it reproducible rather than
    "flaky".
    """
    os.environ.setdefault("PYTHONHASHSEED", "0")
