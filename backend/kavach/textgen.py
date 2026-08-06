"""One plain-text LLM call, against whichever provider has a key.

Two things in this system need free-form text rather than a tagged token
stream: challenge questions (`challenge.py`) and attacker answers
(`attacks/text.py`). Both were written against the Anthropic Messages API and
fell back to a fixed template bank everywhere else -- which made a paid key the
difference between the designed system and its fallback, on a student project
whose corpus was already annotated for free on Gemini.

The templates are not a substitute. Both modules say so in their own words: a
fixed question bank has a bounded, enumerable challenge space, and enumerating
it is exactly the attack the LLM path exists to defeat. So "no paid key" was
quietly costing the security claim rather than costing convenience.

This module is the missing adapter. It is deliberately small -- one call, no
JSON schema, no batching -- because that is all those two callers need, and
because `lid/llm.py` already owns the hard parts (provider table, key loading,
retry classification, pacing) and they should not be reimplemented here.

`generator` on the result records which path produced the text, and stays as
load-bearing as it was: a run with many template fallbacks is a run whose
unpredictability claim is weaker, and that must be visible rather than
inferred.
"""

from __future__ import annotations

from typing import Any

from .lid.llm import (
    PROVIDERS_BY_NAME,
    Pacer,
    available_providers,
    load_api_keys,
    with_retries,
)

#: Enough for one question or one short conversational answer. Unlike tagging,
#: neither caller's output scales with input length, so a flat ceiling is
#: right here -- see `lid.llm.output_budget` for the case where it is not.
DEFAULT_MAX_TOKENS = 2048


class TextError(RuntimeError):
    """The provider was reached and returned nothing usable."""


class TextGenerator:
    """A single `complete()` against Anthropic or an OpenAI-compatible provider.

    Construction never fails and never contacts the network, so a caller can
    build one unconditionally and let `complete()` raise. That matters because
    both callers treat "no text" as a fallback rather than an error, and an
    exception at construction time would have to be handled somewhere that has
    no fallback to offer.
    """

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        anthropic_model: str | None = None,
        effort: str = "medium",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.requested_provider = provider
        self.model = model
        self.anthropic_model = anthropic_model
        """Used only when Anthropic is the resolved provider.

        Separate from `model` because callers hold a configured id like
        `claude-opus-5` that is meaningless -- and rejected -- anywhere else.
        Forwarding it as `model` would send an Anthropic name to Gemini's
        endpoint the moment the key changed, and the failure would surface as
        a 404 mid-login rather than as a configuration error."""
        self.effort = effort
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None
        self._resolved: str | None = None
        self._pacer: Pacer | None = None
        self.retries = 0

    # ------------------------------------------------------------- resolution

    def resolve(self) -> str | None:
        """Name the provider that will serve `complete()`, without calling it.

        Returns "anthropic", an OpenAI-compatible provider name, or None when
        no key is present anywhere. Cached: the answer cannot change within a
        process and both callers ask on every single challenge.
        """
        if self._resolved is not None:
            return self._resolved or None

        load_api_keys()
        import os
        from importlib.util import find_spec

        wanted = self.requested_provider
        has_anthropic = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if wanted == "anthropic" or (
            wanted is None and has_anthropic and find_spec("anthropic") is not None
        ):
            self._resolved = "anthropic"
            return self._resolved

        if wanted is not None and wanted != "anthropic":
            if wanted not in PROVIDERS_BY_NAME:
                raise ValueError(
                    f"unknown provider {wanted!r}; choose from "
                    f"{['anthropic', *PROVIDERS_BY_NAME]}"
                )
            self._resolved = wanted
            return self._resolved

        found = [p.name for p in available_providers()]
        if found and find_spec("openai") is not None:
            self._resolved = found[0]
            return self._resolved

        self._resolved = ""  # cached negative; `or None` above converts it
        return None

    @property
    def model_id(self) -> str | None:
        """Provider-qualified model id, for the provenance record."""
        name = self.resolve()
        if name is None:
            return None
        if name == "anthropic":
            from .lid.llm import DEFAULT_MODEL

            return self.model or self.anthropic_model or DEFAULT_MODEL
        return f"{name}/{self.model or PROVIDERS_BY_NAME[name].default_model}"

    # ------------------------------------------------------------------ call

    def complete(self, *, system: str, user: str) -> str:
        """Return the model's text.

        Raises:
            TextError: No provider is configured, or the provider returned
                empty or truncated text. Callers are expected to catch this
                and fall back -- a login must not fail because a question
                generator was unreachable.
        """
        name = self.resolve()
        if name is None:
            raise TextError(
                "no LLM provider key found; set ANTHROPIC_API_KEY or a key for one of "
                f"{list(PROVIDERS_BY_NAME)} in .env"
            )
        if name == "anthropic":
            return self._complete_anthropic(system, user)
        return self._complete_openai(name, system, user)

    def _note_retry(self, attempt: int, delay: float, exc: BaseException) -> None:
        self.retries += 1

    def _complete_anthropic(self, system: str, user: str) -> str:
        import anthropic

        if self._client is None:
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        from .lid.llm import DEFAULT_MODEL

        def call() -> Any:
            return self._client.messages.create(
                model=self.model or self.anthropic_model or DEFAULT_MODEL,
                max_tokens=self.max_tokens,
                # Cached: the system prompt is identical on every challenge in
                # a session, and it is the larger half of the request.
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            )

        response = with_retries(call, on_retry=self._note_retry)
        if getattr(response, "stop_reason", None) == "refusal":
            raise TextError("the model refused the request")
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        if not text:
            raise TextError("the model returned no text")
        return text

    def _complete_openai(self, name: str, system: str, user: str) -> str:
        from openai import OpenAI

        provider = PROVIDERS_BY_NAME[name]
        if self._client is None:
            key = self._api_key or provider.api_key()
            if not key:
                raise TextError(
                    f"No API key for provider {name!r}. Set "
                    f"{' or '.join(provider.env_keys)}; free keys at {provider.signup}"
                )
            self._client = OpenAI(api_key=key, base_url=provider.base_url)
            self._pacer = Pacer(provider.min_interval_seconds)

        def call() -> Any:
            # Paced inside the retried callable, so a retry waits its turn too.
            if self._pacer is not None:
                self._pacer.wait()
            return self._client.chat.completions.create(
                model=self.model or provider.default_model,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

        response = with_retries(call, on_retry=self._note_retry)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if choice.finish_reason == "length" and not text:
            raise TextError(
                f"{name} hit the output limit before emitting any text; raise "
                "TextGenerator(max_tokens=...)"
            )
        if not text:
            raise TextError(
                f"{name} returned no text (finish_reason={choice.finish_reason!r})"
            )
        return text


__all__ = ["TextGenerator", "TextError", "DEFAULT_MAX_TOKENS"]
