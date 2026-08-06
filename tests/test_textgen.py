"""Tests for the provider-neutral text call.

The property under test throughout is that **"which provider" and "did it
work" are separate questions**. Collapsing them is what the module exists to
undo: challenge and attack text used to treat "no Anthropic key" as "no LLM",
so a machine with a working free-tier key silently produced template output and
recorded it as if templates had been chosen.
"""

from __future__ import annotations

from typing import Any

import pytest

from kavach.textgen import TextError, TextGenerator


class FakeOpenAIResponse:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        message = type("M", (), {"content": text})()
        self.choices = [type("C", (), {"message": message, "finish_reason": finish_reason})()]


class FakeOpenAIClient:
    def __init__(self, text: str = "ok", finish_reason: str = "stop") -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text
        self._finish = finish_reason
        outer = self

        class Completions:
            def create(self, **kwargs: Any) -> FakeOpenAIResponse:
                outer.calls.append(kwargs)
                return FakeOpenAIResponse(outer._text, outer._finish)

        self.chat = type("Chat", (), {"completions": Completions()})()


def gemini_gen(monkeypatch: pytest.MonkeyPatch, **kw: Any) -> TextGenerator:
    """A generator pinned to gemini with a fake client already installed."""
    gen = TextGenerator(provider="gemini", **kw)
    gen._resolved = "gemini"
    return gen


class TestResolution:
    def test_no_key_anywhere_resolves_to_none(self) -> None:
        """The normal state of a fresh clone, and not an error."""
        assert TextGenerator().resolve() is None

    def test_none_is_distinguishable_from_a_failure(self) -> None:
        """`complete` raises rather than returning empty, so a caller can tell
        "nothing configured" from "the model said nothing"."""
        with pytest.raises(TextError, match="no LLM provider key"):
            TextGenerator().complete(system="s", user="u")

    def test_an_unknown_provider_is_a_programming_error(self) -> None:
        """Not a fallback: a typo in a config should not silently downgrade the
        system to templates and report itself as offline."""
        with pytest.raises(ValueError, match="unknown provider"):
            TextGenerator(provider="geminii").resolve()

    def test_explicit_provider_is_honoured(self) -> None:
        assert TextGenerator(provider="groq").resolve() == "groq"

    def test_resolution_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both callers ask on every challenge; re-reading `.env` each time
        would put a filesystem hit in the login path."""
        gen = TextGenerator()
        calls = []
        import kavach.lid.llm as llm_mod

        monkeypatch.setattr(llm_mod, "load_api_keys", lambda: calls.append(1))
        gen.resolve()
        gen.resolve()
        gen.resolve()
        assert len(calls) <= 1

    def test_model_id_is_provider_qualified(self) -> None:
        """`gemini-3.1-flash-lite` alone does not say who served it."""
        assert TextGenerator(provider="groq", model="m").model_id == "groq/m"

    def test_an_anthropic_model_id_never_reaches_another_provider(self) -> None:
        """Callers hold a configured `claude-*` id. Forwarding it as the model
        for Gemini's endpoint would 404 mid-login, and only once the key
        changed -- so the provider's own default has to win."""
        gen = TextGenerator(provider="gemini", anthropic_model="claude-opus-5")
        assert gen.model_id is not None
        assert "claude" not in gen.model_id
        assert gen.model_id.startswith("gemini/")

    def test_that_same_id_is_used_when_anthropic_does_resolve(self) -> None:
        gen = TextGenerator(provider="anthropic", anthropic_model="claude-opus-5")
        assert gen.model_id == "claude-opus-5"

    def test_model_id_is_none_when_nothing_is_configured(self) -> None:
        assert TextGenerator().model_id is None


class TestOpenAICompatibleCall:
    def test_returns_the_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gen = gemini_gen(monkeypatch)
        gen._client = FakeOpenAIClient("என்ன சாப்பிட்டீங்க?")
        assert gen.complete(system="s", user="u") == "என்ன சாப்பிட்டீங்க?"

    def test_sends_system_and_user_separately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gen = gemini_gen(monkeypatch)
        client = FakeOpenAIClient()
        gen._client = client
        gen.complete(system="SYS", user="USR")
        roles = {m["role"]: m["content"] for m in client.calls[0]["messages"]}
        assert roles == {"system": "SYS", "user": "USR"}

    def test_empty_text_raises_rather_than_returning_blank(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blank question would be issued to a real user as a real challenge."""
        gen = gemini_gen(monkeypatch)
        gen._client = FakeOpenAIClient("   ")
        with pytest.raises(TextError):
            gen.complete(system="s", user="u")

    def test_truncation_names_the_constant_to_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tagger learned this the hard way: a truncated response is valid
        text and invalid everything else, and the default error names neither
        the cause nor a fix."""
        gen = gemini_gen(monkeypatch)
        gen._client = FakeOpenAIClient("", finish_reason="length")
        with pytest.raises(TextError, match="max_tokens"):
            gen.complete(system="s", user="u")

    def test_partial_text_at_the_limit_is_still_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated *question* is still a usable question, unlike truncated
        JSON. Discarding it would fall back to a template for no reason."""
        gen = gemini_gen(monkeypatch)
        gen._client = FakeOpenAIClient("நீங்க எந்த", finish_reason="length")
        assert gen.complete(system="s", user="u") == "நீங்க எந்த"

    def test_max_tokens_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        gen = gemini_gen(monkeypatch, max_tokens=99)
        client = FakeOpenAIClient()
        gen._client = client
        gen.complete(system="s", user="u")
        assert client.calls[0]["max_tokens"] == 99


class TestChallengeIntegration:
    """The generator's own reporting must stay truthful about which path ran."""

    def test_no_provider_reports_plain_template_not_an_error(self) -> None:
        """"Nothing configured" and "configured and broken" are different
        findings; a report that conflates them sends someone debugging a
        network problem that does not exist."""
        from test_pipeline_layers import make_skg

        from kavach.challenge import ChallengeGenerator

        challenge = ChallengeGenerator().generate("s1", make_skg())
        assert challenge.generator == "template"

    def test_use_llm_false_never_builds_a_generator(self) -> None:
        from kavach.challenge import ChallengeGenerator

        assert ChallengeGenerator(use_llm=False)._text is None


class TestAttackIntegration:
    def test_provenance_names_the_model_that_wrote_the_text(self) -> None:
        """An attack corpus is evidence about a specific adversary. Recording
        an Anthropic id on free-tier text attributes the IAPMR to the wrong
        system."""
        from kavach.attacks.text import AttackTextGenerator

        gen = AttackTextGenerator(seed=0)
        gen._text._resolved = "gemini"
        gen._text.model = "gemini-x"
        assert gen._model_used() == "gemini/gemini-x"

    def test_an_explicit_key_still_means_anthropic(self) -> None:
        """Passing `api_key` is a deliberate choice and is not second-guessed."""
        from kavach.attacks.text import AttackTextGenerator

        gen = AttackTextGenerator(seed=0, api_key="sk-test", model="claude-x")
        assert gen._model_used() == "claude-x"
