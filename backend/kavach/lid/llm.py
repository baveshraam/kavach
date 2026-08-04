"""LLM-based word-level language ID and semantic-class tagging.

This is the second stage of the LID pipeline, and the only place an LLM does
load-bearing work in KAVACH. It resolves what `rules.py` could not:

* Latin-script tokens that may be English or romanised Tamil
* Semantic class for every token (rules never determine this)
* Named-entity detection

Cost design
-----------
Annotating a 30-speaker corpus is roughly 25k tokens of adjudication. Three
things keep that affordable, and all three matter:

1.  **Rules first.** `rules.py` resolves the large majority of tokens for
    free; only the remainder reaches the model.
2.  **Prompt caching.** The ontology block is byte-identical on every request
    and sits behind a `cache_control` breakpoint, so it is written once and
    read at ~0.1x thereafter. Verify this is working -- see
    `TaggingStats.cache_hit_rate`; a zero rate means something volatile
    leaked into the prefix and you are paying full price on every call.
3.  **Batch API for corpus annotation.** Half price, and annotating a corpus
    is not latency-sensitive. Interactive login-time tagging uses the
    synchronous path; bulk annotation should always use `submit_batch`.

Structured output guarantees the response parses. Without it this module
would need a retry loop around `json.loads`, which is exactly the kind of
scaffolding that becomes dead weight.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..csbg.ontology import (
    CLASS_DESCRIPTIONS,
    CLASS_ORDER,
    Language,
    SemanticClass,
)
from ..csbg.tokens import Token

#: Default model. Tagging is a high-volume, well-specified task, so it runs at
#: low effort -- but with adaptive thinking left ON. Disabling thinking on
#: Opus 5 risks internal tags leaking into the visible response, and lowering
#: effort achieves the same cost saving without that failure mode.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "low"

#: Max tokens for a tagging response. One utterance is at most ~40 tokens, and
#: each produces a small JSON object, so this is generous.
DEFAULT_MAX_TOKENS = 4096


class TaggedToken(BaseModel):
    """One token as tagged by the model."""

    text: str
    language: Language
    semantic_class: SemanticClass
    confidence: float = Field(default=1.0)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        # Structured-output JSON Schema does not support numeric bounds
        # (no minimum/maximum), so the range is enforced here rather than by
        # the API. A model that returns 1.5 is clamped, not rejected --
        # failing a whole utterance over a confidence value would be worse.
        return max(0.0, min(1.0, v))


class TaggingResponse(BaseModel):
    """Full model response for one utterance."""

    tokens: list[TaggedToken]


def _json_schema() -> dict[str, Any]:
    """JSON Schema for the structured-output constraint.

    Hand-built rather than derived from the Pydantic model because the API's
    structured-output subset is narrower than full JSON Schema: every object
    needs `additionalProperties: false` and a complete `required` list, and
    numeric bounds are unsupported. Generating this from Pydantic would emit
    constraints the API rejects.
    """
    return {
        "type": "object",
        "properties": {
            "tokens": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The token, copied exactly from the input.",
                        },
                        "language": {
                            "type": "string",
                            "enum": [lang.value for lang in Language],
                        },
                        "semantic_class": {
                            "type": "string",
                            "enum": [c.value for c in CLASS_ORDER],
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in the language tag, 0.0 to 1.0.",
                        },
                    },
                    "required": ["text", "language", "semantic_class", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tokens"],
        "additionalProperties": False,
    }


def build_system_prompt() -> str:
    """The cached instruction block.

    MUST be byte-identical across requests or prompt caching silently stops
    working. Nothing dynamic goes in here -- no timestamps, no speaker IDs, no
    per-utterance context. Those belong in the user turn, after the cache
    breakpoint.
    """
    class_lines = "\n".join(
        f"- {cls_.value}: {CLASS_DESCRIPTIONS[cls_]}" for cls_ in CLASS_ORDER
    )
    return f"""You are annotating Tamil-English code-switched speech transcripts for a \
speaker-verification research corpus. For each token you assign a language tag and a \
semantic concept class.

# Language tags

- TA: Tamil. Includes Tamil script AND romanised Tamil written in Latin letters \
(e.g. "naan", "romba", "panren", "appuram"). Romanised Tamil is common in ASR output \
and must be tagged TA, not EN.
- EN: English.
- NEUTRAL: Language-independent tokens: digits, punctuation, symbols, and \
interjections shared by both languages.
- NAMED_ENTITY: Proper names of people, places, brands, organisations, and titled \
works. Tag these NAMED_ENTITY regardless of what script they are written in -- saying \
"Chennai" is not a language choice.

## Intra-word code-mixing

An English stem with a Tamil suffix ("college-la", "bus-ku", "phone-oda") is tagged by \
its STEM, so these are EN. The Tamil morphology supplies the grammatical frame, but the \
concept lives in the stem, which is what this annotation measures.

# Semantic classes

{class_lines}

# Rules

1. Return exactly one entry per input token, in the same order. Copy `text` verbatim.
2. Never merge, split, skip, or reorder tokens. The count must match the input exactly.
3. `confidence` reflects certainty about the LANGUAGE tag only. Use below 0.7 when a \
Latin-script token is genuinely ambiguous between English and romanised Tamil.
4. Choose the most specific applicable semantic class. Use OTHER only when nothing \
else fits.
5. Judge romanised Tamil by how the word sounds, not by whether it is a valid English \
word. "enna" is Tamil ("what"), not a name.
"""


@dataclass(slots=True)
class TaggingStats:
    """Token accounting across a tagging session.

    `cache_hit_rate` is the metric that matters: if it stays near zero across
    repeated calls, prompt caching is broken and cost is roughly 10x what it
    should be.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from cache. Should approach 1.0."""
        total = self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens
        return self.cache_read_tokens / total if total else 0.0

    def record(self, usage: Any) -> None:
        self.requests += 1
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    def summary(self) -> str:
        return (
            f"{self.requests} requests | in {self.input_tokens} "
            f"cache_write {self.cache_creation_tokens} cache_read {self.cache_read_tokens} "
            f"out {self.output_tokens} | cache hit {self.cache_hit_rate:.1%}"
        )


class LLMTagger:
    """Anthropic-backed token tagger.

    The `anthropic` package is imported lazily so the CSBG core stays
    installable and testable without it.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._client: Any = None
        self.stats = TaggingStats()
        self._system_prompt = build_system_prompt()

    @property
    def client(self) -> Any:
        """Lazily-constructed Anthropic client.

        Constructed with no explicit key when none was supplied, so the SDK's
        own credential resolution applies (ANTHROPIC_API_KEY, then
        ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile). Do not
        require the env var to be set -- a configured OAuth profile is a valid
        credential source and demanding a key would break it.
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ImportError(
                    "The `anthropic` package is required for LLM tagging. "
                    "Install it with `pip install anthropic`, or use "
                    "kavach.lid.rules for rule-only tagging."
                ) from exc
            self._client = (
                anthropic.Anthropic(api_key=self._api_key)
                if self._api_key
                else anthropic.Anthropic()
            )
        return self._client

    # ------------------------------------------------------------- requests

    def _system_blocks(self) -> list[dict[str, Any]]:
        """System prompt with a cache breakpoint on the last block."""
        return [
            {
                "type": "text",
                "text": self._system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _output_config(self) -> dict[str, Any]:
        return {
            "format": {"type": "json_schema", "schema": _json_schema()},
            "effort": self.effort,
        }

    @staticmethod
    def _user_content(tokens: list[str], context: str | None = None) -> str:
        numbered = "\n".join(f"{i}\t{t}" for i, t in enumerate(tokens))
        parts = []
        if context:
            parts.append(f"Utterance: {context}\n")
        parts.append(f"Tag these {len(tokens)} tokens:\n{numbered}")
        return "\n".join(parts)

    def _request_params(self, tokens: list[str], context: str | None) -> dict[str, Any]:
        """Request body shared by the sync and batch paths.

        Kept in one place so the two paths cannot drift -- a divergence would
        mean batch-annotated and login-annotated tokens were produced under
        different instructions, silently corrupting the corpus.
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self._system_blocks(),
            "output_config": self._output_config(),
            "messages": [{"role": "user", "content": self._user_content(tokens, context)}],
        }

    # ---------------------------------------------------------- synchronous

    def tag(self, tokens: list[str], *, context: str | None = None) -> list[TaggedToken]:
        """Tag one utterance's tokens. Used at login time, where latency matters.

        Args:
            tokens: Surface forms, in order.
            context: Optional full transcript, which materially improves
                disambiguation of short Latin-script tokens.

        Returns:
            One TaggedToken per input token, in order.

        Raises:
            ValueError: If the model returns a different number of tokens.
                This is a hard failure by design: silently padding or
                truncating would misalign every token's tag with its surface
                form and corrupt the CSBG in a way that is invisible
                downstream.
        """
        if not tokens:
            return []

        response = self.client.messages.create(**self._request_params(tokens, context))
        self.stats.record(response.usage)

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Tagging request refused (category="
                f"{getattr(response.stop_details, 'category', None)}). "
                "This should not occur for linguistic annotation; inspect the input."
            )

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError("Tagging response contained no text block.")

        parsed = TaggingResponse.model_validate_json(text)
        self._check_alignment(tokens, parsed.tokens)
        return parsed.tokens

    @staticmethod
    def _check_alignment(inputs: list[str], outputs: list[TaggedToken]) -> None:
        if len(outputs) != len(inputs):
            raise ValueError(
                f"Tagger returned {len(outputs)} tokens for {len(inputs)} inputs. "
                "Token alignment is required -- a mismatch would attach tags to the "
                "wrong surface forms. Inputs: {inputs!r}"
            )

    # --------------------------------------------------------------- batch

    def submit_batch(
        self, utterances: dict[str, list[str]], *, contexts: dict[str, str] | None = None
    ) -> str:
        """Submit corpus annotation as a batch. USE THIS FOR BULK ANNOTATION.

        Half the cost of the synchronous path and no rate-limit pressure.
        Corpus annotation is not latency-sensitive, so there is no reason to
        pay double for it.

        Args:
            utterances: {utterance_id: tokens}. The id becomes the batch
                `custom_id` and is how results are matched back.
            contexts: Optional {utterance_id: transcript}.

        Returns:
            Batch ID. Poll with `poll_batch`, collect with `collect_batch`.
        """
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        contexts = contexts or {}
        requests = [
            Request(
                custom_id=utt_id,
                params=MessageCreateParamsNonStreaming(
                    **self._request_params(tokens, contexts.get(utt_id))
                ),
            )
            for utt_id, tokens in utterances.items()
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def poll_batch(self, batch_id: str) -> str:
        """Return the batch's processing status (`in_progress`, `ended`, ...)."""
        return self.client.messages.batches.retrieve(batch_id).processing_status

    def collect_batch(
        self, batch_id: str, expected: dict[str, list[str]]
    ) -> tuple[dict[str, list[TaggedToken]], dict[str, str]]:
        """Collect finished batch results.

        Results arrive in arbitrary order and are keyed by `custom_id`, never
        by position.

        Args:
            batch_id: From `submit_batch`.
            expected: The original {utterance_id: tokens}, for alignment checks.

        Returns:
            (successes, failures) -- failures maps utterance_id to an error
            description. Partial failure is normal on large batches; re-submit
            only the failures rather than the whole corpus.
        """
        successes: dict[str, list[TaggedToken]] = {}
        failures: dict[str, str] = {}

        for result in self.client.messages.batches.results(batch_id):
            uid = result.custom_id
            if result.result.type != "succeeded":
                failures[uid] = f"{result.result.type}"
                continue
            message = result.result.message
            text = next((b.text for b in message.content if b.type == "text"), None)
            if text is None:
                failures[uid] = "no_text_block"
                continue
            try:
                parsed = TaggingResponse.model_validate_json(text)
                self._check_alignment(expected[uid], parsed.tokens)
            except (ValueError, KeyError) as exc:
                failures[uid] = str(exc)
                continue
            successes[uid] = parsed.tokens
            self.stats.record(message.usage)

        return successes, failures


def to_tokens(
    tagged: list[TaggedToken], *, timings: list[tuple[int, int]] | None = None
) -> list[Token]:
    """Convert tagger output into CSBG `Token` objects.

    Args:
        tagged: Model output.
        timings: Optional (start_ms, end_ms) per token from ASR word timestamps.
    """
    out: list[Token] = []
    for i, t in enumerate(tagged):
        start, end = timings[i] if timings and i < len(timings) else (0, 0)
        out.append(
            Token(
                text=t.text,
                language=t.language,
                semantic_class=t.semantic_class,
                lid_confidence=t.confidence,
                start_ms=start,
                end_ms=end,
            )
        )
    return out


def estimate_cost(
    n_tokens_to_tag: int,
    *,
    system_prompt_tokens: int = 900,
    cached: bool = True,
) -> dict[str, float]:
    """Rough USD estimate for annotating a corpus. Sanity check, not a bill.

    Uses Claude Opus 5 list pricing ($5/M input, $25/M output) with cache
    reads at ~0.1x and cache writes at ~1.25x. Verify current pricing before
    quoting these numbers anywhere.

    Args:
        n_tokens_to_tag: Word tokens needing LLM adjudication (i.e. *after*
            the rules stage, not the raw corpus size).
        system_prompt_tokens: Size of the cached instruction block.
        cached: Whether prompt caching is working.
    """
    tokens_per_request = 40
    requests = max(1, n_tokens_to_tag // tokens_per_request)

    in_price, out_price = 5.0 / 1e6, 25.0 / 1e6

    if cached:
        prompt_cost = (
            system_prompt_tokens * 1.25 * in_price
            + (requests - 1) * system_prompt_tokens * 0.1 * in_price
        )
    else:
        prompt_cost = requests * system_prompt_tokens * in_price

    content_cost = n_tokens_to_tag * 3 * in_price
    output_cost = n_tokens_to_tag * 25 * out_price
    total = prompt_cost + content_cost + output_cost

    return {
        "requests": float(requests),
        "prompt_usd": round(prompt_cost, 4),
        "content_usd": round(content_cost, 4),
        "output_usd": round(output_cost, 4),
        "total_usd": round(total, 2),
        "total_usd_batch": round(total * 0.5, 2),
    }


__all__ = [
    "TaggedToken",
    "TaggingResponse",
    "TaggingStats",
    "LLMTagger",
    "build_system_prompt",
    "to_tokens",
    "estimate_cost",
    "DEFAULT_MODEL",
]
