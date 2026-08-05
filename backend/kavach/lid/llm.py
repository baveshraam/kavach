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
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

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

#: Floor for a tagging response's output budget. Not the whole story -- see
#: `output_budget`, which scales it with the input.
DEFAULT_MAX_TOKENS = 16384

#: Output tokens to allow per input word.
#:
#: **These are caps, not reservations.** Billing and latency follow the tokens
#: actually generated, so an over-generous ceiling costs nothing and an
#: under-generous one truncates the JSON mid-string and kills the run. They are
#: therefore set well above the observed need, deliberately.
#:
#: Two things make tagging expensive per word here, and both were underestimated
#: on the first attempt. JSON escapes non-ASCII, so a Tamil word arrives as
#: `நேத்து` -- six characters per character,
#: before the surrounding object with its language, semantic_class and
#: confidence fields. And a thinking-capable model spends output tokens on
#: reasoning before it emits any of that.
#:
#: Measured against this corpus: a 37-word utterance overran 4096, which is why
#: 96/word (and its 4096 floor) was not enough.
OUTPUT_TOKENS_PER_WORD = 320


def output_budget(n_tokens: int) -> int:
    """Output token budget for tagging `n_tokens` words.

    Scaled rather than fixed, and generous in both terms. A fixed budget is
    either wasteful on the short utterances or too small on the long ones, and
    "wasteful" here costs nothing because the number is a ceiling: the response
    truncating is the only failure mode with a price.
    """
    return max(DEFAULT_MAX_TOKENS, OUTPUT_TOKENS_PER_WORD * n_tokens + 1024)

#: Attempts per request before giving up.
#:
#: Sized for the case this exists for: annotating a corpus one utterance at a
#: time against a *free* tier. Those allow ten-odd requests a minute, a 56-
#: utterance pass fires far faster than that, and a 429 partway through used to
#: abort the run and discard everything already tagged.
MAX_ATTEMPTS = 6

#: Base for the exponential backoff, in seconds. 2, 4, 8, 16, 32 with jitter --
#: the last of which is longer than most free-tier rate windows, which is the
#: point.
RETRY_BASE_SECONDS = 2.0

#: HTTP statuses worth retrying. 429 is the rate limit; 5xx are the provider's
#: problem and usually transient. **408 and 409 are deliberately absent**: a
#: timeout may have been served and a conflict will not resolve itself.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})

_T = TypeVar("_T")


def _retry_after(exc: BaseException) -> float | None:
    """The provider's own `Retry-After`, in seconds, if it sent one.

    Preferred over the backoff schedule whenever present: the provider knows
    when its window resets and guessing shorter just burns another attempt.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-requests"):
        raw = headers.get(key) if hasattr(headers, "get") else None
        if raw is None:
            continue
        try:
            seconds = float(str(raw).rstrip("s"))
        except ValueError:
            continue  # HTTP-date form; fall back to the schedule
        if seconds >= 0:
            return min(seconds, 120.0)
    return None


def is_transient(exc: BaseException) -> bool:
    """True if `exc` is worth retrying.

    Deliberately structural rather than a list of SDK exception classes: the
    anthropic and openai clients raise different types for the same condition,
    and importing either at module scope would undo the lazy-import that keeps
    this module cheap. Both set `status_code` on API errors, and both name
    their connection failures recognisably.

    A token-alignment `ValueError` and a content refusal are **not** transient.
    They are deterministic, and retrying them burns quota to get the same
    answer five more times.
    """
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    name = type(exc).__name__
    return any(
        marker in name
        for marker in ("RateLimit", "Timeout", "Connection", "APIError", "InternalServer")
    )


def with_retries(
    call: Callable[[], _T],
    *,
    attempts: int = MAX_ATTEMPTS,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """Run `call`, retrying transient failures with exponential backoff.

    Jitter is full-range rather than fixed: without it a batch of requests that
    were rate-limited together retry together, hit the same limit together, and
    the backoff accomplishes nothing.

    Raises:
        The last exception, once attempts are exhausted or it is not transient.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless transient
            if attempt >= attempts or not is_transient(exc):
                raise
            delay = _retry_after(exc)
            if delay is None:
                delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                delay = random.uniform(delay / 2, delay)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)
    raise AssertionError("unreachable: the loop either returns or raises")


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
    """Anthropic-backed token tagger, with prompt caching and the Batch API.

    See `OpenAICompatibleTagger` for the free-tier path (Gemini, Groq, or a
    local Ollama), which shares this class's system prompt, JSON schema and
    alignment check and differs only in transport.

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
        different instructions, silently corrupting the corpus. That includes
        the output budget, which scales with the utterance: see
        `output_budget`.
        """
        return {
            "model": self.model,
            "max_tokens": max(self.max_tokens, output_budget(len(tokens))),
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

        params = self._request_params(tokens, context)
        response = with_retries(lambda: self.client.messages.create(**params))
        self.stats.record(response.usage)

        if response.stop_reason == "refusal":
            raise RuntimeError(
                f"Tagging request refused (category="
                f"{getattr(response.stop_details, 'category', None)}). "
                "This should not occur for linguistic annotation; inspect the input."
            )

        if response.stop_reason == "max_tokens":
            # Same trap as the OpenAI-compatible path: a truncated response is
            # valid text and invalid JSON, and the parser's error names the
            # line number rather than the cause.
            raise RuntimeError(
                f"Hit the output limit tagging {len(tokens)} words, so the JSON "
                f"is truncated. The budget was "
                f"{max(self.max_tokens, output_budget(len(tokens)))} tokens; "
                "non-ASCII is JSON-escaped at six characters per character, so "
                "Tamil is expensive here -- raise OUTPUT_TOKENS_PER_WORD."
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


# --------------------------------------------------------------------------
# OpenAI-compatible providers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Provider:
    """An OpenAI-compatible chat-completions endpoint.

    Gemini and Groq both publish one, which is what makes a single adapter
    enough. They are not equivalent for this task -- see `PROVIDERS`.
    """

    name: str
    base_url: str
    default_model: str
    env_keys: tuple[str, ...]
    signup: str

    def api_key(self) -> str | None:
        for key in self.env_keys:
            value = os.environ.get(key)
            if value:
                return value
        return None


#: Providers this tagger can drive, in the order `resolve_provider` tries them.
#:
#: **Gemini first, and the ordering is a judgement about Tamil.** The task is
#: word-level language ID on romanised Tamil mixed with English -- deciding
#: whether "veetla" is Tamil and "family" is English inside one sentence. That
#: is a low-resource multilingual judgement, not a reasoning task, and the
#: models differ on it far more than they differ on benchmarks. Gemini Flash
#: has substantially more Tamil in its training mix than the Llama models Groq
#: serves. Groq is kept because it is fast and free and worth having when
#: Gemini's daily quota runs out mid-corpus.
#:
#: A wrong tag here is not noise. It moves a token into the wrong language for
#: a semantic class, which is precisely the quantity the CSBG measures, so
#: tagging quality bounds every number downstream.
PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # Not 2.5-flash: Google has closed it to new keys, and a key issued
        # today gets a 404 naming the model rather than an auth error. If this
        # one goes the same way, `client.models.list()` on the base_url below
        # shows what the key can actually reach -- prefer a non-preview flash.
        default_model="gemini-3.6-flash",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        signup="https://aistudio.google.com/apikey",
    ),
    Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        env_keys=("GROQ_API_KEY",),
        signup="https://console.groq.com/keys",
    ),
    Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        env_keys=("OPENAI_API_KEY",),
        signup="https://platform.openai.com/api-keys",
    ),
    Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        env_keys=("OLLAMA_API_KEY",),
        signup="https://ollama.com/download",
    ),
)

PROVIDERS_BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


class OpenAICompatibleTagger:
    """Token tagger against any OpenAI-compatible chat-completions endpoint.

    Exists so the corpus can be annotated on a free tier. `LLMTagger` uses the
    Anthropic Batch API and prompt caching, which halve the cost of a large
    corpus and are worth having -- but a paid key is a hard blocker on a
    student project, and this annotation is the one step nothing downstream
    works without.

    What is shared with `LLMTagger` and must stay shared: the system prompt,
    the JSON schema, and `_check_alignment`. Two taggers that drifted on any of
    those would produce a corpus annotated under two different instructions
    with no record of which token came from which.

    What is not available here: prompt caching and the batch API. Both are
    provider-specific. `TaggingStats.cache_hit_rate` therefore reads 0.0 on
    this path and that is correct rather than broken, so do not go looking for
    the leak that `llm.py`'s module docstring warns about.
    """

    def __init__(
        self,
        *,
        provider: Provider | str = "gemini",
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self.provider = (
            provider if isinstance(provider, Provider) else PROVIDERS_BY_NAME[provider]
        )
        self.model = model or self.provider.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key
        self._client: Any = None
        self.stats = TaggingStats()
        self._system_prompt = build_system_prompt()
        self.retries = 0
        """Transient failures retried over this tagger's life. **Report this
        for a corpus run.** A pass that needed forty retries took its labels
        from the same model as one that needed none, but it also means the run
        was fighting a rate limit, and a future pass on a bigger corpus will
        need pacing rather than luck."""
        self.retry_seconds = 0.0

    def _note_retry(self, attempt: int, delay: float, exc: BaseException) -> None:
        """Count and announce a retry.

        Printed rather than silent: a corpus pass that stalls for thirty
        seconds with no output looks hung, and the operator's next move is to
        kill it -- which is exactly the wrong move.
        """
        self.retries += 1
        self.retry_seconds += delay
        status = getattr(exc, "status_code", None) or type(exc).__name__
        print(
            f"  [{self.provider.name}] {status}; retrying in {delay:.1f}s "
            f"(attempt {attempt}/{MAX_ATTEMPTS})",
            flush=True,
        )

    supports_batch = False
    """No batch API. `LIDPipeline.tag_corpus` checks this and loops instead."""

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ImportError(
                    "The openai package is required for OpenAI-compatible providers. "
                    "Install with `pip install openai` -- it is the client library, "
                    "not a dependency on OpenAI the service, and is what Gemini and "
                    "Groq's compatibility endpoints expect."
                ) from exc

            key = self._api_key or self.provider.api_key()
            if not key:
                raise RuntimeError(
                    f"No API key for provider {self.provider.name!r}. Set "
                    f"{' or '.join(self.provider.env_keys)}; free keys at "
                    f"{self.provider.signup}"
                )
            self._client = OpenAI(api_key=key, base_url=self.provider.base_url)
        return self._client

    def tag(self, tokens: list[str], *, context: str | None = None) -> list[TaggedToken]:
        """Tag one utterance's tokens. Same contract as `LLMTagger.tag`.

        Raises:
            ValueError: If the model returns a different number of tokens.
                Hard failure by design -- padding or truncating would attach
                every tag to the wrong surface form.
        """
        if not tokens:
            return []

        # Retried, because this is the path a free-tier key takes and
        # `tag_corpus` drives it once per utterance with no pacing. See
        # `with_retries`.
        response = with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                max_tokens=max(self.max_tokens, output_budget(len(tokens))),
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": LLMTagger._user_content(tokens, context),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tagging_response",
                        "strict": True,
                        "schema": _json_schema(),
                    },
                },
            ),
            on_retry=self._note_retry,
        )
        self.stats.record(_usage_adapter(response.usage))

        choice = response.choices[0]
        text = choice.message.content
        if not text:
            raise RuntimeError(
                f"{self.provider.name} returned an empty tagging response "
                f"(finish_reason={choice.finish_reason!r})."
            )

        # Caught here rather than left to the JSON parser. A truncated response
        # is valid text and invalid JSON, so pydantic reports "EOF while
        # parsing a string at line 106" -- which names neither the cause nor
        # anything the operator can act on.
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"{self.provider.name} hit the output limit tagging "
                f"{len(tokens)} words, so the JSON is truncated. The budget was "
                f"{max(self.max_tokens, output_budget(len(tokens)))} tokens. "
                "Non-ASCII is JSON-escaped at six characters per character, so "
                "Tamil is expensive here -- raise OUTPUT_TOKENS_PER_WORD."
            )

        parsed = TaggingResponse.model_validate_json(text)
        LLMTagger._check_alignment(tokens, parsed.tokens)
        return parsed.tokens


@dataclass(frozen=True, slots=True)
class _Usage:
    """Anthropic-shaped usage, so `TaggingStats.record` needs no branch."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def _usage_adapter(usage: Any) -> _Usage:
    """Map an OpenAI-style usage object onto the Anthropic field names."""
    if usage is None:
        return _Usage()
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return _Usage(
        input_tokens=(getattr(usage, "prompt_tokens", 0) or 0) - cached,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cached,
    )


_env_file_loaded = False


def load_api_keys() -> None:
    """Merge a `.env` file into `os.environ`, once per process.

    `config.Settings` also reads `.env`, but pydantic-settings only maps the
    `KAVACH_`-prefixed names onto its own fields and puts nothing back into the
    environment -- so an unprefixed `GEMINI_API_KEY` sitting in `.env` was
    invisible to `Provider.api_key()`, and the tagger reported "no key" with
    the key on disk two directories up.

    Real environment variables win over the file: an operator who exported a
    key for one command means that key, not a stale one someone committed to
    their checkout months ago.
    """
    global _env_file_loaded
    if _env_file_loaded:
        return
    _env_file_loaded = True
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is in requirements
        return
    # usecwd: find it from where the command was run, not from this file, so
    # `python -m kavach.annotate` works from the repo root and from backend/.
    load_dotenv(find_dotenv(usecwd=True), override=False)


def available_providers() -> list[Provider]:
    """Providers with a key present in the environment, in preference order."""
    load_api_keys()
    return [p for p in PROVIDERS if p.api_key()]


def make_tagger(
    provider: str | None = None, *, model: str | None = None
) -> LLMTagger | OpenAICompatibleTagger | None:
    """Build whichever tagger the environment can support.

    Args:
        provider: Force one of `PROVIDERS_BY_NAME`, or "anthropic". None picks
            the first provider with a key present, Anthropic first.
        model: Override the provider default.

    Returns:
        A tagger, or None when no key is available anywhere. None is not an
        error -- it is the normal state on a fresh machine, and the caller
        (`annotate._default_pipeline`) degrades to rules-only and says so.
    """
    load_api_keys()
    if provider == "anthropic" or (
        provider is None
        and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    ):
        return LLMTagger(model=model or DEFAULT_MODEL)

    if provider is not None:
        if provider not in PROVIDERS_BY_NAME:
            raise ValueError(
                f"unknown provider {provider!r}; choose from "
                f"{['anthropic', *PROVIDERS_BY_NAME]}"
            )
        return OpenAICompatibleTagger(provider=provider, model=model)

    found = available_providers()
    if not found:
        return None
    return OpenAICompatibleTagger(provider=found[0], model=model)


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
    "OpenAICompatibleTagger",
    "Provider",
    "PROVIDERS",
    "PROVIDERS_BY_NAME",
    "available_providers",
    "load_api_keys",
    "make_tagger",
    "build_system_prompt",
    "to_tokens",
    "estimate_cost",
    "is_transient",
    "with_retries",
    "DEFAULT_MODEL",
    "OUTPUT_TOKENS_PER_WORD",
    "output_budget",
    "MAX_ATTEMPTS",
    "RETRYABLE_STATUS",
]
