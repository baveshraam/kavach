"""Word-level language identification and semantic-class tagging.

Turns an ASR transcript into the annotated `Token` sequence the CSBG is built
from. Two stages:

    rules.py     Script detection + romanised-Tamil lexicon. Free, no API.
    llm.py       LLM adjudication for ambiguous tokens + semantic class.

`rules` has no dependencies beyond the standard library. `llm` imports the
`anthropic` package lazily, so this subpackage can be imported and the rules
tested without it.

    from kavach.lid import LIDPipeline, LLMTagger

    pipeline = LIDPipeline(llm_tagger=LLMTagger())
    utt = pipeline.tag_utterance("naan college-la irundhen", utterance_id="u1")
"""

from . import rules
from .llm import (
    DEFAULT_MODEL,
    LLMTagger,
    TaggedToken,
    TaggingResponse,
    TaggingStats,
    build_system_prompt,
    estimate_cost,
    to_tokens,
)
from .pipeline import LIDPipeline, PipelineStats
from .rules import RuleResult, resolution_rate, simple_tokenise, tag_token, tag_tokens

__all__ = [
    "rules",
    "RuleResult",
    "tag_token",
    "tag_tokens",
    "simple_tokenise",
    "resolution_rate",
    "LLMTagger",
    "TaggedToken",
    "TaggingResponse",
    "TaggingStats",
    "build_system_prompt",
    "to_tokens",
    "estimate_cost",
    "DEFAULT_MODEL",
    "LIDPipeline",
    "PipelineStats",
]
