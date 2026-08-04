"""Two-stage LID pipeline: rules first, LLM for the remainder.

Rules resolve script-unambiguous tokens for free; the LLM handles the
genuinely hard ones and assigns semantic class (which rules never determine).

The pipeline can run without an LLM at all (`llm_tagger=None`), which is what
the test suite and any offline reproduction use. In that mode unresolved
tokens fall back to a low-confidence guess -- usable for smoke-testing the
plumbing, **not** for producing corpus annotations or paper numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..csbg.ontology import Language, SemanticClass
from ..csbg.tokens import Token, UtteranceTokens
from . import rules
from .llm import LLMTagger, TaggedToken, to_tokens


@dataclass(slots=True)
class PipelineStats:
    """Where tags came from. Report these -- they justify the design."""

    total_tokens: int = 0
    resolved_by_rules: int = 0
    resolved_by_llm: int = 0
    fallback_guesses: int = 0
    """Tokens tagged without either stage resolving them. Non-zero means the
    output is not corpus-grade."""

    @property
    def rule_resolution_rate(self) -> float:
        return self.resolved_by_rules / self.total_tokens if self.total_tokens else 0.0

    @property
    def is_corpus_grade(self) -> bool:
        """False if any token was guessed. Gate corpus exports on this."""
        return self.fallback_guesses == 0

    def summary(self) -> str:
        flag = "" if self.is_corpus_grade else f"  [!] {self.fallback_guesses} guessed"
        return (
            f"{self.total_tokens} tokens | rules {self.resolved_by_rules} "
            f"({self.rule_resolution_rate:.1%}) | llm {self.resolved_by_llm}{flag}"
        )


@dataclass(slots=True)
class LIDPipeline:
    """Orchestrates rule-based and LLM-based tagging."""

    llm_tagger: LLMTagger | None = None
    trust_rules_over_llm: bool = True
    """When True, a confident rule result (Tamil script) wins over the LLM.
    Script evidence is definitionally correct and cheaper to trust; the LLM is
    for cases where script is uninformative."""

    stats: PipelineStats = field(default_factory=PipelineStats)

    def tag_utterance(
        self,
        transcript: str,
        *,
        utterance_id: str,
        speaker_id: str | None = None,
        timings: list[tuple[int, int]] | None = None,
    ) -> UtteranceTokens:
        """Tag a transcript into an annotated UtteranceTokens.

        Args:
            transcript: ASR output.
            utterance_id: Identifier for the result.
            speaker_id: Optional speaker identifier.
            timings: Optional per-token (start_ms, end_ms) from ASR.
        """
        surface = rules.simple_tokenise(transcript)
        if not surface:
            return UtteranceTokens(
                utterance_id=utterance_id, tokens=[], speaker_id=speaker_id,
                transcript=transcript,
            )

        rule_results = rules.tag_tokens(surface)
        self.stats.total_tokens += len(surface)

        # Semantic class always needs the LLM, so every token goes to it when
        # one is configured -- the rules stage saves cost on *language*, and
        # its confident language calls still override the model below.
        llm_tags: list[TaggedToken] | None = None
        if self.llm_tagger is not None:
            llm_tags = self.llm_tagger.tag(surface, context=transcript)

        tokens = self._merge(surface, rule_results, llm_tags, timings)
        return UtteranceTokens(
            utterance_id=utterance_id,
            tokens=tokens,
            speaker_id=speaker_id,
            transcript=transcript,
        )

    def _merge(
        self,
        surface: list[str],
        rule_results: list[rules.RuleResult],
        llm_tags: list[TaggedToken] | None,
        timings: list[tuple[int, int]] | None,
    ) -> list[Token]:
        """Combine rule and LLM evidence into final tokens."""
        if llm_tags is not None:
            merged = to_tokens(llm_tags, timings=timings)
            out: list[Token] = []
            for tok, rule in zip(merged, rule_results):
                if (
                    self.trust_rules_over_llm
                    and rule.is_resolved
                    and rule.confidence >= 0.95
                    and rule.language is not tok.language
                ):
                    # Script evidence beats the model. Keep the model's
                    # semantic class -- rules never determine that.
                    out.append(
                        Token(
                            text=tok.text,
                            language=rule.language,  # type: ignore[arg-type]
                            semantic_class=tok.semantic_class,
                            lid_confidence=rule.confidence,
                            start_ms=tok.start_ms,
                            end_ms=tok.end_ms,
                        )
                    )
                else:
                    out.append(tok)
                self.stats.resolved_by_llm += 1
            return out

        # No LLM configured: rules only, with a fallback for the remainder.
        out = []
        for i, (text, rule) in enumerate(zip(surface, rule_results)):
            start, end = timings[i] if timings and i < len(timings) else (0, 0)
            if rule.is_resolved:
                language, confidence = rule.language, rule.confidence
                self.stats.resolved_by_rules += 1
            else:
                # Latin script, unresolved. Guessing EN is the least-bad
                # default (most Latin tokens are English), but the low
                # confidence marks it as untrustworthy and it is counted so
                # corpus exports can be blocked.
                language, confidence = Language.EN, 0.35
                self.stats.fallback_guesses += 1
            out.append(
                Token(
                    text=text,
                    language=language,  # type: ignore[arg-type]
                    semantic_class=SemanticClass.OTHER,
                    lid_confidence=confidence,
                    start_ms=start,
                    end_ms=end,
                )
            )
        return out

    def tag_corpus(
        self, transcripts: dict[str, str], *, speaker_ids: dict[str, str] | None = None
    ) -> dict[str, UtteranceTokens]:
        """Tag many utterances via the Batch API. USE THIS FOR CORPUS ANNOTATION.

        Half the cost of looping `tag_utterance`. Falls back to the
        synchronous path when no LLM tagger is configured.

        Note this submits and blocks until the batch ends; for a large corpus
        prefer driving `submit_batch` / `poll_batch` / `collect_batch`
        yourself so the process can be interrupted and resumed.
        """
        speaker_ids = speaker_ids or {}

        if self.llm_tagger is None:
            return {
                uid: self.tag_utterance(
                    text, utterance_id=uid, speaker_id=speaker_ids.get(uid)
                )
                for uid, text in transcripts.items()
            }

        tokenised = {uid: rules.simple_tokenise(t) for uid, t in transcripts.items()}
        tokenised = {uid: toks for uid, toks in tokenised.items() if toks}

        batch_id = self.llm_tagger.submit_batch(tokenised, contexts=transcripts)
        import time

        while self.llm_tagger.poll_batch(batch_id) != "ended":
            time.sleep(30)

        successes, failures = self.llm_tagger.collect_batch(batch_id, tokenised)
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(tokenised)} utterances failed to annotate: "
                f"{list(failures.items())[:5]}. Re-submit only the failures rather than "
                "the whole corpus."
            )

        out: dict[str, UtteranceTokens] = {}
        for uid, tags in successes.items():
            rule_results = rules.tag_tokens(tokenised[uid])
            self.stats.total_tokens += len(tokenised[uid])
            out[uid] = UtteranceTokens(
                utterance_id=uid,
                tokens=self._merge(tokenised[uid], rule_results, tags, None),
                speaker_id=speaker_ids.get(uid),
                transcript=transcripts[uid],
            )
        return out


__all__ = ["LIDPipeline", "PipelineStats"]
