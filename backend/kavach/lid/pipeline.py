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


#: How confident the LLM must be before it may overrule Tamil script.
#:
#: Script is strong evidence and stays the default, but it is evidence about
#: *what the ASR wrote*, not about what the speaker said. Whisper transliterates
#: English into Tamil script on some utterances -- see
#: `PipelineStats.transliteration_recovered` -- and on those tokens the script
#: is confidently wrong. A high-confidence EN call on a Tamil-script token is
#: essentially only produced by transliteration, so it is allowed to win.
LLM_OVERRIDE_CONFIDENCE = 0.85


@dataclass(slots=True)
class PipelineStats:
    """Where tags came from. Report these -- they justify the design."""

    total_tokens: int = 0
    resolved_by_rules: int = 0
    resolved_by_llm: int = 0
    fallback_guesses: int = 0
    """Tokens tagged without either stage resolving them. Non-zero means the
    output is not corpus-grade."""
    transliteration_recovered: int = 0
    """Tamil-script tokens the LLM confidently called English, i.e. English the
    ASR transliterated. **Report this number.** It is the size of a bias that
    would otherwise be silent and one-directional: every one of these tokens
    would have been recorded as a Tamil choice the speaker did not make, and
    they concentrate in NUMBER and TIME_DATE, two of the most discriminative
    CSBG classes."""

    rewritten_by_model: int = 0
    """Tokens whose text the model changed, and which were restored from the
    surface form.

    Not an error -- the correction is automatic -- but a non-zero count means
    the tagger is editing rather than labelling, which is worth knowing about a
    model before trusting its labels. On the first real corpus pass this was
    the tagger returning an empty string for a `U+FFFD` replacement character;
    `_check_alignment` compares counts, so nothing caught it."""

    @property
    def rule_resolution_rate(self) -> float:
        return self.resolved_by_rules / self.total_tokens if self.total_tokens else 0.0

    @property
    def is_corpus_grade(self) -> bool:
        """False if any token was guessed. Gate corpus exports on this."""
        return self.fallback_guesses == 0

    def summary(self) -> str:
        flag = "" if self.is_corpus_grade else f"  [!] {self.fallback_guesses} guessed"
        translit = (
            f" | translit {self.transliteration_recovered}"
            if self.transliteration_recovered
            else ""
        )
        rewritten = (
            f" | text restored {self.rewritten_by_model}"
            if self.rewritten_by_model
            else ""
        )
        return (
            f"{self.total_tokens} tokens | rules {self.resolved_by_rules} "
            f"({self.rule_resolution_rate:.1%}) | llm {self.resolved_by_llm}"
            f"{translit}{rewritten}{flag}"
        )


@dataclass(slots=True)
class LIDPipeline:
    """Orchestrates rule-based and LLM-based tagging."""

    llm_tagger: LLMTagger | None = None
    trust_rules_over_llm: bool = True
    """When True, a confident rule result (Tamil script) wins over the LLM.

    Script evidence is cheap and nearly always right, and the LLM is mainly
    there for the Latin-script tokens where script says nothing. But it is not
    *definitionally* right, which is what this used to claim: it reports the
    script the ASR chose, and Whisper transliterates English into Tamil script
    on some utterances. A confident LLM call still wins -- see
    `LLM_OVERRIDE_CONFIDENCE`."""

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
            # `surface` is authoritative for the token text -- the model
            # assigns labels, it does not get to rewrite the word. See
            # `to_tokens` and `PipelineStats.rewritten_by_model`.
            merged, rewritten = to_tokens(llm_tags, surface=surface, timings=timings)
            self.stats.rewritten_by_model += rewritten
            out: list[Token] = []
            for tok, rule in zip(merged, rule_results):
                disagree = (
                    self.trust_rules_over_llm
                    and rule.is_resolved
                    and rule.confidence >= 0.95
                    and rule.language is not tok.language
                )
                if disagree and tok.lid_confidence >= LLM_OVERRIDE_CONFIDENCE:
                    # The model is confident against confident script evidence.
                    # On this corpus that means the ASR transliterated an
                    # English word into Tamil script ("மானிங்க் சிக்ஸ்
                    # தெட்டி" for "morning six thirty"), so the script is
                    # describing Whisper's output rather than the speaker's
                    # choice. Take the model and count it.
                    self.stats.transliteration_recovered += 1
                    disagree = False
                if disagree:
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

        # A tagger without a batch API (Gemini, Groq, Ollama -- anything on the
        # OpenAI-compatible path) is driven one utterance at a time. The saving
        # the batch API buys is halved cost, and a provider whose whole appeal
        # is a free tier has nothing to halve.
        if self.llm_tagger is not None and not getattr(self.llm_tagger, "supports_batch", True):
            return {
                uid: self.tag_utterance(
                    text, utterance_id=uid, speaker_id=speaker_ids.get(uid)
                )
                for uid, text in transcripts.items()
            }

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
