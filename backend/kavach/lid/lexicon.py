"""An offline tagger built from the corpus the LLM already tagged.

WHY THIS EXISTS
---------------
Semantic class needs the LLM, and the live system talks to a free-tier
provider. When that provider is down -- measured on 2026-09-29: repeated 503s
and timeouts from Gemini for over an hour -- every login lost its CSBG branch.
But the corpus is ~9,750 tokens the same LLM tagged in context, over the same
speakers and the same topics a login asks about. Those labels are data: a
word's majority (language, class) across the corpus is a reasonable tag for the
same word at login.

WHAT IT IS NOT
--------------
It is not the LLM. It knows no word the corpus did not contain, it cannot use
context (so a word tagged differently in different sentences gets its
majority label), and a Tamil word whose suffix differs from every corpus form
is matched on its longest known prefix, which is a heuristic. `coverage` on
each result says how many tokens it actually knew; the pipeline reports it
and treats a low-coverage answer as unmeasured rather than scoring it.

It is therefore a *fallback for the live demo path*, never an annotation tool:
nothing it tags may enter a corpus or a reported number.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..csbg.ontology import Language, SemanticClass
from . import rules
from .llm import TaggedToken

#: Shortest Tamil prefix a suffix-stripped match may use. Shorter prefixes
#: collide across unrelated words (most Tamil verbs share 2-3 letter onsets).
MIN_PREFIX = 4

#: Coverage below which a lexicon-tagged utterance should not be scored as a
#: CSBG: most tokens would be OTHER, and OTHER is excluded as low-signal anyway.
MIN_USEFUL_COVERAGE = 0.5

_PUNCT = re.compile(r"[^\w஀-௿]+", re.UNICODE)


def normalise(text: str) -> str:
    """Lower-case, NFC, punctuation stripped -- the lexicon key."""
    return _PUNCT.sub("", unicodedata.normalize("NFC", text).lower())


def _is_tamil(word: str) -> bool:
    return any("஀" <= ch <= "௿" for ch in word)


@dataclass(slots=True)
class LexiconTagger:
    """Majority-label lookup over corpus tokens. See the module docstring."""

    entries: dict[str, tuple[Language, SemanticClass, float]] = field(default_factory=dict)
    last_coverage: float = 0.0
    supports_batch: bool = False

    # ---------------------------------------------------------------- build

    @classmethod
    def from_manifests(cls, paths: list[Path]) -> "LexiconTagger":
        votes: dict[str, Counter] = defaultdict(Counter)
        for path in paths:
            if not path.exists():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for utt in manifest.get("utterances", []):
                # Excluded utterances are excluded for their tokens (Whisper
                # translated them), so they must not vote either.
                if utt.get("excluded_reason") or utt.get("annotation_source") != "LLM":
                    continue
                for tok in utt.get("tokens") or []:
                    key = normalise(tok.get("text", ""))
                    if key:
                        votes[key][(tok["language"], tok["semantic_class"])] += 1
        entries: dict[str, tuple[Language, SemanticClass, float]] = {}
        for key, counter in votes.items():
            (lang, cls_), n = counter.most_common(1)[0]
            try:
                entries[key] = (Language(lang), SemanticClass(cls_), n / sum(counter.values()))
            except ValueError:
                continue
        return cls(entries=entries)

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "LexiconTagger":
        return cls.from_manifests(sorted(data_dir.glob("corpus_v*/manifest.json")))

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ tag

    def _lookup(self, word: str) -> tuple[Language, SemanticClass, float] | None:
        key = normalise(word)
        if not key:
            return None
        hit = self.entries.get(key)
        if hit is not None:
            return hit
        if _is_tamil(key):
            # Tamil is agglutinative: "வீட்டுக்கு" and "வீட்டில்" share a stem the
            # corpus may only have seen in one form. Longest known prefix wins.
            for end in range(len(key) - 1, MIN_PREFIX - 1, -1):
                hit = self.entries.get(key[:end])
                if hit is not None:
                    lang, cls_, conf = hit
                    return lang, cls_, conf * 0.8
        return None

    def tag(self, tokens: list[str], *, context: str | None = None) -> list[TaggedToken]:
        """Same contract as `LLMTagger.tag`. Unknown tokens get their language
        from script rules and the class OTHER, at low confidence."""
        rule_results = rules.tag_tokens(tokens)
        out: list[TaggedToken] = []
        known = 0
        for text, rule in zip(tokens, rule_results):
            hit = self._lookup(text)
            if hit is not None:
                lang, cls_, conf = hit
                known += 1
                out.append(TaggedToken(text=text, language=lang, semantic_class=cls_, confidence=max(0.05, min(1.0, conf))))
            else:
                lang = rule.language if rule.is_resolved else Language.EN
                out.append(TaggedToken(text=text, language=lang, semantic_class=SemanticClass.OTHER, confidence=0.35))
        self.last_coverage = known / len(tokens) if tokens else 0.0
        return out


__all__ = ["LexiconTagger", "MIN_USEFUL_COVERAGE", "normalise"]
