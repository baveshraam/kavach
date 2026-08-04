"""Cross-lingual answer verification.

Checks whether a spoken answer matches the fact stored in the SKG. The naive
approach -- exact string equality against the ASR transcript -- fails
constantly here, for reasons that are structural rather than fixable:

* The speaker may answer in Tamil script, romanised Tamil, or English:
  "தஞ்சாவூர்" / "Thanjavur" / "Thanjavoor" are the same answer.
* Tamil ASR misspells proper nouns routinely.
* Answers are conversational ("naan Thanjavur-la irundhu varen"), not bare
  entities.

Four independent matchers, combined by taking the maximum. Any single one
succeeding is sufficient evidence the speaker knew the answer -- requiring
agreement would compound their individual false-rejection rates.

    exact      Normalised string equality. The baseline to beat.
    entity     Expected entity appears anywhere in the answer.
    phonetic   Transliterate both to a common phonetic space, then edit
               distance. Handles cross-script and ASR spelling errors.
    semantic   Multilingual sentence embeddings. Handles translation and
               paraphrase.

**The exact-match failure rate is a publishable result** (proposal 5.4/§4.4).
`compare_matchers` produces exactly that ablation, so run it and report it
rather than only reporting the fused matcher.

Heavy dependencies (sentence-transformers, indic-transliteration) are
optional; each matcher degrades independently and reports whether it ran.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

#: Tamil-script vowel/consonant signs stripped when building a coarse
#: phonetic key. Vowel-length and gemination distinctions survive romanisation
#: inconsistently ("Thanjavur"/"Thanjavoor"), so folding them prevents a
#: spurious mismatch on what is the same spoken answer.
_TAMIL_DIACRITICS = re.compile(r"[ா-்ௗ]")

#: Latin digraphs that romanise the same Tamil sound. Order matters --
#: longer sequences first, or "th" would be consumed by the "t" rule.
_ROMAN_FOLDINGS: tuple[tuple[str, str], ...] = (
    ("zh", "l"),    # ழ often written zh or l
    ("th", "t"),
    ("dh", "d"),
    ("ch", "c"),
    ("sh", "s"),
    ("ph", "p"),
    ("bh", "b"),
    ("gh", "g"),
    ("kh", "k"),
    ("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"),
    ("ai", "a"), ("au", "o"), ("ay", "e"),
    ("v", "w"), ("j", "s"), ("z", "s"),
)


@dataclass(slots=True)
class MatchResult:
    """Outcome of matching one answer against one expected value."""

    score: float
    """Fused score in [0, 1]."""

    method: str
    """Which matcher produced `score`."""

    exact_score: float = 0.0
    entity_score: float = 0.0
    phonetic_score: float = 0.0
    semantic_score: float = 0.0

    available: dict[str, bool] = field(default_factory=dict)
    """Which matchers actually ran. A matcher that could not run scores 0.0,
    and without this flag that is indistinguishable from a confident
    mismatch -- which would silently look like a security failure."""

    normalised_answer: str = ""
    normalised_expected: str = ""

    def passed(self, threshold: float) -> bool:
        return self.score >= threshold

    def explain(self) -> str:
        parts = [
            f"exact={self.exact_score:.2f}",
            f"entity={self.entity_score:.2f}",
            f"phonetic={self.phonetic_score:.2f}",
        ]
        parts.append(
            f"semantic={self.semantic_score:.2f}"
            if self.available.get("semantic")
            else "semantic=unavailable"
        )
        return f"{self.score:.2f} via {self.method} ({', '.join(parts)})"


def normalise(text: str) -> str:
    """Lowercase, strip punctuation and diacritics, collapse whitespace.

    NFC-normalised first so Tamil combining marks compare consistently --
    without it, two visually identical strings can differ byte-wise.
    """
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\s஀-௿]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def phonetic_key(text: str) -> str:
    """Reduce text to a coarse phonetic key for cross-script comparison.

    Tries `indic-transliteration` (ISO-15919) so Tamil script and romanised
    Tamil land in the same space. Without it, falls back to folding the Latin
    digraphs above -- weaker, and cross-script matching degrades to the
    entity matcher.
    """
    text = normalise(text)
    if not text:
        return ""

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        if re.search(r"[஀-௿]", text):
            text = transliterate(text, sanscript.TAMIL, sanscript.ISO)
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))
    except ImportError:
        text = _TAMIL_DIACRITICS.sub("", text)

    text = text.lower()
    for src, dst in _ROMAN_FOLDINGS:
        text = text.replace(src, dst)
    text = re.sub(r"(.)\1+", r"\1", text)  # collapse doubled letters
    return re.sub(r"[^a-z]", "", text)


def levenshtein(a: str, b: str) -> int:
    """Edit distance. Iterative two-row form -- no recursion, no dependency."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def similarity_ratio(a: str, b: str) -> float:
    """1 - normalised edit distance, in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def exact_match(answer: str, expected: str) -> float:
    """Normalised string equality. THE BASELINE -- report its failure rate."""
    return 1.0 if normalise(answer) == normalise(expected) else 0.0


def entity_match(answer: str, expected: str) -> float:
    """Whether the expected entity appears within the answer.

    Handles the common case of a conversational reply that contains the
    entity: "naan Thanjavur-la irundhu varen" for expected "Thanjavur".

    Multi-word expected values score by the fraction of their words present,
    so a partial recall degrades smoothly instead of dropping to zero.
    """
    a, e = normalise(answer), normalise(expected)
    if not a or not e:
        return 0.0
    if e in a:
        return 1.0

    words = e.split()
    if len(words) > 1:
        present = sum(1 for w in words if w in a)
        return present / len(words)

    # Single expected word: allow a fuzzy hit against any answer token, so
    # ASR misspellings still count.
    return max((similarity_ratio(e, tok) for tok in a.split()), default=0.0) if len(e) > 3 else 0.0


def phonetic_match(answer: str, expected: str) -> float:
    """Compare phonetic keys. Handles cross-script and ASR spelling errors.

    Scores the best-matching window of the answer rather than the whole
    string, so a short expected entity is not penalised for appearing inside
    a long conversational reply.
    """
    a_key, e_key = phonetic_key(answer), phonetic_key(expected)
    if not a_key or not e_key:
        return 0.0

    whole = similarity_ratio(a_key, e_key)
    if len(a_key) <= len(e_key):
        return whole

    window = len(e_key)
    best = max(
        similarity_ratio(a_key[i : i + window], e_key)
        for i in range(0, len(a_key) - window + 1)
    )
    return max(whole, best)


class SemanticMatcher:
    """Multilingual sentence-embedding matcher.

    Handles the case the other three cannot: the speaker answers in a
    different language from the stored fact ("சென்னை" vs "Chennai"), or
    paraphrases it. Model loads lazily; unavailability is reported, never
    silently scored as a mismatch.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._failed = False

    @property
    def available(self) -> bool:
        if self._failed:
            return False
        if self._model is not None:
            return True
        try:
            self._load()
            return True
        except Exception:
            self._failed = True
            return False

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def similarity(self, answer: str, expected: str) -> float:
        """Cosine similarity, rescaled to [0, 1].

        Returns 0.0 when the model is unavailable -- callers must consult
        `available` to distinguish that from a genuine mismatch.
        """
        if not self.available:
            return 0.0
        import numpy as np

        vecs = self._model.encode([answer, expected], normalize_embeddings=True)
        cos = float(np.dot(vecs[0], vecs[1]))
        # LaBSE cosines for unrelated multilingual text sit well above 0, so a
        # raw cosine would make everything look like a partial match.
        # Rescaling from 0.5 puts the useful range across [0, 1].
        return max(0.0, min(1.0, (cos - 0.5) / 0.5))


@dataclass(slots=True)
class AnswerMatcher:
    """Fuses the four matchers.

    Weights scale each matcher's raw score by how much a hit from it should
    be trusted. Exact and entity matches are near-conclusive; phonetic and
    semantic are strong but noisier, so a marginal hit from them does not
    clear the threshold alone.
    """

    semantic_matcher: SemanticMatcher | None = None
    exact_weight: float = 1.0
    entity_weight: float = 1.0
    phonetic_weight: float = 0.95
    semantic_weight: float = 0.90

    def match(self, answer: str, expected: str) -> MatchResult:
        """Score a spoken answer against the expected SKG value."""
        exact = exact_match(answer, expected)
        entity = entity_match(answer, expected)
        phonetic = phonetic_match(answer, expected)

        semantic = 0.0
        semantic_available = False
        if self.semantic_matcher is not None and self.semantic_matcher.available:
            semantic = self.semantic_matcher.similarity(answer, expected)
            semantic_available = True

        weighted = {
            "exact": exact * self.exact_weight,
            "entity": entity * self.entity_weight,
            "phonetic": phonetic * self.phonetic_weight,
            "semantic": semantic * self.semantic_weight,
        }
        method, score = max(weighted.items(), key=lambda kv: kv[1])

        return MatchResult(
            score=min(1.0, score),
            method=method,
            exact_score=exact,
            entity_score=entity,
            phonetic_score=phonetic,
            semantic_score=semantic,
            available={
                "exact": True,
                "entity": True,
                "phonetic": True,
                "semantic": semantic_available,
            },
            normalised_answer=normalise(answer),
            normalised_expected=normalise(expected),
        )


def compare_matchers(pairs: list[tuple[str, str, bool]]) -> dict[str, dict[str, float]]:
    """Ablate the matchers. PRODUCES A PAPER TABLE.

    Quantifies how badly exact match fails on code-mixed answers, which is a
    reportable result in its own right for a low-resource venue.

    Args:
        pairs: (spoken_answer, expected_value, should_match) triples. Build
            these from real login recordings, including genuine answers that
            an exact matcher rejects and impostor answers it must keep
            rejecting.

    Returns:
        {matcher_name: {accuracy, false_reject_rate, false_accept_rate}}.
    """
    matcher = AnswerMatcher(semantic_matcher=SemanticMatcher())
    fns = {
        "exact": lambda a, e: exact_match(a, e),
        "entity": lambda a, e: entity_match(a, e),
        "phonetic": lambda a, e: phonetic_match(a, e),
        "fused": lambda a, e: matcher.match(a, e).score,
    }

    out: dict[str, dict[str, float]] = {}
    for name, fn in fns.items():
        tp = fp = tn = fn_ = 0
        for answer, expected, should in pairs:
            accepted = fn(answer, expected) >= 0.7
            if should and accepted:
                tp += 1
            elif should and not accepted:
                fn_ += 1
            elif not should and accepted:
                fp += 1
            else:
                tn += 1
        total = tp + fp + tn + fn_
        out[name] = {
            "accuracy": (tp + tn) / total if total else 0.0,
            "false_reject_rate": fn_ / (tp + fn_) if (tp + fn_) else 0.0,
            "false_accept_rate": fp / (fp + tn) if (fp + tn) else 0.0,
            "n": float(total),
        }
    return out


__all__ = [
    "MatchResult",
    "AnswerMatcher",
    "SemanticMatcher",
    "normalise",
    "phonetic_key",
    "levenshtein",
    "similarity_ratio",
    "exact_match",
    "entity_match",
    "phonetic_match",
    "compare_matchers",
]
