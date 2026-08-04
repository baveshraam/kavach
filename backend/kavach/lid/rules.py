"""Rule-based word-level language identification.

The cheap first stage of the LID pipeline. Every token this resolves
confidently is one the LLM never has to see, which matters: annotating a
30-speaker corpus is ~25k tokens of adjudication, and script detection alone
resolves the large majority of them for free.

The three cases, in order of difficulty:

1.  **Tamil script** (U+0B80-U+0BFF) -> TA. Unambiguous.
2.  **Latin script** -> EN *or* romanised Tamil. Ambiguous, and this is the
    hard case: Whisper frequently romanises Tamil ("naan enna panren"), which
    a naive script check would tag as English and silently corrupt every
    downstream statistic. Resolved by lexicon lookup, then LLM.
3.  **Digits, punctuation, symbols** -> NEUTRAL. No language choice was made.

Nothing here decides semantic class -- that always needs the LLM.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..csbg.ontology import Language

#: Tamil block. Also covers Tamil digits and the Tamil Supplement is
#: deliberately excluded -- it is historical numerals, effectively unused in
#: speech transcripts.
_TAMIL_RANGE = (0x0B80, 0x0BFF)

#: Other Indic blocks. A Whisper transcript of Tamil speech occasionally emits
#: Devanagari or Malayalam for a borrowed word. Treated as TA rather than
#: NEUTRAL: the speaker made a non-English choice, which is what the CSBG
#: measures. Logged so it can be audited -- a spike here means the ASR is
#: misidentifying the language and the audio needs checking.
_OTHER_INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)

_LATIN_RE = re.compile(r"^[a-zA-Z][a-zA-Z''\-.]*$")
_DIGIT_RE = re.compile(r"^[\d௦-௯.,:/-]+$")
_PUNCT_ONLY_RE = re.compile(r"^[^\w஀-௿]+$")


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Outcome of rule-based tagging for one token."""

    language: Language | None
    """Resolved language, or None if the LLM must adjudicate."""

    confidence: float
    """0-1. Script-based results are 1.0; lexicon hits are lower."""

    reason: str
    """Which rule fired. Kept for the LID validation set -- when you
    hand-check 500 tokens for the paper, this tells you which stage to fix."""

    @property
    def is_resolved(self) -> bool:
        return self.language is not None


def _in_range(cp: int, lo: int, hi: int) -> bool:
    return lo <= cp <= hi


def script_of(token: str) -> str:
    """Dominant script of a token: 'tamil', 'latin', 'other_indic', or 'none'.

    Uses the majority of *letter* characters, so a token like "college-la"
    (Latin + Tamil suffix romanised) resolves by its bulk rather than by its
    first character.
    """
    tamil = latin = other_indic = 0
    for ch in token:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _in_range(cp, *_TAMIL_RANGE):
            tamil += 1
        elif any(_in_range(cp, lo, hi) for lo, hi in _OTHER_INDIC_RANGES):
            other_indic += 1
        elif "LATIN" in unicodedata.name(ch, ""):
            latin += 1

    if tamil == latin == other_indic == 0:
        return "none"
    if tamil >= latin and tamil >= other_indic:
        return "tamil"
    if other_indic >= latin:
        return "other_indic"
    return "latin"


#: Romanised Tamil function words and very high-frequency content words.
#: These are the tokens Whisper most often romanises, and they are frequent
#: enough that resolving them here measurably cuts LLM cost.
#:
#: NOT exhaustive and NOT a substitute for the LLM stage -- it is a
#: high-precision shortlist. Extend it from your own corpus's actual
#: romanisation patterns rather than from a dictionary; what matters is what
#: your ASR emits, not what is orthographically correct.
ROMANISED_TAMIL: frozenset[str] = frozenset(
    {
        # pronouns / copula
        "naan", "nan", "naa", "nee", "neenga", "ninga", "avan", "aval", "avanga",
        "adhu", "athu", "idhu", "ithu", "namma", "enakku", "unakku", "avanukku",
        "en", "un", "avan", "nama", "naanga", "neengal",
        # common verbs
        "irukku", "iruku", "irukken", "irukeen", "panren", "pannren", "panna",
        "pannu", "poren", "poi", "poga", "varen", "vandhen", "vanthen", "sollu",
        "sonnen", "solren", "theriyum", "theriyala", "vendam", "venum", "mudiyala",
        "paaru", "paathen", "kudu", "kuduthen", "vaanga", "vanga", "seri",
        # discourse / connectives
        "appuram", "apuram", "aana", "aanaa", "ippo", "ippodhu", "innum",
        "romba", "rombha", "konjam", "konjum", "sari", "seri", "ama", "aama",
        "illa", "illai", "enna", "yen", "yaen", "epdi", "eppadi", "enga", "engae",
        "adhaan", "athan", "dhan", "than", "mattum", "kooda", "vera",
        # kinship
        "amma", "appa", "anna", "akka", "thambi", "thangai", "thangachi",
        "mama", "mami", "chithi", "chithappa", "paati", "thatha",
        # everyday nouns
        "veedu", "veetla", "oor", "ooru", "saapadu", "sapadu", "thanni",
        "kadai", "kaasu", "panam", "neram", "naal", "varusham", "maasam",
        "vela", "velai", "padam", "paatu", "pasanga", "ponnu", "paiyan",
    }
)

#: Latin-script tokens that look like romanised Tamil but are English.
#: Checked before ROMANISED_TAMIL. Small by design -- collisions are rare, and
#: each entry should come from an observed misclassification, not speculation.
ENGLISH_HOMOGRAPHS: frozenset[str] = frozenset({"enna", "anna", "amma", "manna", "senna"})

#: Tamil case/postposition suffixes that attach to English stems -- the
#: signature of intra-word code-mixing ("college-la", "bus-ku", "phone-oda").
#: The stem is English, the morphology is Tamil.
#:
#: How to tag these is a genuine linguistic judgement, not an obvious call.
#: Under Myers-Scotton's Matrix Language Frame model, Tamil is the matrix
#: language supplying morphosyntax while the English stem is an embedded
#: insertion. We tag by the *stem* (EN), because the CSBG asks "which language
#: does this speaker choose for this concept?" and the concept lives in the
#: stem. Tagging by suffix would make almost every content word Tamil and
#: erase the signal entirely.
#:
#: State this choice explicitly in the paper -- a reviewer familiar with MLF
#: will ask, and "we tag by stem, here is why" is a much better answer than
#: silence.
#: Grouped by grammatical function, longest variant first within each group.
#: The dative in particular has three surface forms depending on the stem
#: ending (-ukku / -kku / -ku) and all three occur on English stems
#: ("college-ukku", "bus-ku") -- omitting any one silently drops those tokens
#: back to the LLM stage.
TAMIL_SUFFIXES: frozenset[str] = frozenset(
    {
        # dative "to/for"
        "ukku", "kku", "ku",
        # benefactive "for the sake of"
        "kaaga", "kaga", "kaka",
        # sociative / possessive "with, of"
        "oda", "ode", "odu", "kooda", "kuda",
        # locative "in, at"
        "la", "le", "la", "il", "ile", "ill",
        # allative / adessive "to, near"
        "kitta", "kitte", "gitta", "kittae",
        # instrumental / ablative "by, from"
        "aal", "aala", "irundhu", "irundu",
        # accusative
        "ai", "aye", "ya",
        # clitics: also / question / adverbial / adjectival
        "um", "vum", "aa", "ah", "aana", "aaga", "aay",
    }
)


def strip_tamil_suffix(token: str) -> tuple[str, str | None]:
    """Split a hyphenated Tamil suffix off an English stem.

    Returns (stem, suffix) where suffix is None if none was found. Only splits
    on an explicit hyphen ("college-la"); un-hyphenated agglutination
    ("collegela") is left to the LLM, since splitting it heuristically would
    mangle genuine English words ending in these letter sequences.
    """
    if "-" not in token:
        return token, None
    stem, _, tail = token.rpartition("-")
    if not stem:
        return token, None
    if tail.lower() in TAMIL_SUFFIXES:
        return stem, tail
    return token, None


def tag_token(token: str) -> RuleResult:
    """Apply the rule cascade to one token.

    Returns:
        A RuleResult. `language is None` means "send this to the LLM".
    """
    stripped = token.strip()
    if not stripped:
        return RuleResult(Language.NEUTRAL, 1.0, "empty")

    if _PUNCT_ONLY_RE.match(stripped):
        return RuleResult(Language.NEUTRAL, 1.0, "punctuation")

    if _DIGIT_RE.match(stripped):
        # A digit is language-neutral *as written*, but the speaker said it in
        # some language -- ASR just normalised it away. This is a real
        # measurement gap for the NUMBER class, which the challenge generator
        # specifically targets. Prefer an ASR configured to emit number words;
        # see asr.transcribe(suppress_numerals=True).
        return RuleResult(Language.NEUTRAL, 1.0, "numeric")

    script = script_of(stripped)

    if script == "tamil":
        return RuleResult(Language.TA, 1.0, "tamil_script")

    if script == "other_indic":
        return RuleResult(Language.TA, 0.6, "other_indic_script")

    if script == "none":
        return RuleResult(Language.NEUTRAL, 1.0, "no_letters")

    # Latin script: English, or romanised Tamil.
    lower = stripped.lower().strip(".,!?;:'\"")

    stem, suffix = strip_tamil_suffix(lower)
    if suffix is not None:
        # Tamil morphology on an English stem -> tag by stem (see TAMIL_SUFFIXES).
        if stem in ROMANISED_TAMIL:
            return RuleResult(Language.TA, 0.9, "romanised_tamil_with_suffix")
        return RuleResult(Language.EN, 0.75, "english_stem_tamil_suffix")

    if lower in ENGLISH_HOMOGRAPHS:
        return RuleResult(None, 0.0, "homograph_needs_llm")

    if lower in ROMANISED_TAMIL:
        return RuleResult(Language.TA, 0.9, "romanised_tamil_lexicon")

    # Unresolved Latin token. Most are English, but guessing here would bias
    # every romanised-Tamil token toward EN -- exactly the failure this module
    # exists to prevent. Defer.
    return RuleResult(None, 0.0, "latin_needs_llm")


def tag_tokens(tokens: list[str]) -> list[RuleResult]:
    """Apply `tag_token` across a token sequence."""
    return [tag_token(t) for t in tokens]


def resolution_rate(results: list[RuleResult]) -> float:
    """Fraction resolved without the LLM. Report this -- it is the cost driver."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.is_resolved) / len(results)


def simple_tokenise(text: str) -> list[str]:
    """Whitespace + punctuation tokenisation, keeping intra-word hyphens.

    Deliberately simple. Tamil is agglutinative and a morphological analyser
    would split differently, but the CSBG counts language choices per
    *orthographic word*, and the ASR emits orthographic words. Using a
    morphological analyser here would change what a "token" means without
    changing what is being measured.
    """
    return [t for t in re.findall(r"[\w஀-௿]+(?:[-'][\w஀-௿]+)*|[^\s\w]", text) if t.strip()]
