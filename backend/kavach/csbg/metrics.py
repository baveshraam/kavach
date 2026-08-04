"""Corpus-level code-mixing metrics.

These are *established, published* measures -- not KAVACH contributions. They
are reproduced here because they become graph-level attributes of the CSBG
and because reviewers will check the formulas against the source papers.

Each function documents its source. Where a paper leaves an edge case
undefined (empty input, single language present) the choice made here is
stated explicitly, because those cases occur constantly on short
authentication utterances of 5-15 tokens and silently returning 0.0 vs
NaN changes downstream scores.

References:
    Gambäck & Das (2014), "On Measuring the Complexity of Code-Mixing", ICON.
    Barnett et al. (2000), "The LIDES Coding Manual" -- M-index.
    Guzmán et al. (2017), "Metrics for Modeling Code-Switching Across
        Corpora", Interspeech -- I-index and burstiness.

    VERIFY these citations against the originals before submission. They are
    reproduced from secondary description and the exact venue/year strings
    should be confirmed.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .ontology import CHOICE_LANGUAGES, Language
from .tokens import UtteranceTokens, count_switch_points


@dataclass(frozen=True, slots=True)
class CodeMixingMetrics:
    """Graph-level code-mixing attributes of a speaker or an utterance."""

    cmi: float
    """Code-Mixing Index, 0-100. 0 = monolingual, higher = more balanced mixing."""

    m_index: float
    """Multilingual index, 0-1. 0 = monolingual, 1 = perfectly balanced."""

    i_index: float
    """Integration index, 0-1. Fraction of token boundaries that are switches."""

    burstiness: float
    """-1 (perfectly periodic switching) to +1 (very bursty). 0 = Poisson-like."""

    n_tokens: int
    """Total tokens including language-independent ones."""

    n_choice_tokens: int
    """Tokens representing an actual TA/EN choice."""

    n_switches: int
    """Total switch points, counted within utterances only."""

    ta_fraction: float
    """Share of choice tokens that are Tamil, 0-1."""

    @property
    def is_reliable(self) -> bool:
        """Whether there is enough data for these numbers to mean anything.

        Below ~20 choice tokens these metrics are dominated by sampling
        noise. Short authentication responses routinely fall under this, which
        is exactly why the CSBG scorer weights the metric term low by default
        (see scoring.ScoringWeights).
        """
        return self.n_choice_tokens >= 20


def compute_cmi(utterances: list[UtteranceTokens]) -> float:
    """Code-Mixing Index (Gambäck & Das 2014), aggregated over utterances.

        CMI = 100 * (1 - max(w_i) / (n - u))    when n > u
        CMI = 0                                  when n == u

    where n = total tokens, u = language-independent tokens, and max(w_i) is
    the token count of the most frequent language.

    Aggregation: computed over the pooled token counts of all utterances
    rather than as a mean of per-utterance CMIs. Per-utterance averaging
    over-weights short utterances, and a 3-token utterance has an essentially
    meaningless CMI.

    Returns:
        0.0 for empty input or when no language-choice tokens exist -- a
        speaker who said nothing has not mixed.
    """
    counts: Counter[Language] = Counter()
    for utt in utterances:
        for tok in utt.choice_tokens:
            counts[tok.language] += 1

    n_choice = sum(counts.values())
    if n_choice == 0:
        return 0.0

    max_lang = max(counts.values())
    return 100.0 * (1.0 - max_lang / n_choice)


def compute_m_index(utterances: list[UtteranceTokens]) -> float:
    """Multilingual index (Barnett et al. 2000).

        M = (1 - sum(p_i^2)) / ((k - 1) * sum(p_i^2))

    with p_i the proportion of language i and k the number of languages
    (2 here). Normalised so 0 = monolingual and 1 = perfectly balanced.

    Returns:
        0.0 when fewer than two languages are present.
    """
    counts: Counter[Language] = Counter()
    for utt in utterances:
        for tok in utt.choice_tokens:
            counts[tok.language] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    k = len(CHOICE_LANGUAGES)
    sum_sq = sum((counts.get(lang, 0) / total) ** 2 for lang in CHOICE_LANGUAGES)

    # sum_sq == 1.0 exactly when one language holds every token; the formula
    # then gives 0/0. The limit is 0 (no mixing), which we return directly.
    if math.isclose(sum_sq, 1.0) or sum_sq <= 0.0:
        return 0.0

    return (1.0 - sum_sq) / ((k - 1) * sum_sq)


def compute_i_index(utterances: list[UtteranceTokens]) -> float:
    """Integration index (Guzmán et al. 2017): switch points per boundary.

        I = (number of switch points) / (number of token boundaries)

    Boundaries are counted *within* utterances only. Pooling across
    utterances would fabricate a switch at every join.

    Returns:
        0.0 when there are no boundaries (every utterance has <2 tokens).
    """
    switches = 0
    boundaries = 0
    for utt in utterances:
        choice = utt.choice_tokens
        if len(choice) < 2:
            continue
        switches += count_switch_points(choice)
        boundaries += len(choice) - 1

    if boundaries == 0:
        return 0.0
    return switches / boundaries


def compute_burstiness(utterances: list[UtteranceTokens]) -> float:
    """Burstiness of language spans (Guzmán et al. 2017).

        B = (sigma_tau / mu_tau - 1) / (sigma_tau / mu_tau + 1)

    where tau are the lengths of consecutive same-language spans. Ranges from
    -1 (perfectly regular alternation) through 0 (Poisson-like) to +1 (highly
    bursty -- long monolingual stretches broken by rapid switching).

    This distinguishes two speakers with identical CMI but different
    *rhythm*: one alternating steadily, another speaking Tamil for a minute
    then English for a minute. That difference is exactly the kind of
    idiosyncrasy KAVACH is looking for.

    Returns:
        0.0 when fewer than two spans exist or the mean span length is zero.
    """
    spans: list[int] = []
    for utt in utterances:
        choice = utt.choice_tokens
        if not choice:
            continue
        run = 1
        for prev, cur in zip(choice, choice[1:]):
            if cur.language is prev.language:
                run += 1
            else:
                spans.append(run)
                run = 1
        spans.append(run)

    if len(spans) < 2:
        return 0.0

    mean = sum(spans) / len(spans)
    if mean <= 0:
        return 0.0

    var = sum((s - mean) ** 2 for s in spans) / len(spans)
    std = math.sqrt(var)
    ratio = std / mean
    return (ratio - 1.0) / (ratio + 1.0)


def compute_all_metrics(utterances: list[UtteranceTokens]) -> CodeMixingMetrics:
    """Compute every code-mixing metric in one pass over the utterances."""
    counts: Counter[Language] = Counter()
    n_tokens = 0
    n_switches = 0
    for utt in utterances:
        n_tokens += len(utt)
        choice = utt.choice_tokens
        for tok in choice:
            counts[tok.language] += 1
        if len(choice) >= 2:
            n_switches += count_switch_points(choice)

    n_choice = sum(counts.values())
    ta_fraction = counts.get(Language.TA, 0) / n_choice if n_choice else 0.0

    return CodeMixingMetrics(
        cmi=compute_cmi(utterances),
        m_index=compute_m_index(utterances),
        i_index=compute_i_index(utterances),
        burstiness=compute_burstiness(utterances),
        n_tokens=n_tokens,
        n_choice_tokens=n_choice,
        n_switches=n_switches,
        ta_fraction=ta_fraction,
    )
