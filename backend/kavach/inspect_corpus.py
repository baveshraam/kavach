"""Read a corpus with your eyes.

Every other tool here reduces the corpus to a number. This one does the
opposite: it puts the transcripts, the tags and the per-speaker code-mixing
profile on screen so a human can tell whether the numbers are describing
anything real.

That check cannot be automated away. The pipeline will happily produce an EER
from transcripts that are fluent nonsense, tags that are all `OTHER`, or four
speakers who code-switch identically -- and only the last of those is a result.

Three views, each answering one question:

    --summary   (default) Do these speakers differ at all? Per-speaker CMI,
                Tamil share and class coverage. This is the go/no-go read.
    --text      Did the ASR hear them? Transcripts, grouped by speaker.
    --tags      Did the tagger label them? Token tables with language and class.
    --translit  How much English did the ASR write in Tamil script? The
                tokens behind `transliteration_recovered`, listed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .csbg.metrics import compute_all_metrics
from .csbg.ontology import Language, SemanticClass
from .csbg.tokens import UtteranceTokens
from .lid import rules

#: Class coverage below this many tokens is too thin for the class to carry
#: evidence about a speaker. Matches the spirit of `CoverageReport`'s own
#: threshold -- restated here rather than imported so this stays a read-only
#: view that cannot be broken by a change to the scoring config.
THIN_CLASS_TOKENS = 5


def _tokens_by_speaker(corpus: Any) -> dict[str, list[UtteranceTokens]]:
    out: dict[str, list[UtteranceTokens]] = {}
    for u in corpus.utterances:
        if not u.tokens:
            continue
        out.setdefault(u.speaker_id, []).append(
            UtteranceTokens(
                utterance_id=u.utterance_id,
                tokens=list(u.tokens),
                speaker_id=u.speaker_id,
                transcript=u.transcript,
            )
        )
    return out


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summary(corpus: Any) -> str:
    """Per-speaker code-mixing profile. The go/no-go read.

    If every speaker's Tamil share and class distribution look the same, the
    CSBG has nothing to separate them with, and no amount of downstream
    machinery will change that. Better to see it here than in an EER of 50%.
    """
    by_speaker = _tokens_by_speaker(corpus)
    lines = [
        f"# {corpus.name} ({corpus.provenance.value})",
        "",
        f"{len(corpus.speakers)} speakers, {len(corpus.sessions)} sessions, "
        f"{len(corpus.utterances)} utterances, "
        f"{sum(1 for u in corpus.utterances if u.tokens)} tagged.",
        "",
    ]

    untagged = [u.utterance_id for u in corpus.utterances if not u.tokens]
    if untagged:
        lines += [
            f"**{len(untagged)} utterances have no tokens** and are absent from "
            f"everything below: {', '.join(untagged[:5])}"
            + (" ..." if len(untagged) > 5 else ""),
            "",
        ]

    if not by_speaker:
        return "\n".join(lines + ["Nothing is tagged. Run `--stage tag` first."])

    lines += [
        "## Code-mixing profile",
        "",
        "| speaker | utts | tokens | choice | TA share | CMI | I-index | burst | switches |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, utterances in sorted(by_speaker.items()):
        m = compute_all_metrics(utterances)
        flag = "" if m.is_reliable else " *"
        lines.append(
            f"| {sid} | {len(utterances)} | {m.n_tokens} | {m.n_choice_tokens}{flag} | "
            f"{m.ta_fraction:.2f} | {m.cmi:.1f} | {m.i_index:.3f} | "
            f"{m.burstiness:+.2f} | {m.n_switches} |"
        )
    lines += [
        "",
        "`*` = fewer than 20 choice tokens, so that row's metrics are sampling noise.",
        "",
    ]

    shares = [compute_all_metrics(u).ta_fraction for u in by_speaker.values()]
    if len(shares) > 1:
        spread = max(shares) - min(shares)
        lines += [
            f"Tamil share spans {spread:.2f} across speakers "
            f"({min(shares):.2f} to {max(shares):.2f}). "
            + (
                "That is a real difference to separate on."
                if spread >= 0.10
                else "**That is very little to separate on** -- if the class "
                "distributions below are also flat, the CSBG has no signal here "
                "and the honest read is that these speakers code-switch alike."
            ),
            "",
        ]

    lines += _class_table(by_speaker)
    return "\n".join(lines)


def _class_table(by_speaker: dict[str, list[UtteranceTokens]]) -> list[str]:
    """Which language each speaker picks per semantic class.

    This *is* the CSBG, printed. A class where every speaker picks the same
    language carries no evidence no matter how many tokens it has, and a class
    with three tokens carries none either -- both are visible here and neither
    is visible in a score.
    """
    per_speaker: dict[str, Counter] = {}
    totals: Counter = Counter()
    for sid, utterances in by_speaker.items():
        counts: Counter = Counter()
        for utt in utterances:
            for tok in utt.choice_tokens:
                counts[(tok.semantic_class, tok.language)] += 1
                totals[tok.semantic_class] += 1
        per_speaker[sid] = counts

    speakers = sorted(per_speaker)
    lines = [
        "## Language choice per class",
        "",
        "Tamil share of choice tokens, with the token count. `-` = no tokens.",
        "",
        "| class | n | " + " | ".join(speakers) + " |",
        "|---|---|" + "---|" * len(speakers),
    ]

    for cls in SemanticClass:
        if not totals[cls]:
            continue
        cells = []
        for sid in speakers:
            counts = per_speaker[sid]
            ta = counts[(cls, Language.TA)]
            en = counts[(cls, Language.EN)]
            n = ta + en
            if n == 0:
                cells.append("-")
            elif n < THIN_CLASS_TOKENS:
                cells.append(f"{ta / n:.2f} ({n})*")
            else:
                cells.append(f"{ta / n:.2f} ({n})")
        lines.append(f"| {cls.value} | {totals[cls]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        f"`*` = fewer than {THIN_CLASS_TOKENS} tokens for that speaker in that "
        "class; the share is one or two words, not a habit.",
        "",
    ]
    return lines


# --------------------------------------------------------------------------
# Transcripts and tags
# --------------------------------------------------------------------------


def transcripts(corpus: Any, *, speaker: str | None = None) -> str:
    """The transcripts, grouped by speaker. Read them."""
    lines = [f"# Transcripts -- {corpus.name}", ""]
    by_speaker: dict[str, list[Any]] = {}
    for u in corpus.utterances:
        if speaker and u.speaker_id != speaker:
            continue
        by_speaker.setdefault(u.speaker_id, []).append(u)

    for sid, utterances in sorted(by_speaker.items()):
        lines += [f"## {sid}", ""]
        for u in utterances:
            head = f"**{u.utterance_id}**"
            if u.prompt_id:
                head += f"  `{u.prompt_id}`"
            if u.duration_sec:
                head += f"  {u.duration_sec:.1f}s"
            lines.append(head)
            lines.append(u.transcript.strip() or "_(no transcript)_")
            if u.reference_transcript:
                lines.append(f"> script: {u.reference_transcript.strip()}")
            lines.append("")
    return "\n".join(lines)


def tags(corpus: Any, *, speaker: str | None = None, limit: int | None = None) -> str:
    """Token tables. The only view where a tagging failure is visible."""
    lines = [f"# Tags -- {corpus.name}", ""]
    shown = 0
    for u in corpus.utterances:
        if speaker and u.speaker_id != speaker:
            continue
        if not u.tokens:
            continue
        if limit is not None and shown >= limit:
            break
        shown += 1
        lines += [f"## {u.utterance_id}", "", f"> {u.transcript.strip()}", ""]
        lines += ["| token | lang | conf | class | script |", "|---|---|---|---|---|"]
        for tok in u.tokens:
            script = rules.script_of(tok.text)
            # The disagreement worth seeing: Tamil script, English choice.
            mark = " **<-**" if script == "tamil" and tok.language is Language.EN else ""
            lines.append(
                f"| {tok.text} | {tok.language.value} | {tok.lid_confidence:.2f} | "
                f"{tok.semantic_class.value} | {script}{mark} |"
            )
        lines.append("")
    if not shown:
        lines.append("_Nothing tagged matches that filter._")
    return "\n".join(lines)


def transliteration(corpus: Any) -> str:
    """Every Tamil-script token the tagger called English.

    These are the tokens `LIDPipeline` recovered from Whisper transliterating
    English into Tamil script. Listing them is how the recovery gets checked by
    a human rather than trusted: a wrong one here is an English label on a word
    the speaker really did say in Tamil, which is the same bias pointing the
    other way.
    """
    found: list[tuple[str, str, str]] = []
    for u in corpus.utterances:
        for tok in u.tokens or []:
            if tok.language is Language.EN and rules.script_of(tok.text) == "tamil":
                found.append((u.speaker_id, tok.text, tok.semantic_class.value))

    lines = [f"# Transliterated English -- {corpus.name}", ""]
    if not found:
        lines += [
            "None found. Either the ASR wrote no English in Tamil script, or the "
            "pipeline is not recovering it -- check a `--tags` dump for Tamil-script "
            "tokens labelled TA that you can read as English words.",
        ]
        return "\n".join(lines)

    lines += [
        f"{len(found)} tokens. Every one would otherwise have been recorded as a "
        "Tamil choice the speaker did not make.",
        "",
    ]

    by_class = Counter(cls for _, _, cls in found)
    lines += ["| class | n |", "|---|---|"]
    lines += [f"| {cls} | {n} |" for cls, n in by_class.most_common()]
    lines += ["", "| speaker | token | class |", "|---|---|---|"]
    lines += [f"| {sid} | {text} | {cls} |" for sid, text, cls in found]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.inspect_corpus",
        description="Read a corpus with your eyes, not through a metric.",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--speaker", default=None, help="Only this speaker id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="--tags only: stop after this many utterances.")
    parser.add_argument("--out", type=Path, help="Write to a file as well as stdout.")

    view = parser.add_mutually_exclusive_group()
    view.add_argument("--summary", action="store_true",
                      help="Per-speaker code-mixing profile (default).")
    view.add_argument("--text", action="store_true", help="Transcripts.")
    view.add_argument("--tags", action="store_true", help="Token tables.")
    view.add_argument("--translit", action="store_true",
                      help="Tamil-script tokens tagged English.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .corpus import load_manifest

    corpus = load_manifest(args.manifest)

    if args.text:
        out = transcripts(corpus, speaker=args.speaker)
    elif args.tags:
        out = tags(corpus, speaker=args.speaker, limit=args.limit)
    elif args.translit:
        out = transliteration(corpus)
    else:
        out = summary(corpus)

    print(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "THIN_CLASS_TOKENS",
    "build_parser",
    "main",
    "summary",
    "tags",
    "transcripts",
    "transliteration",
]
