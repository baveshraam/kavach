"""Hand-labelled word-level LID: export, load, score.

Every language and semantic-class label in the corpus comes from an LLM. That
makes the whole CSBG an estimate built on an unmeasured estimate, and "the
tagger is probably fine" is not a number a reviewer can check. Track 1 of the
proposal is exactly that number: word-level LID accuracy against human labels
on a subset.

The workflow is three commands and one human:

    python -m kavach.goldset export --manifest ... --out gold.tsv --utterances 20
    <a Tamil-English bilingual edits gold.tsv>
    python -m kavach.goldset score --manifest ... --gold gold.tsv

Two design decisions carry the validity of the resulting number.

**Labels are blank by default.** Prefilling the annotator's file with the
pipeline's own guesses roughly halves the labelling time and biases the result
in one direction: a corrector who sees "TA" agrees with it far more often than
a labeller starting from nothing, so the measured accuracy drifts toward 100%
whatever the tagger actually does. `--prefill` exists because there are honest
uses for it (adjudicating a second pass, sanity-checking the format), and every
file records which mode produced it so a prefilled gold set can never be
reported as a blind one.

**The scorer reports the transliteration slice separately.** Whisper writes some
English words in Tamil script, so a token can be Tamil-script and English-choice
at once. Aggregate accuracy hides this because those tokens are a small
fraction; they are not a small fraction of NUMBER and TIME_DATE, which are the
classes that separate speakers best. `GoldScore.transliterated` is that slice.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .csbg.ontology import Language, SemanticClass
from .lid import rules

#: Written into the header of an exported file and echoed in the score, so a
#: gold set built against one ontology cannot be silently scored against a
#: later one.
GOLDSET_FORMAT = "kavach-goldset-v1"

#: Column names. `lang` and `class` are what the human fills in; the rest are
#: context and must not be edited.
COLUMNS = ("utterance_id", "index", "token", "lang", "class", "note")

_LANGUAGE_HELP = "/".join(l.value for l in Language)
_BLANK = ""


class GoldsetError(ValueError):
    """A gold file that cannot be trusted. Always names the row."""


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GoldToken:
    """One hand-labelled token."""

    utterance_id: str
    index: int
    token: str
    language: Language | None = None
    semantic_class: SemanticClass | None = None
    note: str = ""

    @property
    def is_labelled(self) -> bool:
        """Both fields filled. A row with only one is not usable for either
        metric -- language accuracy conditioned on the rows where the annotator
        also got round to the class would be a different population."""
        return self.language is not None and self.semantic_class is not None


@dataclass(slots=True)
class Goldset:
    """A loaded gold file."""

    tokens: list[GoldToken] = field(default_factory=list)
    prefilled: bool = False
    """True if the file was exported with the pipeline's guesses in place.
    Carried into `GoldScore` so the caveat travels with the number."""

    source: str = ""

    @property
    def labelled(self) -> list[GoldToken]:
        return [t for t in self.tokens if t.is_labelled]

    @property
    def utterance_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.tokens:
            seen.setdefault(t.utterance_id, None)
        return list(seen)

    def by_utterance(self) -> dict[str, list[GoldToken]]:
        out: dict[str, list[GoldToken]] = {}
        for t in self.tokens:
            out.setdefault(t.utterance_id, []).append(t)
        for toks in out.values():
            toks.sort(key=lambda t: t.index)
        return out


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def sample_utterances(
    corpus: Any, *, n: int, seed: int = 0, per_speaker: bool = True
) -> list[Any]:
    """Pick utterances to label, spread across speakers.

    Spread deliberately: a random sample of 20 from a 4-speaker corpus can land
    12 on one speaker, and LID accuracy is a per-speaker property here -- one
    speaker's romanised Tamil habit is another's Tamil script. A number
    dominated by one voice is not a corpus number.
    """
    with_text = [u for u in corpus.utterances if (u.transcript or "").strip()]
    if not with_text:
        raise GoldsetError(
            "No utterance in this corpus has a transcript. Run "
            "`python -m kavach.annotate --stage asr` first -- there is nothing "
            "to label yet."
        )

    rng = random.Random(seed)
    if not per_speaker:
        rng.shuffle(with_text)
        return with_text[:n]

    by_speaker: dict[str, list[Any]] = {}
    for u in with_text:
        by_speaker.setdefault(u.speaker_id, []).append(u)
    for utterances in by_speaker.values():
        rng.shuffle(utterances)

    # Round-robin so the sample stays balanced at any n, including n smaller
    # than the speaker count.
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    out: list[Any] = []
    depth = 0
    while len(out) < n:
        added = False
        for sid in speakers:
            if depth < len(by_speaker[sid]) and len(out) < n:
                out.append(by_speaker[sid][depth])
                added = True
        if not added:
            break
        depth += 1
    return out


def export(
    corpus: Any,
    out_path: str | Path,
    *,
    n_utterances: int = 20,
    seed: int = 0,
    prefill: bool = False,
) -> Path:
    """Write a TSV for a human to label.

    TSV rather than CSV because Tamil answers contain commas far more often
    than tabs, and a quoted CSV is harder to edit in a spreadsheet without the
    editor helpfully reformatting it.

    Args:
        prefill: Fill `lang` and `class` with the pipeline's existing labels.
            Biases the result toward agreement -- see the module docstring. The
            file records that it was prefilled either way.
    """
    out_path = Path(out_path)
    chosen = sample_utterances(corpus, n=n_utterances, seed=seed)

    rows: list[list[str]] = []
    for utterance in chosen:
        surface = rules.simple_tokenise(utterance.transcript)
        existing = list(utterance.tokens or [])
        for i, text in enumerate(surface):
            lang = cls = _BLANK
            if prefill and i < len(existing) and existing[i].text == text:
                lang = existing[i].language.value
                cls = existing[i].semantic_class.value
            rows.append([utterance.utterance_id, str(i), text, lang, cls, ""])

    header = _header(corpus, chosen, prefill=prefill)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return out_path


def _header(corpus: Any, chosen: Sequence[Any], *, prefill: bool) -> str:
    """Comment block at the top of the file. It is the whole instruction sheet:
    a labeller should not need anything else open."""
    speakers = sorted({u.speaker_id for u in chosen})
    lines = [
        f"# {GOLDSET_FORMAT}",
        f"# prefilled: {'yes' if prefill else 'no'}",
        f"# corpus: {corpus.name} ({len(chosen)} utterances, "
        f"{len(speakers)} speakers)",
        "#",
        "# Fill in the `lang` and `class` columns. Leave a row blank if you are",
        "# not sure -- a blank row is dropped, a guessed row becomes a silent",
        "# error in the accuracy number.",
        "#",
        f"# lang:  {_LANGUAGE_HELP}",
        "#   TA / EN         the speaker chose that language for this word",
        "#   NEUTRAL         belongs to neither (numerals as digits, 'ok', 'mm')",
        "#   NAMED_ENTITY    a proper noun; saying it is not a language choice",
        "#",
        "#   Label what the SPEAKER said, not the script it is written in.",
        "#   Whisper writes English in Tamil script: 'மானிங்க் சிக்ஸ்' is the",
        "#   English 'morning six' and its lang is EN. This case is the point",
        "#   of the exercise -- do not normalise it away.",
        "#",
        "# class: " + ", ".join(c.value for c in SemanticClass),
        "#",
        "# Do not edit utterance_id, index or token. Use `note` freely.",
        "#",
    ]
    if prefill:
        lines += [
            "# WARNING: this file was prefilled with the system's own labels.",
            "# Agreement measured against it overstates accuracy. Any number",
            "# from it must be reported as adjudication, not as blind labelling.",
            "#",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------


def load(path: str | Path) -> Goldset:
    """Read a labelled file back, refusing anything ambiguous.

    Raises:
        GoldsetError: On an unknown label, a bad index, or a missing column.
            Every message names the line number -- a gold file is edited by
            hand in a spreadsheet, so "row 3 of the parsed data" is useless and
            "line 47" is actionable.
    """
    path = Path(path)
    prefilled = False
    tokens: list[GoldToken] = []

    with path.open(encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    body: list[str] = []
    for line in raw_lines:
        if line.startswith("#"):
            if "prefilled: yes" in line:
                prefilled = True
            continue
        body.append(line)

    if not body:
        raise GoldsetError(f"{path}: no rows, only comments.")

    reader = csv.reader(body, delimiter="\t")
    header = next(reader, None)
    if header is None or tuple(h.strip() for h in header[: len(COLUMNS)]) != COLUMNS:
        raise GoldsetError(
            f"{path}: expected the header row {COLUMNS}, found {header}. "
            "Re-export rather than repairing by hand -- a shifted column "
            "silently relabels every token."
        )

    # Line number of the header, so reported numbers match the editor's gutter.
    offset = len(raw_lines) - len(body) + 1

    for n, row in enumerate(reader, start=offset + 1):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) < 3:
            raise GoldsetError(f"{path}:{n}: expected {len(COLUMNS)} columns, got {len(row)}.")
        row = list(row) + [""] * (len(COLUMNS) - len(row))
        uid, index, text, lang, cls, note = (c.strip() for c in row[: len(COLUMNS)])

        try:
            idx = int(index)
        except ValueError:
            raise GoldsetError(f"{path}:{n}: index {index!r} is not a number.") from None

        tokens.append(
            GoldToken(
                utterance_id=uid,
                index=idx,
                token=text,
                language=_parse(Language, lang, path, n, "lang"),
                semantic_class=_parse(SemanticClass, cls, path, n, "class"),
                note=note,
            )
        )

    return Goldset(tokens=tokens, prefilled=prefilled, source=str(path))


def _parse(enum: Any, value: str, path: Path, line: int, column: str) -> Any:
    if not value:
        return None
    try:
        return enum(value.upper())
    except ValueError:
        valid = ", ".join(m.value for m in enum)
        raise GoldsetError(
            f"{path}:{line}: {column}={value!r} is not one of: {valid}"
        ) from None


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GoldScore:
    """Word-level LID and semantic-class accuracy against human labels.

    **This is the Track 1 number.** Report `language_accuracy` with `n_tokens`
    and `n_utterances` beside it; an accuracy over 300 tokens from 4 speakers is
    a pilot figure and saying so costs nothing.
    """

    n_tokens: int = 0
    n_utterances: int = 0
    n_speakers: int = 0
    n_unlabelled: int = 0
    """Rows the annotator left blank. Dropped, not counted as errors -- "I am
    not sure" is not a system failure -- but reported, because a gold set that
    is 40% blank is measuring the easy tokens."""

    language_correct: int = 0
    class_correct: int = 0
    both_correct: int = 0

    language_confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    """(gold, predicted) -> count. Sorted worst-first by `worst_confusions`."""

    class_confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    transliterated_total: int = 0
    """Tamil-script tokens whose gold language is EN: English the ASR
    transliterated. The size of the problem."""

    transliterated_correct: int = 0
    """...and how many of those the pipeline still called EN. The gap between
    these two is a one-directional bias that lands on NUMBER and TIME_DATE."""

    missing_utterances: list[str] = field(default_factory=list)
    misaligned: list[str] = field(default_factory=list)
    """Utterances whose gold tokens do not line up with the corpus tokens,
    usually because the transcript was re-run after the export. Excluded from
    every count rather than scored against the wrong words."""

    prefilled: bool = False
    goldset_source: str = ""

    @property
    def language_accuracy(self) -> float:
        return self.language_correct / self.n_tokens if self.n_tokens else 0.0

    @property
    def class_accuracy(self) -> float:
        return self.class_correct / self.n_tokens if self.n_tokens else 0.0

    @property
    def joint_accuracy(self) -> float:
        return self.both_correct / self.n_tokens if self.n_tokens else 0.0

    @property
    def transliteration_recall(self) -> float:
        """Of the transliterated English the gold set found, how much the
        pipeline recovered. 0.0 with a non-zero total means script evidence is
        winning every one of them."""
        if not self.transliterated_total:
            return 0.0
        return self.transliterated_correct / self.transliterated_total

    def worst_confusions(self, n: int = 5) -> list[tuple[str, str, int]]:
        return [(g, p, c) for (g, p), c in self.language_confusion.most_common(n)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": GOLDSET_FORMAT,
            "source": self.goldset_source,
            "prefilled": self.prefilled,
            "n_tokens": self.n_tokens,
            "n_utterances": self.n_utterances,
            "n_speakers": self.n_speakers,
            "n_unlabelled": self.n_unlabelled,
            "language_accuracy": round(self.language_accuracy, 4),
            "class_accuracy": round(self.class_accuracy, 4),
            "joint_accuracy": round(self.joint_accuracy, 4),
            "language_confusion": {f"{g}->{p}": c for (g, p), c in self.language_confusion.items()},
            "class_confusion": {f"{g}->{p}": c for (g, p), c in self.class_confusion.items()},
            "transliterated_total": self.transliterated_total,
            "transliterated_correct": self.transliterated_correct,
            "transliteration_recall": round(self.transliteration_recall, 4),
            "missing_utterances": self.missing_utterances,
            "misaligned": self.misaligned,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Word-level LID against human labels",
            "",
            f"- {self.n_tokens} labelled tokens from {self.n_utterances} utterances, "
            f"{self.n_speakers} speakers",
            f"- **language accuracy {self.language_accuracy:.1%}**",
            f"- semantic-class accuracy {self.class_accuracy:.1%}",
            f"- both correct {self.joint_accuracy:.1%}",
        ]
        if self.n_unlabelled:
            lines.append(
                f"- {self.n_unlabelled} rows left blank by the annotator and dropped"
            )
        if self.transliterated_total:
            lines += [
                "",
                "## Transliterated English",
                "",
                f"{self.transliterated_total} tokens are written in Tamil script but "
                f"were spoken in English. The pipeline recovered "
                f"{self.transliterated_correct} of them "
                f"({self.transliteration_recall:.1%}). Every miss is recorded as a "
                "Tamil choice the speaker did not make.",
            ]
        if self.language_confusion:
            lines += ["", "## Language confusions", "", "| gold | predicted | n |", "|---|---|---|"]
            lines += [f"| {g} | {p} | {c} |" for g, p, c in self.worst_confusions(8)]
        if self.misaligned:
            lines += [
                "",
                f"**{len(self.misaligned)} utterances excluded for misalignment.** "
                "The transcript changed after the gold set was exported; re-export "
                "and re-label those, or the labels belong to different words: "
                + ", ".join(self.misaligned[:5]),
            ]
        if self.missing_utterances:
            lines += [
                "",
                f"**{len(self.missing_utterances)} gold utterances are not in the "
                "corpus** and were skipped: " + ", ".join(self.missing_utterances[:5]),
            ]
        if self.prefilled:
            lines += [
                "",
                "> This gold set was **prefilled** with the system's own labels, so "
                "the annotator corrected rather than labelled. Agreement measured "
                "this way overstates accuracy and may not be reported as blind "
                "word-level LID accuracy.",
            ]
        return "\n".join(lines) + "\n"


def score(gold: Goldset, corpus: Any) -> GoldScore:
    """Compare a gold set against the corpus's current tokens.

    Scores what is *in the manifest*, not a fresh pipeline run, so the number
    describes the annotations the CSBG was actually built from.
    """
    by_id = {u.utterance_id: u for u in corpus.utterances}
    result = GoldScore(prefilled=gold.prefilled, goldset_source=gold.source)

    speakers: set[str] = set()
    scored_utterances = 0

    for uid, rows in gold.by_utterance().items():
        utterance = by_id.get(uid)
        if utterance is None:
            result.missing_utterances.append(uid)
            continue

        predicted = list(utterance.tokens or [])
        if not predicted:
            result.missing_utterances.append(uid)
            continue

        # Alignment check before anything is counted. Scoring token 5's gold
        # label against token 5 of a re-transcribed utterance compares two
        # different words and produces a number that looks fine.
        if any(
            r.index >= len(predicted) or predicted[r.index].text != r.token
            for r in rows
        ):
            result.misaligned.append(uid)
            continue

        labelled = [r for r in rows if r.is_labelled]
        result.n_unlabelled += len(rows) - len(labelled)
        if not labelled:
            continue

        scored_utterances += 1
        speakers.add(utterance.speaker_id)

        for row in labelled:
            token = predicted[row.index]
            result.n_tokens += 1

            lang_ok = token.language is row.language
            class_ok = token.semantic_class is row.semantic_class
            result.language_correct += lang_ok
            result.class_correct += class_ok
            result.both_correct += lang_ok and class_ok

            if not lang_ok:
                result.language_confusion[
                    (row.language.value, token.language.value)  # type: ignore[union-attr]
                ] += 1
            if not class_ok:
                result.class_confusion[
                    (row.semantic_class.value, token.semantic_class.value)  # type: ignore[union-attr]
                ] += 1

            if row.language is Language.EN and rules.script_of(row.token) == "tamil":
                result.transliterated_total += 1
                result.transliterated_correct += lang_ok

    result.n_utterances = scored_utterances
    result.n_speakers = len(speakers)
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.goldset",
        description="Hand-labelled word-level LID: the Track 1 number.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="write a TSV for a human to label")
    exp.add_argument("--manifest", required=True, type=Path)
    exp.add_argument("--out", required=True, type=Path)
    exp.add_argument("--utterances", type=int, default=20,
                     help="how many to sample, spread across speakers")
    exp.add_argument("--seed", type=int, default=0)
    exp.add_argument(
        "--prefill", action="store_true",
        help="fill in the system's own labels. Biases the result toward "
             "agreement; the file and the score both record that it was used",
    )

    sc = sub.add_parser("score", help="score a labelled file against the manifest")
    sc.add_argument("--manifest", required=True, type=Path)
    sc.add_argument("--gold", required=True, type=Path)
    sc.add_argument("--report", type=Path, help="write the markdown report here")
    sc.add_argument("--json", type=Path, help="write the numbers as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .corpus import load_manifest

    corpus = load_manifest(args.manifest)

    if args.command == "export":
        try:
            path = export(
                corpus, args.out, n_utterances=args.utterances,
                seed=args.seed, prefill=args.prefill,
            )
        except GoldsetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        n_rows = sum(1 for _ in path.read_text(encoding="utf-8").splitlines()) - 1
        print(f"wrote {path} ({n_rows} rows to label)")
        if args.prefill:
            print(
                "warning: prefilled with the system's own labels. The result is "
                "adjudication, not blind labelling.",
                file=sys.stderr,
            )
        return 0

    try:
        gold = load(args.gold)
    except GoldsetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = score(gold, corpus)
    if result.n_tokens == 0:
        print(
            "error: no gold token could be scored. "
            + (
                f"{len(result.misaligned)} utterances were misaligned"
                if result.misaligned
                else "every row was left blank or names an utterance not in "
                "this manifest"
            )
            + ".",
            file=sys.stderr,
        )
        return 2

    print(result.to_markdown())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(result.to_markdown(), encoding="utf-8")
    if args.json:
        import json

        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLUMNS",
    "GOLDSET_FORMAT",
    "GoldScore",
    "GoldToken",
    "Goldset",
    "GoldsetError",
    "build_parser",
    "export",
    "load",
    "main",
    "sample_utterances",
    "score",
]
