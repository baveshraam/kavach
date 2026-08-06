"""Annotation: audio in a manifest becomes tagged tokens in the same manifest.

Two stages, and they fail differently, so they are separable on the command
line and recorded separately in the manifest.

  1. **ASR.** `asr.WhisperASR` over each utterance's audio. Slow, offline,
     deterministic given a checkpoint, and free. Writes `transcript`.
  2. **LID + semantic tagging.** `lid.pipeline` over the transcript. Fast,
     needs an API key for the part that matters, and costs money. Writes
     `tokens`.

WHY THE SPLIT IS NOT COSMETIC
-----------------------------
Stage 1 takes about an hour on this corpus and a crash in it is expensive.
Stage 2 takes seconds per utterance and will be re-run whenever the ontology
or the prompt changes. Coupling them means re-transcribing to re-tag.

WHAT STAGE 2 CANNOT DO WITHOUT A KEY
------------------------------------
`lid.rules` resolves *language* from script evidence and explicitly decides no
semantic class -- "that always needs the LLM", in its own words. So a
rules-only pass assigns `SemanticClass.OTHER` to every token, and a CSBG built
from it has one class containing everything. That is not a weak graph, it is
not a graph: every speaker's is identical and every LLR is zero.

So a rules-only run is worth doing -- it exercises the plumbing, produces the
transcripts, and tells you whether the audio is tractable -- and its output is
worth exactly nothing as a verification result. `PipelineStats.is_corpus_grade`
already refuses it and `Corpus.reportability()` repeats the refusal. This
module adds a third statement of it at the point where somebody would
otherwise read an EER off a rules-only corpus and believe it.

WORD ERROR RATE IS A RESULT, NOT DIAGNOSTICS
--------------------------------------------
For read speech the reference transcript is known, so ASR is scoreable. Tamil-
English code-mixed WER on phone recordings is a number this field wants and
does not have much of, it does not depend on the CSBG separating anybody, and
it is the Track-1 contribution standing on its own. It is computed here per
utterance because the interesting variation is per prompt.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .audio import load_audio
from .corpus import (
    AnnotationSource,
    Corpus,
    UtteranceRecord,
    load_manifest,
    save_manifest,
)
from .lid.pipeline import LIDPipeline

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


# --------------------------------------------------------------------------
# Word error rate
# --------------------------------------------------------------------------


def normalise_for_wer(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace.

    Deliberately *not* language-aware. A normaliser that transliterated
    romanised Tamil to Tamil script before comparing would be scoring its own
    transliterator, and the resulting WER would not be comparable to anything
    else reported on code-mixed speech.

    Unicode is NFC-normalised first so that a Tamil vowel sign written as a
    combining sequence compares equal to its composed form -- otherwise
    identical-looking words differ and the WER is inflated by an encoding
    detail.
    """
    text = unicodedata.normalize("NFC", text.lower())
    return _PUNCTUATION.sub(" ", text).split()


def is_latin(word: str) -> bool:
    """True if every letter in `word` is Latin script.

    The test is on letters only, so `six-forty-five` and `WhatsApp'la` count
    while `நான்` does not.
    """
    letters = [c for c in word if c.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(c, "") for c in letters)


def latin_fraction(text: str) -> float:
    """Share of tokens written in Latin script, 0 when there are none."""
    words = normalise_for_wer(text)
    return sum(is_latin(w) for w in words) / len(words) if words else 0.0


def transcripts_are_comparable(
    reference: str, hypothesis: str, *, tolerance: float = 0.5
) -> bool:
    """Whether a WER between these two would mean anything.

    It would not, for this corpus, and the reason is structural rather than
    fixable. The read-speech scripts romanise Tamil -- "Naan enga family kooda
    dhaan iruken" -- because participants have to read them aloud. Whisper
    writes Tamil in Tamil script. The reference is therefore ~100% Latin and
    the hypothesis ~20%, no Tamil word can ever align, and the WER is pinned
    near or above 100% however good the transcription is. Measured on the first
    three utterances here: 281%, 221%, 201%.

    Restricting the comparison to Latin tokens does not rescue it, which is
    worth stating because it is the obvious repair and it is wrong. Romanised
    Tamil *is* Latin, so the filter removes nothing from the reference and
    removes all the Tamil from the hypothesis, leaving the full script compared
    against a handful of English words. That scored 96% and looked like a
    result.

    Transliterating one side would make the number a measurement of the
    transliterator, and romanised Tamil has no standard orthography to
    transliterate to -- the same word is spelled a dozen ways by a dozen
    people, which is exactly why the scripts had to be written by hand.

    So this returns False, `asr_wer` stays None, and the report says why. The
    Track-1 word-level LID contribution needs **hand-corrected transcripts on a
    subset**, which is a human annotation task and the normal way this is done.
    That subset is cheap here precisely because the reference exists: a human
    is correcting a transcript, not producing one.
    """
    return abs(latin_fraction(reference) - latin_fraction(hypothesis)) <= tolerance


def word_error_rate(reference: str, hypothesis: str, *, latin_only: bool = False) -> float:
    """Levenshtein distance over words, divided by reference length.

    Args:
        reference: Ground truth.
        hypothesis: What the ASR heard.
        latin_only: Compare only Latin-script tokens on both sides. Valid when
            both sides use the same orthography for the same language; see
            `transcripts_are_comparable` for why that does not hold against
            the romanised reference scripts.

    Returns 0.0 when both are empty and 1.0 when only the reference is, which
    keeps the aggregate finite on a failed transcription instead of producing
    a division by zero in the middle of a corpus run.
    """
    ref, hyp = normalise_for_wer(reference), normalise_for_wer(hypothesis)
    if latin_only:
        ref = [w for w in ref if is_latin(w)]
        hyp = [w for w in hyp if is_latin(w)]
    if not ref:
        return 0.0 if not hyp else 1.0

    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                previous[j - 1] if r == h else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1] / len(ref)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class AnnotationReport:
    """What one annotation run did, and why its output may not be trusted."""

    transcribed: int = 0
    tagged: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)
    wer_by_prompt: dict[str, list[float]] = field(default_factory=dict)

    not_comparable: int = 0
    """Utterances with a reference that WER cannot be computed against,
    because the two sides do not share an orthography. See
    `transcripts_are_comparable` -- for a romanised script this is all of
    them, and saying so is more useful than a number that is not one."""
    total_tokens: int = 0
    guessed_tokens: int = 0
    resolved_by_rules: int = 0
    used_llm: bool = False
    tagger: str = ""
    """Which provider tagged. Recorded because tagging quality bounds every
    number downstream, so "annotated with Gemini Flash" and "annotated with
    Llama 3.3 on Groq" are different corpora and the paper has to say which."""

    tagger_model: str = ""

    transliteration_recovered: int = 0
    """Tamil-script tokens the model confidently called English, i.e. English
    the ASR transliterated. **Report this.** Every one of them would otherwise
    have been recorded as a Tamil choice the speaker did not make, and they
    concentrate in NUMBER and TIME_DATE -- two of the most discriminative CSBG
    classes. See `LIDPipeline.stats.transliteration_recovered`."""

    retries: int = 0
    """Transient API failures retried during the pass. Non-zero means the run
    was fighting a rate limit; the labels are the same but a bigger corpus will
    need pacing rather than luck."""

    degenerate: list[tuple[str, str, float]] = field(default_factory=list)
    """`(utterance_id, token, share)` for transcripts Whisper looped on.

    **These must be re-transcribed or dropped, not annotated.** A loop is a
    plausible word repeated, so every copy lands in the same semantic class and
    counts as a real language choice: sixty hallucinated `Rs` would make MONEY
    the most confident cell in that speaker's graph, built from a word nobody
    said. See `Transcript.repetition_loop`."""

    @property
    def is_corpus_grade(self) -> bool:
        """The same rule `PipelineStats.is_corpus_grade` applies, restated
        rather than re-derived: any guessed token disqualifies the pass."""
        return self.used_llm and self.guessed_tokens == 0

    @property
    def mean_wer(self) -> float | None:
        values = [w for ws in self.wer_by_prompt.values() for w in ws]
        return sum(values) / len(values) if values else None

    def to_markdown(self) -> str:
        lines = ["# Annotation report", ""]
        lines.append(
            f"{self.transcribed} transcribed, {self.tagged} tagged, "
            f"{self.skipped} skipped, {len(self.failures)} failed."
        )
        lines.append("")

        if self.tagger:
            lines += [f"Tagged with **{self.tagger}** / `{self.tagger_model}`.", ""]

        if self.transliteration_recovered:
            lines += [
                f"**{self.transliteration_recovered} tokens were English written in "
                "Tamil script** by the ASR, and were recovered as English rather than "
                "recorded as a Tamil choice the speaker did not make. They concentrate "
                "in NUMBER and TIME_DATE. Report this number: without the recovery it "
                "is a silent, one-directional bias on two of the most discriminative "
                "CSBG classes.",
                "",
            ]

        if self.retries:
            lines += [
                f"{self.retries} API calls were retried after a transient failure "
                "(rate limit or server error). The labels are unaffected, but the pass "
                "was fighting a rate limit -- a larger corpus needs pacing.",
                "",
            ]

        if self.degenerate:
            lines += [
                f"## {len(self.degenerate)} transcripts degenerated — do not annotate these",
                "",
                "Whisper looped on these recordings, emitting one fragment over and "
                "over. **Re-transcribe or drop them.** A loop is a plausible word "
                "repeated, so every copy lands in the same semantic class and is "
                "counted as a real language choice -- the speaker's graph ends up "
                "most confident about the word they never said.",
                "",
                "| utterance | token | share of transcript |",
                "|---|---|---|",
            ]
            lines += [
                f"| {uid} | `{token}` | {share:.1%} |"
                for uid, token, share in sorted(self.degenerate, key=lambda r: -r[2])
            ]
            lines.append("")

        if not self.used_llm and self.tagged:
            lines += [
                "> **Not corpus-grade.** No LLM tagger was configured, so every",
                "> token carries `SemanticClass.OTHER` and the CSBG has one class",
                "> containing everything. Every speaker's graph is identical and",
                "> every log-likelihood ratio is zero.",
                ">",
                "> Set any one of these and re-run `--stage tag`; the transcripts",
                "> do not need redoing:",
                ">",
                "> - `GEMINI_API_KEY` — free, https://aistudio.google.com/apikey",
                "> - `GROQ_API_KEY` — free, https://console.groq.com/keys",
                "> - `ANTHROPIC_API_KEY` — paid, adds prompt caching and the batch API",
                "",
            ]
        elif self.guessed_tokens:
            lines += [
                f"> **Not corpus-grade.** {self.guessed_tokens} of "
                f"{self.total_tokens} tokens were guessed by the LID fallback.",
                "",
            ]

        if self.not_comparable:
            lines += [
                "## Word error rate: not computed",
                "",
                f"{self.not_comparable} utterances have a reference script that "
                "WER cannot be measured against. The scripts romanise Tamil "
                "because participants read them aloud; Whisper writes Tamil in "
                "Tamil script. No Tamil word can align, so any WER between them "
                "describes an orthography mismatch rather than an ASR system, "
                "and restricting to Latin tokens does not help -- romanised "
                "Tamil is Latin.",
                "",
                "The Track-1 word-level LID contribution needs **hand-corrected "
                "transcripts on a subset**. That is cheap here precisely because "
                "the reference exists: a human corrects a transcript rather than "
                "producing one.",
                "",
            ]

        if self.wer_by_prompt:
            lines += ["## Word error rate", ""]
            lines += ["| prompt | n | WER |", "|---|---|---|"]
            for prompt_id in sorted(self.wer_by_prompt):
                values = self.wer_by_prompt[prompt_id]
                lines.append(
                    f"| {prompt_id} | {len(values)} | {sum(values) / len(values):.1%} |"
                )
            mean = self.mean_wer
            if mean is not None:
                lines += ["", f"Overall: **{mean:.1%}** over {len(self._all_wer())} utterances."]
            lines.append("")

        if self.total_tokens:
            lines += [
                "## Tagging",
                "",
                f"- {self.total_tokens} tokens",
                f"- {self.resolved_by_rules} resolved by script evidence "
                f"({self.resolved_by_rules / self.total_tokens:.1%})",
                f"- {self.guessed_tokens} guessed",
                "",
            ]

        if self.failures:
            lines += ["## Failures", ""] + [f"- {f}" for f in self.failures] + [""]
        return "\n".join(lines)

    def _all_wer(self) -> list[float]:
        return [w for ws in self.wer_by_prompt.values() for w in ws]


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def transcribe_corpus(
    corpus: Corpus,
    *,
    model_size: str = "large-v3",
    compute_type: str = "int8",
    device: str = "auto",
    language: str | None = None,
    force: bool = False,
    limit: int | None = None,
    report: AnnotationReport | None = None,
    on_progress=None,
    save_every: int = 1,
    manifest_path: Path | None = None,
) -> AnnotationReport:
    """Stage 1. Fill in `transcript`, and `asr_wer` where a reference exists.

    Saves after every `save_every` utterances when `manifest_path` is given.
    An hour of CPU transcription that is lost to a crash at utterance 50 is an
    hour nobody spends twice, so the default is to save every one.
    """
    from .asr import WhisperASR

    report = report or AnnotationReport()
    backend = WhisperASR(
        model_size=model_size,
        compute_type=compute_type,
        device=device,
        language=language,
    )
    root = corpus.root or Path()

    pending = [u for u in corpus.utterances if force or not u.transcript]
    if limit is not None:
        pending = pending[:limit]
    report.skipped += len(corpus.utterances) - len(pending)

    for i, utterance in enumerate(pending, start=1):
        if not utterance.audio_path:
            report.failures.append(f"{utterance.utterance_id}: no audio path")
            continue
        try:
            audio = load_audio(root / utterance.audio_path)
            transcript = backend.transcribe(audio)
        except Exception as exc:  # noqa: BLE001 - one bad file must not end the run
            report.failures.append(f"{utterance.utterance_id}: {type(exc).__name__}: {exc}")
            continue

        utterance.transcript = transcript.text
        loop = transcript.repetition_loop()
        if loop is not None:
            token, share = loop
            report.degenerate.append((utterance.utterance_id, token, share))
        reference = utterance.reference_transcript
        if reference:
            if transcripts_are_comparable(reference, transcript.text):
                wer = word_error_rate(reference, transcript.text)
                utterance.asr_wer = round(wer, 4)
                report.wer_by_prompt.setdefault(utterance.prompt_id, []).append(wer)
            else:
                utterance.asr_wer = None
                report.not_comparable += 1
        report.transcribed += 1

        if on_progress:
            on_progress(i, len(pending), utterance)
        if manifest_path and save_every and i % save_every == 0:
            save_manifest(corpus, manifest_path)

    if manifest_path:
        save_manifest(corpus, manifest_path)
    return report


def tag_corpus(
    corpus: Corpus,
    *,
    force: bool = False,
    resume: bool = False,
    limit: int | None = None,
    report: AnnotationReport | None = None,
    pipeline: LIDPipeline | None = None,
    manifest_path: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    on_progress=None,
) -> AnnotationReport:
    """Stage 2. Fill in `tokens`, `annotation_source` and `n_guessed_tokens`.

    Without an LLM tagger this assigns `SemanticClass.OTHER` to everything and
    the result is unusable as a CSBG -- see the module docstring. It still
    runs, because the alternative is that nobody discovers the transcripts are
    empty until after paying for a tagging batch.

    Args:
        force: Re-tag everything, including utterances already LLM-tagged.
        resume: Tag anything not already LLM-tagged. What to use after a run
            died partway, and what to use over rules-only tokens left by a
            smoke test. See `_needs_tagging`.
        on_progress: Called with (index, total, utterance) before each request.
            The ASR stage has had this since it was written; the tag stage
            needs it for the same reason. A 56-utterance pass against a free
            tier runs for minutes with a stall on every rate limit, and a
            silent process is one an operator kills.
    """
    report = report or AnnotationReport()
    pipeline = pipeline if pipeline is not None else _default_pipeline(provider, model)
    report.used_llm = pipeline.llm_tagger is not None
    if pipeline.llm_tagger is not None:
        report.tagger = getattr(
            getattr(pipeline.llm_tagger, "provider", None), "name", "anthropic"
        )
        report.tagger_model = getattr(pipeline.llm_tagger, "model", "")

    pending = [
        u
        for u in corpus.utterances
        if u.transcript and _needs_tagging(u, force=force, resume=resume)
    ]
    if limit is not None:
        pending = pending[:limit]

    def finish() -> AnnotationReport:
        report.total_tokens = pipeline.stats.total_tokens
        report.guessed_tokens = pipeline.stats.fallback_guesses
        report.resolved_by_rules = pipeline.stats.resolved_by_rules
        report.transliteration_recovered = pipeline.stats.transliteration_recovered
        report.retries = getattr(pipeline.llm_tagger, "retries", 0)
        if manifest_path:
            save_manifest(corpus, manifest_path)
        return report

    for i, utterance in enumerate(pending, start=1):
        if on_progress is not None:
            on_progress(i, len(pending), utterance)
        before = pipeline.stats.fallback_guesses
        try:
            tagged = pipeline.tag_utterance(
                utterance.transcript,
                utterance_id=utterance.utterance_id,
                speaker_id=utterance.speaker_id,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised after saving
            # Save what is already tagged before letting this propagate. A
            # corpus pass against a free tier can die on the fiftieth
            # utterance, and losing the first forty-nine turns a rate limit
            # into an hour of re-tagging -- against the same rate limit.
            finish()
            raise RuntimeError(
                f"Tagging failed on {utterance.utterance_id!r} after "
                f"{report.tagged} of {len(pending)} utterances: {exc}\n"
                + (
                    f"The {report.tagged} already tagged are saved to "
                    f"{manifest_path}. Re-run with `--stage tag --resume` to "
                    "continue from here rather than starting over."
                    if manifest_path
                    else "Nothing was saved: no manifest path was given."
                )
            ) from exc

        utterance.tokens = list(tagged.tokens)
        utterance.n_guessed_tokens = pipeline.stats.fallback_guesses - before
        utterance.annotation_source = (
            AnnotationSource.LLM if report.used_llm else AnnotationSource.RULES
        )
        report.tagged += 1

        # Periodic checkpoint. Every utterance would rewrite the whole manifest
        # 56 times for a pass this size; every tenth bounds the loss to nine
        # utterances without making the write the slow part.
        if manifest_path and i % 10 == 0:
            save_manifest(corpus, manifest_path)

    return finish()


def _needs_tagging(utterance: UtteranceRecord, *, force: bool, resume: bool) -> bool:
    """Whether this utterance should be sent to the tagger.

    `resume` exists because `force` is too blunt after a partial run: the
    utterances tagged before the failure carry `AnnotationSource.LLM`, and
    re-running with `--force` sends them all back to the model against the same
    rate limit that stopped it. Plain `--stage tag` is too blunt in the other
    direction -- it skips anything with tokens, including the rules-only tokens
    a no-key smoke test leaves behind, which are exactly what needs replacing.
    """
    if force:
        return True
    if resume:
        return utterance.annotation_source is not AnnotationSource.LLM
    return utterance.tokens is None


def _default_pipeline(provider: str | None = None, model: str | None = None) -> LIDPipeline:
    """An LLM-backed pipeline when any provider key is present, rules-only
    otherwise.

    Absence of a key is not an error here. It is the normal state on a laptop,
    and the run that follows is honestly labelled rather than refused.
    """
    try:
        from .lid.llm import make_tagger

        return LIDPipeline(llm_tagger=make_tagger(provider, model=model))
    except ValueError:
        raise
    except Exception:  # noqa: BLE001 - a missing package is not a reason to stop
        return LIDPipeline()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.annotate",
        description="Transcribe and tag the utterances in a corpus manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Stages are separable because they fail differently:\n"
            "  --stage asr   slow, offline, free; re-run only if the audio changes\n"
            "  --stage tag   fast, needs a provider key; re-run on any ontology change\n"
            "\n"
            "Tagging providers, first with a key present wins:\n"
            "  GEMINI_API_KEY  free   https://aistudio.google.com/apikey\n"
            "  GROQ_API_KEY    free   https://console.groq.com/keys\n"
            "  ANTHROPIC_API_KEY      adds prompt caching and the batch API\n"
            "\n"
            "Without any key the tagging stage assigns OTHER to every token and the\n"
            "CSBG has one class containing everything. The run is labelled, not refused."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--stage", default="both", choices=("asr", "tag", "both")
    )
    parser.add_argument("--model", default=None, help="Whisper checkpoint (default: from Settings).")
    parser.add_argument(
        "--provider",
        default=None,
        help="Tagging provider: gemini, groq, openai, ollama or anthropic. "
        "Default picks the first one with a key in the environment.",
    )
    parser.add_argument(
        "--tagger-model", default=None, help="Override the provider's default model."
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--language", default=None, help="Force an ISO code; default auto-detect.")
    parser.add_argument("--limit", type=int, default=None, help="Only this many utterances.")
    parser.add_argument(
        "--force", action="store_true", help="Redo utterances that already have output."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Tag stage only: tag anything not already LLM-tagged. Use after a "
             "run died partway (--force would re-send the ones that succeeded, "
             "against the same rate limit that stopped it), and over rules-only "
             "tokens left by a no-key smoke test.",
    )
    parser.add_argument("--report", type=Path, help="Write the report as markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    corpus = load_manifest(args.manifest)
    problems = corpus.validate()
    if problems:
        print("error: manifest is not structurally sound:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2

    from .config import get_settings

    settings = get_settings()
    report = AnnotationReport()

    # Shared by both stages: `--stage tag` alone would otherwise print nothing
    # for minutes.
    def progress(i: int, total: int, utterance: UtteranceRecord) -> None:
        print(f"[{i}/{total}] {utterance.utterance_id}", flush=True)

    if args.stage in ("asr", "both"):
        transcribe_corpus(
            corpus,
            model_size=args.model or settings.whisper_model,
            compute_type=settings.whisper_compute_type,
            device=args.device or settings.whisper_device,
            language=args.language if args.language is not None else settings.whisper_language,
            force=args.force,
            limit=args.limit,
            report=report,
            on_progress=progress,
            manifest_path=args.manifest,
        )

    if args.stage in ("tag", "both"):
        try:
            tag_corpus(
                corpus,
                force=args.force,
                resume=args.resume,
                limit=args.limit,
                report=report,
                manifest_path=args.manifest,
                provider=args.provider,
                model=args.tagger_model,
                on_progress=progress,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except RuntimeError as exc:
            # Already saved what it got -- the message says how far and how to
            # resume. Still an error exit: the corpus is partially tagged.
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print()
    print(report.to_markdown())

    reasons = corpus.reportability()
    if reasons:
        print("## Not reportable as it stands\n")
        for r in reasons:
            print(f"- {r}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_markdown(), encoding="utf-8")

    return 0 if not report.failures else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
