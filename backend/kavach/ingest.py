"""Ingest: a folder of phone recordings becomes a manifest the evaluation reads.

`corpus.py` defines what a corpus *is* and refuses to certify a bad one. This
module is the only thing that creates one from real returns: fourteen files in
a folder named after a participant, recorded on a phone, in whatever format
that phone writes.

WHAT THIS REFUSES TO CARRY FORWARD
----------------------------------
**The folder name.** Participants send `jai/` or `priya_recordings.zip`, and a
first name in a path is a direct identifier sitting next to a voiceprint. The
speaker CSV maps folder to pseudonym and *stays out of the manifest*; the
manifest and the converted audio use `speaker_id` alone. That mapping file is
the same artefact as the consent register and lives with it, outside this
repository -- which is what makes a deletion request executable.

WHY FILENAMES RESOLVE THROUGH `PROTOCOL_V1` ORDER
-------------------------------------------------
Participants are asked for `01`-`14` and send back `1.m4a`, `01.m4a`,
`Recording 3.m4a` and `14 (1).m4a`. The number is an *ordinal position* in
`PROTOCOL_V1`, not a prompt id, so `03` is `p03_commute` because it is third.
That indirection is deliberate and it is also a hazard: reordering
`PROTOCOL_V1` silently relabels every file already collected. `corpus.py` says
this too. It is repeated here because this is the module that would do the
relabelling.

WHY CONVERSION HAPPENS AT INGEST
--------------------------------
Phones write AAC in an `.m4a` container. Nothing downstream reads that:
`soundfile` handles WAV and FLAC, ECAPA and Whisper both want 16 kHz mono. So
ingest writes a 16 kHz mono WAV *processing copy* into the corpus and leaves
the participant's original untouched wherever it landed. The original is the
archival copy and the only one that still carries the recorder's own artefacts,
which is why `integrity.py`'s codec baseline must be told about the source
format rather than reading the WAV and concluding the audio was never
compressed.

WHAT IT MEASURES ON THE WAY THROUGH
-----------------------------------
Every file gets a signal check, and the results are advisory rather than fatal.
A 30-speaker collection where ingest rejects a file outright means asking a
participant to record again, which mostly means losing the participant. So bad
files load with a recorded warning and `IngestReport.problems` is what a human
reads before deciding. The one exception is a file that cannot be decoded at
all, which is not a quality judgement.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .audio import (
    TARGET_SAMPLE_RATE,
    Audio,
    AudioError,
    check_quality,
    estimate_snr_db,
    leading_silence_sec,
    load_audio,
    save_wav,
)
from .corpus import (
    PROTOCOL_V1,
    Corpus,
    Environment,
    Provenance,
    SessionRecord,
    SpeakerRecord,
    UtteranceRecord,
    save_manifest,
)

#: Extensions a phone recorder plausibly writes. `.opus` is included so that a
#: WhatsApp voice note is *ingested and flagged* rather than silently skipped:
#: a participant who sent one needs telling, and a folder that looks empty
#: looks like a participant who never replied.
AUDIO_SUFFIXES = frozenset({".m4a", ".mp4", ".aac", ".wav", ".mp3", ".flac", ".ogg", ".opus", ".amr", ".3gp"})

#: Formats that mean the file was re-encoded in transit rather than by the
#: recorder. Opus at 16 kHz is what WhatsApp voice notes become.
TRANSIT_ENCODED_SUFFIXES = frozenset({".opus", ".ogg", ".amr", ".3gp"})

#: Collection standard, applied per utterance. Nothing here rejects a file --
#: see the module docstring. `MIN_LEAD_SILENCE_SEC` is the one participants
#: most often miss and the one `integrity.py` most wants.
MIN_DURATION_SEC = 3.0
MAX_DURATION_SEC = 120.0
MIN_SNR_DB = 15.0
MIN_LEAD_SILENCE_SEC = 0.25

_NUMBER_IN_NAME = re.compile(r"(\d{1,2})")


# --------------------------------------------------------------------------
# Filename -> prompt
# --------------------------------------------------------------------------


def resolve_prompt_id(filename: str) -> str | None:
    """Map a returned filename to a `PROTOCOL_V1` prompt id.

    Accepts the prompt id itself (`p03_commute.m4a`), or any name containing a
    number that is a valid 1-based position in the protocol -- `3.m4a`,
    `03.wav`, `Recording 03.m4a`, `03 (1).m4a`.

    Returns None when nothing resolves, which the caller reports rather than
    guesses at. A file assigned to the wrong prompt is worse than a file left
    out: it puts one speaker's festival answer in another's commute class and
    the corpus still validates.
    """
    stem = Path(filename).stem
    lowered = stem.lower()

    for prompt in PROTOCOL_V1:
        if prompt.prompt_id.lower() in lowered:
            return prompt.prompt_id

    # `03 (1)` -- a duplicate download. The first number is the prompt; a
    # trailing counter in brackets is not.
    without_counter = re.sub(r"\(\d+\)\s*$", "", stem).strip()
    match = _NUMBER_IN_NAME.search(without_counter)
    if not match:
        return None

    position = int(match.group(1))
    if not 1 <= position <= len(PROTOCOL_V1):
        return None
    return PROTOCOL_V1[position - 1].prompt_id


# --------------------------------------------------------------------------
# Reference transcripts from the read-speech scripts
# --------------------------------------------------------------------------

_SCRIPT_HEADING = re.compile(r"^###\s+(\d{1,2})\s*$")
_SCRIPT_ASIDE = re.compile(r"^\*\(.*\)\*$")


def parse_script(path: str | Path) -> dict[str, str]:
    """Read `participant_scripts/SPEAKER_X.md` into `{prompt_id: text}`.

    The scripts are sent to participants verbatim, so the parser has to
    tolerate the parts that are addressed to a human: the `### 01` headings are
    the answers, the italic asides (`*(Shorter -- about 20 seconds.)*`) are
    instructions and are dropped, and everything before the first heading is
    the recording rules.

    What comes back is the *reference* transcript -- the words the speaker was
    asked to say, not the words they said. The two differ, on purpose: the
    script tells readers to substitute a word that feels wrong in their mouth.
    Storing it lets ASR be scored against a known target, and that is the whole
    reason a scripted corpus is worth collecting.
    """
    text = Path(path).read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is None:
            return
        body = " ".join(line.strip() for line in buffer if line.strip())
        if body and 1 <= current <= len(PROTOCOL_V1):
            out[PROTOCOL_V1[current - 1].prompt_id] = body

    for raw in text.splitlines():
        heading = _SCRIPT_HEADING.match(raw.strip())
        if heading:
            flush()
            current = int(heading.group(1))
            buffer = []
            continue
        if current is None:
            continue
        line = raw.strip()
        if line.startswith("---") or line.startswith("#"):
            flush()
            current = None
            buffer = []
            continue
        if _SCRIPT_ASIDE.match(line):
            continue
        buffer.append(line)

    flush()
    return out


def find_script(script_id: str, *, scripts_dir: Path) -> Path | None:
    """Locate the script file for a speaker's profile letter."""
    if not script_id:
        return None
    candidate = scripts_dir / f"SPEAKER_{script_id.strip().upper()}.md"
    return candidate if candidate.exists() else None


# --------------------------------------------------------------------------
# The speaker table
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SpeakerSpec:
    """One row of the speaker CSV: everything a filename cannot carry.

    `folder` is the only field that identifies a person, and it is the only
    field that does not reach the manifest.
    """

    speaker_id: str
    folder: str
    script_id: str = ""
    consent_ref: str = ""
    gender: str = ""
    age_band: str = ""
    dominant_language: str = ""
    device: str = ""
    environment: str = ""
    recorded_on: str = ""
    session_id: str = ""
    notes: str = ""

    def resolved_session_id(self) -> str:
        return self.session_id or f"{self.speaker_id}_s1"

    def resolved_environment(self) -> Environment | None:
        if not self.environment:
            return None
        try:
            return Environment(self.environment.strip().upper())
        except ValueError:
            return None


CSV_COLUMNS = (
    "speaker_id",
    "folder",
    "script_id",
    "consent_ref",
    "gender",
    "age_band",
    "dominant_language",
    "device",
    "environment",
    "recorded_on",
    "session_id",
    "notes",
)


def read_speaker_csv(path: str | Path) -> list[SpeakerSpec]:
    """Load the speaker table.

    Raises:
        ValueError: If `speaker_id` or `folder` is missing from the header.
            Both are structural -- without them there is nothing to key on and
            nothing to read from.
    """
    rows: list[SpeakerSpec] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = {h.strip() for h in (reader.fieldnames or [])}
        missing = {"speaker_id", "folder"} - headers
        if missing:
            raise ValueError(
                f"speaker CSV {path} is missing required column(s): {sorted(missing)}"
            )
        for row in reader:
            clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if not clean.get("speaker_id"):
                continue
            rows.append(
                SpeakerSpec(**{k: v for k, v in clean.items() if k in CSV_COLUMNS})
            )
    return rows


def write_template_csv(audio_root: str | Path, path: str | Path) -> list[str]:
    """Write a speaker CSV skeleton, one row per folder found.

    `speaker_id` is pre-filled with `S01`, `S02`, ... in sorted folder order so
    the pseudonyms are assigned by the tool rather than by whoever fills in the
    sheet, and `folder` is the participant's own name. Everything else is left
    blank for a human, because everything else is a fact only a human has.
    """
    folders = sorted(p.name for p in Path(audio_root).iterdir() if p.is_dir())
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, folder in enumerate(folders, start=1):
            writer.writerow({"speaker_id": f"S{i:02d}", "folder": folder})
    return folders


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FileOutcome:
    """What happened to one returned file."""

    source: str
    speaker_id: str = ""
    prompt_id: str = ""
    duration_sec: float = 0.0
    snr_db: float = 0.0
    lead_silence_sec: float = 0.0
    clipping_ratio: float = 0.0
    ingested: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestReport:
    """Everything a human needs before trusting the manifest.

    Kept separate from `Corpus.validate()` because the two answer different
    questions. `validate` asks whether the manifest is internally coherent;
    this asks whether the *recordings* are what the protocol asked for, which
    is a question only the ingest step can see -- the manifest has no field for
    "this file starts mid-syllable".
    """

    files: list[FileOutcome] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ingested(self) -> list[FileOutcome]:
        return [f for f in self.files if f.ingested]

    @property
    def is_clean(self) -> bool:
        return not self.problems and not any(f.warnings for f in self.ingested)

    def to_markdown(self) -> str:
        lines = ["# Ingest report", ""]
        ok = self.ingested
        lines.append(f"{len(ok)} utterances ingested, {len(self.skipped)} files skipped.")
        lines.append("")

        if self.problems:
            lines += ["## Problems", ""]
            lines += [f"- {p}" for p in self.problems] + [""]

        by_speaker: dict[str, list[FileOutcome]] = {}
        for f in ok:
            by_speaker.setdefault(f.speaker_id, []).append(f)

        lines += ["## Per speaker", ""]
        lines += ["| speaker | files | total | mean SNR | flagged |", "|---|---|---|---|---|"]
        for speaker_id in sorted(by_speaker):
            fs = by_speaker[speaker_id]
            total = sum(f.duration_sec for f in fs)
            snr = sum(f.snr_db for f in fs) / len(fs)
            flagged = sum(1 for f in fs if f.warnings)
            lines.append(
                f"| {speaker_id} | {len(fs)}/{len(PROTOCOL_V1)} | {total / 60:.1f} min "
                f"| {snr:.1f} dB | {flagged} |"
            )
        lines.append("")

        flagged = [f for f in ok if f.warnings]
        if flagged:
            lines += ["## Flagged recordings", ""]
            for f in flagged:
                lines.append(f"- **{f.speaker_id} / {f.prompt_id}** ({Path(f.source).name})")
                lines += [f"  - {w}" for w in f.warnings]
            lines.append("")

        if self.skipped:
            lines += ["## Skipped", ""] + [f"- {s}" for s in self.skipped] + [""]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


class ConversionError(RuntimeError):
    """No available tool could decode a participant's file."""


def _convert_to_wav(src: Path, dst: Path, *, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Decode any recorder format to mono WAV at `sample_rate`.

    Tries ffmpeg, then macOS `afconvert`. Both are external because neither
    `soundfile` nor `librosa` decodes AAC without one of them anyway -- making
    it explicit here produces an error naming the missing tool instead of a
    stack trace from inside a loader.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[tuple[list[str], str]] = [
        (
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)],
            "ffmpeg",
        ),
        (
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}", "-c", "1",
             str(src), str(dst)],
            "afconvert",
        ),
    ]
    errors: list[str] = []
    for command, name in attempts:
        if shutil.which(command[0]) is None:
            continue
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and dst.exists():
            return
        errors.append(f"{name}: {result.stderr.strip()[:200]}")

    if not errors:
        raise ConversionError(
            f"Cannot decode {src.name}: install ffmpeg (`brew install ffmpeg`). "
            "Phone recordings are AAC in an .m4a container and no Python audio "
            "library reads that without it."
        )
    raise ConversionError(f"Failed to convert {src.name}. " + "; ".join(errors))


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def _check_signal(audio: Audio, source: Path, outcome: FileOutcome) -> None:
    """Fill in the measurements and the advisory warnings for one file."""
    quality = check_quality(audio, min_seconds=MIN_DURATION_SEC, max_seconds=MAX_DURATION_SEC)
    outcome.duration_sec = audio.duration_sec
    outcome.clipping_ratio = audio.clipping_ratio
    outcome.snr_db = estimate_snr_db(audio)
    outcome.lead_silence_sec = leading_silence_sec(audio)

    if quality.is_silent:
        outcome.warnings.append("silent or near-silent -- check the microphone")
    if quality.is_too_short:
        outcome.warnings.append(
            f"{audio.duration_sec:.1f}s is shorter than the {MIN_DURATION_SEC:.0f}s floor"
        )
    if audio.duration_sec > MAX_DURATION_SEC:
        outcome.warnings.append(
            f"{audio.duration_sec:.0f}s is far longer than the protocol asks for -- "
            "possibly two answers in one file"
        )
    if quality.is_clipped:
        outcome.warnings.append(f"{audio.clipping_ratio:.2%} of samples clipped")
    if outcome.snr_db < MIN_SNR_DB:
        outcome.warnings.append(
            f"SNR {outcome.snr_db:.0f} dB is below the {MIN_SNR_DB:.0f} dB standard -- "
            "noisy room, or an answer with no pauses in it"
        )
    if outcome.lead_silence_sec < MIN_LEAD_SILENCE_SEC:
        outcome.warnings.append(
            f"speech starts at {outcome.lead_silence_sec:.2f}s; the protocol asks for a "
            "beat of silence first, which integrity.py uses as its noise-floor baseline"
        )
    if source.suffix.lower() in TRANSIT_ENCODED_SUFFIXES:
        outcome.warnings.append(
            f"{source.suffix} is a transit format, not a recorder format -- this was "
            "probably sent as a voice note and re-encoded on the way"
        )


def ingest(
    audio_root: str | Path,
    speakers: Sequence[SpeakerSpec],
    out_root: str | Path,
    *,
    name: str = "corpus_v1",
    provenance: Provenance = Provenance.SCRIPTED,
    scripts_dir: str | Path | None = None,
    reference_transcripts: bool = True,
    copy_audio: bool = True,
) -> tuple[Corpus, IngestReport]:
    """Build a manifest from returned recordings.

    Args:
        audio_root: Directory holding one subdirectory per participant.
        speakers: The speaker table; `folder` names the subdirectory.
        out_root: Corpus root. Converted WAVs go to `<out_root>/audio/<id>/`.
        name: Corpus name recorded in the manifest.
        provenance: SCRIPTED for read speech, RECORDED for spontaneous. Not
            inferred -- see `Provenance`, the whole point of the field is that
            it cannot be changed by moving files around.
        scripts_dir: Where `SPEAKER_A.md`... live. Defaults to
            `participant_scripts/` beside the repository root.
        reference_transcripts: Attach the script text as
            `UtteranceRecord.transcript`. Only meaningful for SCRIPTED.
        copy_audio: Write converted WAVs. False does the resolution and the
            signal checks without touching the disk, which is what a dry run
            wants.

    Returns:
        The corpus and the report. The corpus is *not* validated here -- the
        caller decides what to do about a manifest with problems, and printing
        both lists together is more useful than raising on the first.
    """
    audio_root = Path(audio_root)
    out_root = Path(out_root)
    scripts_path = (
        Path(scripts_dir)
        if scripts_dir is not None
        else Path(__file__).resolve().parents[2] / "participant_scripts"
    )

    report = IngestReport()
    corpus = Corpus(name=name, provenance=provenance, root=out_root)

    for spec in speakers:
        folder = audio_root / spec.folder
        if not folder.is_dir():
            report.problems.append(
                f"speaker {spec.speaker_id}: folder {spec.folder!r} not found under {audio_root}"
            )
            continue

        script_text: dict[str, str] = {}
        if reference_transcripts and spec.script_id:
            script_file = find_script(spec.script_id, scripts_dir=scripts_path)
            if script_file is None:
                report.problems.append(
                    f"speaker {spec.speaker_id}: script {spec.script_id!r} has no file in "
                    f"{scripts_path}"
                )
            else:
                script_text = parse_script(script_file)

        corpus.speakers.append(
            SpeakerRecord(
                speaker_id=spec.speaker_id,
                consent_ref=spec.consent_ref,
                gender=spec.gender,
                age_band=spec.age_band,
                dominant_language=spec.dominant_language,
                script_id=spec.script_id,
                notes=spec.notes,
            )
        )
        session_id = spec.resolved_session_id()
        corpus.sessions.append(
            SessionRecord(
                session_id=session_id,
                speaker_id=spec.speaker_id,
                recorded_on=spec.recorded_on,
                device=spec.device,
                environment=spec.resolved_environment(),
            )
        )
        if spec.environment and spec.resolved_environment() is None:
            report.problems.append(
                f"speaker {spec.speaker_id}: environment {spec.environment!r} is not one of "
                f"{[e.value for e in Environment]}"
            )

        candidates = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
        )
        claimed: dict[str, Path] = {}
        for source in candidates:
            prompt_id = resolve_prompt_id(source.name)
            if prompt_id is None:
                report.skipped.append(
                    f"{spec.speaker_id}: {source.name} -- no prompt number in the filename"
                )
                continue
            if prompt_id in claimed:
                report.skipped.append(
                    f"{spec.speaker_id}: {source.name} -- {prompt_id} already taken by "
                    f"{claimed[prompt_id].name}"
                )
                continue
            claimed[prompt_id] = source

        for prompt in PROTOCOL_V1:
            source = claimed.get(prompt.prompt_id)
            if source is None:
                report.problems.append(
                    f"speaker {spec.speaker_id}: no recording for {prompt.prompt_id}"
                )
                continue

            outcome = FileOutcome(
                source=str(source), speaker_id=spec.speaker_id, prompt_id=prompt.prompt_id
            )
            relative = Path("audio") / spec.speaker_id / f"{session_id}_{prompt.prompt_id}.wav"
            destination = out_root / relative

            try:
                if source.suffix.lower() == ".wav":
                    audio = load_audio(source)
                    if copy_audio:
                        save_wav(audio, destination)
                else:
                    if copy_audio:
                        _convert_to_wav(source, destination)
                        audio = load_audio(destination)
                    else:
                        with tempfile.TemporaryDirectory() as tmp:
                            scratch = Path(tmp) / "probe.wav"
                            _convert_to_wav(source, scratch)
                            audio = load_audio(scratch)
            except (AudioError, ConversionError) as exc:
                report.problems.append(f"{spec.speaker_id}/{source.name}: {exc}")
                report.files.append(outcome)
                continue

            _check_signal(audio, source, outcome)
            outcome.ingested = True
            report.files.append(outcome)

            corpus.utterances.append(
                UtteranceRecord(
                    utterance_id=f"{session_id}_{prompt.prompt_id}",
                    session_id=session_id,
                    speaker_id=spec.speaker_id,
                    prompt_id=prompt.prompt_id,
                    audio_path=str(relative) if copy_audio else "",
                    duration_sec=round(audio.duration_sec, 3),
                    transcript=script_text.get(prompt.prompt_id, ""),
                )
            )

        if script_text:
            absent = [
                p.prompt_id for p in PROTOCOL_V1 if p.prompt_id not in script_text
            ]
            if absent:
                report.problems.append(
                    f"speaker {spec.speaker_id}: script {spec.script_id} has no text for "
                    f"{absent}"
                )

    corpus.notes = (
        f"Ingested from {len(speakers)} speaker folders. "
        f"{len(report.ingested)} utterances, "
        f"{sum(f.duration_sec for f in report.ingested) / 60:.1f} minutes of speech. "
        "Transcripts are the reference scripts, not ASR output."
        if provenance is Provenance.SCRIPTED
        else f"Ingested from {len(speakers)} speaker folders."
    )
    return corpus, report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kavach.ingest",
        description="Turn returned participant recordings into a corpus manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Typical run:\n"
            "  python -m kavach.ingest --audio recordings --emit-template data/speakers.csv\n"
            "  # fill in the CSV: consent_ref, script_id, device, age_band ...\n"
            "  python -m kavach.ingest --audio recordings --speakers data/speakers.csv \\\n"
            "      --out data/corpus_v1 --provenance SCRIPTED\n"
        ),
    )
    parser.add_argument("--audio", required=True, type=Path, help="Directory of speaker folders.")
    parser.add_argument("--speakers", type=Path, help="Speaker CSV.")
    parser.add_argument("--out", type=Path, help="Corpus root to write.")
    parser.add_argument(
        "--emit-template",
        type=Path,
        metavar="CSV",
        help="Write a speaker-CSV skeleton for the folders found, then exit.",
    )
    parser.add_argument(
        "--provenance",
        default=Provenance.SCRIPTED.value,
        choices=[p.value for p in Provenance],
        help="SCRIPTED for read speech (default), RECORDED for spontaneous.",
    )
    parser.add_argument("--name", default="corpus_v1", help="Corpus name in the manifest.")
    parser.add_argument("--scripts-dir", type=Path, help="Where SPEAKER_*.md live.")
    parser.add_argument(
        "--no-reference-transcripts",
        action="store_true",
        help="Do not attach script text as the reference transcript.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and check every file without writing audio or a manifest.",
    )
    parser.add_argument("--report", type=Path, help="Write the ingest report as markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.emit_template:
        folders = write_template_csv(args.audio, args.emit_template)
        print(f"Wrote {args.emit_template} with {len(folders)} rows: {', '.join(folders)}")
        print("Fill in consent_ref and script_id before ingesting; both are required.")
        return 0

    if not args.speakers:
        print("--speakers is required (or --emit-template to create one).", file=sys.stderr)
        return 2
    if not args.out and not args.dry_run:
        print("--out is required unless --dry-run.", file=sys.stderr)
        return 2

    speakers = read_speaker_csv(args.speakers)
    out_root = args.out or Path(".")
    corpus, report = ingest(
        args.audio,
        speakers,
        out_root,
        name=args.name,
        provenance=Provenance(args.provenance),
        scripts_dir=args.scripts_dir,
        reference_transcripts=not args.no_reference_transcripts,
        copy_audio=not args.dry_run,
    )

    print(report.to_markdown())

    problems = corpus.validate()
    if problems:
        print("## Manifest problems\n")
        for p in problems:
            print(f"- {p}")
        print()

    reasons = corpus.reportability()
    if reasons:
        print("## Not reportable as it stands\n")
        for r in reasons:
            print(f"- {r}")
        print()

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_markdown(), encoding="utf-8")

    if args.dry_run:
        print("Dry run: no audio copied, no manifest written.")
        return 0

    if problems:
        print("Refusing to write a manifest with structural problems.", file=sys.stderr)
        return 1

    path = save_manifest(corpus, out_root / "manifest.json")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
