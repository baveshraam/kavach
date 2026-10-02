# Evidence pipeline — design

Status: **draft for review**, 2026-10-02 23:50. Sub-project 1 of 4 (see §8). Nothing here is implemented.

## 1. Outcome, and what is assumed

**Why.** The review is tomorrow. The presenter (S04) can record as much of their own voice as needed,
tonight, solo, for about two to three hours. The goal is a defensible answer to "does this system
reliably accept you and reject other people?", with numbers, instead of a demo that merely worked once.

**Outcome.** By tomorrow: (a) the presenter has recorded a labelled, session-structured dataset of their
own voice through the *same browser-microphone path the demo uses*; (b) one command evaluates the real
voiceprint on it and prints false-reject and false-accept rates with honest confidence intervals, per
condition; (c) the numbers and the caveats are in a report the presenter can show.

**Decisions made by the user** (not assumptions): approach A (single-enrollee proof); solo and a few
hours tonight; impostors are the existing corpus speakers plus attacks generated later plus the panel live
tomorrow; build order evidence pipeline → interruption guard → read-a-sentence mode → repo and write-up.

**Assumptions, open to correction:**
- The voice factor (ECAPA) is what this pipeline evaluates. The knowledge and code-switch (CSBG) branches
  are out of scope here (§7); the Studio *captures* what a later CSBG analysis needs, but does not run it.
- "Reliable" means measured on **held-out sessions**: enrolment and test never share a sitting.
- Every recording is of the presenter, for themselves. No third party is recorded by this pipeline.

## 2. What the evidence can and cannot claim

It can claim, for **one enrolled speaker**: the rate at which that speaker's genuine probes are rejected
(across devices, rooms, days), and the rate at which each of N other recorded speakers is accepted. It
**cannot** claim that code-switching or the voiceprint works for people in general (one enrollee), nor
report a population EER. The report states this itself, in its own text, so a screenshot cannot drop it.

Two further limits are printed with every table: the impostors are 11 people, so the false-accept interval
is wide and is bootstrapped over *speakers*, not trials; and the corpus impostors read different material
from the presenter (not the same sentence), so this is not yet the "same sentence" test. That test is the
panel, live, tomorrow (sub-project 3 supplies the read-a-sentence mode for it).

## 3. Data model

A **session** is one sitting with one device in one room. A **clip** is one recording.

```
data/studio/<pseudonym>/
  index.jsonl                 append-only, one JSON object per line (never rewritten)
  <session_id>/
    <clip_id>.wav             16 kHz mono PCM, what every consumer reads
    <clip_id>.orig            the bytes as the browser sent them (webm/opus, m4a, wav)
```

`index.jsonl` records, per clip: `clip_id, session_id, kind, prompt_id, text_hint, device, environment,
state_note, recorded_at (UTC ISO), duration_sec, sha256_wav, orig_ext`. Kinds:

| kind | what | purpose |
|---|---|---|
| `read` | a fixed sentence read aloud (the demo sentences; many repetitions) | voice-only, same-sentence test |
| `free` | an answer to one of the 14 bilingual `PROTOCOL_V1` prompts | CSBG data; natural speech |
| `fact` | an answer to a question about one of the presenter's own facts | the demo's login path |

`device` is a closed set (`DEMO_LAPTOP_MIC`, `PHONE`, `HEADSET`, `OTHER`); `environment` reuses
`corpus.Environment` (`QUIET_ROOM`, `OFFICE`, `CORRIDOR`, ...). Closed sets, because a free-text field
produces thirty groups of one and no slice large enough to compare (the same reason `Environment` exists).
`state_note` is free text for the presenter ("tired", "hurried"); it is never sliced on.

Everything lives under `data/`, which is git-ignored in its entirety (voiceprints). The index stores no
transcript: transcripts are produced later by the existing annotate stages.

## 4. The Recording Studio

**API** (all gated; see §6): `GET /api/studio/plan?speaker=S04`, `POST /api/studio/clips`
(multipart: audio, session_id, kind, prompt_id, device, environment, state_note),
`GET /api/studio/summary?speaker=S04`.

- `POST` decodes with the same `decode_bytes` the login uses (so a browser WebM/Opus clip is exercised
  through exactly the demo's decoder), refuses audio under one second or silent audio with a reason, writes
  the `.wav` and `.orig`, appends one line to the index, and returns the clip with running counts.
  The index line is written last: a crash leaves an orphan file, never an index entry pointing at nothing.
- `plan` returns the recording plan (§5) for the next session: an ordered list of
  `{kind, prompt_id, text_en, text_ta, repeat}`. The `read` sentences and the `fact` questions come from
  the presenter's own facts and a fixed sentence bank; the `free` prompts are `PROTOCOL_V1`.
- `summary` returns minutes and clip counts per session, device, environment and kind, so the presenter
  can see what is missing (for example "no noisy-room clips yet").

**UI** (`/studio`): a setup card (session label is generated; pick device, environment, optional note),
then a guided queue that shows the prompt in Tamil and English, uses the existing `AudioRecorder`, saves on
Accept and advances. A "re-record" control discards the last clip *before* it is saved (never after: the
index is append-only). A running progress bar and a per-session minutes counter.

## 5. The recording plan (what the presenter actually does)

About two to three hours, in four sessions. The **last session is the held-out test**: it differs from the
enrolment sessions in time and, where possible, device, because that is the condition that fails in a
demo.

| Session | Device / room | Contents | ~Time |
|---|---|---|---|
| S1 | demo laptop mic, quiet room | 3 demo sentences × 15 reps; 14 free prompts × 2; each fact × 4 | 35 min |
| S2 | phone, quiet room | the same | 35 min |
| S3 | demo laptop mic, a second room or posture (standing, farther from the mic) | 3 sentences × 10; 14 free × 1; facts × 3 | 25 min |
| S4 (held out) | after a real break (≥ 30 min); laptop mic, as the demo will be | 3 sentences × 20; free × 1; facts × 3 | 25 min |

S1-S3 enrol; S4 is never enrolled and is scored against the template built from S1-S3. If time allows,
a second held-out session on the headset or in a noisier room is worth more than more enrolment.

## 6. Safety, consent and fail-closed

- `Settings.studio_speakers: list[str] = []` (empty: the Studio records nobody) and
  `Settings.studio_enabled: bool = False`. Both routes and the page are off until the presenter's pseudonym
  is listed, exactly as `clone_victims` and `demo_attack_bank` are. `/api/health` reports the flag.
- The Studio records the allowlisted presenter only; there is no way to name another speaker.
- Index lines and filenames carry the pseudonym only, never a name. Nothing leaves the machine.
- `run_demo.ps1` does **not** enable the Studio; a separate switch does, so a recording session is a
  deliberate act and the demo build never accepts uploads into the dataset.

## 7. The evaluation (`python -m kavach.eval.enrollee`)

```
--studio data/studio/S04  --enrol-sessions S1,S2,S3  --test-sessions S4
--impostors data/corpus_v2/manifest.json --impostors data/corpus_v3/manifest.json
--threshold 0.62   --seed 7   --out paper/results_s04/
```

**Procedure.**
1. **Enrol.** Embed every clip of the enrolment sessions with the real ECAPA model, through the same
   `prepare_for_embedding` the login uses, and build a `SpeakerTemplate` (centroid), as production does.
2. **Genuine trials.** Every clip of the test sessions, scored against the template.
3. **Impostor trials.** Every clip of every *other* speaker in the impostor manifests (11 speakers),
   scored against the same template.
4. **Operating points.** Reported at (i) the **system threshold** the demo actually uses (0.62), and
   (ii) a threshold fitted on a **dev** partition (the last *enrolment* session held out of the template
   for this fit, against a random half of the impostor *speakers*) and applied unchanged to the **test**
   partition (the held-out session against the other half of the speakers). Split by speaker and by
   session, never by trial; a trial straddling the boundary is discarded, as `eval/ablation.py` does.
5. **Intervals.** False-reject: Wilson over clips *and* a session-cluster bootstrap. False-accept: Wilson
   over trials *and* a speaker-cluster bootstrap (resampling the impostor speakers). The report prints
   both and says which to trust (the cluster one, because clips from one sitting are not independent).
6. **Slices.** FRR, mean and minimum genuine score by `device × environment × kind`, dropping groups under
   a minimum size rather than printing a rate of one trial; and impostor scores by speaker, so the person
   who scores highest is named by pseudonym.
7. **Separation.** Genuine mean/min, impostor mean/max, d′, and the gap between the minimum genuine and the
   maximum impostor (the number that decides whether a threshold exists at all).

**Output.** `report.md` and `results.json` (aggregate numbers only; no audio, no per-clip biometric
vector) under `paper/results_s04/`, tracked in git like the other results. Per-trial scores go to
`data/studio/<pseudonym>/eval/` (git-ignored). Each report begins with the §2 limits.

**Not in this pipeline (and said so in the report):** CSBG and knowledge branches, fusion, attacks
(replay, overlap, clone), the same-sentence impostor test.

## 8. Decomposition

1. **This document**: capture and measure the voice.
2. Interruption guard: a single-speaker check (embed 3 s windows; reject clips whose windows disagree),
   validated on synthetic mixtures and the presenter's recordings.
3. Read-a-sentence login mode: a random code-mixed sentence, voiceprint plus text match.
4. Results write-up and slides.

## 9. Testing

All offline; no model and no network. The embedder is injected, so the evaluation is tested on synthetic
vectors whose geometry is known.

- Index: append-only, one line per clip, orphan files never indexed, an unreadable or silent upload refused
  with a reason and nothing written, the pseudonym only in lines and paths.
- Routes: 404 while `studio_enabled` is off; refuses a pseudonym not in `studio_speakers`; a real WebM/Opus
  and a no-faststart M4A round-trip through `decode_bytes` into a 16 kHz WAV of the right duration.
- Plan: counts match §5; every fact yields a question; `PROTOCOL_V1` prompts all present.
- Evaluation: Wilson and cluster-bootstrap against hand-computed values; the dev/test split shares no
  speaker and no session; a perfectly separated synthetic set gives FRR 0 and FAR 0 and a positive gap; an
  overlapping one does not; slices drop small groups; the report text contains the §2 limits verbatim.
- Real checkpoints only behind `@pytest.mark.models`, never in the default run.

## 10. Risks and open questions

- **Impostor population of 11, with different material.** The false-accept interval will be wide. That is a
  true statement about the evidence, and the report says it.
- **S4 may fail where S1-S3 pass.** That is the point of a held-out session. If it does, the response is
  more enrolment variety (device, distance), not a lower threshold; the report shows the per-condition
  table that says which.
- **Demo laptop microphone vs enrolment audio.** The enrolled voice is phone audio from one sitting. If the
  held-out laptop-mic session scores far below phone-session genuine scores, the fix is S1/S3 enrolment on
  the laptop mic, which is why they are in the plan.
- **Open for the reviewer:** (a) are three demo sentences right, or does the presenter want to choose them?
  (b) should the fitted-threshold operating point replace the fixed 0.62 in the demo if it is clearly
  better? (That would be a separate, deliberate change; this pipeline only measures.)
