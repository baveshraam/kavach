# HANDOFF

State of the work as of the last commit, and what to do next. Read
[PROJECT.md](PROJECT.md) for the design reasoning; this file is only about where
things stand and what is left.

---

## Update 2026-10-03 -- the evidence pipeline: record yourself, measure the voice

Branch `feature/evidence-pipeline` (pushed to `baveshraam/kavach`, spec and plan in `docs/superpowers/`).
1150 tests pass offline. This answers "can the system be shown to work for the presenter, with numbers?".

- **Recording Studio** (`/studio`, `kavach/studio/`): guided sessions recorded through the browser microphone
  (the demo's own path, decoded by the login's `decode_bytes`), each clip labelled by session, device
  (`DEMO_LAPTOP_MIC | PHONE | HEADSET | OTHER`) and room, saved under git-ignored `data/studio/S04/` with an
  append-only index written last. Off by default, allowlist-only (`Settings.studio_enabled`,
  `studio_speakers`), never enabled by the demo build. Start it with `run_studio.ps1`; follow `RECORDING_GUIDE.md`
  (five sessions: S1-S3 enrol; **S4 and S5 are held out**, after breaks and on other devices/rooms).
- **Evaluation**: `python -m kavach.eval.enrollee --studio data/studio/S04 --enrol-sessions S1,S2,S3
  --test-sessions S4,S5 --impostors data/corpus_v2/manifest.json --impostors data/corpus_v3/manifest.json
  --exclude-speaker S04`. Enrolment and test never share a session; false-reject and false-accept each get a
  Wilson interval *and* a cluster bootstrap (over sessions / over impostor speakers: clips from one sitting are
  not independent); per-condition and per-impostor tables; a fitted-threshold operating point chosen on a dev
  partition and applied to a test partition split by speaker and session. The report opens with its limits:
  one enrollee, 11 impostors who read different material (not yet the same-sentence test), voice only.
- **Found while building it:** with a single held-out session the false-reject interval over sessions is a
  meaningless 0-100% (one sitting is one cluster), so a second held-out session is part of the plan, and the
  report says so when it has only one.
- **Smoke-tested with real ECAPA** on the presenter's 13 existing clips split into pseudo-sessions (one sitting,
  a plumbing check, NOT a result): 154 impostor trials over 11 speakers, report written. Even there the margin
  between the worst genuine score (0.62) and the best impostor (0.54, S12) was thin: exactly what the held-out
  recordings are for.

**Not built yet** (later sub-projects): the interruption guard (reject a clip with two voices), the "read this
sentence" login mode, the results write-up, and the CSBG analysis of the free-speech recordings.

---

## Update 2026-10-02 (b) -- the clone-attack bank, a demo preflight, and what rehearsal found

Executed from `docs/superpowers/plans/2026-10-02-clone-attack-bank.md` (spec:
`docs/superpowers/specs/2026-10-02-clone-attack-bank-design.md`) on the branch
`feature/clone-attack-bank`. **Nothing is pushed.** 1097 tests pass offline.

### What exists now

- **Clone bank** (`attacks/bank.py`): clips of the presenter's voice with provenance and a strict
  loader (allowlist, hashes, speaker, path escape). Only annotated clips without a translated/looping
  flag are ever measured or served.
- **Two stages, two environments.** `attacks/make_clone_bank.py` (clone env, `.venv-clone`, kNN-VC)
  and `attacks/annotate_bank.py` (main env: real Whisper, tagger, ECAPA, answer matcher). Setup recipe
  in `requirements-clone.txt`. kNN-VC verified on the GPU: 6 s converted in 6.7 s cold, 1.2 s warm.
- **Gated routes** `GET /api/clone-bank`, `POST /api/clone-bank/match`, `GET /api/clone-bank/{id}/audio`:
  404 unless `Settings.demo_attack_bank`; a clip *says* the answer, so `/api/health` reports the flag.
  `Settings.clone_victims` defaults to `[]` (nobody may be cloned).
- **Attack Lab**: with a bank, A4 rows are *measured* (real ECAPA similarity, real CSBG on the real
  transcript, real answer matcher), trials are capped at the number of distinct clips, yield is printed,
  the run stays `simulated` and `paper_ready()` still refuses it.
- **UI**: a "Clone attack" demo button (only when the bank is on) and an "acoustic measured/modelled"
  label on every lab run. Verified in a real browser against the real backend with a throwaway
  *synthetic* bank (no real person, no real voice): button, match, audio fetch, submit, label; and the
  button absent with the bank off.
- **`python -m kavach.demo_check --presenter S04 --flows`**: the preflight. Names the fix for every
  gap and drives the genuine, replay, impostor and clone logins. **`DEMO_RUNBOOK.md`** records the
  drills as observed.

### Found by rehearsing against the real backend (all fixed, each with a test)

1. **The impostor demo clip was rejected by the integrity gate before the voiceprint ran.** Turning
   splice detection off had not fixed this (the 2026-10-02 (a) note said it probably would). The replay
   detector's envelope similarity is about sqrt(cut / clip): a 20 s cut of a 24 s stored clip scores
   ~0.91 against a 0.85 threshold. Staged clips are now at most 40% of the clip and 12 s
   (`demo_check.cut_plan`, UI `stagingCut`, pinned by `tests/test_demo_staging.py`).
2. **A real `.m4a` failed with "Decoded audio is empty"**: moov atom at the end of the file, ffmpeg
   reading a pipe. `decode_bytes` now uses a seekable temp file; tests encode m4a, webm, ogg and mp3.
3. The first non-16 kHz upload stalled for seconds (librosa's lazy import): warmed at start-up.
   Silence and sub-second clips used to reach Whisper, which invents text: rejected early with a reason.
   Garbage uploads showed raw ffmpeg output: now a plain message.
4. The preflight read an empty bank as "covered" (vacuous truth), and demanded ACCEPT from a stand-in
   that cannot answer the random challenge. Both corrected (a stand-in is expected to be BORDERLINE).

### Blocked on the presenter, not faked

- **S04 has no knowledge-graph facts.** Until they are entered the preflight says `NOT READY` for S04
  and S04 cannot be challenged. Only S08 (4 facts) and S09 (3) have any.
- **No real bank exists.** It needs a teammate's spoken answer per S04 fact
  (`data/clone_sources/<predicate>.wav`), then the two commands in `DEMO_RUNBOOK.md` section 6.
- **S04's agreement to be cloned** is recorded only as the user's decision in this session.
- **The live browser microphone path was not exercised** and must be rehearsed once.
- Stage 2 (IndicF5, behind a go/no-go probe) is unplanned, as the spec says.
- The 88-slide deck's screenshots predate the UI fixes and show `w 0.00`.

### Independent review pass (fresh reviewer, whole branch) and what it changed

0 Critical, 4 Important, 14 Minor. Fixed, each with a test that failed first (suite 1097/1097):

1. A clip replaced or deleted *after* the bank was first loaded was still served (200) or raised a 500: the
   hash was checked only at load and the loaded bank is cached. `verified_audio_file` now re-hashes on
   every serve; a changed or missing file is a 503 with the reason.
2. A garbage upload spent the challenge, so the retry the polite 400 invites was rejected as a replay.
   `verify` now decodes before consuming (only for a challenge that could still be consumed).
3. The Clone button showed whenever the flag was on, even with no bank; it now needs a bank that can answer
   a question (`coveredFacts` non-empty). Checked in a browser: bank on, bank empty, bank off.
4. The README's "never distribute" sentence was an invisible HTML comment that also broke the table.
5. Re-graded up from Minor because they affect the honesty of numbers: the note and the row computed yield
   two different ways (now one, against the current threshold); a clip with no real answer score fell back
   to a constant inside a "measured" row (now left out, with a note); the preflight printed READY after a
   503/500 in the clone flow (only a 404 means "no clone for this question"); the runbook claimed a warning
   the preflight never gives.

**Deferred minors** (not fixed; the user decides): lab falls back to modelled without a note when the bank
is another speaker's or has no usable clips for that attack; failed bank loads are never cached (a broken
bank is re-hashed per request) and the cache dict is created lazily inside `clone_bank()`; `demo_check`
crashes with a traceback on a `TransportError` outside the clone loop instead of printing a FAIL line;
`decode_bytes`'s docstring still says piping "avoids writing to disk", and ffmpeg has no timeout, no
upload-size cap and no `-protocol_whitelist file` (a one-line hardening worth taking before exposing the
server beyond localhost); `knn_vc.py` loads the hub repo unpinned (`trust_repo=True`) and never fills
`backend_version`; `make_clone_bank --overwrite` deletes the old WAVs before the new bank is saved, so a
crash mid-run leaves no bank; `annotate_bank` reports "LLM" whenever the tagger has no `last_llm_error`,
uses beam ASR (`fast=False`) where live logins use the greedy path, and repeats the foreign-script filter
instead of calling `Pipeline.annotate`; an expired challenge may get a 404 instead of a 409 on the match
route if the ledger has pruned it; `GET /api/clone-bank` `problems` does not mention flagged clips; weak
tests (one half of `test_ordinary_audio_is_not_caught_by_the_early_reject` can never fail, the "no other
route serves the clone text" sweep checks only health and the speaker routes, the CLI-refusal test does not
`delenv` `KAVACH_CLONE_VICTIMS`, nothing checks that the TypeScript `stagingCut` mirrors `cut_plan`).

### Rehearsal numbers (2026-10-02, S08 as stand-in presenter)

Preflight with `--flows`, three runs in a row: identical results. Replay rejected at the integrity
gate; impostor rejected by the voiceprint (integrity passes); stand-in BORDERLINE 0.52 with the voice
passing. Cold start: health in ~5 s; first login 15 s after launch 2.3 s + 8.3 s, then 1.0 s + 4.9 s.
LLM provider unreachable: challenges fall back to the template bank (6-7 s), logins complete in 8-12 s
and say the CSBG was not scored. Details and the drills table: `DEMO_RUNBOOK.md`.

---

## Update 2026-10-02 -- the demo is verified end to end; splice tests are off, with the measurement

Goal of this pass: finish the demo-ready prototype on the 12 speakers on disk
(no new data). Five commits sit on top of `df0796b`; **nothing has been pushed**.

### Verified live: real backend, no mock

Whisper `small`, ECAPA, Gemini tagging, LaBSE listed, `integrity_check_splice`
at its new default (off). The three demo flows, staged from stored audio the way
the UI's "Demo with stored audio" does it:

| Flow | Decision | What decided it |
|---|---|---|
| Genuine stand-in (20 s cut of the claimed speaker's own audio) | ACCEPT, 0.635 / 0.55 | voice 0.922; CSBG 0.508 (margin **+0.008**); knowledge 0.38 *failed* |
| Replay (stored clip, byte for byte) | REJECT, 7 ms | integrity gate: byte-identical to a stored recording |
| Impostor (another speaker, 20 s cut, re-encoded) | REJECT, 0.299 | voice 0.142 against 0.62; integrity passed, so the voiceprint decides |

Read these honestly. The stand-in's knowledge branch failing is expected -- the
clip does not answer the challenge. The CSBG's +0.008 margin is a coin flip,
which is what 50% EER on free speech means. The first login took 31.5 s cold and
the next 7.6 s. LaBSE is **not** prefetched (1.9 GB), so the knowledge branch
runs on its string matchers only: `run_demo.ps1 -Prefetch` once, on a good
connection, before presenting. These three logins were removed from the demo DB
afterwards (restored from a backup), so its history is as it was.

### The splice tests are off by default, and why

`python -m kavach.calibrate_integrity --manifest data/corpus_v2/manifest.json
--manifest data/corpus_v3/manifest.json` (17 s, no models) reproduces this.

- They reject **167 of 168 genuine clips**. Three causes, none a threshold:
  exact-zero runs sit at the *edges* of 75 clips (decoder padding); the click
  test fires 19-256 times per genuine clip on ordinary fricatives; the
  background-step test sees a median 6 dB step across genuine pauses against a
  4 dB threshold, because real pauses are reverb tails, not stationary room tone.
- Retuning does not rescue it. At a matched duration, splices cut from a
  speaker's own clips are not separable from genuine windows by any cue (AUC
  0.49-0.60; at most 11% detected at a 5% false-reject rate). The best click
  setting finds 59% of known hard-cut joins while flagging every genuine file.
- `calibrate_floor` calls a floor of 0.0 "feasible" -- it meets the false-reject
  budget by rejecting nothing -- so the new command asks the prior question and
  ends in a verdict. Today's is "do not enable".
- Same-sitting splices are the only kind a one-session corpus can build, so this
  says nothing about cross-session splices. It is a limitation to state, not a
  result to claim.
- The checker no longer says "no edit artefacts found" when no edit test ran, and
  the Attack Lab says its A2 integrity column measures nothing about splices
  while the tests are off. With them on it prints the genuine false-reject rate
  beside the catch rate: a detector that flags everything catches 100% of attacks.

### Also fixed

- `w 0.00` on every weighted branch: callers pass a placeholder weight and the
  policy owns the real one, so `fuse` now reports the weight it used.
- `TokenText` had no whitespace between tokens in the DOM (a CSS margin faked it).
- A replay's rejection explanation called it a splice.
- The 21 hallucinated-script tokens (13 utterances, 7 speakers) are out of the
  stored graphs; those speakers were rebuilt CSBG-only, no re-embedding. Backups:
  `data/kavach.db.pre-foreign-rebuild-*`, `data/kavach.db.pre-live-*`.

### Still open

- **Clone attacks (A3-A5 acoustic scores are still modelled).** Needs its own
  spec: kNN-VC is voice *conversion*, so it does not fit `CloneBackend`, which is
  text-to-speech. Consent of whoever is cloned comes first (IndicF5's terms
  forbid cloning without permission).
- Gold-set labelling needs a bilingual human; nothing in code unblocks it.
- The Attack Lab cuts A2 segments from 0.0 s of each clip, which for speakers
  with decoder padding starts inside exact zeros. It only matters if the splice
  tests are ever turned back on; start segments after `calibrate_integrity.EDGE_MARGIN_SEC`.
- The 200+ h video-speech set has not been looked at.
- Push to `PremKxmar/speech.git`.

---

## Update 2026-09-29 — demo build on the existing 12 speakers (no new data)

Goal of this pass: a demo-ready system on the data already collected. Research
notes and the demo script are in [DEMO_PLAN.md](DEMO_PLAN.md).

**Run it:** `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1`
(`-Seed` rebuilds the demo DB first).

### Done

- **Demo DB holds all 12 consented speakers** — `python -m kavach.seed_demo`
  loads corpus_v2 + corpus_v3 (161 utterances, 1.12 h, 9,750 tagged words)
  with their *manifest* transcripts and tags (no ASR/LLM re-run), builds every
  CSBG and ECAPA template (self-consistency 0.77–0.91), and carries over the
  hand-typed SKG facts. The old DB is moved to `data/demo_backup_*`, never deleted.
- **Consent is recorded**: `data/consent_register.csv` is YES for S01–S12
  (the "every row is still PENDING" line further down is out of date).
- **UI rework** (`kavach/`): new design tokens (`src/index.css`), a shared kit
  (`src/components/ui/kit.tsx`), and every page rebuilt on it. Tamil is
  terracotta and English is ink-blue everywhere; no gradients or neon.
  Removed fake content: Enrolment's hard-coded log, "density 0.82" and mock
  triples; an Evaluation fairness chart keyed on Male/Female groups no backend
  emits. New: guided Authenticate flow with file upload (for replay/clone
  clips), SKG fact editor in the Speakers drawer, a "language axis" CSBG
  layout plus a two-speaker dumbbell comparison, and an Evaluation tab that
  renders `paper/*/results.json` with its blockers (`GET /api/offline-results`).
- `@types/react` was missing, so `tsc` had been type-checking JSX loosely; installed.
- The live-EER table printed fractions as percentages (100x too small); fixed.

### Found, and fixed for the demo

1. **The splice gate rejects 167 of 168 genuine recordings** (every speaker).
   Codec output (runs of exact zeros, sample-level jumps) reads as edits;
   `INTEGRITY_FLOOR` was calibrated on synthetic audio only. New
   `Settings.integrity_check_splice` (default on); `run_demo.ps1` turns it
   off. Replay/duplicate detection is unaffected and does catch a resubmitted
   corpus clip. **Done 2026-10-02:** recalibrated on the genuine corpus plus splices built
   from it. Nothing separates them, so the tests stay off (see the update above).
2. **A live login could hang for 10+ minutes.** The OpenAI-compatible client
   used the SDK's 600 s timeout plus 2 hidden retries under our 6. Now 45 s
   per request, SDK retries off, and the live path uses `live_llm_attempts=2`.
   If tagging still fails, the login **degrades**: CSBG reported unmeasured
   (never scored on rules-only OTHER tokens), knowledge still runs, the reason
   appears in the explanation. Corpus annotation keeps its full retry budget
   and still raises.
3. **Uploads froze the whole server.** `/api/authenticate` and
   `/api/utterances` were `async` routes calling the pipeline synchronously;
   now in a threadpool (health stays at ~0.2 s during a login).
4. **Whisper took 219 s for an 18 s clip on CPU** (repetition-loop re-decode
   times temperature fallback). ASR `auto` now detects CUDA through
   CTranslate2 (torch here is a CPU build, so the old check always said CPU)
   and uses the GPU when cuBLAS 12 + cuDNN 9 load (`nvidia-cublas-cu12`,
   `nvidia-cudnn-cu12` in the venv).
5. `Settings.warm_models_on_start` loads the models at server start (on in `run_demo.ps1`). Demo switches live there, not in `.env`, because the test suite reads `.env` (trap 4).

### Round 2 — data collection is over; workarounds so the prototype works

**Standing constraint from the user: no more data collection, ever.** Work
with the 12 speakers on disk; anything that needs more data stays pending.

6. **Live ASR is single-pass greedy** (`Settings.live_fast_asr`,
   `WhisperASR.transcribe(fast=True)`): 6.8 s for an 18 s clip on CPU versus
   49 s at beam 5 and 219 s when a loop triggered the re-decode. CPU threads
   raised from CTranslate2's default 4 to 8. Corpus annotation keeps beam 5.
7. **Offline tagger** (`lid/lexicon.py`): majority (language, class) per word
   over the 9,750 corpus tokens the LLM already tagged — 2,701 entries, 60–86%
   coverage on unseen code-mixed sentences. Used when the LLM fails or no key
   is set; the CSBG is scored if coverage >= 50% and the login explanation
   says it was lexicon-tagged. Live path only, never for annotation.
8. **CSBG veto off for the demo** (`Settings.csbg_veto_enabled`): both
   offline runs fitted it on dev and discarded it; a veto on a chance-level
   branch rejects genuine users at random.
9. **No model download mid-login** (`SemanticMatcher(allow_download=False)`
   on the live path). A login had started the 1.9 GB LaBSE download and sat
   on it; the HF cache still holds a partial copy. Fetch everything once with
   `python -m kavach.prefetch` (or `run_demo.ps1 -Prefetch`); until then the
   knowledge branch uses its three string matchers only.
10. **Hallucinated-script tokens are dropped** at login and when a graph is
    built (`asr.FOREIGN_SCRIPT`), and the login explanation counts them. The
    seeded graphs predate this: rebuild a speaker (Speakers → Rebuild) or
    re-run `seed_demo` to apply it to stored tokens.
11. **Microphone top-up** (Speakers drawer): add 2–3 clips from the demo
    laptop to an enrolled speaker and rebuild, so the voiceprint knows the
    demo microphone — enrolment audio is all phone recordings.
12. **Attack demo from stored audio** (Authenticate → "Demo with stored
    audio"): replay = a stored clip byte-for-byte (duplicate detector);
    impostor = a 20 s cut of another speaker re-encoded in the browser (so the
    voiceprint, not the duplicate check, has to reject it); genuine stand-in =
    a cut of the claimed speaker's own enrolment audio, labelled as such.

**Re-verified end to end on 2026-10-02** after items 6–12 (see the update above).

### Found, not fixed (needs a decision or more work)

- **13 of 161 stored transcripts contain hallucinated script** — Greek,
  Cyrillic, Hangul, CJK, Hebrew letters inside Tamil-English speech (e.g.
  S03_p04, S04_p11, S12_p07). They were tagged and are in the graphs.
  `repetition_loop` / `looks_translated` do not catch this; a foreign-script
  check belongs next to them.
- The CSBG result stands: 50% EER on free speech (5 speakers). Everything that
  needs a second session is still blocked on data (table below).

---

## Current state

- **1097 tests passing**, offline, in about 2-5 minutes on this laptop (as of 2026-10-02).
- Working tree clean as of the last commit; **not pushed** -- see the 2026-10-02 updates.
- Backend runs and serves the UI. All eight pages render, `tsc --noEmit` is
  clean, `vite build` succeeds, and the Graph Explorer now draws the SKG as
  well as the CSBG. `/api/health` reports `connected` on a Gemini key alone.
- **No paid API key is needed anywhere.** Tagging, challenge questions and
  attacker text all run on whichever provider has a key, free tiers included.
  Until recently only tagging did, so on a free-tier machine the other two
  silently served their template banks — which is a weaker *security claim*,
  not a weaker convenience, since a fixed question bank is enumerable and
  enumerating it is the attack the LLM path exists to defeat.
- **The pipeline has been run end to end on real recordings.** 7 speakers ×
  14 utterances, transcribed with `large-v3` (8.6% WER on the new three),
  tagged with `gemini-3.1-flash-lite`. `paper/results/` holds the 4-speaker
  run; `data/corpus_v2` is the 7-speaker corpus.
- **The run is correctly marked unreportable, on four counts**, and one of them
  matters more than the rest — see §5.2.4 of `KAVACH_Project_Idea.md`. The
  Tamil share per speaker is 0.69 / 0.03 / 0.90 / 0.63 / 0.05 / 0.03 / 0.06,
  which looks like a spectacular separation and is an artifact: each speaker
  read a *different* script, and four of the seven scripts are
  English-dominant. **On this corpus a CSBG is a script classifier.**
  Free-speech sessions are not an improvement to the corpus, they are the
  corpus.
- **And the script separation does not even buy verification accuracy.** On the
  seven-speaker run the CSBG alone scores **26.32% EER** — against 50% for
  guessing. So the 0.87 spread that looks like the headline result separates
  *speakers in aggregate* while carrying almost nothing per probe, which is
  what verification actually needs. Anyone tempted to report the spread should
  read that row first.
- **ECAPA alone scores 0.00% EER on this run, and that is not a result
  either.** Enrolment and probe come from the same sitting, so it measures how
  well an embedding memorises one recording session — one microphone, one
  codec, one room. A second session per speaker is what turns it into a number.

### What the first run does establish

Neither of these depends on the script, so both survive into the paper:

- Word-level LID and semantic tagging work on real code-mixed Tamil ASR
  output. `large-v3` transcripts track the reference scripts close to word for
  word; four `U+FFFD` characters across 56 utterances are the only artifact.
- **37 tokens were English that Whisper wrote in Tamil script** — `மாணிங்`,
  `ஏவினிங்`, `நைட்டுக்கு`, `யூஸ்வலி`, `சிக்ஸ்`, `தெட்டிக்கு`,
  `பஸ்டாப்புக்கு`, `போன்ல` — recovered rather than recorded as Tamil choices
  nobody made. `kavach.inspect_corpus --translit` lists them.

```bash
git clone https://github.com/PremKxmar/speech.git
cd speech
pip install -r requirements-core.txt
pytest                    # expect 1097 passed
```

The suite reaches no network and loads no checkpoint. Two autouse fixtures in
`tests/conftest.py` enforce that — `offline_by_default` for model checkpoints
and `no_live_llm_calls` for provider keys. Both exist for the same reason: a
suite whose behaviour changes when an unrelated `pip install` or a `.env` file
appears is testing the machine, not a configuration. Opt in with
`@pytest.mark.models` or `@pytest.mark.llm`.

**Python 3.10+ is required, and it is not optional.** The code uses
`dataclass(slots=True)`; on 3.9 every test file fails at collection with
`TypeError: dataclass() got an unexpected keyword argument 'slots'`. The
machine this was last built on had only the system 3.9, so:

```bash
uv venv --python 3.11 .venv          # or any 3.10+ interpreter
uv pip install --python .venv/bin/python -r requirements-core.txt
.venv/bin/python -m pytest
```

Note `pyproject.toml` sets `addopts = "-q"`, so passing `-q` again suppresses
the pass/fail summary line. Run plain `pytest` to see the count.

---

## The one thing that blocks publication

**There is no corpus.** The code is publishable-grade; the paper is not producible,
because no experiment has been run on human speakers. `simulation.py` generates
speakers who differ *by construction* — that a model separates them proves the
implementation is correct, not that Tamil–English speakers actually have stable,
distinguishable code-switching habits. That is the hypothesis the paper claims, and
it is untested.

Nothing else on this list matters as much. Everything below is preparation for the
day the recordings exist.

---

## What was built last session

All five items on the previous list are done.

| Item | Where |
|---|---|
| Corpus layer | `backend/kavach/corpus.py`, `tests/test_corpus.py` (61 tests) |
| Recording protocol | [RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md) |
| Figures | `backend/kavach/eval/figures.py`, `tests/test_figures.py` (41 tests) |
| Experiment runner | `backend/kavach/experiments.py`, `tests/test_experiments.py` (31 tests) |
| Scoring ablations + stability wiring | `eval/ablation.py::ablate_scoring`, `run_ablation(enrolment=...)` |
| Per-speaker IAPMR | `GET /api/attacks/per-speaker`, rendered in the Attack Lab |

One command now produces the whole evaluation:

```bash
python -m kavach.experiments --out paper/results/ --simulate-branches
# -> results.json, report.md, tables/*.tex, figures/*.pdf
```

Everything it emits is stamped unreportable, in the JSON, in the README it
writes, and as a banner inside every `.tex` file. Removing that banner requires
a corpus, not an edit.

---

## Next tasks, in priority order

### 1. Record the corpus

**In progress.** Four speakers have returned complete sets. Collection is
running on the scripted route: `participant_scripts/SPEAKER_A.md` …
`SPEAKER_J.md`, one language profile per speaker, matrix in
`participant_scripts/README.md`. Known mapping so far: `jai` read script A.

`ingest.py` turns returned folders into a manifest. Read the "What the first
four returns taught us" section of [RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md)
before ingesting anything — two of four sent WhatsApp voice notes, which makes
the codec a speaker attribute, and every numeral transcribed as a digit until
the `suppress_numerals` fix.

**A scripted corpus cannot support §5.1** and `Provenance.SCRIPTED` enforces
that. What it does support: word-level LID on known text, whether the pipeline
recovers a planted profile end to end, and the acoustic/integrity branches,
which do not care that the words were authored. Run the spontaneous protocol in
parallel with anyone who will sit for it — five spontaneous speakers plus ten
scripted is a better paper than either alone.

The go/no-go question is unchanged and now answerable sooner: **do genuine and
impostor CSBG scores separate at all on real speakers?** With one session each,
`--within-session` scores the pilot and stamps the run unreportable, which is
the right trade for a smoke test. §12 of `KAVACH_Project_Idea.md` is the
fallback paper if the answer is no, and it is a respectable one.

### 1a. What is blocked on data, and only on data

The pipeline is finished. Everything below needs recordings, not code.

| Blocked | Unblocked by |
|---|---|
| §5.1 headline claim, any CSBG number | **free-speech** sessions — scripted speech makes the CSBG a script classifier |
| §5.3 cross-session stability | a **second session** per speaker |
| the knowledge branch (scored 0 of 720 trials) | a second session — it needs the claimed speaker's enrolled answer to the probe's prompt, which a within-session split never provides |
| a stable EER | more speakers; 7 now, which leaves 3 dev / 4 test — better than 4 and still thin |
| Track 1 word-level LID accuracy | a bilingual labelling `data/goldset_v1.tsv` (1143 tokens, 20 utterances, 4 speakers; re-export smaller with `--utterances 6`) |
| any `RECORDED` provenance claim | `data/consent_register.csv` — every row is still PENDING, and S05–S07 have no row yet |

Ingest free speech with `python -m kavach.ingest --no-reference-transcripts`;
that is what makes the manifest `RECORDED` rather than `SCRIPTED`.

Run `python -m kavach.inspect_corpus --manifest ... --acoustic` on any folder
that arrives from someone else, **before** ingesting it.

**Never re-emit `data/speakers.csv` over the existing one.** `--emit-template`
numbers folders alphabetically, so on the seven-folder tree it renames
`BaveshRaamS` from S04 to S01 and shuffles the rest. Speaker ids are the join
key for the consent register, the corpus manifest and every recorded result, so
that rename silently reattributes recordings to the wrong people — including
their consent status. Append rows by hand instead; the ids already assigned are
load-bearing.

### 2. Wire the real branches into the experiment runner — DONE

`eval/branches.py` supplies both, behind `--real-branches`: ECAPA-TDNN cosine
against an enrolled template, and the answer matcher against the claimed
speaker's enrolled answer to the same prompt. Off by default because it needs
the corpus audio present and downloads the speechbrain checkpoint on first use.
`--real-branches` and `--simulate-branches` together raise.

Three things there are load-bearing and easy to undo by accident:

- **Unmeasurable trials score `nan`, never `0.0`.** On a [0, 1] branch scale
  `0.0` is maximal evidence *against* the claim, so a missing recording scored
  as `0.0` looks like a confident impostor detection and improves the EER.
- **Cosine maps to [0, 1] affinely, not by clamping negatives.** The affine map
  is strictly monotone so it moves no trial past another; clamping would
  collapse the impostor tail the veto threshold is fitted on.
- **Coverage is counted and blocks reporting.** A branch that scored nothing and
  a branch that scored everything and found no signal produce the same fusion
  table.

**The knowledge branch measures nothing on the current pilot, by construction.**
It needs the claimed speaker to have answered the probe's prompt at enrolment.
A cross-session protocol always gives it that; a within-session split never
does, because a prompt held out as a probe is by definition absent from that
speaker's enrolment. It reports zero coverage and blocks the run rather than
emitting nan noise. Second sessions fix it — nothing in the code will.

### 3. Annotate the corpus

**The Whisper question is closed** — decided on real audio, reasoning in
`Settings.whisper_model`. `large-v3` for annotation (`small` emits fragments of
unrelated languages at code-switch boundaries), auto-detect for language (it
keeps English in Latin script, which is what `lid.rules` reads for free), and
`suppress_numerals` now genuinely suppresses numerals.

`annotate.py` does the pass. Stage 1 (ASR) is built and runs. **Stage 2 needs
`ANTHROPIC_API_KEY` and nothing downstream works without it** — `lid.rules`
decides no semantic class, so a rules-only pass puts every token in
`SemanticClass.OTHER`, the CSBG has one class containing everything, and every
speaker's graph is identical. This is the single blocking dependency for a
first number.

**WER against the read-speech scripts is not computable, and that is a finding
rather than a gap.** The scripts romanise Tamil because participants read them
aloud; Whisper writes Tamil in Tamil script. No Tamil word can align, so the
WER is pinned above 100% (measured: 281%, 221%, 201%) and describes an
orthography mismatch. Restricting to Latin tokens does *not* rescue it —
romanised Tamil is Latin, so the filter strips nothing from the reference and
all the Tamil from the hypothesis; that scored 96% and looked like a result.
`annotate.transcripts_are_comparable` refuses both and `asr_wer` stays None.

The Track-1 contribution therefore needs **hand-labelled word-level LID on a
subset**, and `kavach.goldset` is the tooling for it:

    python -m kavach.goldset export --manifest data/corpus_v1/manifest.json \
        --out data/goldset_v1.tsv --utterances 20
    # a Tamil-English bilingual fills in the lang and class columns
    python -m kavach.goldset score --manifest data/corpus_v1/manifest.json \
        --gold data/goldset_v1.tsv --report paper/results/lid_track1.md

The export samples round-robin across speakers (a uniform sample from four
speakers lands most tokens on one voice, and LID accuracy is per-speaker here)
and writes its whole instruction sheet into the file header.

**Labels are blank by default and `--prefill` is opt-in.** Prefilling turns
labelling into agreeing: a corrector who sees "TA" agrees far more often than a
labeller starting from nothing, so measured accuracy drifts toward 100%
whatever the tagger does. The file records which mode produced it and
`GoldScore` carries the caveat into the report, so a prefilled set can never be
reported as blind.

The scorer reports the **transliteration slice** separately —
`transliterated_total` / `transliteration_recall`, Tamil-script tokens the human
labelled EN. Aggregate accuracy hides them because they are a small fraction
overall and not a small fraction of NUMBER and TIME_DATE.

### 3a. Tools built while waiting for the corpus

Three commands that did not exist before and are needed to read what comes out
of the pipeline.

**`python -m kavach.inspect_corpus --manifest ...`** — read the corpus with your
eyes. `--summary` (default) is the go/no-go read: per-speaker CMI, Tamil share,
switch counts, and the language-choice-per-class table, which is the CSBG
printed. It states the conclusion rather than leaving it to be computed — a
Tamil-share spread under 0.10 across speakers gets "that is very little to
separate on". `--text` for transcripts, `--tags` for token tables, `--translit`
for every Tamil-script token labelled English.

Use `--translit` to *check* the transliteration recovery rather than trust it.
A wrong entry there is an English label on a word the speaker really did say in
Tamil, which is the same bias pointing the other way, and nothing downstream
can tell the two apart.

**`python -m kavach.goldset`** — see §3 above.

**`python -m kavach.experiments --real-branches --goldset ...`** — see §2 and §3.

### 3b. Tagging survives a rate limit now — but know how it behaves

A corpus pass drives one request per utterance with no pacing, against a free
tier that allows ten-odd a minute. Retries are automatic (429/5xx, exponential
backoff with jitter, `Retry-After` honoured and capped at 120s) and printed, so
a stall is visible rather than looking like a hang.

If a run still dies, **it has already saved what it tagged** (checkpoint every
ten utterances, plus a save before the exception propagates). Resume with
`--stage tag --resume`, never `--force`: `--force` re-sends the utterances that
succeeded, against the same rate limit that stopped the run. `--resume` also
does the right thing over rules-only tokens left by a no-key smoke test, which
plain `--stage tag` skips because they already have tokens.

### 4. Smaller items

All three of the previous items here are done:

- **Graph Explorer label overlap** — fixed. Concentric now spaces by label
  width and ranks the language nodes into the middle explicitly (its default
  ranking is node degree, which put them there most of the time anyway and
  silently reordered whenever the edge threshold moved).
- **The Speaker Knowledge Graph view** — implemented as `SKGViz.tsx`. It
  renders only what the wire format carries; inferring each node's semantic
  class would mean duplicating `FACT_TYPES` from `skg.py` in TypeScript, where
  it drifts the first time a fact type is added.
- **`Settings.demo_reveal_answers`** — tested both directions. The useful test
  is a sweep: with the flag off, a distinctive answer string must not appear in
  the body of *any* route that touches a speaker, which catches a future route
  that serialises a challenge wholesale. `/api/speakers/{id}/skg` is asserted to
  still return it — that route is the enrolment editor, and it doubles as the
  positive control proving the sweep can see.

Still open:

- The **Export PNG** button exports PNG, not SVG. cytoscape renders to canvas
  and SVG needs the `cytoscape-svg` extension; adding it is the fix if a vector
  figure is wanted.

---

## Traps already hit — do not re-derive these

0. **Whisper sometimes translates instead of transcribing, and it looks like
   success.** `CODE_MIX_PROMPT` exists to prevent it and is a hint, not a
   guarantee — it did not hold on 2 of the pilot's 98 utterances. The output is
   fluent, correct English, so every downstream check passes and the utterance
   enters the graph as a speaker who chose English for every token. That is a
   fabricated language choice, not noise, and it lands on whichever speaker the
   model found hardest, so it reads as a per-speaker finding. `Transcript
   .looks_translated()` catches it at ASR time **without a reference
   transcript** — which is the point, because free-speech sessions have none.
   `kavach.inspect_corpus --translated` sweeps a manifest after the fact.
   Neither test is absolute: one real case had 7 choice tokens and the other
   scored 0.04 rather than 0.00, so the signal is distance from the *speaker's
   own* baseline.

1. **Do not set a threshold by reasoning about a score scale.** It has been wrong
   twice here (`CSBG_VETO_FLOOR`, `INTEGRITY_FLOOR`), both times because the
   constant lived in a module that could not observe what the other module emitted.
   Measure the distribution first. Both constants now have tests that re-derive them
   from real detector output.

2. **Do not let a grid search pick a floor that sits on a discrete evidence level.**
   `score < floor` means a floor of exactly 0.20 catches none of the probes scoring
   exactly 0.20, while looking optimal. See `integrity.INTEGRITY_FLOOR`.

3. **Do not report an unattainable operating point as a rate.** `format_rate()`
   returns `"n/a"` for NaN. If you add a metric, follow it.

4. **Do not let the test suite inherit its environment.** It used to pass in 48 s
   only because the models were not installed. `KAVACH_OFFLINE` and
   `tests/conftest.py` make that explicit; keep new tests inside it, and mark
   anything needing a real checkpoint `@pytest.mark.models`.

5. **Do not substring-search a serialised structure for a secret.**
   `test_public_dict_hides_the_answer` used `assert answer not in str(public)`, and
   `public_dict()` carries Unix timestamps while one SKG fixture fact is the room
   number `"214"`. A float timestamp is ten digits, so it contains any given
   three-digit run every few hundred seconds — the suite failed roughly once in
   several full runs, on a collision with the clock. The assertion shape is wrong
   in both directions: too weak (an answer split across fields, or case-folded,
   slips through) and too strong (any numeric field can collide). Check the fields
   that could actually carry it.

6. **`data/` is git-ignored in its entirety** because it holds voiceprints next to
   hometowns and family names. `Store.delete_speaker` must erase audio, tokens,
   facts, graphs and history — a deletion request that leaves recordings behind has
   not been honoured.

7. **Never report a number from `simulation.py` as an experimental result.** Its own
   docstring says so.

8. **A normaliser fitted on dev has no statistics for test speakers.**
   `fit_cohort_normaliser(split.dev)` covered *zero* of the eleven test
   speakers, because `split_by_speaker` puts a trial in dev only when both its
   speakers are dev speakers. Every test lookup fell back to (mean 0, std 1),
   so cohort z-norm was reported on the test table and did not happen. Fixed by
   keeping the discarded cross-boundary trials on `Split.cohort` and fitting
   from those whose *probe* is a dev speaker — see `cohort_fitting_trials` for
   why the other half of that cohort is still excluded. **The general trap: a
   lookup with a silent default cannot tell you it missed.**

9. **An ablation measured on the wrong scope reads +0.00 and the zero lies.**
   Every CSBG scoring ablation showed no effect on the fused system, because
   the knowledge branch separates far better and pins the EER. The rows only
   mean something measured on the branch they change. `AblationRow.scope`
   records which system each row is an EER *of*, and the markdown prints it.

10. **A seeded RNG must not mint identifiers.** `run_attack` built its run id
    from the same seeded `random.Random` that makes its scores reproducible, so
    attacking a second speaker with the same attack type produced a duplicate
    primary key and a 500 — on the one workflow the Attack Lab exists for.
    Reproducible scores are the property worth having; a reproducible id is a
    collision.

11. **Escape LaTeX in one pass, not with chained `str.replace`.** Escaping `\`
    first yields `\textbackslash{}`, and a later `{` rule then escapes the
    braces of that replacement. Reordering only moves the collision, because
    `~` and `^` expand to braces too. See `experiments._TEX_ESCAPES`.

12. **The mock hides the degraded path.** Three separate frontend bugs were
    invisible in `VITE_USE_MOCK=true` because the mock always returns a
    populated, well-formed payload: an empty `models` array crashed the whole
    app, a hardcoded `spk_001` made the Graph Explorer draw nothing, and the
    Overview printed invented figures. **Run the UI against a degraded backend
    before demoing it**, not just against a healthy one.

13. **A lazy `__getattr__` must not reach its own submodule with `from . import
    x`.** Before the import machinery will load a submodule it asks
    `hasattr(pkg, "x")`, which re-enters the hook — RecursionError, not an
    import. `kavach.api` had this on the path a server start-up depends on, and
    891 tests passed over it because the shim was `# pragma: no cover`. Use
    `importlib.import_module`. Relatedly: do not re-export a name a submodule
    already owns (`app`), because importing the submodule rebinds it and the
    meaning then depends on import order.

14. **A silent ASR failure looks like data; an empty one does not.** Whisper
    loops, emitting one plausible fragment for sixty tokens. Every copy lands
    in the same semantic class and is counted as a real language choice, so the
    speaker's most confident CSBG cell is built from a word nobody said. Only
    the tagger's alignment check caught the first one, and only by luck of
    token count. `Transcript.repetition_loop` checks for it now.

15. **Never re-emit `data/speakers.csv` with `--emit-template`.** It numbers
    folders alphabetically, so on the seven-folder tree `BaveshRaamS` moves
    from S04 to S01 and the rest shuffle. Speaker ids join the consent
    register, the manifest and every recorded result — that rename
    reattributes recordings, and their consent status, to the wrong people.

16. **A test that patches a class instead of an instance poisons the session.**
    `type(asr).model = property(...)` in a new ASR test took down an unrelated
    numeral-suppression test three classes away. Set the instance's private
    backing field, or use `monkeypatch`.

17. **A catch rate is unreadable without the rate at which the same test flags
    genuine audio.** The splice tests caught 82.5% on the synthetic generator
    and 100% of A2 in the Attack Lab on real clips -- because they also flagged
    167 of 168 genuine recordings. A detector that flags everything catches every
    attack. `calibrate_floor` hid it a second way: it calls a floor of 0.0
    "feasible", since rejecting nothing meets any false-reject budget. Feasible
    is not useful. Ask first whether the scores separate the classes at all, at
    a matched duration (`kavach.calibrate_integrity`), and print the genuine
    rate beside every catch rate.

---

## Standing instructions from the user

- Push to `PremKxmar/speech.git` at every step.
- The UI lives in `kavach/` and is user-supplied. Extend it; do not replace it.
- **No AI-looking design** — no neon colours, no gradient-heavy dashboards. This
  applies to figures too.
- The deadline is the user's concern, not the assistant's. Build the project.

---

## Answered, and what followed from each

1. **How many speakers?** ~25–30, over multiple sessions. The corpus layer and
   `RECORDING_PROTOCOL.md` are written to that number: ≥25 for usable EER
   intervals, ≥2 sessions per speaker weeks apart for §5.3, and
   `Corpus.reportability()` names every single-session speaker because the
   second sitting is the thing that cannot be recovered later.

2. **ANTHROPIC_API_KEY?** Not available yet. So `Corpus.reportability()`
   counts `n_guessed_tokens` and refuses a rules-only annotation, using the
   same rule `PipelineStats.is_corpus_grade` already applies rather than
   inventing a second standard.

3. **Voice cloner?** Not in scope yet. A3–A5 acoustic scores stay drawn from a
   documented distribution, `paper_ready()` keeps refusing those rows, and the
   experiment runner adds its own blocker whenever stand-in branches are used.

## Still open

1. **Which Whisper checkpoint** — see task 3 above. This one needs a decision
   before annotation starts, because it changes every token downstream.

2. **Is a resource-paper track available at SPELLL-2026?** §5.4 treats the
   corpus as the insurance policy; whether it can be submitted as its own
   contribution changes how much rides on the CSBG result.
