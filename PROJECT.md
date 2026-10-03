# KAVACH — Code-Switch Behaviour Graphs for Voice Biometric Security

**Target venue:** SPELLL-2026, 5th International Conference on Speech and Language
Technologies for Low-Resource Languages, Amrita Vishwa Vidyapeetham, Coimbatore,
17–19 December 2026. Theme: *Knowledge Graphs for low-resource languages using LLM
and Multimodal data.*

**Repository:** `PremKxmar/speech.git` (the teammate's original; the work now continues in
`github.com/baveshraam/kavach`, which preserves its history)

> **Status, 2026-10-03.** This file was written in August, when no speech had been recorded. Since then: a
> 12-speaker pilot corpus exists and was annotated (the CSBG is at chance on free speech, so the paper's fallback
> framing applies); a live demo was built and rehearsed on it; and the direction changed from a research prototype
> that averages three factors to **evidence for a single enrolled speaker behind a gated login**. Sections 3.9-3.12
> and 6 (sessions 9-12) carry that; section 4 is rewritten. The reasoning and every measurement of the latest pass
> are in [DEMO_HARDENING.md](DEMO_HARDENING.md); where things stand is the top of [HANDOFF.md](HANDOFF.md).

This document explains what the system is, why each design decision was made, and
what has actually been measured versus what is still assumed. Read
[HANDOFF.md](HANDOFF.md) first if you are picking the work up mid-stream.

---

## 1. The idea in one paragraph

A bilingual Tamil–English speaker does not choose between languages at random.
They have habits: kinship terms in Tamil, technology in English, numbers one way
when counting money and the other way when giving a phone number. Those habits are
stable within a person and different between people. **KAVACH models a speaker's
language-choice habits as a graph over semantic concept classes — the Code-Switch
Behaviour Graph (CSBG) — and uses it as a soft biometric factor alongside a
conventional voiceprint.**

The security argument is what makes this worth publishing. A voice cloner can
reproduce *how someone sounds* from thirty seconds of audio, and an attacker who
scrapes social media can learn *what someone knows*. Neither of those gives them
*how that person distributes languages across meaning*. That is the gap the CSBG
is aimed at, and attack A4 below is the experiment that tests it.

---

## 2. Architecture

```
audio ──► ASR (faster-whisper)  ──► word-level LID ──► semantic tagging ──► tokens
  │                                  (rules + LLM)      (21 concept classes)
  │                                                              │
  ├──► ECAPA-TDNN embedding ──────────────────────┐              ▼
  │                                               │        CSBG (per speaker)
  ├──► signal-integrity tests ────────────┐       │              │
  │    (splice / duplicate artefacts)     │       │              ▼
  │                                       │       │      log-likelihood ratio
  └──► challenge freshness ───────┐       │       │       vs. background model
                                  │       │       │              │
                                  ▼       ▼       ▼              ▼
                              ┌─────────────────────────────────────┐
                              │  GATES          WEIGHTED BRANCHES   │
                              │  liveness       speaker_embedding   │
                              │  integrity      csbg                │
                              │                 knowledge           │
                              └─────────────────────────────────────┘
                                            │
                                     ACCEPT / REJECT / BORDERLINE
```

The **live login** (the Unlock demo) is a gated pipeline rather than the weighted average above, which remains the
research system and the paper's ablations:

```
challenge (random words) ─► recording ─► gates, all of which must pass:
                                           liveness          unused, in time
                                           signal integrity  not a copy of one already seen
                                           phrase            the shown words, in order (ASR transcript)
                                           voice             ECAPA cosine on the span where the words were
                                                             spoken, against a MEASURED threshold
                                         ─► ACCEPT, or BORDERLINE (one stricter second phrase), or REJECT
                          failed attempts at one identity slow down (throttle)
```

### Repository layout

```
backend/kavach/
  csbg/            THE RESEARCH CORE
    ontology.py      21 semantic classes; which are elicitable; which are low-signal
    tokens.py        Token, UtteranceTokens — the unit everything downstream reads
    graph.py         CSBG construction, CMI, I-index, density
    scoring.py       log-likelihood ratio vs. UBM, cohort z-normalisation
    metrics.py       graph-level statistics
  lid/             word-level language ID + semantic tagging (rules, then LLM)
  attacks/         A1–A5 attack construction, splice/replay detection, IAPMR
  corpus.py        recorded-speech manifest, loader, coverage, session split
  experiments.py   one command -> results.json + figures + LaTeX tables
  eval/
    metrics.py       EER, minDCF, DET, bootstrap CIs, Wilson intervals
    ablation.py      dev/test split, fitted weights, fitted veto, ablations
    figures.py       publication figures; greyscale-safe, no chartjunk
  api/             FastAPI layer serving the React UI
    schemas.py       Pydantic contract mirroring the frontend's types.ts
    converters.py    the single place domain ↔ wire conversion happens
    store.py         SQLite records + filesystem audio
    pipeline.py      the runtime: enrolment, challenge, verification
    attacks.py       the Attack Lab
    app.py           19 routes + audio serving
  integrity.py     signal-integrity gate (splice + duplicate detection)
  fusion.py        branch combination, gates, vetoes, the voice gate
  phrase.py        the random words of the Unlock login and their matcher
  throttle.py      failed attempts slow down
  calibrate_voice.py  the measured voice threshold, the policy file, leave-one-session-out
  cohort.py        public impostor corpora, embedded once and cached
  demo_check.py    the preflight; synthetic.py is its text-to-speech stranger
  studio/          the presenter's recording Studio: store, plan, enrol, words_check
  audio.py  asr.py  embedding.py  matcher.py  skg.py  challenge.py  simulation.py
kavach/            the React + Vite + TypeScript UI (user-supplied; /unlock is the demo screen)
tests/             1490 tests
```

---

## 3. Design decisions, and why

These are the decisions a reviewer will ask about. Each one is implemented and
tested; the file that owns it is named.

### 3.1 Unavailable is not zero — `fusion.py`

A branch that could not be measured is dropped and the remaining weights are
renormalised. Scoring it 0.0 would silently convert a missing measurement into
strong evidence of an impostor. The response records which branches actually
contributed.

### 3.2 Liveness is a gate, not a weight — `fusion.py`

An expired or reused challenge is not "some evidence against" — it is
disqualifying, because it is exactly what a replay looks like. Weighted fusion
would let a very strong voice match outvote it.

### 3.3 A weighted average cannot express a veto, and the threat model needs one — `fusion.py`

With weights (0.40, 0.30, 0.30) and threshold 0.55, a branch weighted 0.30 can move
the fused score by at most 0.30. An attacker who fools the acoustic branch (0.85)
and knows the answer (0.90) is already at 0.61 from those two alone — so **no** CSBG
score, not even zero, can pull the total below threshold. That is attack A4, the
one the CSBG exists to stop, and under plain weighted fusion the CSBG is
structurally incapable of stopping it.

The fix follows from what the score *is*: a log-likelihood ratio, so a strongly
negative value is affirmative evidence *against* identity, not weak support for it.
Averaging that away is a category error. `FusionPolicy.veto_thresholds` lets a
branch reject outright below a floor.

### 3.4 Signal integrity is a second gate — `integrity.py`

A spliced file carries a *perfect* voiceprint, because it is made of the victim's
real voice. Averaging a "who is this" score with a "was this edited" score means a
good enough voice buys tolerance for an edit, which is backwards. So integrity is
disqualifying, like liveness.

It is emitted for **every** configuration including the ECAPA-only baseline,
because edit-artefact tests need no challenge, no knowledge graph and no CSBG. The
consequence is that the A1 and A2 rows go flat across all four columns, and that
flatness is the finding: those attacks are stopped by signal evidence, identically
well with or without the research contribution.

### 3.5 Leave-one-out background model — `csbg/scoring.py`, `eval/ablation.py`

The UBM must exclude the claimed speaker. Including them makes the likelihood ratio
compare a speaker against a population containing themselves.

### 3.6 Dev/test split by speaker, never by trial — `eval/ablation.py`

`split_by_speaker` partitions speakers and **discards** trials that straddle the
boundary — roughly half the impostor trials at `dev_fraction=0.4`. Fewer trials is
cheaper than a leak.

### 3.7 Degraded mode is a first-class state — `api/pipeline.py`

Missing model → the branch reports `available=False` and `/api/health` names what is
missing and what each absence costs. An authenticate call in this mode returns
REJECT whose explanation contains "system failure" — a user told "your voice did not
match" when no model ran has been told something false.

### 3.8 The knowledge factor must not be in the response — `api/converters.py`

`types.ts` declares `Challenge.expectedAnswerEntity` and the UI mock fills it with
the real answer, which anyone authenticating could read out of the network tab. The
field stays so TypeScript validates, returns `""` by default, and is gated behind
`Settings.demo_reveal_answers` (default False) which `/api/health` reports so a demo
build announces itself.

### 3.9 The voiceprint is a gate, not a vote — `fusion.py`, `Settings.voice_gate`

Section 3.3 argued that a weighted average cannot express a veto, and applied it to the CSBG. The same argument
applies harder to the identity factor itself: voice 0.45 (under a 0.62 threshold) plus a known answer plus a
coin-flip CSBG scored 0.72 and was accepted, so a judge needed no voice at all, only a hometown from a slide.
`FusionPolicy.voice_gate` makes the voice necessary: below `threshold - margin` is a reject, the band under the
threshold is at best BORDERLINE, an unmeasurable voice fails closed. It is off in `FusionPolicy()` so the paper's
ablations still measure the weighted rule, and on in `Settings` for the live system. (DEMO_HARDENING R1.)

### 3.10 A phrase of random words binds a recording to one challenge — `phrase.py`

A re-recorded replay of the presenter passed voice and knowledge. The Unlock login shows ten random words per
attempt, transcribes the recording as English with no code-mixing prompt, and requires six in order; a recording
made for another attempt cannot contain them (an unrelated ten-word recording never reached five in 100,000
random pairs). It needs no facts, no language tagger and no network. The voice is scored only on the span where
the shown words were spoken, so speech before or after cannot dilute or hijack it. (R2, R5, R10.)

### 3.11 The threshold is measured, with a margin biased toward the owner — `calibrate_voice.py`

A threshold chosen by reasoning about a score scale has been wrong twice here. `kavach.calibrate_voice` measures
the strangers' tail (public speech corpora and the corpus speakers, cut to login length) and the owner's low end
(leave-one-session-out over the presenter's own Studio sessions), and puts the threshold 60% of the way from the
first to the second, never below 0.55; where the classes overlap the strangers' tail wins and the owner's
false-reject rate is reported with the remedy. Probe length is the strongest lever found (margin +0.10 at 4-8 s,
+0.20 at 8-11 s), which is why the phrase is ten words. The result is a policy file the live system loads; a
damaged file falls back to the default and says so. (R6, R10.)

### 3.12 Evidence is for one enrolled speaker, and says so first — `eval/enrollee.py`, `calibrate_voice.py`

Every report opens with its limits: one enrollee; false-accept over recorded strangers who were not recorded
through the demo laptop and so understate a same-room judge; no population EER. Intervals are never narrower than
Wilson's; a cluster bootstrap over fewer than five clusters is labelled not informative. (Studio spec and plan:
`docs/superpowers/`.)

---

## 4. What has been measured, and what has not

**This is the section to read before writing any number into a paper.**

| Component | Status (2026-10-03) |
|---|---|
| CSBG construction, LLR scoring, cohort z-norm | Real, tested. **On real speech: 26.3% EER on scripted speech (a script classifier, not a speaker habit) and 50.0% EER on free speech (5 speakers, one session each: chance).** |
| EER / minDCF / DET / bootstrap CIs | Real, tested |
| Fitted fusion weights, threshold, veto floor | Real, fitted on a dev split for the paper's ablations; the CSBG veto was fitted and discarded. The live login no longer uses the weighted rule: see 3.9 |
| Splice + duplicate detection | Duplicate detection real. **Splice tests are off by default**: they rejected 167 of 168 genuine corpus clips and no cue separated splices from genuine windows (`kavach.calibrate_integrity`) |
| ECAPA-TDNN embeddings | Real. 12-speaker corpus: cross-speaker cosine mean 0.24, max 0.588; public cohorts, 0 of about 31,000 stranger trials reached 0.62 (nearest 0.608); room, noise and codec effects measured on LibriSpeech (DEMO_HARDENING 3). **The presenter's own cross-session scores do not exist yet** |
| Speech corpus | **Exists**: 12 consenting speakers, 1.12 h, one session each (7 scripted, 5 free), plus public corpora as impostors. No second session per speaker, so cross-session claims wait on the presenter's Studio sessions |
| Voice cloning | kNN-VC clone bank implemented (A4); the real bank awaits a teammate's spoken answers, so A3-A5 acoustic scores remain modelled unless a bank exists, and runs stay `simulated` |
| Acoustic + knowledge branches in `experiments.py` | Real branches wired (`--real-branches`); the knowledge branch scores nothing on a one-session corpus by construction |
| LLM semantic tagging | Gemini free tier for annotation; an offline lexicon fallback for live logins; not corpus-grade without a key |
| The gated login (phrase, voice gate, throttle, step-up) | Tested; rehearsed against real Whisper/ECAPA with synthetic voices and in a real browser; owner numbers pending the Studio sessions |

`AttackTable.paper_ready()` refuses every current row, by design.

**The blocking gaps are now second sessions and more speakers, not a missing corpus.** The 12-speaker pilot is real
but single-session; the knowledge branch and cross-session stability need a second sitting per speaker, which
cannot be collected for other people (the pilot is closed). The presenter's own sessions close the gap for one
speaker. `simulation.py` still says in its own docstring that synthetic speakers differ *by construction*, so a
number it produces is never an experimental result.

---

## 5. Findings worth writing up

Three of these came out of building the system and are documented in
`KAVACH_Project_Idea.md` §4.5.2 and §5.1.2.

### 5.1 A threshold is meaningless until you measure what the score does

**The CSBG veto floor.** Set to 0.15 by reasoning about the score scale. Connecting
the scorer to fusion for the first time showed it could never fire: the logistic
squash compresses hard past an LLR of −2, so a maximally-contradicting impostor
scores 0.29, not near zero. The floor sat *below the impostor mode*. It hid because
the fusion tests use hand-written branch scores and the scorer tests never look at
fusion — the constant lived in the module that could not check it. Now 0.35, with
`TestVetoCalibration` asserting it against what the real scorer emits.

**The integrity floor, independently.** Set to 0.55 by the same kind of reasoning.
Measured over 80 clean and 80 spliced recordings:

| floor | FRR | detection |
|---|---|---|
| 0.25 | 0.00% | 82.5% |
| 0.55 | 5.00% | 82.5% |

Identical detection, and one genuine speaker in twenty locked out by a gate no
other branch can overturn. The evidence is bimodal, so raising the floor across the
empty middle buys nothing and only reaches into the clean tail.

**The generalisable point:** a threshold expressed in units of a score is only
meaningful once you have measured what that score actually does. Two correct
modules with a constant passed between them is exactly where this hides.

### 5.2 A grid search will walk onto a knife edge

`calibrate_floor` sometimes returns exactly 0.200 — and the detectors emit fixed
evidence weights for categorical findings (0.9 for inserted digital silence, 0.8 for
a hard cut away from a pause), so those probes score exactly 0.10 and 0.20. The
comparison is `score < floor`, so a floor of exactly 0.20 catches *none* of the
orphan-click cases while looking optimal to the search. The floor must sit strictly
inside the gap between the highest categorical level (0.20) and the lowest clean
score (0.274).

### 5.3 Polishing a splice makes it easier to detect

| attacker | detection | d' |
|---|---|---|
| naive splice, cross-session | 82.5% | 2.60 |
| naive splice, same session | 63.7% | 1.66 |
| **careful** splice, cross-session | **88.8%** | **4.97** |

`SpliceConfig.careful()` inserts a room-tone pause so the join sounds natural — and
a pause at the join is precisely what the background test examines. The naive hard
cut has no pause, so nothing but the click detector ever looks at it. **The
attacker's best move against this defence is the crude one.** The number to quote is
the middle row: an attacker who records all their raw material in one sitting leaves
no background step, and a third of their attempts get through.

### 5.4 The A5 tension — the most interesting thing in the attack table

A5 is the style-adaptive attacker, who estimates the victim's CSBG from speech they
could overhear. The A5-observed curve measures *within-speaker consistency*, which
means: **the more reliably someone code-switches, the better the biometric works and
the more cheaply it is stolen.** The speech a defender needs to enrol a usable graph
is the speech an attacker needs to steal one. Written up in §5.1.2 with a suggested
scatter plot.

A5 runs are labelled ORACLE rather than OBSERVED when the eavesdropping budget
exceeds the victim's recorded speech, because an unbinding budget is the upper bound
wearing the observed row's name.

### 5.5 An unattainable operating point is not a rate

`FAR@1%FRR` was printing `100.00` for vetoed configurations — a "no such operating
point" sentinel rendered as a measured rate, sitting next to a baseline's `46.70`.
Anyone reading that table would conclude fusion is catastrophic at a point fusion
cannot occupy. Now `math.nan` → `format_rate()` → `"n/a"`, with the veto count beside
it.

Related: `_rates` anchored its threshold sweep at `min(all) − 1.0`, and with `−inf`
present that is `−inf`, and `−inf >= −inf` is True — so every vetoed trial was
accepted at the bottom of the sweep and the FRR floor vanished. The first fix
(substituting a low finite stand-in) was also wrong, because *any* finite value is
acceptable at a low enough threshold. The correct fix anchors from the lowest
*finite* score and passes `−inf` straight through.

### 5.6 A normalisation that was reported and never happened

`run_ablation` fitted its cohort z-normaliser on `split.dev` and applied it to
`split.test`, and the report said so in a caveat. It covered **zero of the eleven
test speakers.** `split_by_speaker` assigns a trial to dev only when *both* its
speakers are dev speakers, so dev trials only ever claim dev identities, and
`CohortNormaliser.apply` falls back to (mean 0, std 1) for anyone it has not seen.
Every test score passed through unnormalised.

It was invisible because the fallback is indistinguishable from a speaker whose
cohort happens to be centred, and because the one visible effect — a squash scale
of 1.5 instead of 2.0 — is monotone, so **EER did not move at all**. A bug that
changes no metric is not caught by checking metrics.

The fix uses the trials `split_by_speaker` was already discarding. A cross-boundary
trial pairing a *test speaker's model* with a *dev speaker's probe* uses no test
probe and no test label, which is exactly what Z-norm has always meant: statistics
computed at enrolment against a development cohort. `Split.cohort` keeps them and
`cohort_fitting_trials` selects the safe half — the other half, a dev model against
a test probe, is still excluded, because dev statistics rescale the dev trials that
the weights, threshold and veto floor are all fitted on.

Effect on the table: `+ CSBG only` moved 28.48% → 23.18% EER and full-fusion minDCF
0.696 → 0.552. **The generalisable point: a dictionary lookup with a default cannot
report that it missed.** Assert coverage, or the default becomes the measurement.

### 5.7 An ablation measured on the wrong system reads zero, and the zero lies

Every CSBG scoring ablation — class set, transition stream, graph metrics, z-norm —
came out at exactly `+0.00` against the full system. Not because the components do
nothing: because the knowledge branch separates far better and pins the fused EER,
so nothing done to the CSBG can move it. The delta was also quantised to the width
of one genuine trial (1.5 points at n=66), wider than the effects being looked for.

A table reading "transition stream removed: +0.00" is read as "the transition stream
is worthless". Scoped to the branch it actually changes, the same ablations read
−1.82, −0.30, +0.00 and +3.48 — the lexical stream is carrying the CSBG, which is
the finding. `AblationRow.scope` now records what each EER is an EER *of*, and
`to_markdown` prints it in its own column with a warning that rows in different
scopes are not comparable.

### 5.8 Three bugs the mock could not show

The frontend had never been rendered against the backend. Running it found three
failures, all of which are invisible under `VITE_USE_MOCK=true` because the mock
always returns a populated, well-formed payload:

1. **`health.models[0].toUpperCase()` crashed the entire application.** `models` is
   empty in degraded mode — the documented first-class state (§3.7) and what anyone
   gets from `requirements-core.txt`. The throw took `AppLayout` down, so *every*
   route rendered blank. The mock hard-codes three model names.
2. **The Graph Explorer defaulted to the mock id `spk_001`.** No real backend issues
   it, so the CSBG fetch 404'd and cytoscape never initialised. The `<select>` still
   *showed* a speaker name, because a value matching no option makes the browser
   display the first one — a blank graph under an apparently-chosen speaker.
3. **The Overview printed fabricated figures.** `System EER 3.12%` and `Attack
   Rejection 98.4%` were hardcoded, as were deltas including `380 success` sitting
   beside a real trial count of zero. Derived from the API, the same panel reads
   `n/a` and `40.2%`.

The third is the one that matters most for this project. The repository is arranged
end to end so an unearned number cannot pass as a measured one — `paper_ready()`
refuses rows, `format_rate` renders unattainable points as `n/a`, `provenance`
travels into the LaTeX — and the dashboard was the single place that convention was
not kept. It is also the surface most likely to be screenshotted into a progress
report.

**The generalisable point: a mock is a design preview, and a design preview has no
failure modes.** Run the UI against a *degraded* backend, not merely a healthy one.

---

## 6. Session history

### Session 1 — proposal and literature
Wrote `KAVACH_Project_Idea.md`. A literature search forced a revision of the novelty
claims (commit `13dc4fd`): code-switching as a *soft biometric under adversarial
conditions* is the defensible framing, not code-switching detection generally.

### Session 2 — the research core
`csbg/` (ontology, tokens, graph, scoring, metrics) and word-level LID with semantic
tagging. Commits `b3fcfe5`, `62b31f7`.

### Session 3 — audio, knowledge, decisions
`audio.py`, `asr.py`, `embedding.py`, `skg.py`, `matcher.py`, `challenge.py`,
`fusion.py`. Commit `f511d1f`.

### Session 4 — the attack suite
A1–A5, `attacks/`, and the veto path in fusion. This is where the CSBG veto floor
was found to be uncalibrated. Commits `e6b273c`, `50e478e`.

### Session 5 — the API
`api/` — all 19 endpoints transcribed from the UI's `client.ts` rather than the
reverse, since the UI is a given. Found and closed the `expectedAnswerEntity` leak.
Commit `ce34c77`, 380 tests.

### Session 6 — the offline evaluation
`eval/ablation.py`: all-pairs trials, split by speaker, weights → threshold → veto
fitted on dev only, bootstrap intervals, ablations, stability, fairness slices.
Found the two reporting bugs in §5.5. Commit `3a68d8d`, 413 tests.

Representative output on a 24-speaker **simulated** corpus:

```
Split: dev 10 speakers / 800 trials | test 14 speakers / 1568 trials
weights   : speaker_embedding=0.369, csbg=0.309, knowledge=0.322  (logistic regression on dev)
threshold : 0.5919 (dev EER operating point)
veto      : 0.450 on csbg -- removes 2.36% FAR, costs 1.25% FRR

| Configuration | EER % [95% CI]      | minDCF | FAR@1%FRR | FRR@1%FAR | n gen/imp             |
| ECAPA alone   | 21.98 [19.71-30.36] | 0.9286 | 46.70     | 85.71     | 112/1456              |
| CSBG alone    | 28.37 [23.21-32.49] | 0.9464 | n/a       | 86.61     | 112/1456 (373 vetoed) |
| + Knowledge   |  5.49 [3.57-6.56]   | 0.3901 | 5.49      | 28.57     | 112/1456              |
| + CSBG only   | 22.39 [18.75-26.79] | 0.9011 | n/a       | 83.04     | 112/1456 (373 vetoed) |
| Full fusion   |  3.71 [2.88-6.25]   | 0.3647 | n/a       | 21.43     | 112/1456 (373 vetoed) |
```

**Not a result.** Simulated speakers differ by construction.

### Session 7 — models, integrity, and an honest test environment (this session)

1. **Installed the heavy models.** speechbrain 1.1.0, faster-whisper 1.2.1,
   librosa 0.11.0. ECAPA-TDNN verified: 192-d embeddings, 0.2 s load from cache.
   The acoustic branch is now real rather than modelled.

2. **Built the signal-integrity gate** (`integrity.py`), wired it into
   `pipeline.verify()` before the models run, into `fusion.fuse()` as a second gate,
   and into the Attack Lab where A1 and A2 are now built as **real waveforms** from
   the victim's stored recordings and scored by the real detector. That column needs
   no simulation caveat. The Attack Lab also computes its own ablation every run —
   the same trials scored with the gate off — because a flat row of zeros does not
   say *why* it is flat.

3. **Calibrated the integrity floor empirically** (§5.1, §5.2, §5.3) and wrote
   `tests/test_integrity.py`, 21 tests whose entire purpose is the seam between the
   detectors and the constant that thresholds them.

4. **Found that the test suite was passing by accident.** It ran in ~48 s and never
   touched the network *because speechbrain and faster-whisper were not installed*.
   Nothing in the tests asked for that. Installing the models turned it into a run
   that tried to download a 3 GB Whisper checkpoint and never finished. Added
   `Settings.offline` and `tests/conftest.py` so degraded mode is declared rather
   than inherited, with `@pytest.mark.models` as the opt-in for real checkpoints.

5. **Fixed a phantom-field bug in the frontend contract test.** The TypeScript
   interface parser did not strip `//` comments, so a comment reading "gates, not
   weighted factors: they carry weight 0" injected a field named `factors`. Prose is
   not syntax.

6. **Fixed a latent flake in the challenge-leak test.** It asserted
   `expected_answer not in str(public_dict())`, and the dict carries Unix
   timestamps while an SKG fixture fact is the room number `"214"` — so the suite
   failed roughly once in several full runs on a collision with the clock. See
   HANDOFF.md trap 5.

Test count: **434 passing**, offline, in about 60 seconds, verified across three
consecutive full runs.

### Session 8 — the corpus layer, the figures, the runner, and the first UI run

1. **Corpus layer** (`corpus.py`, 61 tests). Manifest schema, loader, validation,
   consent enforcement, coverage report, and the elicitation protocol as code —
   `PROTOCOL_V1` asserts at import that every `ELICITABLE_CLASS` has a prompt.
   `split_sessions` splits enrolment from probe speech **by session, never by
   utterance**, the same refusal `split_by_speaker` makes one level up.
   `from_simulation` bridges the simulator into the same interface so there is one
   code path, stamped `SIMULATED` so it cannot be reported by accident.

2. **Recording protocol** ([RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md)). Consent
   first, 25–30 speakers, ≥2 sessions weeks apart, the 14 bilingual prompts, and the
   pilot checks to run before recording the other 25.

3. **Figures** (`eval/figures.py`, 41 tests). Six publication figures, greyscale-safe
   by construction: identity is carried by line style and marker, and a test asserts
   the palette is achromatic. The CSBG heatmap hatches cells whose token count sits
   below the smoothing pseudo-count, because a class at the backoff prior shows the
   population's habit, not the speaker's.

4. **Experiment runner** (`experiments.py`, 31 tests). One command emits
   `results.json`, `report.md`, `tables/*.tex` and `figures/*.pdf`, with the git
   commit, seeds and package versions recorded. Provenance travels into the LaTeX as
   a visible banner, because a `.tex` that looks identical whether it came from
   recordings or from `simulation.py` is how a simulated number reaches a submission.

5. **Scoring ablations and the stability curve** wired into `run_ablation`
   (§5.7), and **per-speaker IAPMR** exposed at `GET /api/attacks/per-speaker` and
   rendered in the Attack Lab — with unattacked speakers listed rather than omitted,
   since an unmeasured speaker is not a protected one.

6. **Found that cohort z-normalisation was never happening on test** (§5.6). The
   fix moved `+ CSBG only` from 28.48% to 23.18% EER.

7. **Ran the frontend against the backend for the first time** and found three bugs
   the mock cannot show (§5.8), one of which blanked every page in the application.
   All eight pages now render against a live degraded backend; `tsc --noEmit` clean.

8. **Fixed a duplicate-key crash in the Attack Lab.** `run_attack` minted its run id
   from the seeded RNG that makes its scores reproducible, so attacking a second
   speaker with the same attack type violated the primary key and returned a 500.

Test count: **600 passing**, offline, in about 36 seconds.

Also discovered: the project requires **Python 3.10+** and fails at collection on
3.9 (`dataclass(slots=True)`). Nothing recorded that before, and the machine this
session ran on had only the system 3.9. HANDOFF.md now says so.

### Session 9 — real recordings, annotation, the demo build (August to 2026-09-30)

Four, then seven, then twelve consenting speakers returned recordings (S01-S07 reading scripts, S08-S12 speaking
freely). Transcribed with Whisper `large-v3`, tagged with Gemini, run through `kavach.experiments`: ECAPA 0.00% EER
(same sitting, not a result), CSBG 26.3% on scripts (a script classifier) and 50.0% on free speech. Then, with data
collection closed, a demo build on the existing 12 speakers: `seed_demo`, a reworked UI, live ASR on a fast path,
an offline tagging lexicon, hallucinated-script filtering, a microphone top-up, stored-audio attack staging. The
splice gate was found to reject 167 of 168 genuine clips. Details: HANDOFF, the 2026-09-29 block.

### Session 10 — the clone-attack bank, a preflight, and what rehearsal found (2026-10-02)

kNN-VC clone bank behind an allowlist and a flag, measured Attack Lab rows (still `simulated`), `kavach.demo_check`,
splice calibration (the tests stay off), fixes found by rehearsing against the real backend (the impostor clip
tripping the integrity gate first, a real `.m4a` failing to decode), and an independent review pass. Merged to
`main`. Details: HANDOFF, the 2026-10-02 blocks; the spec and plan are in `docs/superpowers/`.

### Session 11 — the evidence pipeline (2026-10-02 to 2026-10-03)

The presenter's offer to record unlimited samples of themselves became the Recording Studio (`kavach/studio/`, the
`/studio` page, an append-only index) and the single-enrollee evaluation (`kavach.eval.enrollee`), reviewed by a
fresh reviewer and fixed (interval honesty, a degenerate dev split, a refresh that moved a session). Branch
`feature/evidence-pipeline`. Details: HANDOFF, the 2026-10-03 block; spec and plan in `docs/superpowers/`.

### Session 12 — demo hardening: the pivot (2026-10-03)

An autonomous pass with design decisions delegated. Found that the login had no hard voice gate and that a replay
passed; rebuilt it as a gated login (sections 3.9-3.12), added the Unlock screen, throttle, step-up, measured
threshold, Studio `words` task, enrolment from Studio sessions, preflight phrase flows, and the documentation rule
(CLAUDE.md). Found and fixed one real bug by self-review (the generic fused threshold leaking into phrase logins).
Eleven rulings with their costs, and the measurements, are in DEMO_HARDENING.md. Branch `feature/demo-hardening`.

---

## 7. Running it

```bash
# Backend — core only, no heavy models, runs in degraded mode
pip install -r requirements-core.txt
pytest                                   # 1490 tests, offline, ~2-5 min

# Backend — everything
pip install -r requirements.txt
pytest -m models                         # the tests that load real checkpoints

uvicorn kavach.api.app:app --reload --port 8000 --app-dir backend

# Frontend
cd kavach
cp .env.example .env.local               # VITE_USE_MOCK=false
npm install
npm run dev                              # http://localhost:3000
```

`/api/health` reports which branches can run and what each absence costs.

Useful environment variables (all prefixed `KAVACH_`):

| Variable | Effect |
|---|---|
| `KAVACH_OFFLINE=1` | Load no model checkpoints; exercise degraded mode deliberately |
| `KAVACH_WHISPER_MODEL=small` | Smaller ASR checkpoint (default `large-v3`, ~3 GB) |
| `KAVACH_DEMO_REVEAL_ANSWERS=1` | Populate `expectedAnswerEntity`. **Demos only.** |
| `KAVACH_VOICE_GATE` / `KAVACH_PHRASE_WORDS` / `KAVACH_PHRASE_MIN_MATCH` | The voice gate (default on), the words shown (10) and the share needed (0.6) |
| `KAVACH_THROTTLE_ENABLED` / `_FREE_ATTEMPTS` / `_BASE_DELAY` / `_MAX_DELAY` | Failed attempts slow down (3 free, then 5/10/20/30 s) |
| `KAVACH_DEMO_TOOLS=true` | Expose `POST /api/demo/reset-throttle` for the preflight. Demo builds only |
| `KAVACH_STUDIO_ENABLED` / `KAVACH_STUDIO_SPEAKERS` | The recording Studio, allowlist-only; set by `run_studio.ps1` |
| `ANTHROPIC_API_KEY` | Enables LLM tagging, challenges and attacker text; adds prompt caching and the Batch API |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | The same three jobs on a free tier. Any one key is sufficient |
| `KAVACH_LLM_PROVIDER` | Pin one provider when several keys are present (default: first available, Anthropic preferred) |

---

## 8. Reading order for someone new

1. `KAVACH_Project_Idea.md` — the research proposal, the novelty argument, §4.5.2
   and §5.1.2 for the findings above.
2. `backend/kavach/csbg/ontology.py` — the 21 classes and why each is or is not
   elicitable. Everything else reads these.
3. `backend/kavach/csbg/scoring.py` — the likelihood ratio, which is the method.
4. `backend/kavach/fusion.py` — the module docstring is the argument for the veto.
5. `backend/kavach/integrity.py` — the argument for the second gate, and the
   calibration story.
6. `backend/kavach/eval/ablation.py` — how the paper's table is produced.
7. `DEMO_HARDENING.md`, then `backend/kavach/phrase.py` and `calibrate_voice.py` — the live login and how its
   threshold is measured. `CLAUDE.md` first, if you are an assistant picking this up.

Module docstrings carry the reasoning throughout. Where a number was chosen rather
than measured, the docstring says so.
