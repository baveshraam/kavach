# Recording protocol

> **Status, 2026-10-03.** The pilot corpus is final at 12 consenting speakers (S01-S12); collection from other people is closed unless the presenter says otherwise. This protocol remains the consent and recording standard for any future collection, and `corpus.py` its executable half. The presenter's own sessions follow `RECORDING_GUIDE.md` instead (one consenting speaker, recorded through the Studio).


What to record, from whom, how, and what must be signed first. This is the
document the corpus is collected against; `backend/kavach/corpus.py` is its
executable half, and where the two disagree the code is authoritative because
it is the thing that will be run.

**Read [§1 Consent](#1-consent-comes-first) before contacting a single
participant.** Everything else here can be revised after the pilot. Consent
cannot be applied retroactively, and speech recorded without it is unusable no
matter how good it is.

---

## 0. What this corpus has to support

Four results depend on how the recordings are structured, and each imposes a
constraint that is expensive to retrofit:

| Result | Constraint it imposes |
|---|---|
| EER with usable intervals (§5.2) | ≥ 25 speakers |
| Cross-session stability (§5.3) | ≥ 2 sessions per speaker, **weeks apart** |
| Enrolment-budget curve (§5.3, §5.1.2) | ≥ 5 minutes of speech per speaker |
| Fairness audit (§5.5) | device, environment and dominant-language recorded, with ≥ 2 populated groups per condition |

The one that gets skipped under time pressure is the second session, and it is
the one that cannot be recovered: a second sitting three weeks later is not
reproducible by recording longer today. `Corpus.reportability()` names every
single-session speaker for this reason.

---

## 1. Consent comes first

This corpus pairs **voiceprints with personal facts** — hometowns, family
names, schools. That combination is the input to the Speaker Knowledge Graph
and it is exactly what makes the data sensitive. Treat it as biometric
personal data throughout.

**Before any recording:**

- Obtain institutional ethics approval. Start this in week 1; it is the
  longest-latency item in the whole project.
- Every participant signs a consent form covering, explicitly and separately:
  - what is recorded (voice, and personal facts used to generate challenges);
  - that a **voice model** is derived and retained, not only audio;
  - who may access it, and whether the corpus will be **publicly released**;
  - the right to withdraw, and what withdrawal deletes;
  - that recordings will be used to build **attack samples** (replays,
    splices, and — if in scope — synthetic clones of their voice). This one is
    routinely omitted and is the one most likely to be objected to. Ask
    separately; a participant may consent to the corpus and refuse the clone.

**Record the consent reference, not the person.** `SpeakerRecord.consent_ref`
points into a register held **outside this repository and outside `data/`**.
`speaker_id` is a pseudonym (`spk_001`). Nothing in the manifest is a direct
identifier.

`Corpus.validate()` fails a RECORDED manifest whose speakers lack a
`consent_ref`. That check is not paperwork hygiene — it is the only thing
standing between a mislaid spreadsheet and an unusable corpus.

**Withdrawal must be executable.** A participant who withdraws requires their
audio, tokens, facts, graphs and history to be erased —
`api.store.Store.delete_speaker` is the implementation, and a deletion that
leaves recordings behind has not been honoured.

---

## 2. Participants

**Target: 25–30 speakers.** Below ~20 the EER confidence intervals are too
wide to claim anything, and the usable trial count falls faster than the
speaker count suggests: `eval.ablation.split_by_speaker` **discards** trials
straddling the dev/test boundary, roughly half the impostor trials at
`dev_fraction=0.4`.

Eligibility: fluent in **both** Tamil and English, and a habitual code-mixer.
A speaker who genuinely does not code-switch contributes a degenerate graph.
Screen for it in the pre-session questionnaire rather than discovering it
during annotation.

Recruit for **variance in the fairness conditions**, not a representative
sample — the audit needs ≥ 2 populated groups per condition, and a corpus of
28 speakers who all used the same phone in the same room can report no device
slice at all:

| Attribute | Field | Aim for |
|---|---|---|
| Dominant language | `SpeakerRecord.dominant_language` | TA / EN / balanced all present |
| Gender | `SpeakerRecord.gender` | Not overwhelmingly one |
| Age band | `SpeakerRecord.age_band` | Recorded; skew is acceptable if stated |
| Device | `SessionRecord.device` | At least two distinct devices |
| Environment | `SessionRecord.environment` | Quiet room plus one noisier condition |

Self-report dominant language with a concrete question ("which language do you
think in when you are counting money?"), not a scale.

---

## 3. Sessions

**Two sessions per speaker, minimum, separated by two weeks or more.**

The gap is the measurement. §5.3 asks whether a CSBG holds up when recorded
weeks apart or drifts with topic and mood, and two sittings on the same
afternoon cannot answer it — they share a microphone, a room, a mood and
usually a topic. `corpus.split_sessions` holds out the most recent session as
probe speech precisely so the reported EER is a cross-session number.

Record `SessionRecord.recorded_on` for every session. The paper has to state
the median gap, and a corpus that cannot report it cannot make the §5.3 claim.

| | Session 1 | Session 2 |
|---|---|---|
| Purpose | Enrolment | Probe / cross-session |
| Duration | ~5 min speech | ~3 min speech |
| Prompts | All 14 | All 14, same order |
| Also collected | SKG interview, questionnaire | Nothing extra |

Keep the prompt set identical across sessions. Changing prompts between
sessions confounds session drift with topic change, and then the §5.3 result
measures the prompt list.

### Recording settings

- **16 kHz or higher, mono, 16-bit PCM WAV.** No lossy codecs at capture:
  `integrity.py` looks for edit artefacts, and codec artefacts sit in the same
  frequency neighbourhood. Compress derived copies, never the master.
- One file per prompt answer, not one per session. The manifest is indexed by
  utterance and splicing them apart afterwards costs an alignment pass.
- Note the device honestly ("Redmi Note 12 built-in mic"), not "phone".
- Leave ~1 s of room tone at the start of each file. It is the reference the
  background-continuity test uses, and it costs nothing to capture.

---

## 4. Prompts

The 14 prompts are `corpus.PROTOCOL_V1`, bilingual, and cover every class in
`ontology.ELICITABLE_CLASSES`. That coverage is asserted at import time — the
module refuses to load if a prompt is deleted and leaves a class unelicited.

**The rule the prompts are written to.** From §5.1.1:

> The knowledge branch checks the fact. The CSBG checks the wrapper around it.
> The fact is stealable; the wrapper is the biometric.

Nobody answers a question with a bare noun. Asked their mother's name, a
speaker says *"en amma peru Lakshmi"* or *"my mother's name is Lakshmi"* — and
which of those they reliably say is what the CSBG measures. The attacker knows
the name; they do not know the sentence.

So: **every prompt must elicit a phrase.** A prompt answerable in one word
gives the CSBG nothing to read, and its null result is a failure of the
question, not of the hypothesis. `p14_control_name` is deliberately one-word —
it is the control that demonstrates exactly this, and it is flagged
`elicits_phrase=False` so it never appears in the broken-prompt list.

### Running a prompt

- Ask in whichever language the participant greeted you in, then stop talking.
- **Do not model the answer.** If the interviewer asks in English, the
  participant answers in English, and the corpus measures the interviewer.
  Where practical, use the same interviewer for every session.
- If an answer is under ~5 seconds, prompt once with "tell me a bit more" —
  `build_trials` marks the CSBG branch unavailable below `min_scored_tokens=5`,
  so a very short answer is an unscoreable trial.
- Do not correct language choice. Ever. The thing being measured is the thing
  you would be correcting.

### Self-recorded collection

[PARTICIPANT_SHEET.md](PARTICIPANT_SHEET.md) is the forwardable version of this
section: consent block, recording instructions, all 14 prompts bilingually, the
SKG questions and return instructions, written for someone who has never heard
of the project. Send it as-is. Everything in it is deliberate:

- **The prompt numbers `01`–`14` are the ordinal positions of `PROTOCOL_V1`**, so
  a participant's `07.m4a` is `p07_festival` without anyone consulting a table.
  `ingest.py` resolves numeric filenames through that order. Reordering
  `PROTOCOL_V1` silently remaps every file already collected — don't.
- **`p10_numbers` is expanded** into three concrete asks (birth year, idli price,
  wake-up time). The terse `text_en` works when an interviewer is present to
  clarify; alone with a phone it produces three bare numerals, and the class
  needs them wrapped in sentences.
- **The sheet says what is collected, not what is measured.** "Tamil–English
  speech and voice-based login" is true and sufficient for consent. Naming the
  hypothesis — that per-topic language choice identifies a speaker — makes
  participants monitor exactly the behaviour under test. Debrief afterwards.
- **Prompts are given in both languages with an explicit instruction not to
  match the language they read in.** This substitutes for §4's "ask in whichever
  language they greeted you in": with no interviewer, the *sheet* is the thing
  that models the answer, and an English-only sheet buys an English-only corpus.
- **The language instruction is a permission, never a quota.** "Speak the way
  you would to a friend" is the whole of it. "Please mix Tamil and English"
  reads as a task, and a participant performing a mixture to order produces the
  mixture they think is wanted — which is roughly the same mixture for
  everybody. The between-speaker spread *is* the measurement, so an instruction
  that compresses it returns a null result that says nothing about the
  hypothesis. For the same reason nobody is asked to answer twice in two
  languages, and a return that is 90% Tamil or 90% English is a data point, not
  a failed recording: do not ask for it again.
- **Public release is a separate, optional consent line.** Track-1 at SPELLL is a
  language-resources track; releasing the corpus is worth more than the CSBG
  result if the CSBG result is null. Release consent cannot be added
  retroactively, so it is asked at recruitment even though release is undecided.
- **The second session is promised at recruitment**, not sprung two weeks later.
  Attrition between sittings is what kills §5.3, and the cheapest fix is telling
  people up front that there are two.

**No interviewer means no "tell me a bit more."** The sheet compensates with a
per-prompt content hint and a stated floor, but expect self-recorded answers to
run short: check `wrapperless_prompts()` on the first returns, not on all 30.

**Ask for the recorder's native format and forbid conversion.** Phone recorders
mostly emit AAC/`m4a`; that is one lossy generation and `integrity.py` can be
told about it. A participant who helpfully re-exports to "WAV" has stacked a
second generation on top and destroyed the artefact baseline. The instruction to
send via WhatsApp **Document** rather than as a voice note is the same concern —
voice notes are re-encoded to Opus in transit.

### Scripted collection

[participant_scripts/](participant_scripts/) is the other, faster route:
`SPEAKER_A.md`–`SPEAKER_J.md` answer the same 14 prompts with authored text, one
language profile per speaker. What comes back is real speech with invented
content. It is `Provenance.SCRIPTED`, `is_reportable` is False, and the reasoning
is in [participant_scripts/README.md](participant_scripts/README.md) — no §5.1
claim follows from it.

The scripts carry no consent block: they were stripped to the recording rules
and the answers so participants read them rather than skim them. **Send this in
the covering message, and keep the reply — it is what `consent_ref` points at:**

> This is for a final-year speech project. I'll send you a script — you just
> read it out loud into your phone, 14 short recordings, about 15 minutes. None
> of it is about you, it's all made up. Reply **yes** if you agree to be
> recorded, to a voice model being built from your recording, and to both being
> used in the project. Your name is replaced by a code and never appears
> anywhere. There's a second, shorter recording in about two weeks, so only say
> yes if you can do both. You can pull out later and everything derived from
> your voice gets deleted, not just the audio.

A scripted session collects **no personal data** — every fact in the script is
invented, so §5's SKG interview does not apply. The SKG for a scripted speaker
is populated from their own script, which also means the knowledge branch is
answering questions about a person who does not exist. That is fine for A4 (the
attacker still has to hold the answer) and worthless as evidence about real
users.

### What the first four returns taught us

Four participants returned complete sets (56 files, 24 minutes) before any of
this was written down. Everything below is measured, not anticipated.

**Two of the four sent WhatsApp voice notes.** They arrived as Opus at 18 kbps
in an `.ogg`; the other two sent documents and arrived as AAC at 128 and
192 kbps. The instruction to use **Document** is in the sheet and it was still
missed by half the sample, so say it in the covering message too, and ask for
the originals when it happens — the `.m4a` is still on their phone.

It matters more than fidelity. At 16 kHz nothing important is lost: Opus at
18 kbps carries 8.3 kHz of bandwidth, which is the full Nyquist band at the
sample rate everything downstream runs at. The problem is that **codec becomes
a speaker attribute**. Two speakers on Opus and two on AAC gives the acoustic
branch a feature that separates them perfectly and has nothing to do with their
voices, and `integrity.py`'s codec baseline reads the transit encode as if the
recorder had produced it. Ingest resamples everything to 16 kHz mono, which
removes the bandwidth half of the confound; the artefact half survives, so
record `device` honestly and keep the codec split visible in the fairness
slice.

**Nobody left the lead-in silence.** Speech starts between 0.00 s and 0.56 s
across all 56 files, mostly under 0.2 s, and several open mid-syllable.
`ingest.py` flags it and `audio.leading_silence_sec` measures it. Say it as an
action rather than a duration: *press record, count one-two, then start.*

**SNR ran 18–25 dB on the AAC returns.** Usable, and the honest ceiling for
phone recordings in a student room. Not worth re-recording for; worth stating
in the paper rather than implying a studio.

**Numerals came back as digits, and that was a bug in this repository, not in
the recordings.** `Settings.suppress_numerals` had always passed
faster-whisper's own default token set, which suppresses symbols and not
digits, so prompt 10 — the prompt whose only job is to elicit a year, a price
and a time — transcribed as `2004`, `10 Rs.` and `6.45`. A digit is
language-neutral: it records that a speaker said a number and destroys which
language they said it in, which is the entire content of the NUMBER class. Now
fixed in `WhisperASR._numeral_tokens`, and the same audio transcribes as
`two-thousand and four`, `ten rupees`, `six-forty-five`. **Any annotation run
before this fix must be redone** — the affected classes are NUMBER, TIME_DATE
and MONEY_COMMERCE, which is three of the sixteen the CSBG scores.

**Whisper `small` is not adequate for annotation.** On monolingual stretches it
is fine. At code-switch boundaries it emits fragments of unrelated languages —
Vietnamese and Portuguese words inside a Tamil–English answer about a college
timetable. Use `large-v3` for corpus annotation; a hallucinated token is worse
than a missing one because it enters a graph as a real observation. Auto-detect
(`whisper_language=None`) is right: it returns `ta` and keeps English in Latin
script, which is what lets `lid.rules` resolve those tokens without an LLM call.

### After the pilot, before full collection

Run `Corpus.coverage()` on the pilot recordings and read two lists:

- **`starved_classes()`** — elicitable classes no speaker has enough tokens
  for. These sit at the backoff prior for everyone, identical across speakers,
  carrying no biometric signal. Fix the prompt.
- **`wrapperless_prompts()`** — phrase prompts whose mean answer is too short
  to score. Fix the question.

Both are cheap to fix at 5 speakers and not at 30. This check is the entire
reason to run a pilot.

---

## 5. Speaker Knowledge Graph interview

Collected in session 1, alongside the prompts. Around 15 facts per speaker
across the categories `skg.py` models — family, education, places, routine.

Ask for facts the speaker would recognise instantly and a stranger could
plausibly find: that combination is the threat model. A fact nobody could
discover makes the knowledge branch look stronger than it is, and A4 —
clone plus scraped answer, the attack the CSBG exists to stop — becomes
untestable because no realistic attacker could ever hold the answer.

These facts are the most sensitive thing collected. They live in `data/`,
git-ignored in its entirety, and they are covered by the same consent and the
same deletion path.

---

## 6. Annotation

ASR (`asr.py`) → word-level LID + semantic tagging (`lid/`) → `Token` stream.

**The tagging quality bounds every downstream number.** Without an LLM pass,
`lid.rules` resolves Tamil script definitionally but *guesses* unresolved Latin
tokens, and `PipelineStats.is_corpus_grade` refuses to certify a run
containing guesses. `Corpus.reportability()` applies the same rule, counting
`UtteranceRecord.n_guessed_tokens`. A corpus annotated by rules alone is not
publishable, and the code says so rather than leaving it to a reviewer.

**Hand-annotate a validation slice** — 2 utterances per speaker is enough —
and store it with `annotation_source=HUMAN`. Report tagger accuracy against
it. An unvalidated tagger makes the CSBG a measurement of the tagger.

`tokens=None` means "not yet annotated" and `tokens=[]` means "annotated,
nothing scoreable". They are different states and the loader keeps them
distinct: an un-annotated utterance raises rather than silently building a
graph at the prior.

---

## 7. Attack material

Recorded in session 2, after the prompts, and covered by the separate consent
clause in §1.

- **A1 replay** — play a session-1 recording back through a real speaker and
  re-record it. A signal-processed pseudo-replay is not a replay, and
  `AttackTable.paper_ready()` refuses simulated rows.
- **A2 splice** — assemble an answer from words the speaker really said.
  Record the raw material in **both** sessions: §5.3 found that a same-session
  splice is *harder* to detect (63.7% vs 82.5%), because a single sitting
  leaves no background step at the join. The same-session number is the
  honest one to quote, so the material has to exist.
- **A3–A5 clones** — only with the separate consent, and only if a cloner is
  in scope. Report **attack yield** beside every clone row: a clone counts as
  a trial only if the acoustic branch accepts it, and "A4: 0/40 accepted"
  means nothing at 5% yield and everything at 90%.

---

## 8. Manifest

One `manifest.json` per corpus, written by `corpus.save_manifest`, sitting at
the corpus root with audio paths relative to it. The tree will be moved; it
holds voiceprints and lives outside version control.

```
data/corpus_v1/
  manifest.json
  audio/spk_001/s0/p01_family.wav
```

Before treating a corpus as final:

```python
from kavach.corpus import load_manifest

corpus = load_manifest("data/corpus_v1/manifest.json")
print(corpus.validate(check_audio=True))   # structural problems; must be []
print(corpus.reportability())              # publishability; must be []
print(corpus.coverage().to_markdown())     # per-class and per-prompt coverage
```

`validate()` catches a **broken** manifest; `reportability()` catches a
**sound but unpublishable** corpus. A corpus can pass the first and fail the
second, and that is the normal state during collection.

---

## 9. Checklist

Before recording:

- [ ] Ethics approval granted
- [ ] Consent form covers voice model retention, public release, and attack generation as a separate clause
- [ ] Consent register exists outside the repository; pseudonym scheme fixed
- [ ] Recording chain tested end to end on 2 clips through ASR and LID
- [ ] Interviewer briefed not to model the answer language

After the pilot (5 speakers), before full collection:

- [ ] `coverage().starved_classes()` empty, or prompts revised
- [ ] `coverage().wrapperless_prompts()` empty, or prompts revised
- [ ] Genuine and impostor CSBG scores separate at all — the week-2 go/no-go
- [ ] Tagger checked against the hand-annotated slice

Before reporting anything:

- [ ] `validate(check_audio=True)` returns `[]`
- [ ] `reportability()` returns `[]`
- [ ] Median inter-session gap computed and stated
- [ ] Every fairness condition has ≥ 2 groups above `min_trials`
