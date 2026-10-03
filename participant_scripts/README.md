# Participant scripts

> **Status, 2026-10-03: pilot closed.** The scripts were used for the pilot corpus (S01-S07) and are kept as the record of what participants read. `SPEAKER_A.md` ... `SPEAKER_J.md` are the verbatim forwarded files and are never edited.


Read-speech scripts, one per speaker. `SPEAKER_A.md` … `SPEAKER_J.md` are
forwarded to participants verbatim; this file is the design behind them and is
not sent to anyone.

## Why these exist, and what they cost

[PARTICIPANT_SHEET.md](../PARTICIPANT_SHEET.md) asks 14 questions and collects
spontaneous answers. That is the corpus the hypothesis needs. It is also slow:
people have to think, they answer in eight seconds, and half of them never
return the files.

These scripts trade that away for turnaround. The participant reads; nothing is
elicited. **What comes back is real human speech with authored content.**

The consequence is not negotiable and must not be blurred:

> A CSBG built from read speech recovers the profile *we wrote*. It is not
> evidence that Tamil–English speakers have stable code-switching habits.

That is the §5.1 claim, and read speech cannot support it. What read speech
*can* support:

- **Word-level LID accuracy on code-mixed Tamil–English speech.** We authored
  every token, so the language label of every word is known. That is a labelled
  evaluation set, and it is the one thing here that is a contribution in its own
  right — SPELLL Track-1 is a language-resources track.
- **Does the CSBG recover a known profile through the real pipeline?** End to
  end, through a real microphone, real ASR and real LID, on real voices. If it
  cannot recover a profile we planted, it will certainly not find one we did
  not. This is a system-validation result and a genuine negative control.
- **The acoustic and integrity branches**, which do not care that the words were
  authored. ECAPA embeddings, replay and splice detection are unaffected.

Label it. `Provenance.SCRIPTED` exists for this, `is_reportable` is False, and
`Corpus.reportability()` names it. Nothing about a scripted run may appear in a
table as a §5.1 result.

## If both corpora exist, this one is the control

Spontaneous speech from even five speakers plus scripted speech from ten is a
better paper than either alone: the scripted set bounds what the pipeline can do
when the profile is known, and the spontaneous set shows what is left when it
is not. Run the spontaneous protocol in parallel if any participants will sit
for it.

## Design

Every script answers the same 14 prompts of `corpus.PROTOCOL_V1` in the same
order, so filenames `01`–`14` map to `p01_family`–`p14_control_name` exactly as
they do for spontaneous collection.

**Scripts must differ from each other in language choice, not in topic.** If all
ten read one script every CSBG is identical, every LLR is zero and the EER is
50% — the experiment cannot return a result. So each speaker gets a per-class
language assignment, and the scripts are written to it.

Assignments are **linguistically plausible, not random**. Nobody says English
for RELIGION_FESTIVAL and Tamil for TECH_DIGITAL; a profile drawn from a hat
produces speech no annotator would believe and a corpus a reviewer will
disbelieve. The axes below are ones that genuinely vary between Tamil–English
bilinguals.

### Per-speaker language assignment

`T` = Tamil, `E` = English, `M` = mixed within the class.

| Class | A | B | C | D | E | F | G | H | I | J |
|---|---|---|---|---|---|---|---|---|---|---|
| Matrix language | T | E | T | E | T | T | E | T | E | T |
| KINSHIP | T | E | T | T | T | T | E | T | M | T |
| FOOD | T | T | T | T | T | T | M | T | T | T |
| NUMBER | E | E | T | E | T | E | E | T | E | M |
| TIME_DATE | E | E | T | E | M | E | E | T | E | T |
| MONEY_COMMERCE | M | E | T | E | T | M | E | T | E | T |
| QUANTITY_MEASURE | T | E | T | M | T | T | E | T | M | T |
| TECH_DIGITAL | E | E | E | E | E | E | E | M | E | E |
| EDU_WORK | E | E | M | E | E | T | E | E | E | M |
| PLACE_LOCAL | T | T | T | T | T | T | M | T | T | T |
| PLACE_GLOBAL | E | E | E | E | E | E | E | E | E | E |
| TRANSPORT | M | E | T | E | T | T | E | T | M | T |
| RELIGION_FESTIVAL | T | T | T | T | T | T | T | T | T | T |
| MEDIA_ENTERTAIN | E | E | M | E | E | E | E | E | M | E |
| EMOTION_STATE | T | E | T | M | T | T | E | T | T | M |
| BODY_HEALTH | T | E | T | T | T | T | M | T | M | T |

Three columns are deliberately near-invariant across speakers —
RELIGION_FESTIVAL, PLACE_GLOBAL and FOOD sit at one language for almost
everyone, because they do in life. They are the classes `starved_classes()`
would flag as carrying no biometric signal, and their presence is the honest
result: not every class discriminates. A design where all sixteen separate
cleanly would be a design that had assumed its own conclusion.

**A and G are the widest-separated pair; A and E the closest.** A and E differ
only on NUMBER, TIME_DATE and MONEY_COMMERCE — the hard case, and the one worth
reporting.

### Writing rules

- 70–90 words per answer (~30 s read aloud). Prompt 10 ~40 words, prompt 14 one
  line.
- Romanised Tamil, not Tamil script. Participants read it aloud, so it only has
  to be pronounceable; the ASR transcribes the *audio* and never sees this file.
- Content is invented. No participant supplies a real name, hometown or date,
  which removes the personal-data collection entirely — consent for a scripted
  session is materially lighter than for the spontaneous protocol, and the SKG
  facts come from the script rather than from a person.
- Verbs and function words follow the matrix language. Tamil-matrix speakers
  take English *nouns*, not English verbs — "college la poven", never "I went to
  college la". Getting this wrong is what makes synthetic code-mixing sound
  synthetic.
- Vary the invented details across speakers too: different hometowns, prices,
  wake-up times. Identical content with different language assignments lets the
  knowledge branch separate speakers by topic and quietly inflates fusion.

### Consent lives in the covering message, not in the script

The script files are stripped to the recording rules and the fourteen answers,
so participants read rather than skim. Consent is therefore **not** in them and
must be sent alongside — the wording is in
[../RECORDING_PROTOCOL.md](../RECORDING_PROTOCOL.md). A scripted session collects
no personal data (every fact is invented), so the ask is narrower than the
spontaneous protocol's: record + voice model + research use, and a second
session in two weeks. It is still required. `Corpus.validate()` rejects a
manifest whose speakers have no `consent_ref`, and the reply that grants it is
what that field points at.

### Ground truth

Because every token was authored, the intended language of each word is known.
`ingest.py --scripted` reads these files as the reference transcript and emits
`AnnotationSource.SYNTHETIC` token labels, which is what makes the LID
evaluation possible. The label is the *intended* language: a participant who
substitutes a word breaks that alignment, which is why the scripts ask for
substitutions to preserve language even when they change the word.
