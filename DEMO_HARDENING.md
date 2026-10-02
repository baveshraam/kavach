# Demo hardening: the working record

Started 2026-10-03 ~02:30 IST. Mandate (from the presenter, Bavesh / corpus speaker S04): a demo that
works essentially every time for them and does not work at all for people whose voices are in no data
(the judges), presented like a production product. Design decisions were delegated; each one is logged
below with what it costs if wrong.

This file is the memory of the pass. Read it top to bottom to resume.

## 1. Threat model for the live demo

| # | What a judge (or the room) can do | Before this pass | Now |
|---|---|---|---|
| T1 | Speak their own voice, claiming to be the presenter | rejected only if the *weighted average* fell under 0.55 | voice is a **hard gate**: below the floor is a reject whatever else they say |
| T2 | Know or guess the answer to a personal question (it is said aloud, or on a slide) | voice 0.45 + answer 1.0 + coin-flip CSBG = 0.72, **accepted** | rejected: a correct answer cannot outvote a wrong voice |
| T3 | Record the presenter during the demo, play it back | replay caught only if byte-identical; a re-recorded clip passes voice + knowledge | **random words per attempt**, checked against the ASR transcript: a recording made for another attempt cannot contain them |
| T4 | Hammer retries until a similar voice slips through | unlimited free draws | three free, then 5 / 10 / 20 / 30 s waits; a success clears them |
| T5 | Talk over the presenter / interrupt | unhandled | (planned) second-voice guard |
| T6 | Clone the presenter's voice live | measured offline (clone bank) | unchanged; honest limit, see section 4 |
| T7 | The venue's wifi is down | tagger/LLM unreachable: slow, CSBG unmeasured | the phrase login needs no network at all |
| T8 | The presenter is rejected: nerves, room, mic | untested | being measured: see section 3 |

## 2. Decision log (rulings)

- **R1. The voiceprint becomes a hard gate in the live system (`FusionPolicy.voice_gate`, on in `Settings`).**
  Why: the module's own docstring argues a weighted average cannot express a veto and fixes it for the
  CSBG, but the identity factor itself was an average term. Three zones on the voice score: pass at the
  threshold, grey band (`voice_grey_margin`, default 0.08) is at best BORDERLINE (step-up), below that
  REJECT; an unmeasurable voice fails closed. Left off in `FusionPolicy()` so the paper's ablations still
  measure the weighted rule. Cost if wrong: a genuine voice that lands under the floor is rejected;
  mitigated by retries (T4's throttle is mild) and by calibrating the floor on held-out recordings.
- **R2. New login mode `kind="phrase"`: read six random words.** Why: it is the user's own mental model
  ("read a sentence"), needs no facts and no LLM, and the only cheap defence against re-recorded playback.
  Transcribed as English with the code-mixing prompt removed (that prompt biases English words into Tamil
  script). Gate semantics: wrong words reject whatever the voice; no ASR rejects (fail closed). The
  personal-question login stays, as the stronger step-up. Cost if wrong: Whisper mis-hears an accented
  word; one dropped word is forgiven (4 of 6 needed).
- **R3. Throttle is mild and capped (`kavach.throttle`).** Why: an unthrottled login is a lottery with
  unlimited tickets, but locking the owner out in front of the panel is worse than the lottery. Only
  attempts whose *voice was measured* count; silence, bad files, dead challenges, missing models do not.
  Cost if wrong: a judge's failures delay the owner by at most 30 s; `reset()` clears it.
- **R4. Public speech corpora are the impostor cohort.** LibriSpeech dev-clean (40 speakers) and Google's
  Tamil multi-speaker set, OpenSLR SLR65 (25 male + 25 female). Why: the 12 corpus speakers are too few to
  say anything about unseen judges. They are not people we recorded; both are CC-licensed. Cost if wrong:
  studio-quality read speech understates how a same-channel judge scores, so every FAR figure derived from
  them is labelled as optimistic until the presenter's own sessions and a channel test exist.

## 3. Measurements so far (all on this laptop, ECAPA `speechbrain/spkrec-ecapa-voxceleb`, cosine)

- 12 corpus speakers, phone recordings, 1,771 cross-speaker pairs: mean 0.240, sd 0.119, p95 0.450,
  p99 0.505, **max 0.588** against the 0.62 threshold. Worst pairs involve S12 and S10. Same-sitting
  genuine (leave-one-clip-out) mean 0.91, min 0.645 for S04.
- LibriSpeech dev-clean, 30 speakers with 2+ chapters, enrol on other chapters, test on the longest:
  genuine mean 0.806 (p5 0.66 for 4-6 s probes, 0.76 for 9-14 s), impostor mean 0.120, p99.9 0.482,
  EER 0.01%. Clean studio audio separates easily. **The risk is not clean-audio separation; it is probe
  length, channel and noise.** A 5-second phrase probe has a genuine p5 only 0.04 above the threshold.

## 4. Honest limits (state these, do not paper over them)

- One enrolled speaker, a handful of impostor voices of the right kind. Numbers describe the presenter's
  enrolment, not people in general.
- A live voice-conversion attack that also says the shown words defeats the phrase and voice gates; only
  an anti-spoofing model or a secret the cloner lacks (the personal question) would. Not claimed.
- The CSBG is a coin flip on free speech (50% EER). It is shown, never decisive.

## 5. Open work (ordered)

1. Calibrate the voice threshold and grey band on the presenter's held-out sessions + cohorts.
2. Channel and noise robustness of the voice gate (opus codec, noise, reverb) on the cohorts.
3. Enrol-from-Studio tool; calibrated policy file loaded by the pipeline.
4. Second-voice (interruption) guard.
5. Stage-mode UI: one screen, big button, phrase on screen, clear verdicts, judge mode.
6. Real-model rehearsal with synthetic voices (TTS) through the browser path; preflight for phrase mode.
7. Final write-up: what changed, the numbers, the drills, what is still blocked on the presenter.
