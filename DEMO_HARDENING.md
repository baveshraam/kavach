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

- **R5. The voice is scored on the span where the shown words were spoken.** Whisper's word timings
  locate the phrase; the voice is embedded on that span plus 0.2 s each side. Why: speech before or after
  the phrase (a judge saying "let me try") diluted the owner's voiceprint: real-model rehearsal, owner then
  judge, BORDERLINE 0.54; after the change ACCEPT 0.83. A judge speaking *over* the phrase still lowers the
  score (borderline, never accept). Falls back to the whole recording without timings or under 1 s.
- **R6. The voice threshold is measured, midway.** `kavach.calibrate_voice`: the accept threshold is the
  midpoint between the strangers' tail (the 99.9th percentile of impostor scores, or the worst seen when
  there are too few trials) and the owner's low end (99% of their held-out probes reach it); the borderline
  band reaches halfway down to the strangers' tail. If the classes overlap the strangers' tail wins and the
  owner's false-reject rate is reported with the remedy. The result is written to `data/voice_policy.json`
  and loaded at start; a damaged file falls back to 0.62/0.08 and `/api/health` says so. Why not the plain
  impostor quantile: it puts the threshold at the edge of the strangers, no margin for an unseen judge.
  Cost if wrong: a threshold below 0.62 is possible if the owner's scores are low; `MIN_THRESHOLD` is 0.50.
- **R7. Studio gets a `words` kind and the final enrolment uses every session.** The login's own task (six
  random words) is recorded 10-15 times per session, deterministically per session id so a refresh keeps
  its place. Calibration is leave-one-session-out over all five sessions on those clips; the demo template is
  then enrolled from all five (`kavach.studio.enrol`, database backed up first, provenance stored). The
  held-out evidence report (S1-S3 enrol, S4-S5 test) stays as the cleaner proof number.
- **R8. Facts are no longer a hard requirement of the preflight.** The phrase login needs none; the
  presenter's missing facts are a WARN that the stronger personal-question step-up is unavailable. The
  preflight gained: voice gate on, calibrated policy, voiceprint provenance, and phrase flows driven by a
  synthetic stranger (Windows SAPI) that must be rejected on the voice, and a recording of other words
  that must be rejected on the words. Cost if wrong: synthetic speech is not a human; it proves the
  plumbing only.
- **R9. WavLM-SV (and ResNet) are not used.** On the 12 corpus speakers, within-session, ECAPA's EER was
  0.03% and WavLM-SV's 3.73% (raw cosine; centring did not help). ResNet embedding was abandoned for CPU.
  A second embedder is not worth the risk this close to the demo.

## 3. Measurements so far (all on this laptop, ECAPA `speechbrain/spkrec-ecapa-voxceleb`, cosine)

- 12 corpus speakers, phone recordings, 1,771 cross-speaker pairs: mean 0.240, sd 0.119, p95 0.450,
  p99 0.505, **max 0.588** against the 0.62 threshold. Worst pairs involve S12 and S10. Same-sitting
  genuine (leave-one-clip-out) mean 0.91, min 0.645 for S04.
- LibriSpeech dev-clean, 30 speakers with 2+ chapters, enrol on other chapters, test on the longest:
  genuine mean 0.806 (p5 0.66 for 4-6 s probes, 0.76 for 9-14 s), impostor mean 0.120, p99.9 0.482,
  EER 0.01%. Clean studio audio separates easily. **The risk is not clean-audio separation; it is probe
  length, channel and noise.** A 5-second phrase probe has a genuine p5 only 0.04 above the threshold.

- Public cohorts against corpus speakers as pseudo-targets (templates from each speaker's own clips),
  full-clip probes: LibriSpeech (40 speakers, 14,244 trials) p99.9 0.335, max 0.425; Google Tamil male
  (25 speakers, 7,764 trials) mean 0.220, p99.9 0.566, max 0.604; Tamil female (25, 8,868) p99.9 0.557, max
  0.608. **0 of 30,876 trials reached 0.62; the nearest were 0.604 and 0.608.** Tamil clips average 4.3-4.6 s,
  about a phrase. Against S04's own template: Tamil male max 0.476, female 0.330, LibriSpeech 0.316, the 11
  corpus teammates 0.542. The extreme tail is close to 0.62 for hub-like targets, which is why R6 centres
  the threshold between the classes instead of putting it at the tail.
- Real-model rehearsal (Windows David as the owner, Zira as the judge, Whisper small on GPU, ECAPA on CPU,
  opus 24 kbps as the browser sends it): latency median 2.1-2.6 s per login (first call 8 s before the
  warm-up inference was added); owner accepted at 0.74-0.87; judge reading the right words rejected at 0.16;
  owner with the wrong words rejected on the words at voice 0.81; owner with 5 of 6 words accepted, 3 of 6
  rejected, reversed order rejected; noise at SNR 20 dB accepted (0.74), 10 dB (0.67), 5 dB borderline-to-reject
  (0.54-0.60); +18 dB clipping rejected or borderline (0.51-0.56); owner + judge simultaneous borderline (0.59);
  silence rejected in 0.2 s. These are synthetic voices: the voice scores prove the gates, not human variation.
- Browser end to end (Chrome fake microphone fed the TTS of the words shown on screen, real backend):
  owner unlocked (0.82), judge denied (0.21), owner again unlocked (0.78); no page errors, no failing requests.

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
