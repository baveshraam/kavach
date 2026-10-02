# Recording guide — about three hours (you can stop after S3 and come back for S4 and S5)

You are recording your own voice so the system can be measured on **you**, on sessions it has never
heard. Everything stays on this laptop under `data/studio/S04/` (git-ignored). Nothing is uploaded.

## Before you start (15 minutes)

1. **Start the Studio:** `powershell -ExecutionPolicy Bypass -File .\run_studio.ps1`. It opens
   `http://localhost:3000/studio`. (This is a *different* launcher from the demo on purpose.)
2. **Add your personal facts:** Speakers → S04 → knowledge facts. At least five: hometown, college,
   school, favourite food, a sibling's name, anything only you would know. The Studio asks you
   questions about them, and the demo does too.
3. **Read the three sentences once** (the first screen of a session shows them). If one is awkward to
   say, tell me and I will replace it before you record: they should feel like things you would
   actually say.
4. Close other apps that use the microphone. Plug in the headset.

## The five sessions

A **session is one sitting: one device, one room.** Do not change the device or room in the middle.
Set them in the Studio before pressing Start. Take a real break between sessions.

| Session | Device and room | What it is | Time |
|---|---|---|---|
| **S1** | **Demo laptop mic**, quiet room | enrols you | 35 min |
| **S2** | **Headset** microphone (plugged into the laptop), quiet room | enrols you | 35 min |
| **S3** | **Demo laptop mic**, a *different* spot: another room, or standing, or farther from the mic | enrols you | 25 min |
| **S4** | **Demo laptop mic**, as the demo will be, **after a break of at least 30 minutes** | **held out: never enrolled** | 25 min |
| **S5** | **Demo laptop mic in a noisier room** (a corridor, a fan on, people nearby), after another break | **held out #2: never enrolled** | 20 min |

**S4 and S5 are the ones that matter.** Two, because one held-out sitting is one cluster and cannot give a meaningful error interval on its own. They are scored against a voiceprint built from S1-S3 only, so it tells
us what happens when you walk in on demo day. Do not look at any results between S3 and S5, and do not
adjust anything after seeing them.

## How to speak

- **Naturally**, at the volume and distance you will use in the demo. Do not over-enunciate.
- **Free prompts:** answer in your own Tamil-English mix, 20-40 seconds. Do not translate; say it the
  way you would to a friend.
- **Ten words:** every session also asks for ten random words ("Say: tiger, river, mango, ...") a dozen
  times. This is exactly what the Unlock screen asks for, so these clips are what the demo's voice
  threshold is measured on. Read them the way you will on stage: one breath, normal pace, then stop.
- **Read-aloud sentences:** the same sentence many times is the point (it is the same-sentence test).
  Vary it a little between takes the way you naturally would; do not perform it.
- **Mistakes:** if you stumble, press **Redo** before saving. Saved clips are never overwritten.
- A cold, a tired evening or a hurried take are *fine and useful*: write it in the Note field.
  The demo will not happen on your best day.

## If the page is refreshed

The Studio **keeps your place**: after a refresh it says "Recording into S1 ..." and carries on at the
same prompt. It will never move the rest of a sitting into the next session. To start a *different*
sitting (new device, new room, after a break), press **Start a new session instead**. The Studio also
refuses to save a clip whose device or room differs from its session's, so a mix-up cannot be saved.
Nothing already saved is lost or duplicated, and a recording cut short by a crash is quarantined, not
lost.

## When you are done: four commands, in this order

Stop the Studio first (close its window), then from the repo folder:

```
# 1. The evidence for the claim: S1-S3 enrol, S4 and S5 never enrolled
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.eval.enrollee \
    --studio data/studio/S04 --enrol-sessions S1,S2,S3 --test-sessions S4,S5 \
    --impostors data/corpus_v2/manifest.json --impostors data/corpus_v3/manifest.json

# 2. The demo's voice threshold, measured: each session scored against the other four
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.calibrate_voice --sessions S1,S2,S3,S4,S5

# 2b. Does speech recognition hear your ten words? (the login also needs the words, not just the voice)
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.studio.words_check --speaker S04

# 3. Enrol the demo from all five sessions (the database is backed up first)
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.studio.enrol --speaker S04 --sessions S1,S2,S3,S4,S5

# 4. Start the demo (run_demo.ps1), then check it
PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.demo_check --flows
```

What each does:

1. **The evidence report.** It starts with what it can and cannot claim: one enrolled speaker, measured on
   held-out sessions, against other recorded people. It is evidence about *your* enrolment, not about people in
   general, and says so in its first paragraph. The intervals are never narrower than a plain Wilson interval,
   and with fewer than five sessions (or no errors) it says the cluster interval is *not informative* rather than
   printing a tight range. Per-trial scores are written to `data/studio/S04/eval/trials.csv`.
2. **The calibration.** Chooses the voice threshold midway between where strangers stop and where *you* start,
   from your words clips against the public cohorts (LibriSpeech, Google's Tamil speakers) and the 11 corpus
   speakers. Writes `data/voice_policy.json`; the demo loads it at start. If it says `provisional` or not `ready`,
   it tells you why (usually: record another session).
2b. **The words check.** Transcribes your words clips exactly as the login does and reports the share that
   would have cleared the words gate. Under 95% it names the options (a larger Whisper, a looser match).
3. **The enrolment.** Replaces the demo's voiceprint (built from phone recordings) with one built from the
   same browser-microphone path the login uses, and records what it was built from.
4. **The preflight.** Fails loudly if the voice gate is off, the policy is damaged, or a synthetic stranger
   gets in. Then do one live read-through on the real microphone: nothing replaces it.

Numbers from step 1 describe a template built from S1-S3; the demo's final template (step 3) adds S4 and S5, so
it is at least as well informed. Say so when you quote them.
