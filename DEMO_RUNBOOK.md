# DEMO_RUNBOOK — presenting KAVACH without surprises

Everything in the **Observed** column was observed on 2026-10-02 against the real
backend (Whisper `small`, ECAPA, Gemini tagging, LaBSE, RTX 4060), using S08 as a
stand-in presenter because S04 has no knowledge-graph facts yet. Nothing here is
a claim about what *should* happen. Where a drill could not be run, it says so.

## 1. Thirty minutes before

1. `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1` (add `-Prefetch` once, on good
   internet, if LaBSE is missing; `-Seed` only to rebuild the demo database).
2. Wait for the sidebar to turn green (about 20 s after launch).
3. Run the preflight:

   ```
   PYTHONPATH=backend .venv/Scripts/python.exe -m kavach.demo_check --presenter S04 --flows
   ```

   It must end in `READY`. It names the fix for anything that is not a pass. **Do not present on
   `NOT READY`.** Today the real presenter is `NOT READY`, for one reason: S04 has 0 facts
   (see section 6).
4. Rehearse **one live spoken answer** into the demo laptop's microphone. The preflight cannot do
   this, and it is the genuine-login step of the demo script.

## 2. Demo order (about 8 minutes)

From `DEMO_PLAN.md` §5. Overview/Corpus → Graph Explorer (two contrasting CSBGs) → Speakers
(enrolled presenter, edit SKG facts) → **Authenticate, genuine** (answer the challenge live) →
**Impostor** (button) → **Replay** (button) → **Clone attack** (button, once the bank exists) →
Evaluation (the 50% free-speech CSBG EER, and why).

## 3. Drills — what actually happens

Times include issuing the challenge (a Gemini call, 1–3 s) unless stated.

| Drill | Observed |
|---|---|
| A file that is not audio (garbage bytes, or a `.txt` renamed `.wav`) | `400` — "That file could not be read as audio. Use a WAV, MP3, M4A, OGG or WebM recording." (Before 2026-10-02 the box showed raw ffmpeg output.) |
| Empty upload | `400` — "The uploaded audio is empty." |
| A 0.3 s clip | `REJECT` — "the recording is only 0.3s long and at least 1s is needed … record the answer again." No model runs. |
| 3 s of pure silence | `REJECT` — "no speech was detected in the recording … record the answer again." No model runs. (Before: it reached Whisper, which invents text.) |
| 3 s of noise (not silent) | `REJECT` on the voiceprint (cosine −0.15 against a 0.62 threshold). |
| A real `.webm` / `.mp3` upload (10 s cut of another speaker) | `REJECT` on the voiceprint; 5–8 s. |
| A real `.m4a` (phone voice memo, moov atom at the end) | **Failed with "Decoded audio is empty" until commit `e5fd3a8`.** Now decodes and is `REJECT`ed on the voiceprint (0.27 vs 0.62) in ~5 s. |
| A 40 s clip (over the 30 s cap) | Processed, `REJECT` (it was an impostor); 11 s. |
| Same challenge submitted twice | Second is `REJECT` by the liveness gate — "has already been used. Challenges are single-use." |
| Unknown challenge id | `REJECT`, liveness — "Unknown challenge id". |
| A challenge older than its 60 s TTL | `REJECT`, liveness — "expired 2s ago". |
| Two logins at the same moment | Both complete: 4.7 s and 9.4 s (the second waits for the first). |
| LLM provider unreachable (HTTPS sent to a dead proxy) | Challenge falls back to the template bank (6–7 s, it retries first); login completes in 8–12 s; the screen says "The language tagger (LLM) was unavailable for this login … so the code-switch graph was not scored" and the CSBG is **excluded**, not scored. |
| Backend restarted | Health answers in ~5 s. A first login 15 s after launch took 2.3 s (challenge) + 8.3 s (verify); the next 1.0 s + 4.9 s. |
| The impostor button | `REJECT` **by the voiceprint** (integrity passes). Before 2026-10-02 it was rejected by the integrity gate first (see 4). |
| The replay button | `REJECT` at the integrity gate, "Byte-identical to a previously submitted recording", in milliseconds. |
| **Not exercised** | Refreshing the browser mid-login. A black-holed network (as opposed to refused connections): the LLM client allows 45 s per request and two attempts, so the worst case is unmeasured and could be well over a minute — use a phone hotspot you have tested. The live browser microphone (WebM/Opus from a real mic). |

## 4. What rehearsing found (so you know why the code is the way it is)

- **The impostor clip tripped the integrity gate before the voiceprint.** Turning splice detection
  off had *not* fixed this. The replay detector's envelope similarity is about √(cut ÷ clip), so a
  20 s cut of a 24 s stored clip scores ~0.91 against a 0.85 threshold: "the same performance
  submitted twice". Staged clips are now at most 40% of the stored clip and 12 s, from recordings of
  20 s or more (`demo_check.cut_plan`, the UI's `stagingCut`, pinned to the detector by
  `tests/test_demo_staging.py`).
- **A real `.m4a` failed to decode** (moov atom at the end, ffmpeg reading a pipe). Fixed.
- **The "genuine stand-in" is BORDERLINE (0.52) for S08, by design.** A stand-in is a cut of old
  audio; it cannot answer the random challenge, so the knowledge branch scores low. The preflight
  therefore checks that the voiceprint passes and the integrity gate does not trip, and warns that
  the **live spoken answer** needs one rehearsal. If you submit a stand-in on stage, expect
  BORDERLINE, not ACCEPT, and say why.

## 5. Honest answers to the questions you will get

- **"Does the code-switch graph work?"** Not on this data. 50.0% EER on free speech (5 speakers, one
  session each, 36 test trials, so the interval is wide); the scripted 26% is a script classifier,
  not a speaker habit. The demo shows a working *system* with an attack lab; it does not support the
  claim that the CSBG stops clones. Say so before anyone asks.
- **"How good is the voiceprint?"** On these 12 speakers a genuine probe scores ~0.9 and impostor cuts
  0.14–0.27 against a 0.62 threshold. One session per speaker and a same-sitting template flatter
  both, so this is a plumbing check, not an error rate.
- **"The Tamil transcript looks wrong."** Whisper `small` garbles Tamil; it is chosen for demo speed.
  Annotation for the corpus used `large-v3`.
- **"Is the clone real?"** Only once the bank has been generated and annotated (section 6). Then: the
  A4 clips are kNN-VC conversions of one teammate's spoken answers into the presenter's voice, one
  attacker and one victim, and the Attack Lab prints the yield next to every rate. A3 and A5 remain
  modelled and are labelled "acoustic modelled".

## 6. Blocked on the presenter (nothing here can be faked)

1. **Enter S04's facts** in Speakers → S04 → knowledge facts (hometown, college, favouriteFood …). They
   are personal details; until then S04 cannot be challenged and the preflight says `NOT READY`.
2. **A teammate speaks one short answer per fact**, once, in their own Tamil–English style. Save as
   `data/clone_sources/<predicate>.wav`. Then:

   ```
   KAVACH_CLONE_VICTIMS='["S04"]' PYTHONPATH=backend .venv-clone/Scripts/python.exe \
       -m kavach.attacks.make_clone_bank --victim S04 --sources data/clone_sources
   KAVACH_CLONE_VICTIMS='["S04"]' KAVACH_DEMO_ATTACK_BANK=true PYTHONPATH=backend \
       .venv/Scripts/python.exe -m kavach.attacks.annotate_bank --victim S04
   ```

   The clone environment already exists (`.venv-clone`, verified: kNN-VC converts 6 s of speech in
   1.2 s warm). `run_demo.ps1` already sets `KAVACH_DEMO_ATTACK_BANK` and `KAVACH_CLONE_VICTIMS`.
3. **Confirm S04 agrees to being cloned.** The code refuses anyone not on `clone_victims`, but that
   list is only as good as the decision behind it.
4. Rehearse the live microphone answer once (section 1, step 4).

## 7. Never, on stage

- Turn on `KAVACH_DEMO_REVEAL_ANSWERS` (anyone can read the answer in the network tab). The bank
  routes also return audio that says the answer, which is why `/api/health` reports
  `demoAttackBank`. The preflight warns when the answer-reveal setting is on; it does not warn when the bank is on, because the bank being on is the intended demo configuration.
- Skip the preflight.
- Present a number from this lab as a result: every lab run is labelled simulated.
