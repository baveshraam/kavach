# DEMO_PLAN — finishing KAVACH on the data we already have

> **Status, 2026-10-03.** This is the research note of 2026-09-29 that set the demo's course; it is kept as a
> record and is not edited further. Done since: Gap 1 (the 12-speaker demo database, `seed_demo`), Gap 3 (the
> offline-results panel), Gap 2 for kNN-VC (the clone bank, `kavach.attacks`; IndicF5 was not attempted), the
> live-demo fragilities in Gap 4 (see `DEMO_RUNBOOK.md`), and the gold-set export (labelling still needs a bilingual
> human). **Superseded:** the demo script in section 5 is now secondary; the demo is the Unlock screen
> (`DEMO_RUNBOOK.md` section 0), and the direction is evidence for a single enrolled speaker (`DEMO_HARDENING.md`).
> The constraint "no new recordings" was lifted for the presenter's own voice only (2026-10-02).

Research notes, 2026-09-29. Constraint: **no new recordings.** Everything below
uses the 12 consented speakers already on disk.

---

## 1. What we actually have

| | Scripted (corpus_v2) | Free speech (corpus_v3) |
|---|---|---|
| Speakers | 7 (S01–S07) | 5 (S08–S12) |
| Utterances | 98 (97 usable) | 70 (64 usable, 6 excluded as Whisper translations) |
| Speech | 41 min | ~28 min |
| Tagged tokens | ~6.5k (Gemini, 0 guessed) | 3.5k (Gemini, 0 guessed) |
| Sessions | 1 each | 1 each |

- `data/consent_register.csv`: **all 12 rows are now YES** (HANDOFF.md still says
  PENDING; it is out of date).
- Every manifest carries audio path + transcript + tagged tokens, so the demo
  database can be built **without re-running ASR or the LLM**.
- `pretrained_models/` already holds the ECAPA checkpoint; `.env` has a Gemini key.
- Machine: Python 3.11 venv (torch **CPU** build), RTX 4060 8 GB, ffmpeg, Node 22.

## 2. What the data says (and does not)

| Result | Number | Meaning |
|---|---|---|
| ECAPA alone | 0.00% EER | Same-sitting enrol/probe. Plumbing check, not a result. |
| CSBG alone, scripted | 26.3% EER | Script classifier, not a speaker habit (§5.2.4). |
| **CSBG alone, free speech** | **50.0% EER** | **Chance.** 5 speakers, 36 test trials, so the CI is wide. It is still the only CSBG measurement on real speech. |
| Knowledge branch | 8% coverage | Needs cross-session data by construction. |

**Conclusion:** with this data the CSBG-as-biometric claim (§5.1) is **not
supported**, and it cannot be tested further without second sessions. Per §12
of `KAVACH_Project_Idea.md`, the honest framing is:

> A working multi-factor voice-authentication *system* for code-switched
> Tamil–English speakers, with a real attack lab, plus the CSBG reported as a
> negative/inconclusive result with analysis.

The demo should show the *system* working end to end. It should not claim the
CSBG stops clones.

## 3. The demo gaps, and how to close each one without new data

### Gap 1: the demo DB has only 2 speakers (S08, S09)
**Fix:** a `seed_demo.py` script that, for all 12 speakers in corpus_v2 + v3:
`store.create_speaker` → `store.add_utterance` (audio + manifest transcript +
tokens, `annotated=True`) → `complete_enrolment` (CSBG + ECAPA template).
It makes no ASR or LLM calls. ECAPA on CPU over ~220 clips takes a few minutes.
This also lets the CSBG background model be built from 11 others instead of 1.

### Gap 2: A3/A4/A5 acoustic scores are drawn from a distribution, not measured
This is the biggest credibility gap in a live demo, and **it can be closed with
existing audio**:

| Tool | What it gives | Fit |
|---|---|---|
| **IndicF5** (AI4Bharat, MIT, gated HF, 0.4B params, 24 kHz) | Zero-shot TTS clone from *reference audio + its exact transcript*. The manifests have both for every clip. Supports Tamil. | **A3/A4**: victim's voice saying the challenge answer. Code-mixed input is not documented; test it. |
| **kNN-VC** (bshall/knn-vc, WavLM + HiFiGAN) | Converts *any real speech* into the target's voice; works best with ≥5 min of target audio (we have 4–7 min per speaker). | **The A4 story, exactly:** a teammate answers the challenge in *their own* code-switch style and kNN-VC puts it in the victim's voice. Voice matches; wrapper does not. |
| XTTS-v2 | — | **No Tamil support.** Drop it; `clone.py` still defaults to it. |

Both slot in behind the existing `attacks.clone.CloneBackend` protocol. Run them
in a **separate venv** (IndicF5 wants Python 3.10 and its own pins), pre-generate
clips offline, and feed them through `/api/authenticate` like any upload.
Report **attack yield** (the fraction that fooled ECAPA), as §5.1.3 requires.

Ethics: IndicF5's terms prohibit cloning without permission. The register has
`public_release_agreed=YES`, but get explicit OK from the cloned speaker(s)
(ideally just use the presenter's own voice, S04).

### Gap 3: the Evaluation page shows only demo login history
`/api/evaluation` scores whatever logins ran through the UI. The real offline
numbers (`paper/results*/results.json`, DET/heatmap/stability figures) are not
in the UI at all. **Fix:** serve `paper/results*/` statically and add an
"Offline experiments" panel that renders both runs *with their unreportable
banners*.

### Gap 4: live-demo fragility
- **Whisper on CPU:** `KAVACH_WHISPER_MODEL=small` is set for the demo. Optional
  GPU: install CUDA torch + `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` for
  ctranslate2. Only do it if tested a day early.
- **Gemini at login time:** without it tagging falls back to rules → every token
  OTHER → CSBG unmeasured. Keep a phone hotspot, and pre-record fallback clips.
- **Whisper translating Tamil to English** (21/70 free-speech clips!) can happen
  live. The pipeline flags it; rehearse what to say when it does.
- **Trap 12:** rehearse against the real backend, never `VITE_USE_MOCK=true`.

## 4. Extra, reportable work possible on existing data

In order of value per hour:

1. **Label the gold set** (`data/goldset_v1.tsv`, 1143 tokens; `--utterances 6`
   for a smaller one). Any Tamil-English bilingual on the team can do it. That
   gives **word-level LID accuracy**, the first genuinely reportable number.
   It is labelling, not new data.
2. **Real clone attack table** from Gap 2: yield + IAPMR per configuration. A
   low yield is itself the §8 finding ("open TTS can't clone code-switched
   Tamil well enough to beat ECAPA").
3. **ASR findings already measured:** median WER 5.3% (scripted); Whisper
   *translated* 21/70 free-speech clips (30%) and 2/98 scripted; 64
   English-in-Tamil-script tokens recovered (37 + 27). These are solid,
   reportable low-resource findings.
4. **Fairness/confound slices:** ECAPA scores by codec (WhatsApp `.ogg` for S02/S03
   vs `.m4a`) and SNR (S10 at 16.5 dB vs S09 at 47.6 dB).
5. **CSBG negative result with analysis:** per-class separation on free speech,
   and a permutation test of within- vs between-speaker CSBG distance. That says
   *why* it fails (sparse classes, 5 speakers), not just that it fails.

## 5. Suggested live demo script (~8 min)

1. **Overview / Corpus** — 12 speakers, token stats, the code-switch classes.
2. **Graph Explorer** — two contrasting CSBGs side by side (e.g. S09 vs S10).
3. **Speakers** — presenter (S04, BaveshRaamS) is enrolled; show and edit SKG facts.
4. **Authenticate, genuine** — issue a challenge, answer live → ACCEPT, walk the
   branch breakdown (voice / knowledge / CSBG / liveness / integrity).
5. **Impostor** — a teammate claims S04 → REJECT on voice.
6. **Replay** — play an old S04 clip from a phone → rejected (stale challenge / integrity).
7. **Clone (A4)** — the kNN-VC clip: the teammate's answer in S04's voice. Show the
   ECAPA score next to the CSBG score, and say honestly what the CSBG did.
8. **Evaluation** — the offline results, including the 50% free-speech CSBG EER,
   and why (one session, 5 speakers).

## 6. Recommended order of work

1. `seed_demo.py` + run the real backend + full UI rehearsal (Gaps 1, 4). *Demo-critical.*
2. Offline-results panel in Evaluation (Gap 3). *Small.*
3. kNN-VC clone of S04 (Gap 2). *Highest demo impact.* Then IndicF5 if time allows.
4. Gold-set labelling (§4.1), in parallel, by a teammate.
5. Update HANDOFF.md (consent status, corpus_v3, free-speech result).

## Sources
- IndicF5: https://huggingface.co/ai4bharat/IndicF5 , https://github.com/AI4Bharat/IndicF5
- kNN-VC: https://github.com/bshall/knn-vc (cross-lingual limits: https://arxiv.org/pdf/2506.09709)
- XTTS-v2 languages (no Tamil): https://docs.coqui.ai/en/latest/models/xtts.html
