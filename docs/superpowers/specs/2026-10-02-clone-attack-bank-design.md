# Clone-attack bank — design

Status: **draft for review**, 2026-10-02. Nothing in this document is implemented.
Scope: attacks A3–A5 in the Attack Lab and the live demo (`HANDOFF.md`, "Still open").

## 1. Outcome, and what is assumed

**What was asked for.** "Complete the project" was narrowed, by the user, to the
demo-ready prototype plus clone attacks. The demo-ready part is finished and
verified (`HANDOFF.md`, 2026-10-02 update). This document is the clone-attack part.

**Outcome.** The Attack Lab's A3–A5 rows, and a "Clone attack" button in the demo,
run on audio that was actually cloned into the presenter's voice, instead of on an
acoustic score drawn from a documented distribution (`api/attacks.py::ACOUSTIC_MODEL`).
Every rate is printed beside the **attack yield** — the share of clones that got past
ECAPA — because `attacks/clone.py` establishes that a row of zeros is meaningless
without it.

**Decisions made by the user in this session** (not assumptions):

| Decision | Choice |
|---|---|
| Whose voice may be cloned | The presenter's own voice only (corpus pseudonym `S04`) |
| How a clone reaches the login | Pre-generated clips, not live conversion |
| Which approach | Staged: kNN-VC first, IndicF5 second |

**Assumptions, open to correction:**

- This is a demo plus an honest table. Nothing here is a reportable result; the corpus
  is still one session and 12 speakers, so `AttackTable.paper_ready()` keeps refusing.
- No new *corpus* data is collected. A teammate speaking answers is attack source audio,
  not a corpus recording, but it is still someone speaking and the user should say so
  if they disagree.
- Heavy steps (a CUDA torch download, model checkpoints, synthesis on the GPU) are each
  asked for when reached. The user trains on the same laptop; the miniconda environment
  is theirs and is never touched.

**Non-goals.** Live conversion; any victim but the presenter; training or fine-tuning a
voice model; claiming the CSBG stops clones (the A4 row is a measurement, not a
conclusion); changing fusion weights or thresholds.

## 2. Prerequisites that are not code

1. **The presenter has no SKG facts.** In the demo database only S08 (4 facts) and S09
   (3 facts) have any. A challenge needs a fact (`challenge.select_target` raises when
   there is none), so S04 cannot be challenged at all today, and the bank can only cover
   facts S04 has. The presenter must enter them in the Speakers drawer; they are personal
   details only the presenter can supply. Facts are keyed `(speaker_id, predicate)`, so a
   bank clip's key is the predicate (`hometown`, `college`, ...).
2. **A teammate speaks one short answer per fact**, once, in their own code-switching
   style. That is the A4 attacker: a different person, the victim's voice.
3. **IndicF5 (stage 2 only)** is gated on Hugging Face. A token file exists on this
   machine; the user accepts the model's terms on their own account.
4. **A separate environment** (`.venv-clone`, Python 3.11, CUDA torch). The project
   venv is CPU torch; the miniconda base has CUDA torch but is the user's training
   environment and must not be modified.

## 3. Architecture

Two stages, because they need different environments and fail differently — the same
split `annotate.py` already makes between ASR and tagging.

```
data/clone_sources/<fact>.wav      teammate's spoken answers          (stage 0, by hand)
          │
          ▼  STAGE 1 -- generate         runs in .venv-clone
  attacks.make_clone_bank --victim S04 --backend knn_vc
          │   target audio = S04's stored enrolment clips (5.7 min)
          ▼
data/attacks/clones/S04/{audio/*.wav, bank.json}     clips + provenance, no scores yet
          │
          ▼  STAGE 2 -- annotate         runs in the main env
  attacks.annotate_bank --victim S04
          │   real Whisper + tagger, ECAPA vs S04's template (screen_clone),
          │   AnswerMatcher vs the SKG fact
          ▼
bank.json                                             + transcript, tokens, similarity,
          │                                             admissible, answer score
          ├──► Attack Lab    run_attack(A3|A4|A5)       measured columns, yield printed
          └──► /api/authenticate via the demo button    the full real pipeline
```

### 3.1 Interfaces

`attacks/clone.py` keeps `CloneBackend` (text → speech: XTTS, IndicF5) and gains a
sibling for voice conversion, because kNN-VC takes *audio*, not text, so it cannot
satisfy `SynthesisRequest`:

```python
@runtime_checkable
class VoiceConverter(Protocol):
    def convert(self, source: Audio, target_reference: list[Audio]) -> Audio: ...
    def name(self) -> str: ...
```

`screen_clone`, `CloneQualityReport`, `CloneBatchStats` and `MIN_REFERENCE_SEC` are
reused unchanged. `XTTSCloner` is left as it is; its docstring already says XTTS may
lack Tamil, and nothing here depends on it.

New modules:

| Module | Env | Role |
|---|---|---|
| `attacks/bank.py` | main (numpy, stdlib) | `CloneClip`, `CloneBank`: load, validate, query, yield |
| `attacks/backends/knn_vc.py` | clone | `KnnVcConverter` (lazy torch import) |
| `attacks/backends/indicf5.py` | clone | `IndicF5Cloner` (stage 2) |
| `attacks/make_clone_bank.py` | clone | stage 1 CLI |
| `attacks/annotate_bank.py` | main | stage 2 CLI |

`bank.py` imports no model library, so the main environment and the whole test suite
load it without any checkpoint.

### 3.2 Bank format

`data/attacks/clones/<pseudonym>/bank.json`, next to `audio/<clip_id>.wav`. `data/` is
git-ignored in its entirety, which is where this belongs.

```jsonc
{
  "bank_version": "1",
  "victim": "S04",                       // the pseudonym; never a name
  "victim_speaker_id": "spk_...",        // resolved from the DB at build, checked at load
  "synthetic": true,
  "clips": [{
    "clip_id": "clone_ab12cd34",
    "attack": "A4_clone_knowledge",      // AttackType.value: A3_clone | A4_clone_knowledge | A5_style_adaptive
    "backend": "knn_vc", "backend_version": "...",
    "fact_key": "hometown",              // the predicate this clip answers; null for A3
    "source": {"kind": "teammate_speech", "file": "hometown.wav"},   // or {"kind":"tts_text","text":...}
    "audio_path": "audio/clone_ab12cd34.wav", "sha256": "...", "duration_sec": 6.4,
    "created_at": "...",
    // written by stage 2, absent before it:
    "transcript": "...", "tokens": [...], "annotation_source": "LLM|LEXICON",
    "ecapa_similarity": 0.71, "threshold": 0.62, "admissible": true,
    "answer_score": 0.9
  }]
}
```

`CloneBank.load` refuses a bank whose files are missing, whose hashes do not match,
whose `victim` is not on the allowlist, or whose `victim_speaker_id` is not the DB
speaker it claims to be. A5 clips come only from a text-to-speech backend, because A5
is a *text* attack (re-drawn language choices); a kNN-VC clip is whatever the teammate
said.

### 3.3 Settings (fail closed)

| Setting | Default | Purpose |
|---|---|---|
| `clone_victims: list[str]` | `[]` | Pseudonyms that may be cloned. Empty means no cloning at all. `run_demo.ps1` sets `["S04"]`. |
| `demo_attack_bank: bool` | `False` | Enables every bank route and the lab's use of the bank. Reported in `/api/health`. |

The generator and the loader both enforce `clone_victims`; the UI cannot widen it.

### 3.4 API

All three routes return 404 unless `demo_attack_bank` is on.

- `GET /api/clone-bank` — clips with `id`, `attack`, `backend`, `factKey`, `similarity`,
  `admissible`, `durationSec`, plus the yield summary. No answers, no transcripts.
- `POST /api/clone-bank/match` `{challengeId}` — the A4 clip whose `fact_key` equals the
  issued challenge's `expected_predicate` (the admissible one with the highest
  similarity if several exist), or a clear "no clone answers this question".
- `GET /api/clone-bank/{clip_id}/audio` — the WAV.

**This is a leak surface and is gated exactly like `expectedAnswerEntity`.** A clip *says*
the answer, so a route that returns the clip for a live challenge hands whoever calls it
the knowledge factor as audio. That is acceptable only in a demo build that announces
itself, which is what `demo_attack_bank` and `/api/health` are for, and it is the same
reasoning as `Settings.demo_reveal_answers` (`PROJECT.md` §3.8).

The match route is also what makes the pre-generated choice realistic. An attacker who
knows every fact can pre-clone an answer to every possible question; the demo issues an
ordinary random challenge and the attacker's bank is looked up afterwards. Nothing about
the challenge is pinned.

### 3.5 Attack Lab

For A3–A5 against the bank's victim, when the bank holds clips of that attack, each trial
uses a bank clip instead of the modelled draw:

- acoustic score = `ecapa_similarity` (measured), `acoustic_source = "measured"`;
- CSBG probe = the clip's real `tokens` (ASR + tagging of the cloned speech);
- knowledge score = `answer_score` (the real matcher against the SKG fact).

Otherwise the run is exactly what it is today, with a note saying why the bank was not
used. Three rules, each prevents a number that looks better than it is:

1. **Trials are capped at the number of distinct clips** for a measured row. Resampling 7
   clips to 40 trials would give a confidence interval narrower than the evidence;
   `MIN_TRIALS_PER_CELL` counts trials and cannot tell the difference.
2. **Yield is printed with every measured row**: clips admitted / clips generated, and the
   number of distinct source speakers (one teammate is n = 1 attacker).
3. **The run stays `simulated: true`** and `paper_ready()` is untouched. The note lists
   which columns are measured and why the run is still not reportable (single session,
   same-sitting template, 12 speakers, demo thresholds).

### 3.6 Demo UI

`Authenticate.tsx`'s "Demo with stored audio" panel gains a fourth button, **Clone
attack**, shown only when `/api/health` reports `demoAttackBank` and the bank is
non-empty. It uses the challenge already issued on the page: `POST
/api/clone-bank/match`, fetch the audio, submit it as a login claiming the victim. The
result panel's source label reads "Clone attack · synthetic". If no clip answers the issued
question the panel says which questions the bank does cover and offers a new challenge.
The Attack Lab page shows `measured` or `modelled` beside the acoustic column.

## 4. Safety and honesty requirements

1. Only pseudonyms on `clone_victims` can be generated or loaded. Default empty.
2. `bank.json`, file names and logs carry the pseudonym, never a participant's name.
   (Display names in the demo DB are `S04 · <first name>`; nothing here copies them.)
3. Clips are marked synthetic in `bank.json` and in the WAV file name, and live only under
   git-ignored `data/`. `README.md` gains one line saying they are not to be distributed.
4. The three routes are off by default and `/api/health` says when they are on.
5. `paper_ready()` is not changed. Yield is shown next to every measured rate.
6. A clone that the acoustic branch stops is counted in yield, **not** in the attack's
   success rate (`screen_clone` already says this); a low yield is reported as the
   finding it is, not hidden.

## 5. Staging

| Stage | Contents | Heavy / needs the user |
|---|---|---|
| 0 | S04 facts entered; teammate records one answer per fact | the user, by hand |
| 1a | `bank.py`, settings, three routes, lab integration, demo button, all tested with a fake backend and synthetic audio | none |
| 1b | `.venv-clone`, `KnnVcConverter`, `make_clone_bank`, the first real bank. **A4 clips only** — a teammate speaking each fact's correct answer; A3 and A5 stay modelled until stage 2 | CUDA torch (~2.5 GB), WavLM + HiFiGAN checkpoints, GPU time — asked at the time |
| 1c | `annotate_bank`, the three demo flows incl. clone driven end to end, HANDOFF updated | Whisper / Gemini / ECAPA run — asked at the time |
| 2 | **Go/no-go probe first**: three code-mixed sentences through IndicF5, judged by Whisper intelligibility and ECAPA similarity. Only on "go": `IndicF5Cloner`, A3 and A5 clips | HF terms, IndicF5 download (gated), its own Python 3.10 pins — asked at the time |

Stages 0–1c are one implementation plan. Stage 2 gets its own plan once the probe has
answered, because whether it exists at all depends on that answer.

A "no-go" at stage 2 is a result, not a failure: "open TTS cannot clone code-mixed Tamil
well enough to beat a speaker verifier" is the low-resource finding `attacks/clone.py`
already names, and it is recorded as such.

## 6. Testing

No test downloads or loads a model. The suite stays offline and a few minutes long.

- `bank.py`: round-trip, missing file, hash mismatch, wrong victim, victim not on the
  allowlist, unannotated clips excluded from measured rows.
- Yield accounting through `CloneBatchStats`, including "all clips inadmissible".
- Routes: 404 with the setting off; with it on, the match route returns the clip for the
  issued challenge's predicate and a clear message when none exists; **a leak sweep** in
  the style of the existing `demo_reveal_answers` sweep, asserting no bank route is
  reachable and no answer text appears in any other route's body while the setting is off.
- Lab: measured columns when the bank is used; the trial cap; a note when it falls back
  to the modelled draw; `simulated` stays true; `paper_ready()` still refuses.
- Generator / annotator with an in-test fake `VoiceConverter` and synthetic `Audio`.
- Real checkpoints are covered only by `@pytest.mark.models` tests, opt-in, never in the
  default run.

## 7. Risks and open questions

- **kNN-VC target length.** The presenter has 5.7 min of audio; the method works best
  with more and degrades below about five. If yield is low the cause may be the amount of
  target speech rather than the defence — which `clone.py` already warns about, so the
  report says so and does not credit the defence.
- **Cross-lingual conversion.** kNN-VC was built for English; Tamil and code-mixed source
  speech is untested. Stage 1b measures it before anything builds on it.
- **One attacker.** A single teammate gives a yield for one source speaker. It is reported
  as such.
- **The acoustic score is against a same-sitting template** (one session each), which
  flatters the clone and the genuine user alike. Stated beside every measured row.
- **Whisper translating cloned Tamil into English**, as it sometimes does with genuine
  speech (`HANDOFF.md`, trap 0); `looks_translated` runs in stage 2 and flags the clip
  rather than letting it enter the CSBG column.
- **Open for the reviewer:** (a) is capping trials at the clip count acceptable for the
  lab's UI, which currently asks for 10–60 trials? (b) should `clone_victims` default
  empty (fail closed, as written) or default to the presenter?
