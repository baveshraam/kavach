# KAVACH

**K**nowledge-graph **A**nchored **V**oice **A**uthentication for **C**ode-switched,
**H**ybrid-language speakers.

A speaker-verification system for Tamil–English code-mixed speech whose research
contribution is the **Code-Switch Behaviour Graph (CSBG)**: a per-speaker graph
over 21 semantic concept classes recording *which language this speaker chooses
for which subject*, scored as a log-likelihood ratio against a universal
background model.

The claim the whole system is built to test is one sentence:

> a code-switching profile is hard to steal even when the voice and the secret
> have both already been stolen.

## Where to start

| Document | What it is for |
|---|---|
| **[HANDOFF.md](HANDOFF.md)** | Current state, what is left, and the traps already hit. Read this first if you are picking the work up. |
| **[PROJECT.md](PROJECT.md)** | What the system is, every design decision and why, what has been measured versus assumed, and the full session history. |
| `KAVACH_Project_Idea.md` | The research proposal: novelty argument, experiment design, findings write-up. |

**One thing to know before reading any number in this repository:** two corpora
of real speakers now exist — 7 reading scripts (`SCRIPTED`) and 5 speaking freely
(`RECORDED`) — but neither ships here; see [What is not in this
repository](#what-is-not-in-this-repository). `simulation.py` still exists and its
speakers differ *by construction*, so a number it produces is never an
experimental result. See PROJECT.md §4.

---

## Layout

```
backend/kavach/         the system
    csbg/               ontology, graph construction, LLR scoring, metrics
    lid/                word-level language ID and semantic tagging
    attacks/            the A1–A5 threat model and its detectors
    eval/               EER, minDCF, DET curves, ablations, figures
    api/                FastAPI layer serving the frontend
    corpus.py           recorded-speech manifest, loader, elicitation protocol
    ingest.py           returned participant folders -> validated manifest
    annotate.py         audio -> transcripts -> tagged tokens, in the manifest
    experiments.py      one command producing every table and figure
    audio.py asr.py embedding.py matcher.py skg.py challenge.py fusion.py
kavach/                 the frontend (Vite + React + TypeScript)
participant_scripts/    read-speech scripts, one language profile per speaker
tests/                  964 tests, none of which need a GPU
```

Recording a corpus? Read **[RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md)**
before contacting a participant — consent cannot be applied retroactively.

---

## Running it

### Backend

```bash
pip install -r requirements-core.txt      # seconds; no torch
uvicorn kavach.api.app:app --reload --port 8000 --app-dir backend
```

That starts in **degraded mode**: without `speechbrain` and `faster-whisper`
there is no acoustic branch and no transcript, so those branches report
themselves *unmeasured* rather than scoring zero. `GET /api/health` lists what
loaded and what did not. The corpus, the graph explorer and the attack lab all
work in this mode.

For the full system:

```bash
pip install -r requirements.txt           # torch, speechbrain, whisper, LaBSE
export GEMINI_API_KEY=...                 # tagging, challenges, attacker text
```

**Any one provider key is enough, and a free one will do.** The three jobs that
need a language model — semantic tagging, challenge generation and attacker
text — all go through whichever provider has a key: `ANTHROPIC_API_KEY`
(paid; adds prompt caching and the Batch API), `GEMINI_API_KEY` or
`GROQ_API_KEY` (both free), or a local Ollama. Set `KAVACH_LLM_PROVIDER` to pin
one when several are present. With no key at all the system still runs, but
tagging degrades to rules — which is not corpus-grade — and challenges fall
back to a fixed bank whose whole weakness is that it can be enumerated.

Configuration is `backend/kavach/config.py`, overridable by environment
variable with a `KAVACH_` prefix or a `.env` file. Every threshold in there is
a **reasoned starting point, not a fitted value** — `eval/` fits them on a dev
split, and the fitted numbers are what a paper reports.

### Frontend

```bash
cd kavach
cp .env.example .env.local                # VITE_USE_MOCK=false
npm install
npm run dev                               # http://localhost:3000
```

With `VITE_USE_MOCK=true` it runs entirely on `src/api/mock.ts` and needs no
backend. That is a design preview, not the system: the mock's numbers are
hand-written, and its `expectedAnswerEntity` field carries the real challenge
answer, which the backend deliberately never sends.

### Tests

```bash
pytest                                    # 964 passed, ~90s
pytest -m models                          # only the tests needing real checkpoints
```

**Python 3.10+ is required.** On 3.9 every test file fails at collection
(`dataclass(slots=True)`). If the system interpreter is older:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-core.txt
.venv/bin/python -m pytest
```

### Ingesting recordings

Participants return a folder of numbered files. Two commands turn that into a
manifest:

```bash
# 1. discover the folders, get a CSV skeleton with pseudonyms pre-assigned
python -m kavach.ingest --audio recordings --emit-template data/speakers.csv

# 2. fill in consent_ref and script_id, then build the corpus
python -m kavach.ingest --audio recordings --speakers data/speakers.csv \
    --out data/corpus_v1 --provenance SCRIPTED
```

Filenames resolve through `PROTOCOL_V1` **position**, so `03.m4a`, `3.wav` and
`Q3.ogg` all become `p03_commute` — and anything that does not resolve is
skipped loudly rather than guessed at. Add `--dry-run` to check a return
without writing anything.

The participant's folder name never reaches the manifest. It is a first name
sitting beside a voiceprint; `data/speakers.csv` holds the mapping and belongs
with the consent register, not in this repository.

### Annotating them

```bash
python -m kavach.annotate --manifest data/corpus_v1/manifest.json --stage asr
python -m kavach.annotate --manifest ... --stage tag        # any provider key
```

Two stages because they fail differently. ASR is slow, offline and free —
about an hour for a 25-minute corpus on CPU, saved after every utterance so a
crash costs one file. Tagging is seconds per utterance and needs a key; add
`--resume` after a failure so the utterances already tagged are not re-sent
against the same rate limit that stopped the run.

Read the ASR report before tagging. It lists any transcript Whisper
**degenerated** on — a loop like `Rs.Ls.Rs.Rs.Rs.` where a price was spoken.
Re-transcribe or drop those; do not annotate them. Every copy of the looped
word is counted as a real language choice, so the speaker ends up with their
most confident graph cell built from a word they never said.

**Without a key the tagging stage is not usable output.** `lid.rules` resolves
*language* from script evidence and decides no semantic class, so every token
lands in `SemanticClass.OTHER`, the CSBG has one class containing everything,
and every speaker's graph is identical. The run is labelled rather than
refused, in the report and in `Corpus.reportability()`.

### The offline evaluation

```bash
python -m kavach.experiments --out paper/results/
```

Writes `results.json` (the paper reads from here — no number is hand-typed into
LaTeX), `report.md`, `tables/*.tex` and `figures/*.pdf`, alongside the git
commit and every seed. Without `--manifest` it runs on `simulation.py`, and
**every output is stamped unreportable** — in the JSON, in the README it
writes, and as a banner inside each `.tex` file.

---

## What is not in this repository

A clone of this repository **builds and tests clean, and cannot reproduce a
single experimental number.** That is deliberate, not an oversight, and the gap
is exactly one thing: the audio.

| Absent | Why | How a recipient gets it |
|---|---|---|
| `recordings/`, `SpeechData/` — 168 audio files, 49 MB | Participant folder names are people's real first names sitting beside their voiceprints. Direct identifiers; nothing pseudonymises them until `kavach.ingest` runs. | Transferred out of band, by the corpus owner, only to someone the consent register covers. |
| `data/` — 175 MB: `corpus_v1`–`v3`, manifests, `kavach.db` | Voiceprints next to hometowns, schools and family names. Not encrypted. | Rebuilt from the audio with `kavach.ingest`, or transferred with it. |
| `data/speakers*.csv`, `data/consent_register.csv` | The pseudonym→person mapping and the consent record. These are the *only* files that re-identify a speaker; keeping them beside the corpus would defeat the pseudonymisation. | Held by the corpus owner, outside this repository and outside `data/`. |
| `.env` | API keys. `.env.example` lists the variable names. | Recipient supplies their own; any one provider key works and a free one will do. |
| `.venv/`, `kavach/node_modules/`, `pretrained_models/` | Regenerable. | `pip install -r requirements.txt`, `npm install`; the speechbrain checkpoint downloads on first use. |

What *is* here is every line of source, every test, the participant scripts, the
recording protocol, and the finished results of both runs — `paper/results/`
(scripted) and `paper/results_freespeech/` — as `results.json`, `report.md`,
`tables/*.tex` and, for the free-speech run, the figures, since nobody without
the corpus can regenerate those.

So a recipient can read the code, run the tests, build the UI, run the API in
degraded mode and read every number this project has measured. To *extend* the
corpus they need the audio, and the audio needs a consent conversation — see
[RECORDING_PROTOCOL.md](RECORDING_PROTOCOL.md).

---

## Things that are easy to get wrong here

**The demo numbers are not results.** The Attack Lab's acoustic scores are
modelled, not measured — no voice cloner ships with this — so every run carries
`simulated: true` and `AttackTable.paper_ready()` refuses the row. The
Evaluation page scores whatever logins happened to be clicked through, with no
trial design and no dev/test split. Both say so in their own module docstrings.
The reportable numbers come from `eval/` run offline.

**`/api/challenge` does not return the answer.** The frontend's TypeScript says
it does. Filling that field in would let whoever is authenticating read the
expected answer out of the network tab, which does not weaken the knowledge
branch so much as delete it. `Settings.demo_reveal_answers` exists for offline
demos, defaults to `False`, and is reported in `/api/health` so a demo build
announces itself.

**An unmeasured branch is not a failed branch.** A missing model scoring `0.0`
is indistinguishable from an impostor scoring `0.0`, and the first would look
like a working defence in exactly the table this project exists to produce.
`fusion.fuse` renormalises over the branches that ran.

**`data/` holds voiceprints next to hometowns, schools and family names.** It is
git-ignored, it is not encrypted, and a deletion request must remove the audio
and the knowledge graph, not just the speaker row (`Store.delete_speaker`).

**The mock has no failure modes, so run the UI against a *degraded* backend.**
Three frontend bugs survived until the first real run because `mock.ts` always
returns a populated, well-formed payload — one of them crashed every page in
the application when no model was loaded. See PROJECT.md §5.8.
