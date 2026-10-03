# Google AI Studio Prompt — KAVACH Frontend

> **Status, 2026-10-03: historical.** This is the prompt that produced the first version of the UI (2026-08-04). The UI has been reworked by hand since (new design tokens, a shared kit, the Unlock, Studio and Attack Lab changes) and this prompt is not kept in sync. The living description is `kavach/README.md`; the contract with the backend is `kavach/src/api/types.ts`.


Paste everything below the line into Google AI Studio. It is written to produce a restrained, professional research-instrument UI (no AI-slop aesthetics) and to match the backend API contract exactly, so wiring it up afterwards is mechanical.

---

Build a **React + TypeScript + Vite** single-page web application called **KAVACH** — a research console for a multilingual voice authentication system. This is a laboratory instrument used by speech researchers to collect data, run authentication trials, simulate attacks, and read evaluation metrics. It is not a consumer product and must not look like one.

## CRITICAL — Visual design constraints

I have seen a lot of generated UIs and I do not want the default look. Follow these rules strictly.

**Absolutely forbidden:**
- No neon, no glow effects, no `box-shadow` used as a light source
- No purple/violet/indigo gradient backgrounds, no dark "space" or "cyber" themes
- No glassmorphism, no `backdrop-filter: blur`, no translucent frosted panels
- No gradient text, no gradient buttons, no animated gradient borders
- No emoji used as interface icons
- No large hero sections, no marketing copy, no centered oversized headlines
- No rounded-3xl "bubbly" cards, no heavy drop shadows
- No animated background blobs, particles, meshes, or floating orbs
- No "✨ AI-powered" styling cues of any kind

**Required aesthetic — think Bloomberg terminal, Linear, or a lab instrument panel:**
- Light theme by default, on a neutral off-white page background (`#FAFAF9` or similar warm grey). Optional dark mode as a plain neutral dark grey (`#1A1A19`), never blue-black or purple-black.
- **Exactly one accent colour**, desaturated and serious: a deep slate blue (`#2C4A6E`) or a muted forest green. Use it for primary actions, active states, and selected rows — nothing else.
- Semantic colours used **only** for state, muted rather than saturated: accept/pass `#2F6F4E`, reject/fail `#9B3232`, warning/borderline `#8A6A1F`. These may appear as text, a 1px border, or a small filled dot — never as a large coloured block.
- **Borders, not shadows.** 1px solid `#E5E4E1` (light) / `#33322F` (dark) to separate regions. At most one very subtle shadow on modals.
- Border radius: `4px` maximum. Most things `2px` or square.
- Typography: **Inter** (or system UI stack) for text; **IBM Plex Mono** or **JetBrains Mono** for all numbers, scores, IDs, timestamps, file names, and transcripts. Every numeric value in this app is monospaced and tabular-aligned (`font-variant-numeric: tabular-nums`).
- Type scale is small and dense: body 13px, labels 11px uppercase with letter-spacing, headings 15–18px semibold. Do not use 24px+ text anywhere except the app title.
- Data density is a feature. Tables with tight row heights (~32px), thin dividers, right-aligned numeric columns. Do not pad things out into airy cards.
- Icons: a thin-stroke line set (Lucide, 16px, 1.5px stroke), monochrome, inheriting text colour.
- Transitions: 120ms ease on hover/active only. No entrance animations, no page transitions, no spring physics.

The overall impression should be: **quiet, dense, precise, and slightly austere.** A tool that respects the user's time and displays numbers honestly.

## Layout

Fixed left sidebar, 200px wide, with a 1px right border. At the top, the app name `KAVACH` in 14px semibold plus a 10px monospace subtitle `v0.1 · research build`. Below it, a flat vertical nav list (no icons-only rail, no collapsible accordion):

`Overview` · `Enrolment` · `Authenticate` · `Speakers` · `Graph Explorer` · `Attack Lab` · `Evaluation` · `Corpus`

Active item: accent-coloured left border 2px, slightly darker text, no filled pill background.

Main content area: a 48px header bar with the page title on the left and page-level actions on the right, then the page content with 24px padding and a max width of 1400px.

At the very bottom of the sidebar, a small monospace status block showing backend connection state (`● connected` / `● offline`), model names loaded, and device (`cpu` / `cuda`).

---

## Page 1 — Overview

A dense dashboard. Top row: six small stat tiles in a single row, each just a 11px uppercase label, a large monospace number, and a 11px muted delta or context line. No icons, no coloured backgrounds, separated by 1px borders.

Tiles: `Enrolled Speakers`, `Total Utterances`, `Corpus Duration`, `Auth Trials (7d)`, `System EER`, `Attack Rejection Rate`.

Below, two columns:
- **Left (60%):** "Recent Authentication Attempts" — a table with columns `Time`, `Speaker`, `Challenge`, `S_spk`, `S_csbg`, `S_know`, `Fused`, `Decision`. Scores are monospace to 3 decimal places. Decision is a small coloured dot plus the word `ACCEPT` / `REJECT` / `BORDERLINE`. Rows are clickable and open a detail drawer.
- **Right (40%):** "Score Distributions" — a small overlaid histogram of genuine vs. impostor fused scores, with a vertical dashed line at the operating threshold. Muted colours, thin 1px axis lines, no chart gridline clutter, no chart title inside the plot area.

---

## Page 2 — Enrolment

A four-step wizard. The step indicator is a thin horizontal row of numbered steps with 1px connector lines — not large circles, not a progress bar with a gradient.

**Step 1 — Speaker Profile.** A compact form: Speaker ID (auto-generated, monospace, editable), Display Name, Age Range (select), Gender (select, includes "prefer not to say"), Dominant Language (select: Tamil / English / Balanced), Other Languages (multi-input), Device Used (select), Recording Environment (select: quiet room / office / outdoor / noisy). Below the form, a bordered consent panel with the consent text and a required checkbox: "Speaker has given written informed consent for voice biometric collection, personal-fact collection, and anonymised corpus release." The Continue button is disabled until it is checked.

**Step 2 — Voice Samples.** The speaker reads or speaks 6 prompts. Show the prompt list as a table: `#`, `Prompt Text` (in Tamil script / English / code-mixed as appropriate), `Type` (monolingual-ta / monolingual-en / code-mixed / free-speech), `Duration`, `Status`, `Actions`.

The recorder is a horizontal bar fixed above the table: a square Record button (accent when armed, muted red dot when recording), a live monospace timer `00:00.0`, a **live waveform** rendered as thin vertical bars from the Web Audio API `AnalyserNode` — monochrome, no colour cycling, no circular visualiser — and a live input-level meter as a thin horizontal bar with a clipping indicator. After recording, show playback with a static waveform, plus `Re-record` and `Accept` actions.

Show a running total: `Recorded: 4 / 6 · Total duration: 02:41 · Target: 05:00` with a thin progress bar (1px track, accent fill, no rounding).

**Step 3 — Knowledge Graph Interview.** Present 10–12 personal questions one at a time, each answered by voice (same recorder component). Questions like "Which town are you from?", "What is your favourite food?", "Where did you go to school?". As each answer is transcribed, show the extracted triple in a monospace list on the right side:
```
:speaker_07  :hometown       :Thanjavur
:speaker_07  :favouriteFood  :Kothu_Parotta
```
Each extracted triple has a small inline edit control and a delete control, because ASR will get things wrong and the researcher must correct them. Include a clear privacy notice at the top of this step in a bordered panel: these facts are stored encrypted and are never included in the released corpus.

**Step 4 — Build & Review.** Show a processing checklist that fills in as the backend works — each line has a small square status glyph and monospace timing:
```
[✓] Transcribed 6 utterances                    4.2s
[✓] Word-level language ID (1,204 tokens)       2.8s
[✓] Semantic class tagging                      3.1s
[✓] ECAPA-TDNN embeddings extracted             1.4s
[✓] Code-Switch Behaviour Graph constructed     0.3s
[!] Sparse data: 4 classes below threshold
```
Then a summary panel showing the speaker's headline code-switch statistics (CMI, I-index, matrix-language ratio, tokens per class) as a compact stat table, and a preview of their CSBG (see Page 5). Warn clearly if enrolment duration is below the recommended minimum.

---

## Page 3 — Authenticate

The core demo screen. Two-column layout.

**Left column — the challenge flow:**
1. A speaker selector (searchable dropdown of enrolled speakers) and a `Issue Challenge` button.
2. Once issued, a prominent bordered panel showing the challenge question in code-mixed Tamil–English at 18px, e.g. `உங்க college-la first year-la எந்த hostel-la இருந்தீங்க?` With, in small muted monospace below it: `challenge_id: chg_8f2a1c · target_class: PLACE_LOCAL · issued: 14:32:07 · expires in 0:28` with a live countdown.
3. The recorder component (same as enrolment).
4. After recording, the transcript appears **with each token colour-coded by detected language** — Tamil tokens in the default text colour, English tokens in the accent colour, named entities underlined with a dotted border, neutral tokens muted. Hovering a token shows a tooltip with its semantic class and LID confidence. This is the most visually distinctive element in the app and it must be understated: colour-coding via text colour only, never background highlights.

**Right column — live score breakdown.** Four horizontal score rows, each showing: branch name (`Speaker Embedding`, `Code-Switch Graph`, `Knowledge Match`, `Liveness`), a monospace score to 3 decimals, a thin horizontal bar with a vertical tick marking that branch's threshold, and the branch weight in small muted monospace. Below them, a 1px divider, then the fused score, larger and monospace, and the final decision rendered as text in the semantic colour with a small filled square: `■ ACCEPT` / `■ REJECT`.

Below the decision, an "Explanation" panel — plain sentences generated from the graph, e.g.:
```
Rejected. Code-switch divergence in classes:
  PLACE_LOCAL   expected TA (0.91)  observed EN (0.88)   JSD 0.74
  NUMBER        expected EN (0.94)  observed EN (0.90)   JSD 0.02
Knowledge answer matched (semantic 0.88, phonetic 0.71).
Speaker embedding matched (cosine 0.79, threshold 0.62).
```
This explainability panel is a key selling point — make it clear and readable, not a raw JSON dump.

---

## Page 4 — Speakers

A dense table of all enrolled speakers: `ID`, `Name`, `Dominant Lang`, `Utterances`, `Duration`, `CMI`, `I-index`, `Matrix Lang`, `CSBG Density`, `Enrolled`, `Actions`. Sortable columns, a text filter, and a multi-select with bulk actions. Clicking a row opens a full-height right drawer with tabs: `Profile`, `Recordings`, `Knowledge Graph`, `CSBG`, `Auth History`. Include a `Delete speaker & all data` action in the drawer, styled as a plain destructive text button with a confirmation dialog — a data-subject deletion request must be one click away.

---

## Page 5 — Graph Explorer

Interactive knowledge graph visualisation using **Cytoscape.js** (or react-force-graph if simpler).

Controls in a thin toolbar: a speaker selector, a graph-type toggle (`Code-Switch Behaviour Graph` / `Speaker Knowledge Graph`), an edge-weight threshold slider, a layout selector (force / concentric / hierarchical), and a `Compare with…` selector that loads a second speaker's graph side by side.

**CSBG rendering:** semantic-class nodes as small squares with a 1px border and a text label outside the node; language nodes (`TA`, `EN`, `NEUTRAL`) as slightly larger nodes in the accent colour. Edge thickness maps to probability; edge opacity maps to observation count. Hovering an edge shows `P(EN | NUMBER) = 0.94, n = 31`. No rainbow node colouring — the whole graph is monochrome plus the single accent, with edge thickness carrying the information.

**Comparison mode:** two graphs side by side with a shared layout, and edges that differ significantly between the two speakers outlined in the reject colour. Below the graphs, a divergence table: `Class`, `Speaker A P(EN)`, `Speaker B P(EN)`, `JSD`, sorted descending by JSD.

A `Export as SVG` button — these are going into a paper.

---

## Page 6 — Attack Lab

A console for generating and running spoofing attacks against enrolled speakers.

Top section, "Generate Attack": a target speaker selector and five attack-type cards laid out in a row — again as bordered panels, not colourful cards. Each shows the attack code, name, a one-line description, required inputs, and a `Generate` action:
- `A1 · Replay` — replay a previously recorded utterance
- `A2 · Splice` — concatenate recorded word segments into a target answer
- `A3 · Clone (naive)` — TTS voice clone, no knowledge of the answer
- `A4 · Clone + Knowledge` — TTS clone that speaks the correct answer
- `A5 · Style-Adaptive Clone` — clone that also imitates the speaker's code-switch pattern

Bottom section, "Attack Results": a matrix table with attack types as rows and defence configurations as columns (`ECAPA only`, `+ Knowledge`, `+ CSBG`, `Full fusion`), each cell showing the attack success rate as a monospace percentage with a subtle background tint proportional to the value — use a single-hue tint from transparent to a very light red, not a rainbow heatmap. Below it, per-attack detail rows expandable to show individual trial scores.

Include a clearly worded ethics notice panel at the top of this page stating that generated attack audio is for evaluating defences, is stored separately, and is not released.

---

## Page 7 — Evaluation

The results page that feeds the paper. All charts use thin 1px lines, monochrome plus the accent, no chart junk, no legends inside the plot area, and axis labels in 11px.

- **DET curves** — one plot with a curve per system configuration (ECAPA only, CSBG only, each fusion), log-scaled axes, EER points marked with a small square, and a legend to the right of the plot as a plain list.
- **Ablation table** — rows are system configurations, columns are `EER %`, `minDCF`, `FAR@FRR=1%`, `FRR@FAR=1%`. Best value in each column in semibold, not highlighted with colour.
- **CSBG stability curve** — x-axis enrolment duration (30s → 5min), y-axis EER, with a shaded confidence band in a very light grey.
- **Fairness breakdown** — a grouped bar chart of EER by condition (monolingual Tamil / monolingual English / code-mixed) and by gender, with the sample count printed above each bar because small-n results must be read with caution.
- **Score distributions** — genuine vs. impostor histograms, one small multiple per branch, arranged in a 2×2 grid.

Each chart has an `Export SVG` and `Export CSV` control in a thin header row above it.

---

## Page 8 — Corpus

Data management. A table of all recordings: `File`, `Speaker`, `Type`, `Duration`, `Sample Rate`, `Transcript`, `LID Tokens`, `Annotated`, `Actions`. Filters by speaker, type, and annotation status. A summary strip at the top with total duration, speaker count, token count, and the TA/EN/NEUTRAL token split as a single thin stacked bar. Export controls for the corpus manifest (JSON / CSV) and an annotation-progress indicator.

---

## API contract

Generate a `src/api/client.ts` with typed functions for every endpoint below, and a `src/api/types.ts` with these exact types. Point the base URL at `http://localhost:8000` via `import.meta.env.VITE_API_BASE_URL` with that as the fallback.

```typescript
export type Language = 'TA' | 'EN' | 'NEUTRAL' | 'NAMED_ENTITY';

export type SemanticClass =
  | 'NUMBER' | 'TIME_DATE' | 'KINSHIP' | 'FOOD' | 'PLACE_LOCAL' | 'PLACE_GLOBAL'
  | 'TECH_DIGITAL' | 'EDU_WORK' | 'MONEY_COMMERCE' | 'EMOTION_STATE'
  | 'BODY_HEALTH' | 'TRANSPORT' | 'RELIGION_FESTIVAL' | 'MEDIA_ENTERTAIN'
  | 'DISCOURSE_MARKER' | 'POLITENESS' | 'QUANTITY_MEASURE' | 'ACTION_VERB'
  | 'FUNCTION_WORD' | 'NAMED_ENTITY' | 'OTHER';

export interface Speaker {
  id: string;
  displayName: string;
  ageRange: string;
  gender: string;
  dominantLanguage: 'Tamil' | 'English' | 'Balanced';
  otherLanguages: string[];
  device: string;
  environment: string;
  consentGiven: boolean;
  enrolledAt: string;          // ISO 8601
  utteranceCount: number;
  totalDurationSec: number;
  cmi: number;                 // 0..100
  iIndex: number;              // 0..1
  matrixLanguageRatio: number; // 0..1, fraction Tamil-matrix
  csbgDensity: number;         // 0..1
}

export interface Token {
  text: string;
  language: Language;
  semanticClass: SemanticClass;
  lidConfidence: number;       // 0..1
  startMs: number;
  endMs: number;
}

export interface Utterance {
  id: string;
  speakerId: string;
  type: 'monolingual-ta' | 'monolingual-en' | 'code-mixed' | 'free-speech' | 'auth-response';
  audioUrl: string;
  durationSec: number;
  sampleRate: number;
  transcript: string;
  tokens: Token[];
  annotated: boolean;
  recordedAt: string;
}

export interface Triple { subject: string; predicate: string; object: string; }

export interface Challenge {
  id: string;
  speakerId: string;
  questionText: string;        // code-mixed Tamil-English
  targetClass: SemanticClass;
  expectedAnswerEntity: string;
  issuedAt: string;
  expiresAt: string;
}

export interface BranchScore {
  name: 'speaker_embedding' | 'csbg' | 'knowledge' | 'liveness';
  score: number;               // 0..1
  threshold: number;
  weight: number;
  passed: boolean;
}

export interface ClassDivergence {
  semanticClass: SemanticClass;
  expectedLanguage: Language;
  expectedProb: number;
  observedLanguage: Language;
  observedProb: number;
  jsd: number;
  tokenCount: number;
}

export interface AuthResult {
  id: string;
  speakerId: string;
  challengeId: string;
  transcript: string;
  tokens: Token[];
  branches: BranchScore[];
  fusedScore: number;
  fusedThreshold: number;
  decision: 'ACCEPT' | 'REJECT' | 'BORDERLINE';
  divergences: ClassDivergence[];
  explanation: string[];       // human-readable sentences
  latencyMs: number;
  timestamp: string;
}

export interface CSBGNode { id: string; kind: 'class' | 'language'; label: string; tokenCount: number; }
export interface CSBGEdge { source: string; target: string; probability: number; observationCount: number; edgeType: 'lexical_choice' | 'switch_transition'; }
export interface CSBG {
  speakerId: string;
  nodes: CSBGNode[];
  edges: CSBGEdge[];
  cmi: number;
  iIndex: number;
  matrixLanguageRatio: number;
  sparseClasses: SemanticClass[];
}

export type AttackType = 'A1_REPLAY' | 'A2_SPLICE' | 'A3_CLONE_NAIVE' | 'A4_CLONE_KNOWLEDGE' | 'A5_CLONE_ADAPTIVE';

export interface AttackRun {
  id: string;
  attackType: AttackType;
  targetSpeakerId: string;
  trials: number;
  successRateByConfig: Record<'ecapa_only' | 'plus_knowledge' | 'plus_csbg' | 'full_fusion', number>;
  generatedAt: string;
}

export interface EvalMetrics {
  configurations: Array<{
    name: string;
    eer: number;
    minDcf: number;
    farAtFrr1: number;
    frrAtFar1: number;
    detCurve: Array<{ far: number; frr: number }>;
  }>;
  stabilityCurve: Array<{ durationSec: number; eer: number; ciLow: number; ciHigh: number }>;
  fairness: Array<{ condition: string; group: string; eer: number; sampleCount: number }>;
  scoreDistributions: Array<{ branch: string; genuine: number[]; impostor: number[] }>;
}
```

**Endpoints:**

```
GET    /api/health                          -> { status, models: string[], device: string }
GET    /api/speakers                        -> Speaker[]
POST   /api/speakers                        -> Speaker
GET    /api/speakers/{id}                   -> Speaker
DELETE /api/speakers/{id}                   -> { deleted: true }
GET    /api/speakers/{id}/utterances        -> Utterance[]
GET    /api/speakers/{id}/csbg              -> CSBG
GET    /api/speakers/{id}/skg               -> Triple[]
PUT    /api/speakers/{id}/skg               -> Triple[]
POST   /api/speakers/{id}/enrol/complete    -> { csbg: CSBG, warnings: string[] }

POST   /api/utterances            (multipart: audio file + speakerId + type) -> Utterance
GET    /api/utterances                      -> Utterance[]
DELETE /api/utterances/{id}                 -> { deleted: true }

POST   /api/challenge             { speakerId }                      -> Challenge
POST   /api/authenticate          (multipart: audio + challengeId)   -> AuthResult
GET    /api/auth-history?limit=50                                    -> AuthResult[]

POST   /api/attacks/generate      { attackType, targetSpeakerId, trials } -> AttackRun
GET    /api/attacks                         -> AttackRun[]

GET    /api/evaluation                      -> EvalMetrics
GET    /api/corpus/manifest?format=json|csv -> file download
```

## Implementation requirements

- **Mock data mode.** Because the backend does not exist yet, implement a `src/api/mock.ts` with realistic mock data for every type — 12 mock speakers with plausible Tamil names and varied code-switch statistics, realistic Tamil–English code-mixed transcripts with proper token-level tags, believable score values, and full CSBG graphs. Toggle with `VITE_USE_MOCK=true` (default true). Every screen must render fully and look finished with mock data alone. Keep all mock data in that one file so it can be deleted cleanly later.
- **Audio recording** via `navigator.mediaDevices.getUserMedia` + `MediaRecorder`, producing WebM/Opus, with a reusable `<AudioRecorder>` component exposing `onRecordingComplete(blob, durationMs)`. Include the live waveform and level meter via `AnalyserNode`. Handle permission denial with a plain inline error message.
- **State:** React Query (TanStack Query) for server state, plain `useState`/`useReducer` for local UI state. No Redux.
- **Styling:** Tailwind CSS, but define the palette, spacing, and type scale as CSS custom properties in `index.css` and reference them via the Tailwind theme so the whole design can be retuned from one file. Do not scatter arbitrary hex values through the JSX.
- **Charts:** Recharts, restyled to the constraints above — thin axes, no default purple/blue palette, no drop shadows on bars, no rounded bar tops, no legend chrome.
- **Graph viz:** Cytoscape.js via `cytoscape` + a thin React wrapper.
- **Loading states:** thin 2px indeterminate bar under the page header, or muted skeleton rows in tables. No spinners in the middle of the page, no pulsing gradient skeletons.
- **Empty states:** one line of muted 13px text plus a single text-button action. No illustrations, no large icons.
- **Errors:** an inline bordered panel with the reject colour on a 1px border and plain text. No toast pop-ups for errors that belong on the page.
- **Accessibility:** full keyboard navigation, visible 2px accent focus rings, correct ARIA on the recorder and the wizard, and semantic table markup. Verify contrast ratios meet WCAG AA — the muted palette makes this easy to get wrong.
- **Structure:** `src/pages/`, `src/components/`, `src/api/`, `src/hooks/`, `src/lib/`. One component per file, typed props, no `any`.
- The app must build and run with `npm install && npm run dev` with no further setup.

Produce the complete, working project — every page implemented, no `TODO` placeholders, no "rest of implementation omitted" comments.
