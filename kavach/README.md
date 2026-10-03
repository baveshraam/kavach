# kavach/ — the frontend

React + Vite + TypeScript, Tailwind v4, TanStack Query, lucide icons. It began as a Google AI Studio export
(`../UI_PROMPT_GOOGLE_AI_STUDIO.md` is the original prompt, kept as a record) and has since been reworked by hand;
extend it rather than replace it, and keep the restrained look (no neon, no gradient-heavy dashboards).

## Run it

```bash
cd kavach
cp .env.example .env.local        # VITE_USE_MOCK=false, VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev                       # http://localhost:3000   (port 3000 is what the backend's CORS allows)
npm run lint                      # tsc --noEmit
npm run build                     # vite build
```

From the repository root, `run_demo.ps1` starts the backend and this UI together and opens `/unlock`;
`run_studio.ps1` does the same for recording sessions (`/studio`).

`VITE_USE_MOCK=true` runs on `src/api/mock.ts` with no backend. That is a design preview, not the system: the mock's
numbers are hand-written, it has no failure modes, and its `expectedAnswerEntity` carries a real answer that the
backend never sends. **Rehearse against the real backend, ideally a degraded one.**

## Pages (`src/pages/`)

| Route | Page | For |
|---|---|---|
| `/unlock` | **Unlock** | **The demo screen.** Ten random words, record, verdict with the voice meter, the gates, the words heard, an attempts list, and what the threshold was measured on |
| `/authenticate` | Authenticate | The personal-question login (the step-up) with the full branch breakdown and the staged attack buttons |
| `/studio` | Recording Studio | The presenter's guided sessions (needs the backend started with `run_studio.ps1`; place is kept across a refresh) |
| `/` | Overview | Corpus and system summary |
| `/enrolment` | Enrol a speaker | Enrolment |
| `/speakers` | Speakers | Speaker drawer: knowledge facts, microphone top-up, rebuild |
| `/graph-explorer` | Graph Explorer | CSBG and SKG |
| `/attack-lab` | Attack Lab | A1-A5, with the clone bank when it exists (every lab run is labelled simulated or measured) |
| `/evaluation` | Evaluation | Offline results from `paper/results*/` with their unreportable banners |
| `/corpus` | Corpus | The recorded corpus |

## The contract with the backend

`src/api/types.ts` mirrors `backend/kavach/api/schemas.py` (camelCase on the wire). A drifted field name fails
`tests/test_api.py::TestFrontendContract` rather than blanking a panel on stage, so change both together. API calls
live in `src/api/client.ts`. The Unlock page reads `/api/health` (thresholds, policy source), `/api/voice-policy`
(the measured numbers), `/api/speakers/{id}/voiceprint` (how the voiceprint was built) and posts to
`/api/challenge` (`kind: "phrase"`, `stepUp`) and `/api/authenticate`.

## Microphone path

`Unlock.tsx` and the Studio record through `getUserMedia({ audio: true })` and `MediaRecorder` (`audio/webm;codecs=opus`),
so a voiceprint enrolled from Studio sessions meets the same audio chain at login. Do not change the capture
constraints in one place only. Use Chrome, and the same browser for the Studio and the demo.
