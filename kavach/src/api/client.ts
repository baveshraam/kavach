import { AuthResult, Challenge, Voiceprint, VoiceEvidence, CloneBankInfo, CloneMatch, CSBG, StudioPlan, StudioSummary, EvalMetrics, Speaker, Utterance, Triple, AttackRun, AttackType, PerSpeakerIapmr, OfflineRun } from './types';
import { mockSpeakers, mockUtterances, mockTriples, mockCSBG, mockAuthResults, mockAttacks, mockEvalMetrics } from './mock';

// @ts-ignore
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
// @ts-ignore
export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/** Backend paths (`/api/audio/...`) made absolute; blob and data URLs pass through. */
export const assetUrl = (url: string) => (url.startsWith('/') ? `${API_BASE}${url}` : url);

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (!response.ok) {
    // FastAPI puts the reason in `detail`. That reason is usually the useful
    // part ("this speaker has no knowledge-graph facts"), so surface it.
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

export const apiClient = {
  health: async (): Promise<{ status: string, models: string[], device: string, demoAttackBank?: boolean, voiceGate?: boolean, phraseWords?: number, voiceThreshold?: number, voiceGreyMargin?: number, voicePolicySource?: string, voicePolicyProvisional?: boolean, voicePolicyError?: string }> => {
    if (USE_MOCK) {
      await delay(200);
      return { status: 'connected', models: ['ecapa-tdnn-v2', 'wav2vec2-large-xlsr-ta', 'llama-3-8b-instruct'], device: 'cuda:0' };
    }
    return fetchApi('/api/health');
  },

  studioPlan: async (speaker: string, session?: string): Promise<StudioPlan> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    const q = new URLSearchParams({ speaker });
    if (session) q.set('session', session);
    return fetchApi(`/api/studio/plan?${q}`);
  },

  studioSummary: async (speaker: string): Promise<StudioSummary> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    return fetchApi(`/api/studio/summary?speaker=${encodeURIComponent(speaker)}`);
  },

  studioUpload: async (v: { speaker: string; sessionId: string; kind: string; promptId: string; device: string; environment: string; textHint: string; stateNote: string; blob: Blob; filename?: string }): Promise<{ summary: StudioSummary }> => {
    if (USE_MOCK) throw new Error('The Studio needs the real backend.');
    const f = new FormData();
    f.append('audio', v.blob, v.filename ?? 'clip.webm');
    f.append('speaker', v.speaker); f.append('session_id', v.sessionId); f.append('kind', v.kind);
    f.append('prompt_id', v.promptId); f.append('device', v.device); f.append('environment', v.environment);
    f.append('text_hint', v.textHint); f.append('state_note', v.stateNote);
    return fetchApi('/api/studio/clips', { method: 'POST', body: f });
  },

  /** What the clone bank can answer (demo builds only; 404 otherwise). */
  cloneBankInfo: async (): Promise<CloneBankInfo> => {
    if (USE_MOCK) throw new Error('The clone bank needs the real backend.');
    return fetchApi('/api/clone-bank');
  },

  /** The attacker's cloned answer to an issued challenge (demo builds only; 404 otherwise). */
  matchCloneClip: async (challengeId: string): Promise<CloneMatch> => {
    if (USE_MOCK) throw new Error('The clone bank needs the real backend.');
    return fetchApi('/api/clone-bank/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challengeId }),
    });
  },

  getSpeakers: async (): Promise<Speaker[]> => {
    if (USE_MOCK) return delay(400).then(() => [...mockSpeakers]);
    const rows = await fetchApi<Speaker[]>('/api/speakers');
    return rows.sort((a, b) => a.displayName.localeCompare(b.displayName, undefined, { numeric: true }));
  },

  getSpeaker: async (id: string): Promise<Speaker> => {
    if (USE_MOCK) {
      const spk = mockSpeakers.find(s => s.id === id);
      if (!spk) throw new Error('Not found');
      return delay(200).then(() => ({ ...spk }));
    }
    return fetchApi(`/api/speakers/${id}`);
  },

  createSpeaker: async (speaker: Omit<Speaker, 'id' | 'enrolledAt' | 'utteranceCount' | 'totalDurationSec' | 'cmi' | 'iIndex' | 'matrixLanguageRatio' | 'csbgDensity'>): Promise<Speaker> => {
    if (USE_MOCK) {
      const newSpk: Speaker = {
        ...speaker,
        id: `spk_${Math.random().toString(36).substring(2, 8)}`,
        enrolledAt: new Date().toISOString(),
        utteranceCount: 0,
        totalDurationSec: 0,
        cmi: 0,
        iIndex: 0,
        matrixLanguageRatio: 0,
        csbgDensity: 0
      };
      return delay(500).then(() => newSpk);
    }
    return fetchApi('/api/speakers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(speaker)
    });
  },

  deleteSpeaker: async (id: string): Promise<{ deleted: true }> => {
    if (USE_MOCK) return delay(400).then(() => ({ deleted: true }));
    return fetchApi(`/api/speakers/${id}`, { method: 'DELETE' });
  },

  getSpeakerUtterances: async (id: string): Promise<Utterance[]> => {
    if (USE_MOCK) return delay(300).then(() => mockUtterances.filter(u => u.speakerId === id));
    return fetchApi(`/api/speakers/${id}/utterances`);
  },

  getSpeakerCSBG: async (id: string): Promise<CSBG> => {
    if (USE_MOCK) return delay(300).then(() => mockCSBG);
    return fetchApi(`/api/speakers/${id}/csbg`);
  },

  getSpeakerSKG: async (id: string): Promise<Triple[]> => {
    if (USE_MOCK) return delay(200).then(() => [...mockTriples]);
    return fetchApi(`/api/speakers/${id}/skg`);
  },

  updateSpeakerSKG: async (id: string, triples: Triple[]): Promise<Triple[]> => {
    if (USE_MOCK) return delay(400).then(() => triples);
    return fetchApi(`/api/speakers/${id}/skg`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(triples)
    });
  },

  completeEnrolment: async (id: string): Promise<{ csbg: CSBG, warnings: string[] }> => {
    if (USE_MOCK) return delay(1500).then(() => ({ csbg: mockCSBG, warnings: [] }));
    return fetchApi(`/api/speakers/${id}/enrol/complete`, { method: 'POST' });
  },

  uploadUtterance: async (speakerId: string, type: string, blob: Blob, filename = 'recording.webm'): Promise<Utterance> => {
    if (USE_MOCK) {
      return delay(800).then(() => ({
        ...mockUtterances[0],
        id: `utt_${Math.random().toString(36).substr(2, 6)}`,
        speakerId,
        type: type as any,
        audioUrl: URL.createObjectURL(blob),
        durationSec: 4.5,
        recordedAt: new Date().toISOString()
      }));
    }
    const formData = new FormData();
    formData.append('audio', blob, filename);
    formData.append('speakerId', speakerId);
    formData.append('type', type);
    return fetchApi('/api/utterances', { method: 'POST', body: formData });
  },

  getUtterances: async (): Promise<Utterance[]> => {
    if (USE_MOCK) return delay(400).then(() => [...mockUtterances]);
    return fetchApi('/api/utterances');
  },

  deleteUtterance: async (id: string): Promise<{ deleted: true }> => {
    if (USE_MOCK) return delay(300).then(() => ({ deleted: true }));
    return fetchApi(`/api/utterances/${id}`, { method: 'DELETE' });
  },

  /** The measured operating point and the numbers behind it. */
  voicePolicy: async (): Promise<VoiceEvidence> => {
    if (USE_MOCK) throw new Error('The evidence needs the real backend.');
    return fetchApi('/api/voice-policy');
  },

  /** How the enrolled voiceprint was built (clip count, Studio sessions, devices). */
  voiceprint: async (speakerId: string): Promise<Voiceprint> => {
    if (USE_MOCK) return { nClips: 13, selfConsistency: 0.85, provenance: null };
    return fetchApi(`/api/speakers/${speakerId}/voiceprint`);
  },

  issueChallenge: async (speakerId: string, kind: 'question' | 'phrase' = 'question', stepUp = false): Promise<Challenge> => {
    if (USE_MOCK) {
      return delay(300).then(() => ({
        id: `chg_${Math.random().toString(36).substr(2, 6)}`,
        speakerId,
        questionText: kind === 'phrase' ? 'Please read aloud, clearly: tiger, river, mango, window, candle, silver' : 'உங்க college-la first year-la எந்த hostel-la இருந்தீங்க?',
        kind,
        phrase: kind === 'phrase' ? ['tiger', 'river', 'mango', 'window', 'candle', 'silver'] : [],
        stepUp,
        targetClass: 'PLACE_LOCAL',
        expectedAnswerEntity: 'Hostel A',
        issuedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 60000).toISOString()
      }));
    }
    return fetchApi('/api/challenge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speakerId, kind, stepUp })
    });
  },

  authenticate: async (challengeId: string, blob: Blob, filename = 'response.webm'): Promise<AuthResult> => {
    if (USE_MOCK) return delay(1200).then(() => mockAuthResults[0]);
    const formData = new FormData();
    formData.append('audio', blob, filename);
    formData.append('challengeId', challengeId);
    return fetchApi('/api/authenticate', { method: 'POST', body: formData });
  },

  getAuthHistory: async (): Promise<AuthResult[]> => {
    if (USE_MOCK) return delay(400).then(() => [...mockAuthResults]);
    return fetchApi('/api/auth-history?limit=50');
  },

  generateAttack: async (attackType: string, targetSpeakerId: string, trials: number): Promise<AttackRun> => {
    if (USE_MOCK) {
      return delay(2000).then(() => ({
        id: `atk_${Math.random().toString(36).substr(2, 6)}`,
        attackType: attackType as AttackType,
        targetSpeakerId,
        trials,
        successRateByConfig: { ecapa_only: 0.85, plus_knowledge: 0.12, plus_csbg: 0.08, full_fusion: 0.01 },
        generatedAt: new Date().toISOString()
      }));
    }
    return fetchApi('/api/attacks/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ attackType, targetSpeakerId, trials })
    });
  },

  getAttacks: async (): Promise<AttackRun[]> => {
    if (USE_MOCK) return delay(400).then(() => [...mockAttacks]);
    return fetchApi('/api/attacks');
  },

  // Per-speaker attack success. Read this before quoting any mean: a system
  // that stops every attack on 24 speakers and none on the 25th reports 96%.
  getPerSpeakerIapmr: async (): Promise<PerSpeakerIapmr> => {
    if (USE_MOCK) {
      await delay(300);
      return {
        speakers: [], worstSpeakerId: '', meanIapmr: null,
        unmeasuredSpeakerIds: [], minTrialsPerCell: 30, simulated: true,
        notes: ['Mock mode: no attacks have been run.'],
      };
    }
    return fetchApi('/api/attacks/per-speaker');
  },

  getEvaluation: async (): Promise<EvalMetrics> => {
    if (USE_MOCK) return delay(400).then(() => mockEvalMetrics);
    return fetchApi('/api/evaluation');
  },

  getOfflineResults: async (): Promise<OfflineRun[]> => {
    if (USE_MOCK) return delay(200).then(() => []);
    return fetchApi('/api/offline-results');
  },

  offlineFigureUrl: (run: string, name: string) => `${API_BASE}/api/offline-results/${run}/figures/${name}.png`,
};
