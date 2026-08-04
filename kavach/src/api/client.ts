import { AuthResult, Challenge, CSBG, EvalMetrics, Speaker, Utterance, Triple, AttackRun, AttackType } from './types';
import { mockSpeakers, mockUtterances, mockTriples, mockCSBG, mockAuthResults, mockAttacks, mockEvalMetrics } from './mock';

// @ts-ignore
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
// @ts-ignore
const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (!response.ok) {
    throw new Error(`API Error: ${response.statusText}`);
  }
  return response.json();
}

export const apiClient = {
  health: async (): Promise<{ status: string, models: string[], device: string }> => {
    if (USE_MOCK) {
      await delay(200);
      return { status: 'connected', models: ['ecapa-tdnn-v2', 'wav2vec2-large-xlsr-ta', 'llama-3-8b-instruct'], device: 'cuda:0' };
    }
    return fetchApi('/api/health');
  },

  getSpeakers: async (): Promise<Speaker[]> => {
    if (USE_MOCK) return delay(400).then(() => [...mockSpeakers]);
    return fetchApi('/api/speakers');
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

  uploadUtterance: async (speakerId: string, type: string, blob: Blob): Promise<Utterance> => {
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
    formData.append('audio', blob);
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

  issueChallenge: async (speakerId: string): Promise<Challenge> => {
    if (USE_MOCK) {
      return delay(300).then(() => ({
        id: `chg_${Math.random().toString(36).substr(2, 6)}`,
        speakerId,
        questionText: 'உங்க college-la first year-la எந்த hostel-la இருந்தீங்க?',
        targetClass: 'PLACE_LOCAL',
        expectedAnswerEntity: 'Hostel A',
        issuedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 60000).toISOString()
      }));
    }
    return fetchApi('/api/challenge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speakerId })
    });
  },

  authenticate: async (challengeId: string, blob: Blob): Promise<AuthResult> => {
    if (USE_MOCK) return delay(1200).then(() => mockAuthResults[0]);
    const formData = new FormData();
    formData.append('audio', blob);
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

  getEvaluation: async (): Promise<EvalMetrics> => {
    if (USE_MOCK) return delay(400).then(() => mockEvalMetrics);
    return fetchApi('/api/evaluation');
  }
};
