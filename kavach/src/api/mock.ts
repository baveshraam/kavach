import {
  AuthResult,
  Challenge,
  CSBG,
  EvalMetrics,
  Speaker,
  Utterance,
  Triple,
  AttackRun,
  Token
} from './types';

const now = new Date().toISOString();

export const mockSpeakers: Speaker[] = [
  { id: 'spk_001', displayName: 'Anbu M.', ageRange: '20-29', gender: 'Male', dominantLanguage: 'Tamil', otherLanguages: ['English'], device: 'iPhone 13', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-01T10:00:00Z', utteranceCount: 24, totalDurationSec: 320.5, cmi: 28.4, iIndex: 0.12, matrixLanguageRatio: 0.85, csbgDensity: 0.65 },
  { id: 'spk_002', displayName: 'Balaji K.', ageRange: '30-39', gender: 'Male', dominantLanguage: 'Balanced', otherLanguages: ['English', 'Telugu'], device: 'Samsung S22', environment: 'office', consentGiven: true, enrolledAt: '2026-07-02T11:30:00Z', utteranceCount: 45, totalDurationSec: 610.2, cmi: 45.2, iIndex: 0.35, matrixLanguageRatio: 0.55, csbgDensity: 0.88 },
  { id: 'spk_003', displayName: 'Chitra S.', ageRange: '20-29', gender: 'Female', dominantLanguage: 'English', otherLanguages: ['Tamil', 'Hindi'], device: 'MacBook Pro', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-03T14:15:00Z', utteranceCount: 18, totalDurationSec: 240.1, cmi: 15.6, iIndex: 0.05, matrixLanguageRatio: 0.15, csbgDensity: 0.42 },
  { id: 'spk_004', displayName: 'Dinesh V.', ageRange: '40-49', gender: 'Male', dominantLanguage: 'Tamil', otherLanguages: ['English'], device: 'Redmi Note 10', environment: 'noisy', consentGiven: true, enrolledAt: '2026-07-04T09:45:00Z', utteranceCount: 32, totalDurationSec: 415.8, cmi: 32.1, iIndex: 0.18, matrixLanguageRatio: 0.78, csbgDensity: 0.71 },
  { id: 'spk_005', displayName: 'Elango P.', ageRange: '50-59', gender: 'Male', dominantLanguage: 'Tamil', otherLanguages: [], device: 'Nokia C20', environment: 'outdoor', consentGiven: true, enrolledAt: '2026-07-05T16:20:00Z', utteranceCount: 12, totalDurationSec: 150.4, cmi: 5.2, iIndex: 0.01, matrixLanguageRatio: 0.98, csbgDensity: 0.25 },
  { id: 'spk_006', displayName: 'Farook A.', ageRange: '30-39', gender: 'Male', dominantLanguage: 'Balanced', otherLanguages: ['English', 'Urdu'], device: 'iPhone 11', environment: 'office', consentGiven: true, enrolledAt: '2026-07-06T10:10:00Z', utteranceCount: 28, totalDurationSec: 380.9, cmi: 38.5, iIndex: 0.22, matrixLanguageRatio: 0.62, csbgDensity: 0.79 },
  { id: 'spk_007', displayName: 'Geetha R.', ageRange: '20-29', gender: 'Female', dominantLanguage: 'Balanced', otherLanguages: ['English'], device: 'OnePlus 9', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-07T13:55:00Z', utteranceCount: 40, totalDurationSec: 520.3, cmi: 42.8, iIndex: 0.28, matrixLanguageRatio: 0.58, csbgDensity: 0.85 },
  { id: 'spk_008', displayName: 'Hari N.', ageRange: '20-29', gender: 'Male', dominantLanguage: 'English', otherLanguages: ['Tamil'], device: 'iPhone 14', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-08T15:30:00Z', utteranceCount: 22, totalDurationSec: 290.7, cmi: 22.4, iIndex: 0.08, matrixLanguageRatio: 0.35, csbgDensity: 0.55 },
  { id: 'spk_009', displayName: 'Ilakkiya T.', ageRange: '30-39', gender: 'Female', dominantLanguage: 'Tamil', otherLanguages: ['English'], device: 'Samsung A52', environment: 'office', consentGiven: true, enrolledAt: '2026-07-09T08:20:00Z', utteranceCount: 35, totalDurationSec: 460.1, cmi: 35.6, iIndex: 0.20, matrixLanguageRatio: 0.72, csbgDensity: 0.76 },
  { id: 'spk_010', displayName: 'Janani L.', ageRange: '40-49', gender: 'Female', dominantLanguage: 'Balanced', otherLanguages: ['English', 'Kannada'], device: 'Pixel 6', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-10T11:45:00Z', utteranceCount: 50, totalDurationSec: 680.5, cmi: 48.9, iIndex: 0.40, matrixLanguageRatio: 0.50, csbgDensity: 0.92 },
  { id: 'spk_011', displayName: 'Karthik C.', ageRange: '20-29', gender: 'Male', dominantLanguage: 'Tamil', otherLanguages: ['English'], device: 'iPhone 12', environment: 'outdoor', consentGiven: true, enrolledAt: '2026-07-11T14:10:00Z', utteranceCount: 26, totalDurationSec: 340.2, cmi: 30.2, iIndex: 0.15, matrixLanguageRatio: 0.80, csbgDensity: 0.68 },
  { id: 'spk_012', displayName: 'Lakshmi V.', ageRange: '50-59', gender: 'Female', dominantLanguage: 'Tamil', otherLanguages: [], device: 'Realme 7', environment: 'quiet room', consentGiven: true, enrolledAt: '2026-07-12T16:00:00Z', utteranceCount: 15, totalDurationSec: 190.8, cmi: 8.5, iIndex: 0.02, matrixLanguageRatio: 0.95, csbgDensity: 0.35 },
];

export const mockUtterances: Utterance[] = mockSpeakers.map(spk => ({
  id: `utt_${Math.random().toString(36).substr(2, 6)}`,
  speakerId: spk.id,
  type: 'code-mixed',
  audioUrl: '/mock-audio.webm',
  durationSec: 8.4,
  sampleRate: 16000,
  transcript: "Naan Thanjavur la irunthu varen, my favourite food is Kothu Parotta.",
  tokens: [
    { text: "Naan", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.98, startMs: 0, endMs: 200 },
    { text: "Thanjavur", language: "NAMED_ENTITY", semanticClass: "PLACE_LOCAL", lidConfidence: 0.95, startMs: 250, endMs: 800 },
    { text: "la", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 800, endMs: 950 },
    { text: "irunthu", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.97, startMs: 1000, endMs: 1400 },
    { text: "varen,", language: "TA", semanticClass: "ACTION_VERB", lidConfidence: 0.96, startMs: 1450, endMs: 1900 },
    { text: "my", language: "EN", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 2200, endMs: 2400 },
    { text: "favourite", language: "EN", semanticClass: "EMOTION_STATE", lidConfidence: 0.98, startMs: 2450, endMs: 3000 },
    { text: "food", language: "EN", semanticClass: "FOOD", lidConfidence: 0.99, startMs: 3050, endMs: 3300 },
    { text: "is", language: "EN", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 3350, endMs: 3500 },
    { text: "Kothu Parotta.", language: "NAMED_ENTITY", semanticClass: "FOOD", lidConfidence: 0.94, startMs: 3550, endMs: 4500 }
  ],
  annotated: true,
  recordedAt: spk.enrolledAt
}));

export const mockTriples: Triple[] = [
  { subject: ':speaker_001', predicate: ':hometown', object: ':Thanjavur' },
  { subject: ':speaker_001', predicate: ':favouriteFood', object: ':Kothu_Parotta' },
  { subject: ':speaker_001', predicate: ':school', object: ':Don_Bosco_Matriculation' },
];

export const mockCSBG: CSBG = {
  speakerId: 'spk_001',
  nodes: [
    { id: 'c_FOOD', kind: 'class', label: 'FOOD', tokenCount: 42 },
    { id: 'c_PLACE_LOCAL', kind: 'class', label: 'PLACE_LOCAL', tokenCount: 35 },
    { id: 'c_NUMBER', kind: 'class', label: 'NUMBER', tokenCount: 88 },
    { id: 'c_ACTION_VERB', kind: 'class', label: 'ACTION_VERB', tokenCount: 156 },
    { id: 'l_TA', kind: 'language', label: 'TA', tokenCount: 1205 },
    { id: 'l_EN', kind: 'language', label: 'EN', tokenCount: 340 }
  ],
  edges: [
    { source: 'c_FOOD', target: 'l_EN', probability: 0.85, observationCount: 36, edgeType: 'lexical_choice' },
    { source: 'c_FOOD', target: 'l_TA', probability: 0.15, observationCount: 6, edgeType: 'lexical_choice' },
    { source: 'c_PLACE_LOCAL', target: 'l_TA', probability: 0.70, observationCount: 24, edgeType: 'lexical_choice' },
    { source: 'c_PLACE_LOCAL', target: 'l_EN', probability: 0.30, observationCount: 11, edgeType: 'lexical_choice' },
    { source: 'c_NUMBER', target: 'l_EN', probability: 0.92, observationCount: 81, edgeType: 'lexical_choice' },
    { source: 'c_NUMBER', target: 'l_TA', probability: 0.08, observationCount: 7, edgeType: 'lexical_choice' },
    { source: 'c_ACTION_VERB', target: 'l_TA', probability: 0.95, observationCount: 148, edgeType: 'lexical_choice' },
    { source: 'c_ACTION_VERB', target: 'l_EN', probability: 0.05, observationCount: 8, edgeType: 'lexical_choice' },
    { source: 'l_TA', target: 'l_EN', probability: 0.12, observationCount: 145, edgeType: 'switch_transition' },
    { source: 'l_EN', target: 'l_TA', probability: 0.45, observationCount: 153, edgeType: 'switch_transition' }
  ],
  cmi: 28.4,
  iIndex: 0.12,
  matrixLanguageRatio: 0.85,
  sparseClasses: ['KINSHIP', 'RELIGION_FESTIVAL']
};

export const mockAuthResults: AuthResult[] = [
  {
    id: 'auth_1',
    speakerId: 'spk_001',
    challengeId: 'chg_1',
    transcript: 'Naan Thanjavur la irunthu varen.',
    tokens: [
       { text: "Naan", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.98, startMs: 0, endMs: 200 },
       { text: "Thanjavur", language: "NAMED_ENTITY", semanticClass: "PLACE_LOCAL", lidConfidence: 0.95, startMs: 250, endMs: 800 },
       { text: "la", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 800, endMs: 950 },
       { text: "irunthu", language: "TA", semanticClass: "FUNCTION_WORD", lidConfidence: 0.97, startMs: 1000, endMs: 1400 },
       { text: "varen.", language: "TA", semanticClass: "ACTION_VERB", lidConfidence: 0.96, startMs: 1450, endMs: 1900 },
    ],
    branches: [
      { name: 'speaker_embedding', score: 0.82, threshold: 0.65, weight: 0.4, passed: true },
      { name: 'csbg', score: 0.78, threshold: 0.50, weight: 0.3, passed: true },
      { name: 'knowledge', score: 1.0, threshold: 0.80, weight: 0.2, passed: true },
      { name: 'liveness', score: 0.95, threshold: 0.70, weight: 0.1, passed: true }
    ],
    fusedScore: 0.857,
    fusedThreshold: 0.65,
    decision: 'ACCEPT',
    divergences: [],
    explanation: [
      'Speaker embedding matched (cosine 0.82, threshold 0.65).',
      'Code-switch behaviour consistent with profile.',
      'Knowledge answer matched (semantic 1.0, phonetic 0.88).'
    ],
    latencyMs: 840,
    timestamp: new Date(Date.now() - 3600000).toISOString()
  },
  {
    id: 'auth_2',
    speakerId: 'spk_002',
    challengeId: 'chg_2',
    transcript: 'My favourite food is actually biryani.',
    tokens: [
      { text: "My", language: "EN", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 0, endMs: 200 },
      { text: "favourite", language: "EN", semanticClass: "EMOTION_STATE", lidConfidence: 0.98, startMs: 250, endMs: 600 },
      { text: "food", language: "EN", semanticClass: "FOOD", lidConfidence: 0.99, startMs: 650, endMs: 900 },
      { text: "is", language: "EN", semanticClass: "FUNCTION_WORD", lidConfidence: 0.99, startMs: 950, endMs: 1100 },
      { text: "actually", language: "EN", semanticClass: "DISCOURSE_MARKER", lidConfidence: 0.97, startMs: 1150, endMs: 1500 },
      { text: "biryani.", language: "NAMED_ENTITY", semanticClass: "FOOD", lidConfidence: 0.95, startMs: 1550, endMs: 2100 }
    ],
    branches: [
      { name: 'speaker_embedding', score: 0.68, threshold: 0.65, weight: 0.4, passed: true },
      { name: 'csbg', score: 0.42, threshold: 0.50, weight: 0.3, passed: false },
      { name: 'knowledge', score: 0.0, threshold: 0.80, weight: 0.2, passed: false },
      { name: 'liveness', score: 0.91, threshold: 0.70, weight: 0.1, passed: true }
    ],
    fusedScore: 0.489,
    fusedThreshold: 0.65,
    decision: 'REJECT',
    divergences: [
      { semanticClass: 'FOOD', expectedLanguage: 'TA', expectedProb: 0.8, observedLanguage: 'EN', observedProb: 1.0, jsd: 0.45, tokenCount: 2 }
    ],
    explanation: [
      'Rejected. Knowledge mismatch (wrong answer).',
      'Code-switch divergence in classes:',
      '  FOOD expected TA (0.80) observed EN (1.0) JSD 0.45',
      'Speaker embedding borderline (cosine 0.68, threshold 0.65).'
    ],
    latencyMs: 920,
    timestamp: new Date(Date.now() - 7200000).toISOString()
  },
  {
    id: 'auth_3',
    speakerId: 'spk_003',
    challengeId: 'chg_3',
    transcript: 'School... ah... Don Bosco.',
    tokens: [
      { text: "School...", language: "EN", semanticClass: "EDU_WORK", lidConfidence: 0.98, startMs: 0, endMs: 400 },
      { text: "ah...", language: "NEUTRAL", semanticClass: "DISCOURSE_MARKER", lidConfidence: 0.90, startMs: 500, endMs: 800 },
      { text: "Don Bosco.", language: "NAMED_ENTITY", semanticClass: "PLACE_LOCAL", lidConfidence: 0.95, startMs: 900, endMs: 1500 }
    ],
    branches: [
      { name: 'speaker_embedding', score: 0.64, threshold: 0.65, weight: 0.4, passed: false },
      { name: 'csbg', score: 0.52, threshold: 0.50, weight: 0.3, passed: true },
      { name: 'knowledge', score: 0.85, threshold: 0.80, weight: 0.2, passed: true },
      { name: 'liveness', score: 0.88, threshold: 0.70, weight: 0.1, passed: true }
    ],
    fusedScore: 0.670,
    fusedThreshold: 0.65,
    decision: 'BORDERLINE',
    divergences: [],
    explanation: [
      'Borderline. Speaker embedding failed (cosine 0.64, threshold 0.65).',
      'Knowledge answer matched (semantic 0.85).',
      'CSBG score low confidence due to short utterance.'
    ],
    latencyMs: 780,
    timestamp: new Date(Date.now() - 14400000).toISOString()
  }
];

export const mockAttacks: AttackRun[] = [
  { id: 'atk_1', attackType: 'A1_REPLAY', targetSpeakerId: 'spk_001', trials: 100, successRateByConfig: { ecapa_only: 0.85, plus_knowledge: 0.12, plus_csbg: 0.08, full_fusion: 0.01 }, generatedAt: now },
  { id: 'atk_2', attackType: 'A2_SPLICE', targetSpeakerId: 'spk_001', trials: 100, successRateByConfig: { ecapa_only: 0.65, plus_knowledge: 0.55, plus_csbg: 0.18, full_fusion: 0.05 }, generatedAt: now },
  { id: 'atk_3', attackType: 'A3_CLONE_NAIVE', targetSpeakerId: 'spk_002', trials: 100, successRateByConfig: { ecapa_only: 0.92, plus_knowledge: 0.05, plus_csbg: 0.15, full_fusion: 0.02 }, generatedAt: now },
  { id: 'atk_4', attackType: 'A4_CLONE_KNOWLEDGE', targetSpeakerId: 'spk_002', trials: 100, successRateByConfig: { ecapa_only: 0.92, plus_knowledge: 0.88, plus_csbg: 0.22, full_fusion: 0.11 }, generatedAt: now },
  { id: 'atk_5', attackType: 'A5_CLONE_ADAPTIVE', targetSpeakerId: 'spk_002', trials: 100, successRateByConfig: { ecapa_only: 0.94, plus_knowledge: 0.85, plus_csbg: 0.68, full_fusion: 0.35 }, generatedAt: now }
];

export const mockEvalMetrics: EvalMetrics = {
  configurations: [
    { name: 'ECAPA-TDNN only', eer: 4.82, minDcf: 0.031, farAtFrr1: 18.5, frrAtFar1: 15.2, detCurve: [{far: 0.1, frr: 0.1}, {far: 0.05, frr: 0.05}, {far: 0.01, frr: 0.15}] },
    { name: '+ Knowledge Graph', eer: 2.15, minDcf: 0.014, farAtFrr1: 4.2, frrAtFar1: 3.8, detCurve: [{far: 0.1, frr: 0.02}, {far: 0.05, frr: 0.03}, {far: 0.01, frr: 0.05}] },
    { name: '+ CSBG', eer: 2.95, minDcf: 0.018, farAtFrr1: 8.5, frrAtFar1: 7.2, detCurve: [{far: 0.1, frr: 0.03}, {far: 0.05, frr: 0.04}, {far: 0.01, frr: 0.08}] },
    { name: 'Full Fusion', eer: 1.05, minDcf: 0.007, farAtFrr1: 1.1, frrAtFar1: 1.2, detCurve: [{far: 0.1, frr: 0.01}, {far: 0.05, frr: 0.015}, {far: 0.01, frr: 0.02}] }
  ],
  stabilityCurve: [
    { durationSec: 30, eer: 8.5, ciLow: 7.2, ciHigh: 9.8 },
    { durationSec: 60, eer: 5.2, ciLow: 4.5, ciHigh: 6.0 },
    { durationSec: 120, eer: 3.1, ciLow: 2.8, ciHigh: 3.5 },
    { durationSec: 180, eer: 1.8, ciLow: 1.6, ciHigh: 2.1 },
    { durationSec: 240, eer: 1.2, ciLow: 1.0, ciHigh: 1.4 },
    { durationSec: 300, eer: 1.05, ciLow: 0.9, ciHigh: 1.2 }
  ],
  fairness: [
    { condition: 'Monolingual TA', group: 'Male', eer: 1.1, sampleCount: 450 },
    { condition: 'Monolingual TA', group: 'Female', eer: 1.2, sampleCount: 420 },
    { condition: 'Monolingual EN', group: 'Male', eer: 1.5, sampleCount: 380 },
    { condition: 'Monolingual EN', group: 'Female', eer: 1.4, sampleCount: 395 },
    { condition: 'Code-Mixed', group: 'Male', eer: 0.9, sampleCount: 850 },
    { condition: 'Code-Mixed', group: 'Female', eer: 1.0, sampleCount: 820 }
  ],
  scoreDistributions: [
    { branch: 'ECAPA-TDNN', genuine: [0.6, 0.7, 0.75, 0.8, 0.85, 0.9], impostor: [0.1, 0.2, 0.3, 0.4, 0.5] },
    { branch: 'CSBG', genuine: [0.5, 0.6, 0.7, 0.8, 0.9], impostor: [0.2, 0.3, 0.4, 0.5, 0.6] },
    { branch: 'Knowledge', genuine: [0.8, 0.9, 1.0, 1.0], impostor: [0.0, 0.0, 0.2, 0.4] },
    { branch: 'Fused', genuine: [0.7, 0.8, 0.85, 0.9, 0.95], impostor: [0.1, 0.2, 0.25, 0.3, 0.4] }
  ]
};
