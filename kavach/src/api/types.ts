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
