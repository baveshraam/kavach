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
  questionText: string;        // code-mixed Tamil-English, or the instruction for a phrase challenge
  kind: 'question' | 'phrase'; // 'phrase' = read these random words (no facts, no network)
  phrase: string[];            // the words to read, for a phrase challenge; empty otherwise
  stepUp: boolean;             // the stricter second sample that follows a borderline attempt
  targetClass: SemanticClass;
  expectedAnswerEntity: string;
  issuedAt: string;
  expiresAt: string;
}

export interface BranchScore {
  // 'liveness' and 'signal_integrity' are gates, not weighted factors: they
  // carry weight 0 and reject on their own. Render them as pass/fail, not as
  // a contribution to the fused score.
  name: 'speaker_embedding' | 'csbg' | 'knowledge' | 'liveness' | 'signal_integrity' | 'phrase';
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
  phraseMatched?: string[];    // phrase challenge: which shown words were heard
  phraseMissing?: string[];    // ... and which were not
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
  // Backend extras: every run says whether it was simulated and why.
  simulated?: boolean;
  yieldRate?: number | null;
  acousticSource?: 'measured' | 'modelled';
  notes?: string[];
}

export interface Voiceprint {
  nClips: number; selfConsistency: number;
  provenance: { source: string; sessions: string[]; devices: string[]; n_clips: number } | null;
}
export interface StudioPlanItem { kind: 'read' | 'free' | 'fact' | 'words'; promptId: string; textEn: string; textTa: string; repeat: number }
export interface StudioPlan {
  speaker: string; sessionId: string; hasFacts: boolean; estimatedMinutes: number;
  devices: string[]; environments: string[]; items: StudioPlanItem[];
}
export interface StudioSummary {
  speaker: string; clips: number; minutes: number;
  by_session: Record<string, { clips: number; minutes: number; devices: string[]; environments: string[] }>;
  by_kind: Record<string, { clips: number; minutes: number }>;
  by_device: Record<string, { clips: number; minutes: number }>;
  by_environment: Record<string, { clips: number; minutes: number }>;
}

export interface CloneBankInfo {
  enabled: boolean;
  coveredFacts: string[];
  yieldRate?: number | null;
  problems: string[];
}

export interface CloneMatch {
  clipId: string;
  audioUrl: string;
  attackType: string;
  backend: string;
  similarity: number;
}

export interface SpeakerIapmr {
  speakerId: string;
  name: string;
  trials: number;
  iapmr: number;
  iapmrByConfig: Record<string, number>;
  ciLow: number;
  ciHigh: number;
  belowMinTrials: boolean;
  attackTypes: AttackType[];
}

// Attack success per speaker. The mean hides the case that matters: a system
// stopping every attack on 24 speakers and none on the 25th reports 96% while
// one person is completely unprotected.
export interface PerSpeakerIapmr {
  speakers: SpeakerIapmr[];
  worstSpeakerId: string;
  // null, not 0, when nothing has been measured. A rate of zero is the value
  // that looks like the defence working perfectly.
  meanIapmr: number | null;
  // Enrolled speakers with no attack run. Unmeasured is not protected.
  unmeasuredSpeakerIds: string[];
  minTrialsPerCell: number;
  simulated: boolean;
  notes: string[];
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

// ---------------------------------------------------------------------------
// Offline experiment runs (`python -m kavach.experiments`), passed through
// verbatim from paper/<run>/results.json. Only the fields the UI reads are
// typed; the rest stays available as `unknown`.
// ---------------------------------------------------------------------------

export interface OfflineConfiguration {
  name: string;
  branches: string[];
  eer: number;
  eer_ci: [number, number];
  min_dcf: number;
  far_at_frr_1pct: number | null;
  frr_at_far_1pct: number | null;
  auc: number;
  n_genuine: number;
  n_impostor: number;
  n_vetoed: number;
  is_reliable: boolean;
}

export interface OfflineResults {
  reportable: boolean;
  blockers: string[];
  environment: { generated_utc: string; git_commit: string };
  corpus: {
    name: string; provenance: string; n_speakers: number; n_sessions: number;
    n_utterances: number; n_annotated: number; session_split: string; cross_session: boolean;
    branch_coverage: { name: string; measured: number; unavailable: number; coverage: number }[];
  };
  split: { dev_speakers: string[]; test_speakers: string[]; n_dev_trials: number; n_test_trials: number };
  fitted: { weights: Record<string, number>; threshold: number; veto_floor: number | null; weights_source: string };
  configurations: OfflineConfiguration[];
  ablations: { name: string; scope: string; eer: number; delta: number; note: string }[];
  stability: { n_utterances: number; approx_seconds: number; eer: number; ci_low: number; ci_high: number }[];
  coverage: {
    total_tokens: number; total_choice_tokens: number; n_speakers: number;
    classes: { class: string; n_tokens: number; n_speakers_with_own_evidence: number }[];
  };
  speaker_consistency?: Record<string, number>;
  caveats: string[];
}

export interface OfflineRun {
  id: string;
  results: OfflineResults;
  figures: string[];
}
