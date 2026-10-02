import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, assetUrl } from '../api/client';
import { trimToWav, fetchExact, stagingCut, STAGING_MIN_SOURCE_SEC } from '../lib/audioClip';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { AudioRecorder } from '../components/ui/AudioRecorder';
import { Challenge, AuthResult, BranchScore } from '../api/types';
import {
  Card, CardHeader, CardBody, Button, Field, Select, Badge, DecisionBadge, Notice, EmptyState, TokenText, LangLegend,
  Spinner, branchLabel, titleCase, cn,
} from '../components/ui/kit';
import { RefreshCw, ShieldCheck, Timer, CheckCircle2, XCircle, MinusCircle, ArrowRight } from 'lucide-react';

const GATES = ['liveness', 'signal_integrity'];

export function Authenticate() {
  const queryClient = useQueryClient();
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const [speakerId, setSpeakerId] = useState('');
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [result, setResult] = useState<AuthResult | null>(null);
  const [now, setNow] = useState(Date.now());

  const { data: skg } = useQuery({
    queryKey: ['skg', speakerId],
    queryFn: () => apiClient.getSpeakerSKG(speakerId),
    enabled: !!speakerId,
  });

  const { data: health } = useQuery({ queryKey: ['health'], queryFn: apiClient.health });
  // The flag alone is not enough: with the bank on but empty the button could
  // only ever fail, so it appears only when some clone can answer a question.
  const { data: cloneBank } = useQuery({
    queryKey: ['cloneBank'],
    queryFn: apiClient.cloneBankInfo,
    enabled: !!health?.demoAttackBank,
  });

  const issue = useMutation({
    mutationFn: (id: string) => apiClient.issueChallenge(id),
    onSuccess: data => { setChallenge(data); setResult(null); },
  });

  const [source, setSource] = useState<string>('Live answer');
  const auth = useMutation({
    mutationFn: ({ blob, name }: { blob: Blob; name?: string; label?: string }) => apiClient.authenticate(challenge!.id, blob, name),
    onMutate: ({ label }) => setSource(label ?? 'Live answer'),
    onSuccess: data => {
      setResult(data);
      queryClient.invalidateQueries({ queryKey: ['authHistory'] });
    },
  });

  useEffect(() => {
    if (!challenge || result) return;
    const t = window.setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [challenge, result]);

  const ttl = challenge ? (new Date(challenge.expiresAt).getTime() - new Date(challenge.issuedAt).getTime()) : 1;
  const remaining = challenge ? Math.max(0, new Date(challenge.expiresAt).getTime() - now) : 0;
  const expired = !!challenge && remaining <= 0 && !result;
  const speaker = speakers?.find(s => s.id === speakerId);
  const noFacts = !!speakerId && skg !== undefined && skg.length === 0;

  const choose = (id: string) => {
    setSpeakerId(id);
    setChallenge(null);
    setResult(null);
    issue.reset();
    auth.reset();
  };

  return (
    <>
      <PageHeader
        eyebrow="Live system"
        title="Authenticate"
        description="Claim an identity, answer a one-time spoken challenge, and see how each branch voted. Challenges expire, so a recording made earlier cannot be replayed."
      />
      <PageBody>
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] gap-6 items-start">
          {/* ---------------------------------------------------------- flow */}
          <div className="flex flex-col gap-4">
            <Step n={1} title="Who are you claiming to be?" done={!!speakerId}>
              <Field label="Enrolled speaker">
                <Select value={speakerId} onChange={e => choose(e.target.value)}>
                  <option value="" disabled>Select a speaker…</option>
                  {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
                </Select>
              </Field>
              {speaker && (
                <div className="flex flex-wrap gap-2 mt-3">
                  <Badge>{speaker.utteranceCount} enrolment clips</Badge>
                  <Badge>{(speaker.totalDurationSec / 60).toFixed(1)} min of speech</Badge>
                  <Badge tone={skg?.length ? 'accept' : 'warning'}>{skg?.length ?? 0} known facts</Badge>
                </div>
              )}
              {noFacts && (
                <Notice tone="warning" className="mt-3" title="This speaker has no knowledge facts yet">
                  Challenges are generated from the speaker’s own facts (hometown, favourite food…).{' '}
                  <Link to={`/speakers?speaker=${speakerId}`} className="text-app-accent font-medium hover:underline">Add a few on the Speakers page</Link>, then come back.
                </Notice>
              )}
            </Step>

            <Step n={2} title="Get a challenge" done={!!challenge} disabled={!speakerId || noFacts}>
              {!challenge ? (
                <Button variant="primary" onClick={() => issue.mutate(speakerId)} loading={issue.isPending} disabled={!speakerId || noFacts} icon={<ShieldCheck className="w-4 h-4" />}>
                  Issue challenge
                </Button>
              ) : (
                <div className="flex flex-col gap-4">
                  <p className="font-serif text-[22px] leading-snug text-app-text">{challenge.questionText}</p>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge tone="accent">targets {titleCase(challenge.targetClass)}</Badge>
                    <Badge className="mono">{challenge.id}</Badge>
                  </div>
                  {!result && (
                    <div className="flex items-center gap-3">
                      <Timer className={cn('w-4 h-4', expired ? 'text-app-reject' : 'text-app-text-subtle')} />
                      <div className="flex-1 h-1.5 rounded-full bg-app-surface-muted overflow-hidden">
                        <div className={cn('h-full transition-[width] duration-200', remaining / ttl < 0.25 ? 'bg-app-reject' : 'bg-app-accent')} style={{ width: `${(remaining / ttl) * 100}%` }} />
                      </div>
                      <span className={cn('tnum text-[12.5px] w-24 text-right', expired ? 'text-app-reject font-medium' : 'text-app-text-muted')}>
                        {expired ? 'expired' : `${Math.ceil(remaining / 1000)}s left`}
                      </span>
                    </div>
                  )}
                  <div>
                    <Button size="sm" variant="ghost" icon={<RefreshCw className="w-3.5 h-3.5" />} onClick={() => issue.mutate(speakerId)} loading={issue.isPending}>
                      New challenge
                    </Button>
                  </div>
                </div>
              )}
              {issue.error && <Notice tone="reject" className="mt-3" title="Could not issue a challenge">{(issue.error as Error).message}</Notice>}
            </Step>

            <Step n={3} title="Answer out loud" done={!!result} disabled={!challenge}>
              {challenge && !result ? (
                <>
                  {expired && (
                    <Notice tone="warning" className="mb-3" title="This challenge has expired">
                      You can still submit — the liveness gate will reject it, which is how a replay looks. Or issue a new one.
                    </Notice>
                  )}
                  <AudioRecorder allowUpload busy={auth.isPending} acceptLabel="Verify"
                    onAccept={(blob, _d, name) => auth.mutate({ blob, name, label: name ? `Uploaded file · ${name}` : 'Live answer' })} />
                  <DemoClips claimedId={speakerId} speakers={speakers ?? []} busy={auth.isPending}
                    challengeId={challenge.id} cloneEnabled={!!health?.demoAttackBank && (cloneBank?.coveredFacts.length ?? 0) > 0}
                    onSubmit={(blob, name, label) => auth.mutate({ blob, name, label })} />
                  {auth.isPending && <div className="mt-3"><Spinner label="Transcribing, tagging and scoring… (a few seconds on CPU)" /></div>}
                  {auth.error && <Notice tone="reject" className="mt-3" title="Verification failed">{(auth.error as Error).message}</Notice>}
                </>
              ) : result ? (
                <div className="flex items-center justify-between">
                  <span className="text-[13px] text-app-text-muted">Answer submitted and scored.</span>
                  <Button size="sm" variant="secondary" onClick={() => issue.mutate(speakerId)} icon={<RefreshCw className="w-3.5 h-3.5" />}>Try again</Button>
                </div>
              ) : (
                <p className="text-[13px] text-app-text-subtle">Issue a challenge first.</p>
              )}
            </Step>
          </div>

          {/* -------------------------------------------------------- result */}
          <div className="xl:sticky xl:top-6">
            {result ? <ResultPanel result={result} source={source} /> : (
              <Card className="min-h-[420px] flex items-center justify-center">
                <EmptyState icon={<ShieldCheck className="w-5 h-5" />} title="The decision appears here">
                  Each branch is scored separately, then fused. Gates such as liveness can reject on their own, whatever the other scores say.
                </EmptyState>
              </Card>
            )}
          </div>
        </div>
      </PageBody>
    </>
  );
}

function Step({ n, title, done, disabled, children }: { n: number; title: string; done?: boolean; disabled?: boolean; children: React.ReactNode }) {
  return (
    <Card className={cn('transition-opacity', disabled && 'opacity-55')}>
      <div className="flex items-center gap-3 px-5 pt-4 pb-3">
        <span className={cn('w-6 h-6 rounded-full flex items-center justify-center text-[12px] font-semibold shrink-0',
          done ? 'bg-app-accept text-white' : 'bg-app-surface-muted text-app-text-muted border border-app-border')}>
          {done ? '✓' : n}
        </span>
        <h3 className="text-[14px] font-semibold">{title}</h3>
      </div>
      <CardBody className="pl-14">{children}</CardBody>
    </Card>
  );
}

/**
 * Stage attacks from audio already in the corpus -- no new recordings needed.
 *
 *   Replay   the claimed speaker's stored clip, byte for byte. The duplicate
 *            detector should stop it before any model runs.
 *   Impostor a 20 s cut of a *different* speaker, re-encoded in the browser so
 *            it is not a byte duplicate. The voiceprint has to reject it.
 *   Stand-in a cut of the claimed speaker's own recording, for when that person
 *            is not in the room. Labelled as such: the voice matches trivially
 *            (it is enrolment audio), and it does not answer the question.
 */
function DemoClips({ claimedId, speakers, busy, onSubmit, challengeId, cloneEnabled }: {
  claimedId: string;
  challengeId: string;
  cloneEnabled: boolean;
  speakers: import('../api/types').Speaker[];
  busy: boolean;
  onSubmit: (blob: Blob, name: string, label: string) => void;
}) {
  const [impostorId, setImpostorId] = useState('');
  const [preparing, setPreparing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const others = speakers.filter(s => s.id !== claimedId);
  const impostor = impostorId && impostorId !== claimedId ? impostorId : others[0]?.id ?? '';
  const nameOf = (id: string) => speakers.find(s => s.id === id)?.displayName ?? id;

  const run = async (kind: 'replay' | 'impostor' | 'standin' | 'clone') => {
    setError(null);
    setPreparing(kind);
    try {
      if (kind === 'clone') {
        // The attacker's pre-cloned answer to the challenge just issued. A 404
        // here carries a useful reason ("the bank covers: ..."), shown below.
        const match = await apiClient.matchCloneClip(challengeId);
        onSubmit(
          await fetchExact(assetUrl(match.audioUrl)),
          `clone_${match.clipId}.wav`,
          `Clone attack · synthetic ${match.backend} clip of ${nameOf(claimedId)} (voiceprint ${match.similarity.toFixed(2)})`,
        );
        return;
      }
      const ownerId = kind === 'impostor' ? impostor : claimedId;
      const utts = await apiClient.getSpeakerUtterances(ownerId);
      // A replay submits a whole stored clip. The others are *cuts*, and a cut
      // that is most of a stored clip is itself caught as a replay, so they are
      // drawn only from recordings long enough to cut safely (see stagingCut).
      const pool = kind === 'replay' ? utts.filter(u => u.durationSec >= 8) : utts.filter(u => stagingCut(u.durationSec));
      if (!pool.length) throw new Error(`That speaker has no stored recording of ${STAGING_MIN_SOURCE_SEC} s or more to cut from.`);
      const pick = pool[Math.floor(Math.random() * pool.length)];
      const url = assetUrl(pick.audioUrl);
      if (kind === 'replay') {
        onSubmit(await fetchExact(url), `replay_${pick.id}.wav`, `Replay attack · stored clip of ${nameOf(ownerId)}`);
      } else {
        const cut = stagingCut(pick.durationSec)!;
        const blob = await trimToWav(url, cut.start, cut.length);
        onSubmit(blob, `${kind}_${pick.id}.wav`, kind === 'impostor'
          ? `Impostor · ${nameOf(ownerId)} claiming to be ${nameOf(claimedId)}`
          : `Stand-in · cut of ${nameOf(ownerId)}’s own enrolment audio`);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPreparing(null);
    }
  };

  return (
    <div className="mt-4 rounded-lg border border-dashed border-app-border-strong p-4 flex flex-col gap-3">
      <div>
        <div className="text-[13px] font-semibold">Demo with stored audio</div>
        <div className="text-[12px] text-app-text-muted">Stage an attack from recordings already in the corpus, instead of speaking.</div>
      </div>
      <div className={cn('grid grid-cols-1 gap-2', cloneEnabled ? 'sm:grid-cols-4' : 'sm:grid-cols-3')}>
        <Button size="sm" variant="secondary" disabled={busy || !!preparing} loading={preparing === 'replay'} onClick={() => run('replay')}>
          Replay attack
        </Button>
        <Button size="sm" variant="secondary" disabled={busy || !!preparing || !impostor} loading={preparing === 'impostor'} onClick={() => run('impostor')}>
          Impostor voice
        </Button>
        <Button size="sm" variant="ghost" disabled={busy || !!preparing} loading={preparing === 'standin'} onClick={() => run('standin')}>
          Genuine stand-in
        </Button>
        {cloneEnabled && (
          <Button size="sm" variant="secondary" disabled={busy || !!preparing} loading={preparing === 'clone'} onClick={() => run('clone')}>
            Clone attack
          </Button>
        )}
      </div>
      <Field label="Impostor speaker">
        <Select value={impostor} onChange={e => setImpostorId(e.target.value)}>
          {others.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
        </Select>
      </Field>
      {error && <p className="text-[12px] text-app-reject">{error}</p>}
    </div>
  );
}

function ResultPanel({ result, source }: { result: AuthResult; source: string }) {
  const gates = result.branches.filter(b => GATES.includes(b.name));
  const branches = result.branches.filter(b => !GATES.includes(b.name));
  const tone = result.decision === 'ACCEPT' ? 'accept' : result.decision === 'REJECT' ? 'reject' : 'warning';

  return (
    <div className="flex flex-col gap-4 animate-in">
      <Card className={cn('overflow-hidden border-t-4', {
        'border-t-app-accept': tone === 'accept', 'border-t-app-reject': tone === 'reject', 'border-t-app-warning': tone === 'warning',
      })}>
        <div className="px-5 pt-4 -mb-1 text-[12px] text-app-text-muted">Submitted: <span className="font-medium text-app-text">{source}</span></div>
        <div className="p-5 flex items-center justify-between gap-4">
          <DecisionBadge decision={result.decision} size="lg" />
          <div className="text-right">
            <div className="text-[12px] text-app-text-muted">Fused score / threshold</div>
            <div className="text-[26px] font-semibold tnum leading-tight">
              {result.fusedScore.toFixed(3)}
              <span className="text-[15px] text-app-text-subtle font-normal"> / {result.fusedThreshold.toFixed(2)}</span>
            </div>
          </div>
        </div>
        <div className="px-5 pb-5">
          <Meter score={result.fusedScore} threshold={result.fusedThreshold} passed={result.fusedScore >= result.fusedThreshold} tall />
          {gates.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-4">
              {gates.map(g => (
                <Badge key={g.name} tone={g.passed ? 'accept' : 'reject'}>
                  {g.passed ? <CheckCircle2 className="w-3.5 h-3.5" /> : <XCircle className="w-3.5 h-3.5" />}
                  {branchLabel[g.name] ?? g.name} {g.passed ? 'passed' : 'failed'}
                </Badge>
              ))}
              <Badge className="tnum">{(result.latencyMs / 1000).toFixed(1)} s</Badge>
            </div>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader title="Branch scores" subtitle="Bar = score, tick = that branch’s threshold, w = its weight in the fusion." />
        <CardBody className="flex flex-col gap-4">
          {branches.map(b => <BranchRow key={b.name} b={b} />)}
          {(['speaker_embedding', 'csbg', 'knowledge'] as const)
            .filter(n => !branches.some(b => b.name === n))
            .map(n => (
              <div key={n} className="flex items-center justify-between text-[13px] text-app-text-subtle">
                <span className="flex items-center gap-2"><MinusCircle className="w-4 h-4" /> {branchLabel[n]}</span>
                <span>not measured — dropped from fusion</span>
              </div>
            ))}
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="What was heard" actions={<LangLegend />} />
        <CardBody>
          {result.tokens.length ? <TokenText tokens={result.tokens} /> : <p className="text-[13px] text-app-text-muted">{result.transcript || 'No transcript.'}</p>}
        </CardBody>
      </Card>

      {result.divergences.length > 0 && (
        <Card>
          <CardHeader title="Where the switching differed" subtitle="Classes whose language choice diverged most from the enrolled graph." />
          <CardBody className="flex flex-col gap-2">
            {result.divergences.slice(0, 5).map(d => (
              <div key={d.semanticClass} className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-3 text-[13px]">
                <span className="font-medium">{titleCase(d.semanticClass)}</span>
                <span className="text-app-text-muted">expected <LangTag l={d.expectedLanguage} /> {Math.round(d.expectedProb * 100)}%</span>
                <ArrowRight className="w-3.5 h-3.5 text-app-text-subtle" />
                <span className="text-app-text-muted">heard <LangTag l={d.observedLanguage} /></span>
              </div>
            ))}
          </CardBody>
        </Card>
      )}

      <Card>
        <CardHeader title="Why" />
        <CardBody>
          <ul className="flex flex-col gap-2">
            {result.explanation.map((line, i) => (
              <li key={i} className="flex gap-2.5 text-[13px] leading-relaxed text-app-text-muted">
                <span className="w-1 h-1 rounded-full bg-app-text-subtle mt-[9px] shrink-0" />
                {line}
              </li>
            ))}
          </ul>
        </CardBody>
      </Card>
    </div>
  );
}

function LangTag({ l }: { l: string }) {
  return <span className={cn('font-semibold', l === 'TA' ? 'text-app-ta' : l === 'EN' ? 'text-app-en' : 'text-app-text')}>{l}</span>;
}

function BranchRow({ b }: { b: BranchScore }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[13px] font-medium">{branchLabel[b.name] ?? b.name}</span>
        <span className="flex items-baseline gap-3">
          <span className="text-[12px] text-app-text-subtle tnum">w {b.weight.toFixed(2)}</span>
          <span className={cn('text-[15px] font-semibold tnum', b.passed ? 'text-app-text' : 'text-app-reject')}>{b.score.toFixed(3)}</span>
        </span>
      </div>
      <Meter score={b.score} threshold={b.threshold} passed={b.passed} />
    </div>
  );
}

function Meter({ score, threshold, passed, tall }: { score: number; threshold: number; passed: boolean; tall?: boolean }) {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  return (
    <div className={cn('relative w-full rounded-full bg-app-surface-muted', tall ? 'h-3' : 'h-2')}>
      <div className={cn('h-full rounded-full', passed ? 'bg-app-accept' : 'bg-app-reject')} style={{ width: `${clamp(score) * 100}%` }} />
      <div className="absolute -top-1 -bottom-1 w-[2px] bg-app-text rounded" style={{ left: `calc(${clamp(threshold) * 100}% - 1px)` }} title={`threshold ${threshold.toFixed(2)}`} />
    </div>
  );
}
