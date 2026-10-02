import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Mic, Square, CheckCircle2, XCircle, AlertTriangle, RefreshCw, Lock, LockOpen, Timer, MinusCircle } from 'lucide-react';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Card, CardHeader, CardBody, Button, Field, Select, Badge, Notice, cn } from '../components/ui/kit';
import type { AuthResult, BranchScore, Challenge } from '../api/types';

const MIN_MS = 2000;   // shorter than this cannot hold six words; ask again without spending the challenge
const MAX_MS = 15000;  // a recorder left running

/**
 * Microphone capture through the same path the Studio uses (default constraints, Opus in WebM), so
 * a voiceprint enrolled from Studio sessions meets the same audio chain at login.
 */
function useCapture(onDone: (blob: Blob, ms: number) => void) {
  const [recording, setRecording] = useState(false);
  const [ms, setMs] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  const rec = useRef<MediaRecorder | null>(null);
  const ctx = useRef<AudioContext | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const raf = useRef(0);
  const timer = useRef(0);
  const chunks = useRef<BlobPart[]>([]);
  const started = useRef(0);

  const cleanup = useCallback(() => {
    cancelAnimationFrame(raf.current);
    window.clearTimeout(timer.current);
    stream.current?.getTracks().forEach(t => t.stop());
    ctx.current?.close().catch(() => undefined);
    stream.current = null;
    ctx.current = null;
  }, []);

  const stop = useCallback(() => {
    if (rec.current && rec.current.state !== 'inactive') rec.current.stop();
  }, []);

  const start = useCallback(async () => {
    try {
      setError(null);
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.current = s;
      const ac = new AudioContext();
      ctx.current = ac;
      const an = ac.createAnalyser();
      an.fftSize = 1024;
      ac.createMediaStreamSource(s).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const mr = new MediaRecorder(s, { mimeType: 'audio/webm;codecs=opus' });
      chunks.current = [];
      mr.ondataavailable = e => { if (e.data.size > 0) chunks.current.push(e.data); };
      mr.onstop = () => {
        const dur = performance.now() - started.current;
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        cleanup();
        setRecording(false);
        setLevel(0);
        doneRef.current(blob, dur);
      };
      rec.current = mr;
      mr.start();
      started.current = performance.now();
      setRecording(true);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) { const d = (v - 128) / 128; sum += d * d; }
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
        setMs(performance.now() - started.current);
        raf.current = requestAnimationFrame(tick);
      };
      tick();
      timer.current = window.setTimeout(stop, MAX_MS);
    } catch {
      cleanup();
      setError('The microphone is not available. Allow microphone access for this page and try again.');
    }
  }, [cleanup, stop]);

  useEffect(() => cleanup, [cleanup]);
  return { recording, ms, level, error, start, stop };
}

type Attempt = { at: string; decision: AuthResult['decision']; voice?: number; note: string };

const retrySeconds = (msg: string) => {
  const m = /wait (\d+) s/i.exec(msg);
  return m ? Number(m[1]) : 0;
};

export function Unlock() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: apiClient.health });
  const { data: evidence } = useQuery({ queryKey: ['voicePolicy'], queryFn: apiClient.voicePolicy, retry: false });
  const [speakerId, setSpeakerId] = useState('');
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [result, setResult] = useState<AuthResult | null>(null);
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [now, setNow] = useState(Date.now());
  const [wait, setWait] = useState(0);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    if (!speakerId && speakers?.length) {
      setSpeakerId((speakers.find(s => /bavesh/i.test(s.displayName)) ?? speakers[0]).id);
    }
  }, [speakers, speakerId]);

  const speaker = speakers?.find(s => s.id === speakerId);
  const { data: vp } = useQuery({ queryKey: ['voiceprint', speakerId], queryFn: () => apiClient.voiceprint(speakerId), enabled: !!speakerId, retry: false });
  const firstName = speaker?.displayName.split(/[\s(·-]/)[0] ?? 'this speaker';

  const issue = useMutation({
    mutationFn: (stepUp: boolean) => apiClient.issueChallenge(speakerId, 'phrase', stepUp),
    onSuccess: c => { setChallenge(c); setResult(null); setNote(null); setWait(0); },
    onError: e => setWait(retrySeconds((e as Error).message)),
  });

  const auth = useMutation({
    mutationFn: (blob: Blob) => apiClient.authenticate(challenge!.id, blob),
    onSuccess: r => {
      setResult(r);
      const voice = r.branches.find(b => b.name === 'speaker_embedding');
      setAttempts(prev => [
        { at: new Date().toLocaleTimeString(), decision: r.decision, voice: voice?.score, note: r.explanation[0] ?? '' },
        ...prev,
      ].slice(0, 6));
    },
  });

  const capture = useCapture((blob, ms) => {
    if (ms < MIN_MS) { setNote('That was very short. Read all the words, then press stop.'); return; }
    setNote(null);
    auth.mutate(blob);
  });

  // challenge countdown
  useEffect(() => {
    if (!challenge || result) return;
    const t = window.setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(t);
  }, [challenge, result]);
  // throttle countdown
  useEffect(() => {
    if (wait <= 0) return;
    const t = window.setTimeout(() => setWait(w => Math.max(0, w - 1)), 1000);
    return () => clearTimeout(t);
  }, [wait]);
  // space bar starts and stops the recording while a phrase is on screen
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || !challenge || result || auth.isPending) return;
      const el = e.target as HTMLElement;
      if (el && /input|select|textarea|button/i.test(el.tagName)) return;
      e.preventDefault();
      if (capture.recording) capture.stop(); else void capture.start();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [challenge, result, auth.isPending, capture]);

  const ttl = challenge ? new Date(challenge.expiresAt).getTime() - new Date(challenge.issuedAt).getTime() : 1;
  const remaining = challenge ? Math.max(0, new Date(challenge.expiresAt).getTime() - now) : 0;
  const expired = !!challenge && !result && remaining <= 0 && !capture.recording && !auth.isPending;

  const reset = (id: string) => { setSpeakerId(id); setChallenge(null); setResult(null); setNote(null); issue.reset(); auth.reset(); };

  return (
    <>
      <PageHeader
        eyebrow="Voice access"
        title="Unlock"
        description="Read the words on the screen. The system checks that they are fresh, that the recording is untouched, and that the voice is the enrolled one. A correct answer cannot make up for a different voice."
      />
      <PageBody>
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] gap-6 items-start">
          <Card className="min-h-[520px]">
            <CardHeader
              title={<span className="flex items-center gap-2">{result?.decision === 'ACCEPT' ? <LockOpen className="w-4 h-4 text-app-accept" /> : <Lock className="w-4 h-4 text-app-text-subtle" />} {speaker ? `${firstName}'s voiceprint` : 'No speaker enrolled'}</span>}
              subtitle={speaker ? (vp?.provenance
                ? `Voiceprint from ${vp.nClips} clips · Studio sessions ${vp.provenance.sessions.join(', ')} · ${vp.provenance.devices.join(', ').toLowerCase().replace(/_/g, ' ')}`
                : `Voiceprint from ${vp?.nClips ?? speaker.utteranceCount} enrolment clips`) : undefined}
              actions={speakers && speakers.length > 1 ? (
                <Select value={speakerId} onChange={e => reset(e.target.value)} className="!h-8 !text-[12.5px] w-44">
                  {speakers.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
                </Select>
              ) : undefined}
            />
            <CardBody className="flex flex-col gap-5">
              {!speaker && <Notice tone="warning" title="Nobody is enrolled">Enrol a speaker first.</Notice>}

              {speaker && !challenge && !result && (
                <div className="flex flex-col items-center text-center gap-4 py-10">
                  <div className="w-20 h-20 rounded-full bg-app-surface-muted flex items-center justify-center"><Lock className="w-8 h-8 text-app-text-muted" /></div>
                  <div>
                    <div className="font-serif text-[26px] leading-tight">Say it to open it</div>
                    <p className="text-[13.5px] text-app-text-muted mt-1.5 max-w-sm">You will be shown {health ? 'six random words' : 'a few random words'} to read aloud, about five seconds. They are different every time.</p>
                  </div>
                  <Button variant="primary" size="lg" onClick={() => issue.mutate(false)} loading={issue.isPending} disabled={wait > 0} icon={<Mic className="w-4 h-4" />}>
                    {wait > 0 ? `Wait ${wait} s` : `Unlock as ${firstName}`}
                  </Button>
                </div>
              )}

              {speaker && challenge && !result && (
                <div className="flex flex-col items-center gap-6 py-4">
                  <div className="flex items-center gap-3 w-full max-w-md">
                    <Timer className={cn('w-4 h-4', expired ? 'text-app-reject' : 'text-app-text-subtle')} />
                    <div className="flex-1 h-1.5 rounded-full bg-app-surface-muted overflow-hidden">
                      <div className={cn('h-full transition-[width] duration-200', remaining / ttl < 0.25 ? 'bg-app-reject' : 'bg-app-accent')} style={{ width: `${(remaining / ttl) * 100}%` }} />
                    </div>
                    <span className="tnum text-[12.5px] w-16 text-right text-app-text-muted">{expired ? 'expired' : `${Math.ceil(remaining / 1000)} s`}</span>
                  </div>

                  {challenge.stepUp && <Badge tone="warning">Second sample: the voice must match outright this time</Badge>}

                  <div className="flex flex-wrap justify-center gap-x-5 gap-y-3 max-w-xl" aria-label="Words to read">
                    {challenge.phrase.map((w, i) => (
                      <span key={i} className="font-serif text-[40px] leading-none tracking-tight text-app-text">{w}</span>
                    ))}
                  </div>

                  {expired ? (
                    <Notice tone="warning" title="That phrase has expired">Phrases only last a minute, so a recording made earlier is useless. Get a new one.</Notice>
                  ) : (
                    <div className="flex flex-col items-center gap-3">
                      <button
                        type="button"
                        disabled={auth.isPending}
                        onClick={() => (capture.recording ? capture.stop() : void capture.start())}
                        className={cn(
                          'relative w-[76px] h-[76px] rounded-full flex items-center justify-center transition-all disabled:opacity-50',
                          capture.recording ? 'bg-app-reject text-white' : 'bg-app-accent text-app-on-accent hover:bg-app-accent-hover',
                        )}
                        style={capture.recording ? { boxShadow: `0 0 0 ${6 + capture.level * 22}px rgba(160, 50, 40, 0.16)` } : undefined}
                        title={capture.recording ? 'Stop and check' : 'Start recording'}
                      >
                        {capture.recording ? <Square className="w-6 h-6 fill-current" /> : <Mic className="w-7 h-7" />}
                      </button>
                      <div className="text-[13px] text-app-text-muted h-5 tnum">
                        {auth.isPending ? 'Checking…' : capture.recording ? `Recording ${(capture.ms / 1000).toFixed(1)} s · press again when done` : 'Press the microphone (or the space bar), read the words, press again'}
                      </div>
                    </div>
                  )}

                  {note && <Notice tone="warning">{note}</Notice>}
                  {capture.error && <Notice tone="reject" title="No microphone">{capture.error}</Notice>}
                  {auth.error && <Notice tone="reject" title="Could not check that recording">{(auth.error as Error).message}</Notice>}

                  <Button size="sm" variant="ghost" icon={<RefreshCw className="w-3.5 h-3.5" />} onClick={() => issue.mutate(false)} loading={issue.isPending} disabled={capture.recording || auth.isPending || wait > 0}>
                    {wait > 0 ? `Wait ${wait} s` : 'Different words'}
                  </Button>
                </div>
              )}

              {speaker && result && challenge && (
                <Verdict
                  result={result}
                  challenge={challenge}
                  name={firstName}
                  threshold={health?.voiceThreshold ?? 0.62}
                  margin={health?.voiceGreyMargin ?? 0.08}
                  onAgain={() => issue.mutate(false)}
                  onStepUp={() => issue.mutate(true)}
                  busy={issue.isPending}
                  wait={wait}
                />
              )}

              {issue.error && wait === 0 && <Notice tone="reject" title="Could not start a new attempt">{(issue.error as Error).message}</Notice>}
              {issue.error && wait > 0 && <Notice tone="warning" title="Slow down">Several attempts in a row did not match. The next one opens in {wait} s.</Notice>}
            </CardBody>
          </Card>

          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader title="What is checked" subtitle="Every gate must pass. None can be outvoted." />
              <CardBody className="flex flex-col gap-2.5 text-[13px]">
                <GateLine title="Fresh" text="The phrase is drawn at random for this attempt, single-use, and expires in a minute." />
                <GateLine title="Untouched" text="The recording is not a copy of one already seen." />
                <GateLine title="Said now" text="The words you speak are the words on screen, in order. A recording made for another attempt cannot contain them." />
                <GateLine title="Your voice" text={`The voiceprint must reach ${(health?.voiceThreshold ?? 0.62).toFixed(2)}. Below ${((health?.voiceThreshold ?? 0.62) - (health?.voiceGreyMargin ?? 0.08)).toFixed(2)} it is a flat no; in between you are asked once more.`} />
                <div className="pt-1 flex flex-wrap gap-2">
                  <Badge tone={health?.voicePolicySource === 'calibrated' && !health?.voicePolicyProvisional ? 'accept' : 'warning'}>
                    {health?.voicePolicySource === 'calibrated' ? (health?.voicePolicyProvisional ? 'threshold measured (provisional)' : 'threshold measured on held-out sessions') : 'default threshold, not yet measured'}
                  </Badge>
                  {health?.voicePolicyError && <Badge tone="reject">policy file ignored</Badge>}
                </div>
              </CardBody>
            </Card>
            <EvidenceCard e={evidence} />
            <Card>
              <CardHeader title="Attempts this session" />
              <CardBody>
                {attempts.length === 0 ? <p className="text-[13px] text-app-text-subtle">None yet.</p> : (
                  <ul className="flex flex-col gap-2">
                    {attempts.map((a, i) => (
                      <li key={i} className="flex items-center gap-2 text-[12.5px]">
                        {a.decision === 'ACCEPT' ? <CheckCircle2 className="w-4 h-4 text-app-accept shrink-0" /> : a.decision === 'REJECT' ? <XCircle className="w-4 h-4 text-app-reject shrink-0" /> : <AlertTriangle className="w-4 h-4 text-app-warning shrink-0" />}
                        <span className="tnum text-app-text-subtle whitespace-nowrap shrink-0">{a.at}</span>
                        <span className="tnum whitespace-nowrap shrink-0 font-medium">{a.voice !== undefined ? a.voice.toFixed(2) : '—'}</span>
                        <span className="text-app-text-muted truncate min-w-0" title={a.note}>{a.decision === 'ACCEPT' ? 'unlocked' : a.note.replace(/^Rejected: /, '')}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </CardBody>
            </Card>
          </div>
        </div>
      </PageBody>
    </>
  );
}

function GateLine({ title, text }: { title: string; text: string }) {
  return (
    <div className="flex gap-3">
      <span className="w-[78px] shrink-0 font-semibold">{title}</span>
      <span className="text-app-text-muted">{text}</span>
    </div>
  );
}

/** Which of the shown words the transcript contains; decoration only, the phrase gate decides. */
function heard(expected: string[], transcript: string) {
  const toks = transcript.toLowerCase().match(/[a-z]+/g) ?? [];
  return expected.map(w => toks.some(t => t === w || (t.length >= 4 && Math.abs(t.length - w.length) <= 2 && (t.startsWith(w.slice(0, -1)) || w.startsWith(t.slice(0, -1))))));
}

function Verdict({ result, challenge, name, threshold, margin, onAgain, onStepUp, busy, wait }: {
  result: AuthResult; challenge: Challenge; name: string; threshold: number; margin: number;
  onAgain: () => void; onStepUp: () => void; busy: boolean; wait: number;
}) {
  const by = (n: BranchScore['name']) => result.branches.find(b => b.name === n);
  const voice = by('speaker_embedding');
  const words = by('phrase');
  const live = by('liveness');
  const integ = by('signal_integrity');
  const total = challenge.phrase.length;
  const ok = result.decision === 'ACCEPT';
  const border = result.decision === 'BORDERLINE';
  const tone = ok ? 'accept' : border ? 'warning' : 'reject';
  const stripped = (result.explanation[0] ?? '').replace(/^(Rejected|ACCEPT|BORDERLINE|REJECT): ?/, '');
  const first = stripped.charAt(0).toUpperCase() + stripped.slice(1);
  // The backend says which words it matched; the browser-side guess is only for an old server.
  const got = result.phraseMatched
    ? challenge.phrase.map(w => result.phraseMatched!.includes(w))
    : heard(challenge.phrase, result.transcript);

  const Row = ({ label, b, detail }: { label: string; b?: BranchScore; detail?: string }) => (
    <div className="flex items-center gap-3 text-[13px]">
      {!b ? <MinusCircle className="w-4 h-4 text-app-text-subtle shrink-0" /> : b.passed ? <CheckCircle2 className="w-4 h-4 text-app-accept shrink-0" /> : <XCircle className="w-4 h-4 text-app-reject shrink-0" />}
      <span className="w-40 shrink-0 font-medium">{label}</span>
      <span className="text-app-text-muted tnum">{b ? detail ?? (b.passed ? 'passed' : 'failed') : 'not reached'}</span>
    </div>
  );

  return (
    <div className="flex flex-col gap-5">
      <div className={cn('rounded-lg border px-5 py-4 flex items-start gap-4',
        tone === 'accept' && 'bg-app-accept-soft border-app-accept/25', tone === 'warning' && 'bg-app-warning-soft border-app-warning/25', tone === 'reject' && 'bg-app-reject-soft border-app-reject/25')}>
        {ok ? <LockOpen className="w-8 h-8 text-app-accept shrink-0 mt-0.5" /> : border ? <AlertTriangle className="w-8 h-8 text-app-warning shrink-0 mt-0.5" /> : <Lock className="w-8 h-8 text-app-reject shrink-0 mt-0.5" />}
        <div className="min-w-0">
          <div className="font-serif text-[30px] leading-tight">{ok ? 'Unlocked' : border ? 'Not sure yet' : 'Access denied'}</div>
          <p className="text-[13.5px] text-app-text-muted mt-1">
            {ok ? `The voice matched ${name}'s voiceprint and the words were the ones shown.` : border ? `The voice is close to ${name}'s but not conclusive. One more phrase, and it has to match outright.` : first}
          </p>
        </div>
      </div>

      {voice && <VoiceMeter score={voice.score} threshold={voice.threshold || threshold} margin={margin} />}

      <div className="flex flex-col gap-2.5">
        <Row label="Fresh challenge" b={live} detail="unused and in time" />
        <Row label="Recording untouched" b={integ} detail="not a copy of one seen before" />
        <Row label="Words said" b={words} detail={words ? `${Math.round(words.score * total)} of ${total} words, in order` : undefined} />
        <Row label="Voice" b={voice} detail={voice ? `${voice.score.toFixed(2)} (needs ${(voice.threshold || threshold).toFixed(2)})` : undefined} />
      </div>

      {result.transcript && (
        <div className="text-[13px]">
          <div className="text-app-text-subtle mb-1.5">What was heard</div>
          <div className="flex flex-wrap gap-1.5">
            {challenge.phrase.map((w, i) => (
              <Badge key={i} tone={got[i] ? 'accept' : 'reject'}>{got[i] ? w : `${w} · not heard`}</Badge>
            ))}
          </div>
          <p className="mt-2 text-app-text-muted italic">“{result.transcript}”</p>
        </div>
      )}

      <div className="flex items-center gap-3">
        {border && <Button variant="primary" onClick={onStepUp} loading={busy} disabled={wait > 0} icon={<Mic className="w-4 h-4" />}>Say one more phrase</Button>}
        {!border && <Button variant={ok ? 'secondary' : 'primary'} onClick={onAgain} loading={busy} disabled={wait > 0} icon={<RefreshCw className="w-4 h-4" />}>{wait > 0 ? `Wait ${wait} s` : ok ? 'Lock and try again' : 'Try again'}</Button>}
        <span className="text-[12px] text-app-text-subtle tnum">decided in {(result.latencyMs / 1000).toFixed(1)} s</span>
      </div>
    </div>
  );
}

/** The voice score against the three zones it is judged in: no / once more / yes. */
function VoiceMeter({ score, threshold, margin }: { score: number; threshold: number; margin: number }) {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  const floor = threshold - margin;
  return (
    <div>
      <div className="flex justify-between text-[12px] text-app-text-subtle mb-1.5">
        <span>Voice match</span>
        <span className="tnum text-app-text font-medium">{score.toFixed(2)}</span>
      </div>
      <div className="relative h-3 rounded-full overflow-hidden flex">
        <div className="bg-app-reject-soft" style={{ width: `${clamp(floor) * 100}%` }} />
        <div className="bg-app-warning-soft" style={{ width: `${(clamp(threshold) - clamp(floor)) * 100}%` }} />
        <div className="bg-app-accept-soft flex-1" />
        <div className="absolute top-0 bottom-0 w-[3px] bg-app-text rounded" style={{ left: `calc(${clamp(score) * 100}% - 1.5px)` }} />
      </div>
      <div className="relative h-4 text-[11px] text-app-text-subtle tnum">
        <span className="absolute" style={{ left: `${clamp(floor) * 100}%`, transform: 'translateX(-50%)' }}>{floor.toFixed(2)}</span>
        <span className="absolute" style={{ left: `${clamp(threshold) * 100}%`, transform: 'translateX(-50%)' }}>{threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}

const pctText = (v: number, digits = 1) => `${(100 * v).toFixed(digits)}%`;

/** The numbers behind the threshold, stated with their limits; nothing here is a claim about people in general. */
function EvidenceCard({ e }: { e?: import('../api/types').VoiceEvidence }) {
  if (!e) return null;
  return (
    <Card>
      <CardHeader title="What it has been measured on" subtitle={e.measured ? `Threshold ${e.threshold.toFixed(2)}, chosen from data` : 'Not measured yet'} />
      <CardBody className="flex flex-col gap-3 text-[13px]">
        {!e.measured ? (
          <p className="text-app-text-muted">The threshold is the default, 0.62, a starting point. Record the Studio sessions and run the calibration to replace it with a measured one.</p>
        ) : (
          <>
            {e.frr && (
              <div>
                <div className="font-semibold">The enrolled speaker</div>
                <div className="text-app-text-muted">Turned away {pctText(e.frr.rate)} of {e.nGenuine} held-out attempts (95% interval {pctText(e.frr.low)}–{pctText(e.frr.high)}), across {e.sessions.length} recording sessions.</div>
              </div>
            )}
            {e.far && (
              <div>
                <div className="font-semibold">Other people</div>
                <div className="text-app-text-muted">Let in {pctText(e.far.rate, 2)} of {e.nImpostor.toLocaleString()} attempts by recorded strangers (95% interval up to {pctText(e.far.high, 2)}).</div>
              </div>
            )}
            <div className="flex flex-wrap gap-1.5">
              {e.provisional && <Badge tone="warning">provisional</Badge>}
              {e.cohorts.map(c => <Badge key={c}>{c}</Badge>)}
            </div>
            {e.limits && <p className="text-[12px] text-app-text-subtle leading-relaxed">{e.limits}</p>}
          </>
        )}
      </CardBody>
    </Card>
  );
}
