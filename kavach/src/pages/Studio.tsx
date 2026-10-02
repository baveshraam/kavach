import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { AudioRecorder } from '../components/ui/AudioRecorder';
import { Card, CardHeader, CardBody, Button, Field, Select, Input, Badge, Notice, Spinner } from '../components/ui/kit';
import type { StudioPlanItem } from '../api/types';

const KIND_LABEL: Record<string, string> = { read: 'Read aloud', free: 'Speak freely', fact: 'Answer about yourself', words: 'Read the six words' };

/** One queue entry per repetition, in the plan's order. */
function expand(items: StudioPlanItem[]) {
  return items.flatMap(it => Array.from({ length: it.repeat }, (_, rep) => ({ ...it, rep: rep + 1 })));
}

/**
 * Where you were, kept across a refresh. Without it a refresh forgot the session id and the
 * server offered the *next unused* one, so the rest of a sitting landed in the next session --
 * and a refresh during S3 would have put it in S4, the held-out session.
 */
const SAVED_KEY = 'kavach.studio.v1';
type Saved = { speaker?: string; session?: string; index?: number; started?: boolean; device?: string; environment?: string };
function loadSaved(): Saved {
  try { return JSON.parse(localStorage.getItem(SAVED_KEY) ?? '{}'); } catch { return {}; }
}

export function Studio() {
  const queryClient = useQueryClient();
  const saved = useMemo(loadSaved, []);
  const [speaker, setSpeaker] = useState(saved.speaker ?? 'S04');
  const [session, setSession] = useState(saved.session ?? '');
  const [device, setDevice] = useState(saved.device ?? 'DEMO_LAPTOP_MIC');
  const [environment, setEnvironment] = useState(saved.environment ?? 'QUIET_ROOM');
  const [note, setNote] = useState('');
  const [started, setStarted] = useState(saved.started ?? false);
  const [index, setIndex] = useState(saved.index ?? 0);

  useEffect(() => {
    localStorage.setItem(SAVED_KEY, JSON.stringify({ speaker, session, index, started, device, environment }));
  }, [speaker, session, index, started, device, environment]);

  const plan = useQuery({ queryKey: ['studioPlan', speaker, session], queryFn: () => apiClient.studioPlan(speaker, session || undefined), retry: false });
  const summary = useQuery({ queryKey: ['studioSummary', speaker], queryFn: () => apiClient.studioSummary(speaker), retry: false, enabled: plan.isSuccess });
  const queue = useMemo(() => expand(plan.data?.items ?? []), [plan.data]);
  const sessionId = session || plan.data?.sessionId || 'S1';
  const current = queue[index];

  const upload = useMutation({
    mutationFn: (v: { blob: Blob; name?: string }) => apiClient.studioUpload({
      speaker, sessionId, kind: current.kind, promptId: current.promptId, device, environment,
      textHint: current.textTa, stateNote: note, blob: v.blob, filename: v.name,
    }),
    onSuccess: () => { setIndex(i => i + 1); queryClient.invalidateQueries({ queryKey: ['studioSummary', speaker] }); },
  });

  if (plan.isError) {
    return (
      <>
        <PageHeader title="Recording Studio" description="Record labelled sessions of your own voice." />
        <PageBody>
          <Notice tone="warning" title="The Studio is not available">
            {(plan.error as Error).message}. Start the backend with <span className="mono">run_studio.ps1</span> (the demo build never accepts recordings).
          </Notice>
        </PageBody>
      </>
    );
  }

  return (
    <>
      <PageHeader title="Recording Studio" description="Record labelled sessions of your own voice through the same browser microphone the demo uses. Every clip is saved with its session, device and room." />
      <PageBody>
        <div className="grid grid-cols-1 xl:grid-cols-[1fr_380px] gap-6">
          <div className="flex flex-col gap-4">
            <Card>
              <CardHeader title="1 · This session" subtitle="Change nothing mid-session: a session is one sitting, one device, one room." />
              <CardBody className="grid grid-cols-1 sm:grid-cols-4 gap-3">
                <Field label="Speaker"><Input value={speaker} onChange={e => { setSpeaker(e.target.value); setStarted(false); setIndex(0); }} /></Field>
                <Field label="Session"><Input value={session || plan.data?.sessionId || ''} onChange={e => { setSession(e.target.value); setIndex(0); }} /></Field>
                <Field label="Device"><Select value={device} onChange={e => setDevice(e.target.value)}>{(plan.data?.devices ?? []).map(d => <option key={d} value={d}>{d}</option>)}</Select></Field>
                <Field label="Room"><Select value={environment} onChange={e => setEnvironment(e.target.value)}>{(plan.data?.environments ?? []).map(d => <option key={d} value={d}>{d}</option>)}</Select></Field>
                <div className="sm:col-span-4"><Field label="Note (optional: tired, hurried, far from the mic …)"><Input value={note} onChange={e => setNote(e.target.value)} /></Field></div>
              </CardBody>
            </Card>

            <Card>
              <CardHeader title="2 · Record" subtitle={plan.data ? `${queue.length} recordings planned, about ${plan.data.estimatedMinutes} min for this session.` : undefined} />
              <CardBody>
                {plan.isLoading && <Spinner label="Loading the plan…" />}
                {plan.data && !plan.data.hasFacts && <Notice tone="warning" className="mb-3" title="No personal facts yet">Add a few on the Speakers page to include the answer-about-yourself recordings.</Notice>}
                {plan.data && !started && (
                  // Pin the id now: from here on this sitting is `sessionId`, whatever the server would offer next.
                  <Button variant="primary" onClick={() => { setSession(plan.data!.sessionId); setStarted(true); setIndex(0); }}>Start session {sessionId}</Button>
                )}
                {plan.data && started && (
                  <p className="text-[12.5px] text-app-text-muted mb-3">
                    Recording into <span className="font-semibold">{sessionId}</span> on {device}, {environment}. A refresh keeps your place.
                    {' '}<button type="button" className="underline" onClick={() => { setSession(''); setStarted(false); setIndex(0); upload.reset(); }}>Start a new session instead</button>
                  </p>
                )}
                {started && current && (
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2"><Badge tone="accent">{KIND_LABEL[current.kind]}</Badge><Badge>{index + 1} of {queue.length}</Badge>{current.repeat > 1 && <Badge>take {current.rep} of {current.repeat}</Badge>}</div>
                    <p className="font-serif text-[24px] leading-snug">{current.textTa}</p>
                    {current.kind !== 'fact' && <p className="text-[13px] text-app-text-muted">{current.textEn}</p>}
                    <AudioRecorder key={index} busy={upload.isPending} acceptLabel="Save" onAccept={(blob, _d, name) => upload.mutate({ blob, name })} />
                    {upload.error && <Notice tone="reject" title="Not saved">{(upload.error as Error).message}</Notice>}
                    <div><Button size="sm" variant="ghost" disabled={upload.isPending} onClick={() => { upload.reset(); setIndex(i => i + 1); }}>Skip this one</Button></div>
                  </div>
                )}
                {started && !current && <Notice tone="accept" title="Session complete">Everything planned for {sessionId} is saved. Change the session, device or room above to start the next one.</Notice>}
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader title="What is saved" subtitle="Across all sessions." />
            <CardBody className="flex flex-col gap-3 text-[13px]">
              {summary.data ? (
                <>
                  <div className="tnum"><span className="text-[22px] font-semibold">{summary.data.clips}</span> clips · <span className="font-semibold">{summary.data.minutes}</span> min</div>
                  {Object.entries(summary.data.by_session).map(([id, s]) => (
                    <div key={id} className="flex justify-between"><span>{id} <span className="text-app-text-subtle">{s.devices.join(', ')} · {s.environments.join(', ')}</span></span><span className="tnum">{s.clips} · {s.minutes} min</span></div>
                  ))}
                  <div className="text-app-text-subtle">By kind: {Object.entries(summary.data.by_kind).map(([k, v]) => `${k} ${v.clips}`).join(' · ') || 'none yet'}</div>
                </>
              ) : <span className="text-app-text-subtle">Nothing recorded yet.</span>}
            </CardBody>
          </Card>
        </div>
      </PageBody>
    </>
  );
}
