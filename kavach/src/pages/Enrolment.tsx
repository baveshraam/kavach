import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { Speaker, Utterance, CSBG } from '../api/types';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { AudioRecorder } from '../components/ui/AudioRecorder';
import { FactEditor } from './Speakers';
import { Card, CardHeader, CardBody, Button, Field, Input, Select, Notice, Badge, LangBar, TokenText, Spinner, Stat, titleCase, cn } from '../components/ui/kit';
import { Check, ArrowRight, Hammer, Network, ShieldCheck } from 'lucide-react';

const STEPS = ['Profile', 'Voice samples', 'Knowledge facts', 'Build'];

/** Minimum enrolment speech (`Settings.min_enrolment_seconds`); below it the graph is mostly prior. */
const TARGET_SEC = 180;

/**
 * Free-speech prompts, one per topic in the recording protocol. Each targets
 * a class the CSBG needs evidence for; read speech would measure the script,
 * not the speaker (PROJECT.md §5.2.4), so none of these has text to read.
 */
const PROMPTS = [
  { topic: 'Family', text: 'Tell us about your family — who lives at home and what they do.', cls: 'KINSHIP' },
  { topic: 'Food', text: 'What did you eat yesterday? Describe your favourite meal.', cls: 'FOOD' },
  { topic: 'Commute', text: 'How do you get to college, and how long does it take?', cls: 'TRANSPORT' },
  { topic: 'Money', text: 'How much do you spend in a normal week, and on what?', cls: 'MONEY_COMMERCE' },
  { topic: 'Phone', text: 'What do you use your phone for most during the day?', cls: 'TECH_DIGITAL' },
  { topic: 'Study', text: 'Describe your timetable on a typical college day.', cls: 'EDU_WORK' },
  { topic: 'Festival', text: 'How does your family celebrate your favourite festival?', cls: 'RELIGION_FESTIVAL' },
  { topic: 'Numbers', text: 'Tell us your hostel room or house number, and a phone number you know by heart (a made-up one is fine).', cls: 'NUMBER' },
];

export function Enrolment() {
  const [step, setStep] = useState(0);
  const [speaker, setSpeaker] = useState<Speaker | null>(null);

  return (
    <>
      <PageHeader
        eyebrow="Live system"
        title="Enrol a speaker"
        description="Four steps: consent and profile, a few minutes of free speech, some personal facts for challenges, then build the voiceprint and code-switch graph."
      />
      <PageBody className="max-w-[1100px]">
        <ol className="grid grid-cols-4 gap-3">
          {STEPS.map((s, i) => (
            <li key={s} className={cn('flex items-center gap-3 rounded-lg border px-4 py-3',
              i === step ? 'border-app-accent/40 bg-app-accent-soft/60' : 'border-app-border bg-app-surface')}>
              <span className={cn('w-7 h-7 rounded-full flex items-center justify-center text-[12.5px] font-semibold shrink-0',
                i < step ? 'bg-app-accept text-white' : i === step ? 'bg-app-accent text-app-on-accent' : 'bg-app-surface-muted text-app-text-subtle border border-app-border')}>
                {i < step ? <Check className="w-3.5 h-3.5" /> : i + 1}
              </span>
              <span className={cn('text-[13.5px] font-medium', i > step && 'text-app-text-subtle')}>{s}</span>
            </li>
          ))}
        </ol>

        {step === 0 && <StepProfile onNext={s => { setSpeaker(s); setStep(1); }} />}
        {step === 1 && speaker && <StepVoice speaker={speaker} onNext={() => setStep(2)} />}
        {step === 2 && speaker && (
          <div className="flex flex-col gap-4">
            <FactEditor speakerId={speaker.id} />
            <div className="flex justify-end"><Button variant="primary" onClick={() => setStep(3)}>Continue <ArrowRight className="w-4 h-4" /></Button></div>
          </div>
        )}
        {step === 3 && speaker && <StepBuild speaker={speaker} />}
      </PageBody>
    </>
  );
}

function StepProfile({ onNext }: { onNext: (s: Speaker) => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState({
    displayName: '', ageRange: '18-25', gender: '', dominantLanguage: 'Balanced',
    otherLanguages: '', device: 'Laptop microphone', environment: 'quiet room',
  });
  const [consent, setConsent] = useState(false);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setForm({ ...form, [k]: e.target.value });

  const create = useMutation({
    mutationFn: () => apiClient.createSpeaker({
      ...form,
      otherLanguages: form.otherLanguages.split(',').map(s => s.trim()).filter(Boolean),
      consentGiven: consent,
      dominantLanguage: form.dominantLanguage as Speaker['dominantLanguage'],
    }),
    onSuccess: s => { queryClient.invalidateQueries({ queryKey: ['speakers'] }); onNext(s); },
  });

  return (
    <Card>
      <CardHeader title="Speaker profile" subtitle="Only the display name is required. Profile fields feed the fairness audit, so leave a field blank rather than guess." />
      <CardBody>
        <form onSubmit={e => { e.preventDefault(); if (consent) create.mutate(); }} className="flex flex-col gap-5">
          <div className="grid grid-cols-2 gap-4">
            <Field label="Display name"><Input required value={form.displayName} onChange={set('displayName')} placeholder="e.g. S13 · Priya" /></Field>
            <Field label="Age range">
              <Select value={form.ageRange} onChange={set('ageRange')}>{['18-25', '26-35', '36-50', '50+'].map(o => <option key={o}>{o}</option>)}</Select>
            </Field>
            <Field label="Gender (optional)">
              <Select value={form.gender} onChange={set('gender')}><option value="">Prefer not to say</option><option>Female</option><option>Male</option><option>Other</option></Select>
            </Field>
            <Field label="Stronger language">
              <Select value={form.dominantLanguage} onChange={set('dominantLanguage')}><option>Tamil</option><option>English</option><option>Balanced</option></Select>
            </Field>
            <Field label="Microphone">
              <Select value={form.device} onChange={set('device')}><option>Laptop microphone</option><option>Phone</option><option>Headset</option></Select>
            </Field>
            <Field label="Room">
              <Select value={form.environment} onChange={set('environment')}><option>quiet room</option><option>classroom</option><option>outdoors</option><option>noisy</option></Select>
            </Field>
          </div>
          <label className="flex items-start gap-3 rounded-lg border border-app-border bg-app-surface-muted/50 p-4 cursor-pointer">
            <input type="checkbox" checked={consent} onChange={e => setConsent(e.target.checked)} className="mt-1 accent-[var(--app-accent)] w-4 h-4" />
            <span className="text-[13px] leading-relaxed">
              The speaker has given informed consent for their voice and personal facts to be stored for authentication research. They can ask for everything to be deleted at any time.
            </span>
          </label>
          {create.error && <Notice tone="reject" title="Could not create the speaker">{(create.error as Error).message}</Notice>}
          <div className="flex justify-end">
            <Button type="submit" variant="primary" disabled={!consent || !form.displayName} loading={create.isPending}>Create speaker <ArrowRight className="w-4 h-4" /></Button>
          </div>
        </form>
      </CardBody>
    </Card>
  );
}

function StepVoice({ speaker, onNext }: { speaker: Speaker; onNext: () => void }) {
  const [active, setActive] = useState(0);
  const [done, setDone] = useState<Record<number, Utterance>>({});
  const total = Object.values(done).reduce((a, u) => a + u.durationSec, 0);

  const upload = useMutation({
    mutationFn: ({ blob, name }: { blob: Blob; name?: string }) => apiClient.uploadUtterance(speaker.id, 'free-speech', blob, name),
    onSuccess: u => {
      setDone(d => ({ ...d, [active]: u }));
      const next = PROMPTS.findIndex((_, i) => i !== active && !done[i]);
      if (next >= 0) setActive(next);
    },
  });

  const current = done[active];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[300px_1fr] gap-6 items-start">
      <Card>
        <CardHeader title="Topics" subtitle={`${Math.round(total)} s of ${TARGET_SEC} s recommended`} />
        <div className="px-5 pb-3"><div className="h-1.5 rounded-full bg-app-surface-muted overflow-hidden"><div className="h-full bg-app-accent" style={{ width: `${Math.min(100, (total / TARGET_SEC) * 100)}%` }} /></div></div>
        <div className="px-2 pb-2 flex flex-col">
          {PROMPTS.map((p, i) => (
            <button key={p.topic} onClick={() => setActive(i)}
              className={cn('flex items-center gap-3 px-3 py-2 rounded-md text-left text-[13px]', i === active ? 'bg-app-accent-soft text-app-accent font-medium' : 'hover:bg-app-surface-muted')}>
              <span className={cn('w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-[11px]', done[i] ? 'bg-app-accept text-white' : 'border border-app-border-strong text-app-text-subtle')}>
                {done[i] ? <Check className="w-3 h-3" /> : i + 1}
              </span>
              <span className="flex-1">{p.topic}</span>
              {done[i] && <span className="text-[11.5px] text-app-text-subtle tnum">{Math.round(done[i].durationSec)}s</span>}
            </button>
          ))}
        </div>
      </Card>

      <div className="flex flex-col gap-4">
        <Card>
          <CardBody className="pt-5 flex flex-col gap-4">
            <div className="flex items-center gap-2"><Badge tone="accent">{PROMPTS[active].topic}</Badge><Badge>elicits {titleCase(PROMPTS[active].cls)}</Badge></div>
            <p className="font-serif text-[21px] leading-snug">{PROMPTS[active].text}</p>
            <p className="text-[12.5px] text-app-text-muted">Speak naturally for 20–40 seconds, mixing Tamil and English the way you normally would. There is no script.</p>
            <AudioRecorder key={active} allowUpload busy={upload.isPending} acceptLabel="Save clip" onAccept={(blob, _d, name) => upload.mutate({ blob, name })} />
            {upload.isPending && <Spinner label="Transcribing and tagging…" />}
            {upload.error && <Notice tone="reject" title="Upload failed">{(upload.error as Error).message}</Notice>}
          </CardBody>
        </Card>

        {current && (
          <Card className="animate-in">
            <CardHeader title="What the system heard" subtitle={current.annotated ? `${current.tokens.length} words tagged` : 'Stored, but not annotated — ASR was unavailable.'} />
            <CardBody className="flex flex-col gap-3">
              {current.tokens.length ? <TokenText tokens={current.tokens} className="text-[14px]" /> : <p className="text-[13px] text-app-text-muted">{current.transcript || '—'}</p>}
              <LangBar ta={current.tokens.filter(t => t.language === 'TA').length} en={current.tokens.filter(t => t.language === 'EN').length} showLabels />
            </CardBody>
          </Card>
        )}

        <div className="flex justify-between items-center">
          <span className="text-[12.5px] text-app-text-muted">{Object.keys(done).length} of {PROMPTS.length} topics recorded</span>
          <Button variant="primary" onClick={onNext} disabled={!Object.keys(done).length}>Continue <ArrowRight className="w-4 h-4" /></Button>
        </div>
      </div>
    </div>
  );
}

function StepBuild({ speaker }: { speaker: Speaker }) {
  const queryClient = useQueryClient();
  const build = useMutation({
    mutationFn: () => apiClient.completeEnrolment(speaker.id),
    onSuccess: () => queryClient.invalidateQueries(),
  });
  const csbg: CSBG | undefined = build.data?.csbg;

  return (
    <Card>
      <CardHeader title={`Build ${speaker.displayName}’s models`} subtitle="Fits the voice template (a centroid of all clips) and the code-switch graph (a smoothed distribution), both from scratch." />
      <CardBody className="flex flex-col gap-5">
        {!build.data && (
          <div>
            <Button variant="primary" size="lg" icon={<Hammer className="w-4 h-4" />} loading={build.isPending} onClick={() => build.mutate()}>
              Build voiceprint and graph
            </Button>
            {build.isPending && <p className="text-[12.5px] text-app-text-muted mt-2">Extracting ECAPA embeddings on CPU — this can take a minute.</p>}
          </div>
        )}
        {build.error && <Notice tone="reject" title="Build failed">{(build.error as Error).message}</Notice>}
        {csbg && (
          <>
            <div className="grid grid-cols-4 gap-4 animate-in">
              <Stat label="Code-mixing index" value={csbg.cmi.toFixed(1)} />
              <Stat label="Integration index" value={csbg.iIndex.toFixed(2)} />
              <Stat label="Tamil-matrix share" value={csbg.matrixLanguageRatio.toFixed(2)} />
              <Stat label="Classes with too little data" value={`${csbg.sparseClasses.length} / 21`} />
            </div>
            {build.data!.warnings.length ? (
              <div className="flex flex-col gap-2">
                {build.data!.warnings.map((w, i) => <Notice key={i} tone="warning">{w}</Notice>)}
              </div>
            ) : <Notice tone="accept" title="Enrolment complete">No warnings.</Notice>}
            <div className="flex gap-3">
              <Link to={`/graph-explorer?speaker=${speaker.id}`}><Button icon={<Network className="w-4 h-4" />}>View the graph</Button></Link>
              <Link to="/authenticate"><Button variant="primary" icon={<ShieldCheck className="w-4 h-4" />}>Try logging in</Button></Link>
            </div>
          </>
        )}
      </CardBody>
    </Card>
  );
}
