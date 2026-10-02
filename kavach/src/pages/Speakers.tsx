import { useEffect, useMemo, useState } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, assetUrl } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Speaker, Triple } from '../api/types';
import {
  Card, CardHeader, CardBody, Input, Button, Badge, Table, Th, Td, LangBar, TokenText, Notice, EmptyState, Spinner, minutes, pct, cn,
} from '../components/ui/kit';
import { X, Trash2, Search, Plus, Save, Network, ShieldCheck, ChevronDown, RefreshCw } from 'lucide-react';
import { AudioRecorder } from '../components/ui/AudioRecorder';

/**
 * Suggested predicates for the fact editor. These are suggestions only -- the
 * backend's `skg.FACT_TYPES` is the authority, and a predicate it does not
 * know is still stored (it just targets the OTHER class).
 */
const FACT_SUGGESTIONS: { predicate: string; question: string }[] = [
  { predicate: 'hometown', question: 'Which town or city are you from?' },
  { predicate: 'favouriteFood', question: 'What is your favourite food?' },
  { predicate: 'comfortFood', question: 'What do you eat when you’re feeling low?' },
  { predicate: 'college', question: 'Which college do you study at?' },
  { predicate: 'school', question: 'Which school did you go to?' },
  { predicate: 'familyRole', question: 'Who do you talk to most at home?' },
  { predicate: 'siblingName', question: 'What are your siblings’ names?' },
  { predicate: 'commute', question: 'How do you travel to college?' },
  { predicate: 'commuteTime', question: 'How long does that journey take?' },
  { predicate: 'hostelRoom', question: 'Hostel room or house number?' },
  { predicate: 'favouriteFestival', question: 'Which festival do you enjoy most?' },
  { predicate: 'favouriteFilm', question: 'A film you have watched many times?' },
  { predicate: 'phoneBrand', question: 'What phone do you use?' },
  { predicate: 'firstJob', question: 'First job or internship?' },
];

export function Speakers() {
  const { data: speakers, isLoading } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: utterances } = useQuery({ queryKey: ['utterances'], queryFn: apiClient.getUtterances });
  const [params, setParams] = useSearchParams();
  const selectedId = params.get('speaker');
  const [search, setSearch] = useState('');

  const split = useMemo(() => {
    const m = new Map<string, { ta: number; en: number }>();
    for (const u of utterances ?? []) {
      const acc = m.get(u.speakerId) ?? { ta: 0, en: 0 };
      for (const t of u.tokens) { if (t.language === 'TA') acc.ta++; else if (t.language === 'EN') acc.en++; }
      m.set(u.speakerId, acc);
    }
    return m;
  }, [utterances]);

  const filtered = speakers?.filter(s =>
    `${s.id} ${s.displayName} ${s.environment}`.toLowerCase().includes(search.toLowerCase()));
  const selected = speakers?.find(s => s.id === selectedId) ?? null;
  const select = (id: string | null) => setParams(id ? { speaker: id } : {});

  return (
    <>
      <PageHeader
        eyebrow="Data"
        title="Speakers"
        description="Everyone enrolled in the system, with how much speech we hold for them and how they split their words between Tamil and English."
        actions={
          <div className="relative w-72">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-app-text-subtle" />
            <Input placeholder="Search speakers" value={search} onChange={e => setSearch(e.target.value)} className="pl-9" />
          </div>
        }
      />
      <PageBody>
        <Card className="overflow-hidden">
          {isLoading ? <div className="p-6"><Spinner label="Loading speakers" /></div> : (
            <Table>
              <thead>
                <tr>
                  <Th>Speaker</Th>
                  <Th>Speech type</Th>
                  <Th align="right">Clips</Th>
                  <Th align="right">Duration</Th>
                  <Th className="w-[240px]">Tamil / English</Th>
                  <Th align="right">CMI</Th>
                  <Th align="right">I-index</Th>
                </tr>
              </thead>
              <tbody>
                {filtered?.map(s => {
                  const c = split.get(s.id) ?? { ta: 0, en: 0 };
                  return (
                    <tr key={s.id} onClick={() => select(s.id)}
                      className={cn('cursor-pointer transition-colors hover:bg-app-surface-muted/70', selectedId === s.id && 'bg-app-accent-soft/60')}>
                      <Td>
                        <div className="font-medium">{s.displayName}</div>
                        <div className="mono text-[11.5px] text-app-text-subtle">{s.id}</div>
                      </Td>
                      <Td><Badge tone={s.environment === 'free speech' ? 'accent' : 'neutral'}>{s.environment === 'free speech' ? 'Free speech' : 'Scripted'}</Badge></Td>
                      <Td align="right" className="tnum">{s.utteranceCount}</Td>
                      <Td align="right" className="tnum">{(s.totalDurationSec / 60).toFixed(1)} min</Td>
                      <Td><LangBar ta={c.ta} en={c.en} showLabels /></Td>
                      <Td align="right" className="tnum">{s.cmi.toFixed(1)}</Td>
                      <Td align="right" className="tnum">{s.iIndex.toFixed(2)}</Td>
                    </tr>
                  );
                })}
              </tbody>
            </Table>
          )}
          {!isLoading && !filtered?.length && <EmptyState title="No speakers match">Try a different search, or enrol someone.</EmptyState>}
        </Card>
      </PageBody>

      {selected && <SpeakerDrawer speaker={selected} onClose={() => select(null)} />}
    </>
  );
}

function SpeakerDrawer({ speaker, onClose }: { speaker: Speaker; onClose: () => void }) {
  const queryClient = useQueryClient();
  const { data: utts } = useQuery({ queryKey: ['utterances', speaker.id], queryFn: () => apiClient.getSpeakerUtterances(speaker.id) });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const remove = useMutation({
    mutationFn: () => apiClient.deleteSpeaker(speaker.id),
    onSuccess: () => { queryClient.invalidateQueries(); onClose(); },
  });

  const ta = utts?.reduce((a, u) => a + u.tokens.filter(t => t.language === 'TA').length, 0) ?? 0;
  const en = utts?.reduce((a, u) => a + u.tokens.filter(t => t.language === 'EN').length, 0) ?? 0;

  return (
    <>
      <div className="fixed inset-0 bg-black/20 z-30 animate-in" onClick={onClose} />
      <aside className="fixed top-0 right-0 bottom-0 w-[560px] max-w-[92vw] bg-app-bg border-l border-app-border z-40 flex flex-col shadow-[-24px_0_48px_rgba(0,0,0,0.08)] animate-in">
        <div className="px-6 py-5 border-b border-app-border bg-app-surface flex items-start justify-between gap-4">
          <div>
            <h2 className="font-serif text-[22px] font-semibold leading-tight">{speaker.displayName}</h2>
            <div className="flex items-center gap-2 mt-1.5">
              <span className="mono text-[12px] text-app-text-subtle">{speaker.id}</span>
              <Badge tone={speaker.consentGiven ? 'accept' : 'warning'}>{speaker.consentGiven ? 'Consent on record' : 'No consent'}</Badge>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-md hover:bg-app-surface-muted text-app-text-muted"><X className="w-4 h-4" /></button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
          <div className="grid grid-cols-2 gap-3">
            <Link to={`/authenticate`}><Button variant="primary" className="w-full" icon={<ShieldCheck className="w-4 h-4" />}>Log in as this speaker</Button></Link>
            <Link to={`/graph-explorer?speaker=${speaker.id}`}><Button className="w-full" icon={<Network className="w-4 h-4" />}>View code-switch graph</Button></Link>
          </div>

          <Card>
            <CardBody className="pt-4 grid grid-cols-3 gap-4">
              <MiniStat label="Speech" value={minutes(speaker.totalDurationSec)} />
              <MiniStat label="CMI" value={speaker.cmi.toFixed(1)} />
              <MiniStat label="I-index" value={speaker.iIndex.toFixed(2)} />
              <div className="col-span-3"><LangBar ta={ta} en={en} showLabels /></div>
            </CardBody>
          </Card>

          <FactEditor speakerId={speaker.id} />

          <TopUp speakerId={speaker.id} />

          <Card>
            <CardHeader title="Recordings" subtitle={`${utts?.length ?? 0} enrolment clips, transcribed and tagged.`} />
            <div className="px-3 pb-3 flex flex-col gap-1">
              {utts?.map((u, i) => <UtteranceRow key={u.id} index={i + 1} u={u} />)}
              {!utts && <div className="px-2 py-3"><Spinner label="Loading recordings" /></div>}
            </div>
          </Card>

          <div className="pt-2">
            <Button variant="danger" size="sm" icon={<Trash2 className="w-3.5 h-3.5" />} loading={remove.isPending}
              onClick={() => { if (confirm(`Delete ${speaker.displayName} and all their audio, facts and graphs? This cannot be undone.`)) remove.mutate(); }}>
              Delete speaker and all data
            </Button>
          </div>
        </div>
      </aside>
    </>
  );
}

/**
 * Add recordings from *this* microphone to an already-enrolled speaker and
 * rebuild their models.
 *
 * The corpus was recorded on phones; a demo login comes through a laptop
 * microphone. A voiceprint is partly a fingerprint of the channel, so a
 * genuine speaker on a new device can score below threshold. Two or three
 * clips from the demo machine fold that channel into the template -- no new
 * data collection, just a minute at the laptop before presenting.
 */
function TopUp({ speakerId }: { speakerId: string }) {
  const queryClient = useQueryClient();
  const [added, setAdded] = useState<import('../api/types').Utterance[]>([]);
  const [recKey, setRecKey] = useState(0);

  const upload = useMutation({
    mutationFn: ({ blob, name }: { blob: Blob; name?: string }) => apiClient.uploadUtterance(speakerId, 'free-speech', blob, name),
    onSuccess: u => { setAdded(a => [...a, u]); setRecKey(k => k + 1); },
  });
  const rebuild = useMutation({
    mutationFn: () => apiClient.completeEnrolment(speakerId),
    onSuccess: () => queryClient.invalidateQueries(),
  });

  return (
    <Card>
      <CardHeader
        title="Add recordings from this microphone"
        subtitle="Enrolment audio came from phones. Record 2–3 short answers here (20 s each, speak naturally), then rebuild — the voiceprint then knows this microphone too."
      />
      <CardBody className="flex flex-col gap-3">
        <AudioRecorder key={recKey} allowUpload busy={upload.isPending} acceptLabel="Add clip" onAccept={(blob, _d, name) => upload.mutate({ blob, name })} />
        {upload.isPending && <Spinner label="Transcribing and tagging…" />}
        {upload.error && <Notice tone="reject" title="Upload failed">{(upload.error as Error).message}</Notice>}
        {added.length > 0 && (
          <div className="flex flex-col gap-1.5">
            {added.map(u => (
              <div key={u.id} className="flex items-center gap-2 text-[12.5px] text-app-text-muted">
                <Badge tone={u.annotated ? 'accept' : 'warning'}>{u.annotated ? 'added' : 'stored, not tagged'}</Badge>
                <span className="truncate">{u.transcript || '—'}</span>
                <span className="tnum text-app-text-subtle ml-auto">{u.durationSec.toFixed(0)}s</span>
              </div>
            ))}
          </div>
        )}
        <div className="flex items-center justify-between gap-3">
          <span className="text-[12px] text-app-text-subtle">{added.length} new clip{added.length === 1 ? '' : 's'} this session</span>
          <Button size="sm" variant={added.length ? 'primary' : 'secondary'} icon={<RefreshCw className="w-3.5 h-3.5" />}
            loading={rebuild.isPending} disabled={!added.length} onClick={() => rebuild.mutate()}>
            Rebuild voiceprint and graph
          </Button>
        </div>
        {rebuild.isSuccess && (
          rebuild.data.warnings.length
            ? rebuild.data.warnings.map((w, i) => <Notice key={i} tone="warning">{w}</Notice>)
            : <Notice tone="accept" title="Rebuilt">Voiceprint and code-switch graph now include the new clips.</Notice>
        )}
        {rebuild.error && <Notice tone="reject" title="Rebuild failed">{(rebuild.error as Error).message}</Notice>}
      </CardBody>
    </Card>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[12px] text-app-text-muted">{label}</div>
      <div className="text-[18px] font-semibold tnum">{value}</div>
    </div>
  );
}

function UtteranceRow({ u, index }: { u: import('../api/types').Utterance; index: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={cn('rounded-md', open && 'bg-app-surface-muted/60')}>
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center gap-3 px-3 py-2 text-left rounded-md hover:bg-app-surface-muted/60">
        <span className="tnum text-[12px] text-app-text-subtle w-5">{index}</span>
        <span className="flex-1 min-w-0 text-[13px] truncate text-app-text-muted">{u.transcript || '—'}</span>
        <span className="tnum text-[12px] text-app-text-subtle">{u.durationSec.toFixed(0)}s</span>
        <ChevronDown className={cn('w-4 h-4 text-app-text-subtle transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="px-3 pb-3 flex flex-col gap-2">
          <audio src={assetUrl(u.audioUrl)} controls preload="none" className="w-full" />
          <TokenText tokens={u.tokens} className="text-[13.5px]" />
        </div>
      )}
    </div>
  );
}

export function FactEditor({ speakerId }: { speakerId: string }) {
  const queryClient = useQueryClient();
  const { data } = useQuery({ queryKey: ['skg', speakerId], queryFn: () => apiClient.getSpeakerSKG(speakerId) });
  const [rows, setRows] = useState<Triple[] | null>(null);
  const facts = rows ?? data ?? [];
  const dirty = rows !== null;

  const save = useMutation({
    mutationFn: (t: Triple[]) => apiClient.updateSpeakerSKG(speakerId, t),
    onSuccess: saved => {
      queryClient.setQueryData(['skg', speakerId], saved);
      setRows(null);
    },
  });

  const update = (i: number, patch: Partial<Triple>) => setRows(facts.map((f, j) => (j === i ? { ...f, ...patch } : f)));
  const used = new Set(facts.map(f => f.predicate));
  const next = FACT_SUGGESTIONS.find(s => !used.has(s.predicate));

  return (
    <Card>
      <CardHeader
        title="Knowledge facts"
        subtitle="Challenge questions are generated from these. The answer itself is never sent to the browser at login."
        actions={dirty && <Button size="sm" variant="primary" icon={<Save className="w-3.5 h-3.5" />} loading={save.isPending} onClick={() => save.mutate(facts)}>Save</Button>}
      />
      <CardBody className="flex flex-col gap-2">
        {facts.length === 0 && (
          <Notice tone="info">No facts yet. Add two or three so this speaker can be challenged at login.</Notice>
        )}
        <datalist id="fact-predicates">
          {FACT_SUGGESTIONS.map(s => <option key={s.predicate} value={s.predicate}>{s.question}</option>)}
        </datalist>
        {facts.map((f, i) => (
          <div key={i} className="grid grid-cols-[170px_1fr_auto] gap-2 items-center">
            <Input list="fact-predicates" value={f.predicate} onChange={e => update(i, { predicate: e.target.value })} className="mono text-[12.5px]" />
            <Input value={f.object} placeholder="answer" onChange={e => update(i, { object: e.target.value })} />
            <button onClick={() => setRows(facts.filter((_, j) => j !== i))} className="p-2 rounded-md text-app-text-subtle hover:text-app-reject hover:bg-app-reject-soft"><X className="w-4 h-4" /></button>
          </div>
        ))}
        <div className="flex items-center justify-between pt-1">
          <Button size="sm" variant="ghost" icon={<Plus className="w-3.5 h-3.5" />}
            onClick={() => setRows([...facts, { subject: '', predicate: next?.predicate ?? '', object: '' }])}>
            Add fact{next ? ` · ${next.question}` : ''}
          </Button>
          {save.error && <span className="text-[12px] text-app-reject">{(save.error as Error).message}</span>}
        </div>
      </CardBody>
    </Card>
  );
}
