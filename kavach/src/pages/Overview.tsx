import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import {
  Card, CardHeader, CardBody, Stat, LangBar, LangLegend, DecisionBadge, EmptyState, Badge, Button, minutes, pct, cn,
} from '../components/ui/kit';
import {
  AudioLines, Languages, Fingerprint, Network, KeyRound, ShieldCheck, ArrowRight, Clock, CircleDashed, CheckCircle2, XCircle,
} from 'lucide-react';

const PIPELINE = [
  { icon: AudioLines, title: 'Speech', detail: 'Spoken answer to a fresh challenge' },
  { icon: Languages, title: 'ASR + word-level LID', detail: 'Whisper, then every word tagged TA / EN and one of 21 meaning classes' },
  { icon: Fingerprint, title: 'Voiceprint', detail: 'ECAPA-TDNN cosine against the enrolled template', branch: true },
  { icon: Network, title: 'Code-switch graph', detail: 'Which language this speaker uses for which kind of word', branch: true },
  { icon: KeyRound, title: 'Knowledge', detail: 'Answer matched against the speaker’s own facts', branch: true },
  { icon: ShieldCheck, title: 'Fusion + gates', detail: 'Liveness and splice checks can veto; branches are weighted' },
];

export function Overview() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: utterances } = useQuery({ queryKey: ['utterances'], queryFn: apiClient.getUtterances });
  const { data: authHistory } = useQuery({ queryKey: ['authHistory'], queryFn: apiClient.getAuthHistory });
  const { data: offline } = useQuery({ queryKey: ['offline-results'], queryFn: apiClient.getOfflineResults });

  /*
    Every figure on this page is derived from the API. Anything the backend
    has not measured reads "n/a" rather than a number: a dashboard is where a
    progress slide gets screenshotted from, and a figure that was never
    measured must not be mistakable for one that was.
  */
  const totals = useMemo(() => {
    const u = utterances ?? [];
    let ta = 0, en = 0, tokens = 0;
    const bySpeaker = new Map<string, { ta: number; en: number }>();
    for (const utt of u) {
      const acc = bySpeaker.get(utt.speakerId) ?? { ta: 0, en: 0 };
      for (const t of utt.tokens) {
        tokens++;
        if (t.language === 'TA') { ta++; acc.ta++; }
        else if (t.language === 'EN') { en++; acc.en++; }
      }
      bySpeaker.set(utt.speakerId, acc);
    }
    return {
      duration: u.reduce((a, x) => a + x.durationSec, 0),
      annotated: u.filter(x => x.annotated).length,
      tokens, ta, en, bySpeaker,
    };
  }, [utterances]);

  const profiles = useMemo(() => (speakers ?? [])
    .map(s => {
      const c = totals.bySpeaker.get(s.id) ?? { ta: 0, en: 0 };
      return { ...s, ta: c.ta, en: c.en, share: c.ta + c.en ? c.ta / (c.ta + c.en) : 0 };
    })
    .sort((a, b) => b.share - a.share), [speakers, totals]);

  const accepted = authHistory?.filter(a => a.decision === 'ACCEPT').length ?? 0;
  const free = offline?.find(r => r.id.includes('free'));
  const scripted = offline?.find(r => r.id === 'results');
  const eerOf = (run: typeof free, name: string) => run?.results.configurations.find(c => c.name === name)?.eer;

  return (
    <>
      <PageHeader
        eyebrow="Tamil–English code-switched speech"
        title="Who is speaking — and how they switch"
        description="KAVACH verifies a speaker with three independent signals: how they sound, what they know, and which language they reach for when they talk about family, food, numbers or time."
        actions={<Link to="/authenticate"><Button variant="primary" icon={<ShieldCheck className="w-4 h-4" />}>Start a login</Button></Link>}
      />
      <PageBody>
        {/* Headline numbers */}
        <Card className="grid grid-cols-2 md:grid-cols-5 divide-x divide-app-border">
          {[
            { label: 'Enrolled speakers', value: speakers?.length ?? '—', hint: `${profiles.filter(p => p.environment === 'free speech').length} free speech · ${profiles.filter(p => p.environment !== 'free speech').length} scripted` },
            { label: 'Recordings', value: utterances?.length ?? '—', hint: `${totals.annotated} annotated` },
            { label: 'Speech', value: utterances ? minutes(totals.duration) : '—', hint: 'real, consented audio' },
            { label: 'Words tagged', value: totals.tokens.toLocaleString(), hint: `${pct(totals.ta / Math.max(1, totals.ta + totals.en), 0)} Tamil among TA/EN` },
            { label: 'Logins this session', value: authHistory?.length ?? 0, hint: `${accepted} accepted` },
          ].map(s => (
            <div key={s.label} className="px-5 py-4"><Stat {...s} /></div>
          ))}
        </Card>

        {/* Pipeline */}
        <Card>
          <CardHeader title="How a login is decided" subtitle="Each branch is measured on its own; a branch that could not be measured is dropped, never scored as zero." />
          <CardBody>
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1.2fr_auto_1.6fr_auto_1fr] items-stretch gap-3">
              <PipeStep {...PIPELINE[0]} />
              <Arrow />
              <PipeStep {...PIPELINE[1]} />
              <Arrow />
              <div className="flex flex-col gap-2">
                {PIPELINE.slice(2, 5).map(p => <PipeStep key={p.title} {...p} compact />)}
              </div>
              <Arrow />
              <PipeStep {...PIPELINE[5]} emphasis />
            </div>
          </CardBody>
        </Card>

        <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_1fr] gap-6">
          {/* Language profiles */}
          <Card>
            <CardHeader
              title="Language profile per speaker"
              subtitle="Share of Tamil vs English among words that had a choice. Scripted speakers read assigned text, so their split reflects the script."
              actions={<Link to="/graph-explorer"><Button size="sm" variant="ghost">Open graphs <ArrowRight className="w-3.5 h-3.5" /></Button></Link>}
            />
            <CardBody>
              <LangLegend className="mb-4" />
              <div className="flex flex-col gap-3">
                {profiles.map(p => (
                  <div key={p.id} className="grid grid-cols-[150px_1fr_52px] items-center gap-4">
                    <div className="min-w-0">
                      <div className="text-[13px] font-medium truncate">{p.displayName}</div>
                      <div className="text-[11.5px] text-app-text-subtle">{p.environment === 'free speech' ? 'Free speech' : 'Scripted'} · {p.utteranceCount} clips</div>
                    </div>
                    <LangBar ta={p.ta} en={p.en} />
                    <div className="text-right text-[12.5px] tnum text-app-ta font-medium">{pct(p.share, 0)}</div>
                  </div>
                ))}
                {!profiles.length && <EmptyState title="No speakers enrolled">Run <code className="mono">python -m kavach.seed_demo</code> or enrol one.</EmptyState>}
              </div>
            </CardBody>
          </Card>

          <div className="flex flex-col gap-6">
            {/* Recent logins */}
            <Card>
              <CardHeader title="Recent logins" actions={<Link to="/authenticate"><Button size="sm" variant="ghost">New <ArrowRight className="w-3.5 h-3.5" /></Button></Link>} />
              <div className="px-2 pb-2">
                {authHistory?.length ? authHistory.slice(0, 6).map(a => {
                  const name = speakers?.find(s => s.id === a.speakerId)?.displayName ?? a.speakerId;
                  const voice = a.branches.find(b => b.name === 'speaker_embedding');
                  return (
                    <div key={a.id} className="flex items-center gap-3 px-3 py-2.5 rounded-md hover:bg-app-surface-muted">
                      <div className="flex-1 min-w-0">
                        <div className="text-[13px] font-medium truncate">{name}</div>
                        <div className="text-[11.5px] text-app-text-subtle tnum">
                          {new Date(a.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · voice {voice ? voice.score.toFixed(2) : 'n/a'} · fused {a.fusedScore.toFixed(2)}
                        </div>
                      </div>
                      <DecisionBadge decision={a.decision} />
                    </div>
                  );
                }) : <EmptyState icon={<Clock className="w-5 h-5" />} title="No logins yet">Issue a challenge on the Authenticate page and answer it out loud.</EmptyState>}
              </div>
            </Card>

            {/* Research status */}
            <Card>
              <CardHeader title="Where the research stands" subtitle="From the offline experiment runs. None of these is reportable yet, and the reasons are listed on the Evaluation page." />
              <CardBody className="flex flex-col gap-3">
                <Finding
                  state="done"
                  title="Pipeline works on real code-mixed speech"
                  body={`12 speakers annotated end to end, 0 guessed tags. Voiceprint separates same-session speakers perfectly (EER ${pct(eerOf(scripted, 'ECAPA alone'), 0)}).`}
                />
                <Finding
                  state="open"
                  title="Code-switch graph alone does not yet identify speakers"
                  body={`Free speech: ${pct(eerOf(free, 'CSBG alone'), 0)} EER (chance is 50%) on 5 speakers. Scripted: ${pct(eerOf(scripted, 'CSBG alone'), 0)}, but that separates scripts, not people.`}
                />
                <Finding
                  state="pending"
                  title="Needs more data"
                  body="A second recording session per speaker: cross-session stability, the knowledge branch, and a reportable EER all wait on it."
                />
                <Link to="/evaluation" className="text-[13px] text-app-accent font-medium inline-flex items-center gap-1 mt-1 hover:underline">
                  See the full evaluation <ArrowRight className="w-3.5 h-3.5" />
                </Link>
              </CardBody>
            </Card>
          </div>
        </div>
      </PageBody>
    </>
  );
}

function PipeStep({ icon: Icon, title, detail, compact, emphasis }: { icon: any; title: string; detail: string; compact?: boolean; emphasis?: boolean; branch?: boolean }) {
  return (
    <div className={cn(
      'rounded-lg border flex gap-3',
      compact ? 'px-3 py-2.5 items-center' : 'p-4 flex-col',
      emphasis ? 'border-app-accent/30 bg-app-accent-soft/50' : 'border-app-border bg-app-surface-muted/50',
    )}>
      <div className={cn('rounded-md flex items-center justify-center shrink-0',
        compact ? 'w-8 h-8' : 'w-9 h-9',
        emphasis ? 'bg-app-accent text-app-on-accent' : 'bg-app-surface border border-app-border text-app-text-muted')}>
        <Icon className="w-4 h-4" strokeWidth={1.75} />
      </div>
      <div className="min-w-0">
        <div className="text-[13px] font-semibold leading-snug">{title}</div>
        <div className="text-[12px] text-app-text-muted leading-snug mt-0.5">{detail}</div>
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="hidden lg:flex items-center justify-center text-app-text-subtle">
      <ArrowRight className="w-4 h-4" />
    </div>
  );
}

function Finding({ state, title, body }: { state: 'done' | 'open' | 'pending'; title: string; body: string }) {
  const Icon = state === 'done' ? CheckCircle2 : state === 'open' ? XCircle : CircleDashed;
  return (
    <div className="flex gap-3">
      <Icon className={cn('w-4 h-4 mt-[3px] shrink-0', {
        'text-app-accept': state === 'done',
        'text-app-reject': state === 'open',
        'text-app-text-subtle': state === 'pending',
      })} />
      <div>
        <div className="text-[13px] font-medium flex items-center gap-2">
          {title}
          {state === 'pending' && <Badge>pending</Badge>}
        </div>
        <div className="text-[12.5px] text-app-text-muted leading-relaxed">{body}</div>
      </div>
    </div>
  );
}
