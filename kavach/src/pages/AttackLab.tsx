import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { AttackRun, AttackType } from '../api/types';
import { Card, CardHeader, CardBody, Button, Select, Field, Badge, Notice, Table, Th, Td, EmptyState, pct, cn } from '../components/ui/kit';
import { Repeat, Scissors, Bot, BookKey, Wand2, Swords, ChevronDown } from 'lucide-react';

const ATTACKS: { id: AttackType; code: string; name: string; desc: string; icon: any; realism: 'real' | 'partial' }[] = [
  { id: 'A1_REPLAY', code: 'A1', name: 'Replay', icon: Repeat, realism: 'real',
    desc: 'Plays back one of the victim’s real recordings.' },
  { id: 'A2_SPLICE', code: 'A2', name: 'Splice', icon: Scissors, realism: 'real',
    desc: 'Cuts and cross-fades the victim’s real audio into a new answer.' },
  { id: 'A3_CLONE_NAIVE', code: 'A3', name: 'Voice clone', icon: Bot, realism: 'partial',
    desc: 'A cloned voice that does not know the answer.' },
  { id: 'A4_CLONE_KNOWLEDGE', code: 'A4', name: 'Clone + knowledge', icon: BookKey, realism: 'partial',
    desc: 'A cloned voice that knows the answer, in the attacker’s own words.' },
  { id: 'A5_CLONE_ADAPTIVE', code: 'A5', name: 'Style-adaptive clone', icon: Wand2, realism: 'partial',
    desc: 'Also imitates the victim’s code-switching, estimated from overheard speech.' },
];

const CONFIGS: { key: keyof AttackRun['successRateByConfig']; label: string }[] = [
  { key: 'ecapa_only', label: 'Voiceprint only' },
  { key: 'plus_knowledge', label: '+ Knowledge' },
  { key: 'plus_csbg', label: '+ Code-switch graph' },
  { key: 'full_fusion', label: 'Full system' },
];

export function AttackLab() {
  const queryClient = useQueryClient();
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: attacks } = useQuery({ queryKey: ['attacks'], queryFn: apiClient.getAttacks });
  const { data: perSpeaker } = useQuery({ queryKey: ['attacks', 'per-speaker'], queryFn: apiClient.getPerSpeakerIapmr });
  const [target, setTarget] = useState('');
  const [running, setRunning] = useState<AttackType | null>(null);

  const generate = useMutation({
    mutationFn: (type: AttackType) => { setRunning(type); return apiClient.generateAttack(type, target, 100); },
    onSettled: () => { setRunning(null); queryClient.invalidateQueries({ queryKey: ['attacks'] }); },
  });

  const name = (id: string) => speakers?.find(s => s.id === id)?.displayName ?? id;

  return (
    <>
      <PageHeader
        eyebrow="Research"
        title="Attack Lab"
        description="Run the five attacks from the threat model against an enrolled speaker and see which part of the system stops each one. A cell is the attack success rate — lower is better."
        actions={
          <Field label="Target speaker" className="w-64">
            <Select value={target} onChange={e => setTarget(e.target.value)}>
              <option value="" disabled>Choose a victim…</option>
              {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
            </Select>
          </Field>
        }
      />
      <PageBody>
        <Notice tone="warning" title="What is real here">
          The code-switch graph and splice-detection scores are computed on real recordings. Voice-clone attacks (A3–A5) use modelled voiceprint scores, because no Tamil voice cloner is wired in yet, so those rows show how the defence behaves, not a measured result. Attack audio never enters the released corpus.
        </Notice>

        <div className="grid grid-cols-1 md:grid-cols-3 xl:grid-cols-5 gap-4">
          {ATTACKS.map(atk => (
            <Card key={atk.id} className="flex flex-col p-4 gap-3">
              <div className="flex items-center justify-between">
                <div className="w-9 h-9 rounded-md bg-app-surface-muted border border-app-border flex items-center justify-center text-app-text-muted">
                  <atk.icon className="w-4 h-4" strokeWidth={1.75} />
                </div>
                <Badge tone={atk.realism === 'real' ? 'accept' : 'warning'}>{atk.realism === 'real' ? 'real audio' : 'modelled voice'}</Badge>
              </div>
              <div>
                <div className="text-[12px] text-app-text-subtle font-medium">{atk.code}</div>
                <div className="text-[14px] font-semibold">{atk.name}</div>
              </div>
              <p className="text-[12.5px] text-app-text-muted leading-relaxed flex-1">{atk.desc}</p>
              <Button size="sm" variant={running === atk.id ? 'primary' : 'secondary'} disabled={!target || (generate.isPending && running !== atk.id)}
                loading={running === atk.id} onClick={() => generate.mutate(atk.id)}>
                Run 100 trials
              </Button>
            </Card>
          ))}
        </div>
        {generate.error && <Notice tone="reject" title="Attack run failed">{(generate.error as Error).message}</Notice>}

        <Card className="overflow-hidden">
          <CardHeader title="Attack results" subtitle="Share of attack trials that were accepted, per system configuration. Expand a row for the run’s own notes." />
          {attacks?.length ? (
            <Table>
              <thead>
                <tr>
                  <Th>Attack</Th>
                  <Th>Target</Th>
                  {CONFIGS.map(c => <Th key={c.key} align="center">{c.label}</Th>)}
                  <Th className="w-10" />
                </tr>
              </thead>
              <tbody>{attacks.map(a => <AttackRow key={a.id} run={a} target={name(a.targetSpeakerId)} />)}</tbody>
            </Table>
          ) : (
            <EmptyState icon={<Swords className="w-5 h-5" />} title="No attacks run yet">Choose a target speaker above and run an attack.</EmptyState>
          )}
        </Card>

        {/* The mean hides the failure: a system that stops every attack on 24
            speakers and none on the 25th reports 96% while one person is
            completely unprotected. Unmeasured speakers get rows of their own. */}
        <Card className="overflow-hidden">
          <CardHeader
            title="Exposure per speaker"
            subtitle="Averages hide the one person left unprotected, so every speaker is listed — including those nobody has attacked yet."
            actions={<Badge tone="neutral">mean {perSpeaker?.meanIapmr == null ? 'not measured' : pct(perSpeaker.meanIapmr)}</Badge>}
          />
          <Table>
            <thead><tr><Th>Speaker</Th><Th align="right">Trials</Th><Th align="center">Attack success</Th><Th align="center">95% interval (Wilson)</Th><Th>Attacks run</Th></tr></thead>
            <tbody>
              {perSpeaker?.speakers.map(s => (
                <tr key={s.speakerId}>
                  <Td className="font-medium">
                    {s.name || s.speakerId}
                    {s.speakerId === perSpeaker.worstSpeakerId && <Badge tone="reject" className="ml-2">most exposed</Badge>}
                  </Td>
                  <Td align="right" className="tnum">{s.trials}{s.belowMinTrials && <span className="text-app-text-subtle" title="too few trials to compare"> *</span>}</Td>
                  <Td align="center"><RateCell rate={s.iapmr} /></Td>
                  <Td align="center" className="tnum text-app-text-muted">{pct(s.ciLow)}–{pct(s.ciHigh)}</Td>
                  <Td className="text-[12.5px] text-app-text-muted">{s.attackTypes.map(t => t.slice(0, 2)).join(', ') || '—'}</Td>
                </tr>
              ))}
              {perSpeaker?.unmeasuredSpeakerIds.map(id => (
                <tr key={id} className="text-app-text-subtle">
                  <Td>{name(id)}</Td>
                  <Td align="right" className="tnum">0</Td>
                  <Td align="center" colSpan={3} className="text-[12.5px]">not measured — no attack has been run against this speaker</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          {!!perSpeaker?.notes?.length && (
            <div className="px-5 py-3 border-t border-app-border flex flex-col gap-1">
              {perSpeaker.notes.map((n, i) => <p key={i} className="text-[12px] text-app-text-muted">{n}</p>)}
            </div>
          )}
        </Card>
      </PageBody>
    </>
  );
}

function RateCell({ rate }: { rate: number }) {
  return (
    <span className={cn('inline-flex min-w-[64px] justify-center rounded-md px-2 py-1 tnum text-[12.5px] font-medium',
      rate >= 0.5 ? 'bg-app-reject-soft text-app-reject' : rate >= 0.1 ? 'bg-app-warning-soft text-app-warning' : 'bg-app-accept-soft text-app-accept')}>
      {pct(rate)}
    </span>
  );
}

function AttackRow({ run, target }: { run: AttackRun; target: string }) {
  const [open, setOpen] = useState(false);
  const atk = ATTACKS.find(x => x.id === run.attackType);
  return (
    <>
      <tr className="hover:bg-app-surface-muted/50 cursor-pointer" onClick={() => setOpen(o => !o)}>
        <Td>
          <div className="font-medium">{atk?.code} · {atk?.name}</div>
          <div className="text-[11.5px] text-app-text-subtle tnum">{run.trials} trials · {new Date(run.generatedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}{run.simulated ? ' · simulated' : ''}</div>
        </Td>
        <Td>{target}</Td>
        {CONFIGS.map(c => <Td key={c.key} align="center"><RateCell rate={run.successRateByConfig[c.key]} /></Td>)}
        <Td><ChevronDown className={cn('w-4 h-4 text-app-text-subtle transition-transform', open && 'rotate-180')} /></Td>
      </tr>
      {open && (
        <tr>
          <td colSpan={7} className="px-5 py-3 bg-app-surface-muted/40 border-b border-app-border">
            {run.yieldRate != null && <p className="text-[12.5px] mb-1.5">Clone yield (fooled the voiceprint): <span className="font-semibold tnum">{pct(run.yieldRate)}</span></p>}
            <ul className="flex flex-col gap-1">
              {(run.notes?.length ? run.notes : ['No notes recorded for this run.']).map((n, i) => (
                <li key={i} className="text-[12.5px] text-app-text-muted leading-relaxed">• {n}</li>
              ))}
            </ul>
          </td>
        </tr>
      )}
    </>
  );
}
