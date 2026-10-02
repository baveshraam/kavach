import { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient, assetUrl } from '../api/client';
import { Utterance } from '../api/types';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Card, Button, Input, Select, Field, Badge, LangBar, LangLegend, TokenText, Table, Th, Td, EmptyState, Spinner, Stat, minutes, pct, cn } from '../components/ui/kit';
import { Download, Search, Trash2, ChevronDown } from 'lucide-react';

export function Corpus() {
  const { data: utterances, isLoading } = useQuery({ queryKey: ['utterances'], queryFn: apiClient.getUtterances });
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const [speaker, setSpeaker] = useState('');
  const [type, setType] = useState('');
  const [q, setQ] = useState('');

  const name = (id: string) => speakers?.find(s => s.id === id)?.displayName ?? id;

  const filtered = useMemo(() => (utterances ?? []).filter(u =>
    (!speaker || u.speakerId === speaker) &&
    (!type || u.type === type) &&
    (!q || u.transcript.toLowerCase().includes(q.toLowerCase()))), [utterances, speaker, type, q]);

  const stats = useMemo(() => {
    let ta = 0, en = 0, other = 0;
    for (const u of utterances ?? []) for (const t of u.tokens) {
      if (t.language === 'TA') ta++; else if (t.language === 'EN') en++; else other++;
    }
    return { ta, en, other, duration: (utterances ?? []).reduce((a, u) => a + u.durationSec, 0) };
  }, [utterances]);

  const exportJson = () => {
    const blob = new Blob([JSON.stringify(filtered.map(({ audioUrl, ...u }) => u), null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `kavach_corpus_${filtered.length}_utterances.json`;
    link.click();
  };

  return (
    <>
      <PageHeader
        eyebrow="Data"
        title="Corpus"
        description="Every enrolled recording with its transcript and word-level tags. Tags came from the annotation pass (Whisper large-v3 + Gemini), not from the live demo models."
        actions={<Button icon={<Download className="w-4 h-4" />} onClick={exportJson} disabled={!filtered.length}>Export JSON</Button>}
      />
      <PageBody>
        <Card className="grid grid-cols-2 md:grid-cols-[1fr_1fr_1fr_2fr] divide-x divide-app-border">
          <div className="px-5 py-4"><Stat label="Recordings" value={utterances?.length ?? '—'} hint={`${new Set(utterances?.map(u => u.speakerId)).size} speakers`} /></div>
          <div className="px-5 py-4"><Stat label="Speech" value={minutes(stats.duration)} /></div>
          <div className="px-5 py-4"><Stat label="Words tagged" value={(stats.ta + stats.en + stats.other).toLocaleString()} hint={`${stats.other.toLocaleString()} neutral / names`} /></div>
          <div className="px-5 py-4 flex flex-col justify-center gap-2">
            <span className="text-[12px] font-medium text-app-text-muted">Tamil vs English</span>
            <LangBar ta={stats.ta} en={stats.en} showLabels />
          </div>
        </Card>

        <Card className="px-5 py-4 flex flex-wrap items-end gap-4">
          <Field label="Search transcripts" className="flex-1 min-w-[240px]">
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-app-text-subtle" />
              <Input value={q} onChange={e => setQ(e.target.value)} placeholder="e.g. amma, bus, biryani" className="pl-9" />
            </div>
          </Field>
          <Field label="Speaker" className="w-56">
            <Select value={speaker} onChange={e => setSpeaker(e.target.value)}>
              <option value="">All speakers</option>
              {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
            </Select>
          </Field>
          <Field label="Speech type" className="w-44">
            <Select value={type} onChange={e => setType(e.target.value)}>
              <option value="">All types</option>
              <option value="free-speech">Free speech</option>
              <option value="code-mixed">Scripted code-mixed</option>
              <option value="auth-response">Login answers</option>
            </Select>
          </Field>
          <div className="text-[12.5px] text-app-text-muted h-9 flex items-center tnum">{filtered.length} of {utterances?.length ?? 0}</div>
          <LangLegend className="h-9" />
        </Card>

        <Card className="overflow-hidden">
          {isLoading ? <div className="p-6"><Spinner label="Loading corpus" /></div> : filtered.length ? (
            <Table>
              <thead>
                <tr>
                  <Th>Speaker</Th>
                  <Th>Transcript</Th>
                  <Th align="right">Length</Th>
                  <Th className="w-[140px]">TA / EN</Th>
                  <Th align="center">Status</Th>
                  <Th className="w-10" />
                </tr>
              </thead>
              <tbody>{filtered.map(u => <Row key={u.id} u={u} speaker={name(u.speakerId)} />)}</tbody>
            </Table>
          ) : <EmptyState title="No recordings match">Clear a filter to see more.</EmptyState>}
        </Card>
      </PageBody>
    </>
  );
}

function Row({ u, speaker }: { u: Utterance; speaker: string }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const ta = u.tokens.filter(t => t.language === 'TA').length;
  const en = u.tokens.filter(t => t.language === 'EN').length;
  const del = useMutation({
    mutationFn: () => apiClient.deleteUtterance(u.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['utterances'] }),
  });
  return (
    <>
      <tr className={cn('cursor-pointer hover:bg-app-surface-muted/50', open && 'bg-app-surface-muted/50')} onClick={() => setOpen(o => !o)}>
        <Td className="whitespace-nowrap">
          <div className="font-medium">{speaker}</div>
          <div className="text-[11.5px] text-app-text-subtle">{u.type === 'free-speech' ? 'Free speech' : u.type === 'code-mixed' ? 'Scripted' : u.type}</div>
        </Td>
        <Td className="max-w-[560px]"><div className="truncate text-app-text-muted">{u.transcript || '—'}</div></Td>
        <Td align="right" className="tnum whitespace-nowrap">{u.durationSec.toFixed(1)} s</Td>
        <Td>
          <div className="flex items-center gap-2">
            <LangBar ta={ta} en={en} className="flex-1" />
            <span className="text-[11.5px] tnum text-app-text-subtle w-8 text-right">{ta + en ? pct(ta / (ta + en), 0) : '—'}</span>
          </div>
        </Td>
        <Td align="center"><Badge tone={u.annotated ? 'accept' : 'warning'}>{u.annotated ? 'tagged' : 'pending'}</Badge></Td>
        <Td><ChevronDown className={cn('w-4 h-4 text-app-text-subtle transition-transform', open && 'rotate-180')} /></Td>
      </tr>
      {open && (
        <tr>
          <td colSpan={6} className="px-5 py-4 border-b border-app-border bg-app-surface-muted/30">
            <div className="flex flex-col gap-3 max-w-4xl">
              <audio src={assetUrl(u.audioUrl)} controls preload="none" className="w-full max-w-lg" />
              {u.tokens.length ? <TokenText tokens={u.tokens} className="text-[14px]" /> : <p className="text-[13px] text-app-text-muted">{u.transcript}</p>}
              <div className="flex items-center justify-between">
                <span className="mono text-[11.5px] text-app-text-subtle">{u.id} · {u.sampleRate / 1000} kHz · {u.tokens.length} tokens</span>
                <Button size="sm" variant="danger" icon={<Trash2 className="w-3.5 h-3.5" />} loading={del.isPending}
                  onClick={e => { e.stopPropagation(); if (confirm('Delete this recording? The speaker’s graph will need rebuilding.')) del.mutate(); }}>
                  Delete
                </Button>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
