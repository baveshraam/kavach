import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import type cytoscape from 'cytoscape';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { CSBG } from '../api/types';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { GraphViz, englishShare } from '../components/ui/GraphViz';
import { SKGViz } from '../components/ui/SKGViz';
import { Card, CardHeader, CardBody, Button, Select, Field, Segmented, Badge, LangLegend, EmptyState, Spinner, titleCase, cn } from '../components/ui/kit';
import { Download } from 'lucide-react';

type Mode = 'single' | 'compare' | 'skg';

/** Classes the ontology marks low-signal (ontology.LOW_SIGNAL_CLASSES); shown faded, not hidden. */
const LOW_SIGNAL = new Set(['FUNCTION_WORD', 'NAMED_ENTITY', 'OTHER']);

export function GraphExplorer() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const [params] = useSearchParams();
  // Starts empty and adopts a real speaker once they load. Defaulting to a
  // hard-coded id drew an empty canvas under an apparently-chosen speaker.
  const [a, setA] = useState(params.get('speaker') ?? '');
  const [b, setB] = useState(params.get('with') ?? '');
  // `?mode=compare&speaker=A&with=B` opens straight into a comparison, so a
  // demo can bookmark the two speakers it wants to show.
  const [mode, setMode] = useState<Mode>((['single', 'compare', 'skg'] as const).find(m => m === params.get('mode')) ?? 'single');
  const [layout, setLayout] = useState('axis');
  const [threshold, setThreshold] = useState(0.05);

  useEffect(() => {
    if (!speakers?.length) return;
    if (!speakers.some(s => s.id === a)) setA(speakers[0].id);
    if (!speakers.some(s => s.id === b)) setB(speakers.find(s => s.id !== (a || speakers[0].id))?.id ?? speakers[0].id);
  }, [speakers, a, b]);

  const { data: csbgA } = useQuery({ queryKey: ['csbg', a], queryFn: () => apiClient.getSpeakerCSBG(a), enabled: !!a });
  const { data: csbgB } = useQuery({ queryKey: ['csbg', b], queryFn: () => apiClient.getSpeakerCSBG(b), enabled: !!b && mode === 'compare' });
  const { data: skg } = useQuery({ queryKey: ['skg', a], queryFn: () => apiClient.getSpeakerSKG(a), enabled: !!a && mode === 'skg' });

  const name = (id: string) => speakers?.find(s => s.id === id)?.displayName ?? id;

  const cyRef = useRef<cytoscape.Core | null>(null);
  const handleReady = useCallback((cy: cytoscape.Core | null) => { cyRef.current = cy; }, []);

  // PNG, not SVG: cytoscape renders to canvas and SVG needs `cytoscape-svg`.
  const exportPng = () => {
    const cy = cyRef.current;
    if (!cy) return;
    const bg = getComputedStyle(document.documentElement).getPropertyValue('--app-surface').trim() || '#FFFFFF';
    const link = document.createElement('a');
    link.href = cy.png({ full: true, scale: 3, bg });
    link.download = `${mode === 'skg' ? 'skg' : 'csbg'}_${name(a)}.png`;
    link.click();
  };

  return (
    <>
      <PageHeader
        eyebrow="Data"
        title="Graph Explorer"
        description="A speaker’s code-switch behaviour graph: for each kind of word, how often they say it in Tamil and how often in English. Compare two speakers to see where their habits part."
        actions={mode !== 'compare' && <Button variant="secondary" icon={<Download className="w-4 h-4" />} onClick={exportPng}>Export PNG</Button>}
      />
      <PageBody>
        {/* Toolbar */}
        <Card className="px-5 py-4 flex flex-wrap items-end gap-5">
          <Field label="View">
            <Segmented<Mode> value={mode} onChange={setMode} options={[
              { value: 'single', label: 'Code-switch graph' },
              { value: 'compare', label: 'Compare two' },
              { value: 'skg', label: 'Knowledge graph' },
            ]} />
          </Field>
          <Field label={mode === 'compare' ? 'Speaker A' : 'Speaker'} className="w-56">
            <Select value={a} onChange={e => setA(e.target.value)}>
              {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
            </Select>
          </Field>
          {mode === 'compare' && (
            <Field label="Speaker B" className="w-56">
              <Select value={b} onChange={e => setB(e.target.value)}>
                {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
              </Select>
            </Field>
          )}
          {mode === 'single' && (
            <>
              <Field label="Layout" className="w-44">
                <Select value={layout} onChange={e => setLayout(e.target.value)}>
                  <option value="axis">Language axis</option>
                  <option value="concentric">Concentric</option>
                  <option value="circle">Circle</option>
                  <option value="cose">Force-directed</option>
                </Select>
              </Field>
              {/* SKG edges are facts, not probabilities: nothing to threshold there. */}
              <Field label={`Hide edges below P = ${threshold.toFixed(2)}`} className="w-52">
                <input type="range" min="0" max="0.5" step="0.01" value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} className="w-full accent-[var(--app-accent)] h-9" />
              </Field>
            </>
          )}
          <div className="flex-1" />
          {mode !== 'skg' && <LangLegend />}
        </Card>

        {mode === 'single' && (
          <div className="grid grid-cols-1 2xl:grid-cols-[1.5fr_1fr] gap-6">
            <Card className="overflow-hidden">
              <CardHeader
                title={name(a)}
                subtitle={layout === 'axis' ? 'Each circle is a kind of word, placed by how often this speaker says it in English. Size = how often it occurs; dashed = too few observations to trust.' : 'Edge width = probability of that language for that class.'}
                actions={csbgA && <div className="flex gap-2"><Badge>CMI {csbgA.cmi.toFixed(1)}</Badge><Badge>I-index {csbgA.iIndex.toFixed(2)}</Badge></div>}
              />
              <div className="h-[640px] border-t border-app-border bg-app-surface-muted/30">
                {csbgA ? <GraphViz data={csbgA} layout={layout} threshold={threshold} onReady={handleReady} /> : <div className="p-6"><Spinner label="Loading graph" /></div>}
              </div>
            </Card>
            <Card>
              <CardHeader title="Language choice by class" subtitle="Sorted by how much evidence we have. Faded rows are classes the ontology treats as low-signal." />
              <CardBody>{csbgA ? <ClassChoices data={csbgA} /> : <Spinner />}</CardBody>
            </Card>
          </div>
        )}

        {mode === 'compare' && (
          <Card>
            <CardHeader
              title={<span>Where <span className="text-app-text">{name(a)}</span> and <span className="text-app-text">{name(b)}</span> differ</span>}
              subtitle="Each row is a word class. The dots mark each speaker’s share of English; a long bar is a habit that tells them apart. Sorted by the size of the gap."
            />
            <CardBody>{csbgA && csbgB ? <Dumbbell a={csbgA} b={csbgB} nameA={name(a)} nameB={name(b)} /> : <Spinner label="Loading graphs" />}</CardBody>
          </Card>
        )}

        {mode === 'skg' && (
          <Card className="overflow-hidden">
            <CardHeader title={`${name(a)} — knowledge graph`} subtitle="The personal facts challenges are generated from. Edit them on the Speakers page." />
            <div className="h-[560px] border-t border-app-border bg-app-surface-muted/30">
              <SKGViz triples={skg ?? null} speakerName={name(a)} layout="concentric" onReady={handleReady} />
            </div>
          </Card>
        )}
      </PageBody>
    </>
  );
}

function classRows(data: CSBG) {
  const share = englishShare(data);
  return data.nodes
    .filter(n => n.kind === 'class')
    .map(n => ({ id: n.id, label: n.label, count: n.tokenCount, pEn: share.get(n.id), sparse: data.sparseClasses.includes(n.label as any) }));
}

function ClassChoices({ data }: { data: CSBG }) {
  const rows = classRows(data).sort((x, y) => y.count - x.count);
  return (
    <div className="flex flex-col gap-2.5">
      {rows.map(r => (
        <div key={r.id} className={cn('grid grid-cols-[130px_1fr_44px] items-center gap-3', (LOW_SIGNAL.has(r.label) || r.sparse) && 'opacity-45')}>
          <span className="text-[12.5px] truncate" title={r.label}>{titleCase(r.label)}</span>
          {r.pEn === undefined ? (
            <span className="text-[12px] text-app-text-subtle">no observations</span>
          ) : (
            <div className="flex h-2.5 rounded-full overflow-hidden bg-app-surface-muted">
              <div className="bg-app-ta" style={{ width: `${(1 - r.pEn) * 100}%` }} />
              <div className="bg-app-en" style={{ width: `${r.pEn * 100}%` }} />
            </div>
          )}
          <span className="text-right text-[12px] text-app-text-subtle tnum">{r.count}</span>
        </div>
      ))}
      <div className="grid grid-cols-[130px_1fr_44px] gap-3 text-[11.5px] text-app-text-subtle pt-1">
        <span />
        <span className="flex justify-between"><span>all Tamil</span><span>all English</span></span>
        <span className="text-right">words</span>
      </div>
    </div>
  );
}

function Dumbbell({ a, b, nameA, nameB }: { a: CSBG; b: CSBG; nameA: string; nameB: string }) {
  const rows = useMemo(() => {
    const ra = new Map(classRows(a).map(r => [r.label, r]));
    const rb = new Map(classRows(b).map(r => [r.label, r]));
    return [...ra.keys()]
      .map(k => ({ label: k, a: ra.get(k)!, b: rb.get(k) }))
      .filter(r => r.a.pEn !== undefined && r.b?.pEn !== undefined && !LOW_SIGNAL.has(r.label))
      .map(r => ({ ...r, gap: Math.abs((r.a.pEn ?? 0) - (r.b!.pEn ?? 0)) }))
      .sort((x, y) => y.gap - x.gap);
  }, [a, b]);

  if (!rows.length) return <EmptyState title="Nothing to compare">These two speakers share no classes with observations.</EmptyState>;

  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-5 text-[12.5px] mb-4">
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-app-text" /> {nameA}</span>
        <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-full border-2 border-app-text bg-app-surface" /> {nameB}</span>
      </div>
      <div className="grid grid-cols-[150px_1fr_60px] gap-x-4 text-[11.5px] text-app-text-subtle pb-2 border-b border-app-border">
        <span>Class</span>
        <span className="flex justify-between"><span className="text-app-ta font-medium">← more Tamil</span><span className="text-app-en font-medium">more English →</span></span>
        <span className="text-right">gap</span>
      </div>
      {rows.map(r => {
        const x1 = (r.a.pEn ?? 0) * 100, x2 = (r.b!.pEn ?? 0) * 100;
        return (
          <div key={r.label} className={cn('grid grid-cols-[150px_1fr_60px] gap-x-4 items-center py-2 border-b border-app-border/60', (r.a.sparse || r.b?.sparse) && 'opacity-50')}>
            <span className="text-[13px] truncate">{titleCase(r.label)}</span>
            <div className="relative h-5">
              <div className="absolute inset-x-0 top-1/2 h-px bg-app-border" />
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-app-border-strong" />
              <div className={cn('absolute top-1/2 h-[3px] -translate-y-1/2 rounded', r.gap > 0.25 ? 'bg-app-text' : 'bg-app-text-subtle')}
                style={{ left: `${Math.min(x1, x2)}%`, width: `${Math.abs(x1 - x2)}%` }} />
              <div className="absolute top-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-app-text" style={{ left: `${x1}%` }} title={`${nameA}: ${x1.toFixed(0)}% English`} />
              <div className="absolute top-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-app-text bg-app-surface" style={{ left: `${x2}%` }} title={`${nameB}: ${x2.toFixed(0)}% English`} />
            </div>
            <span className="text-right text-[12.5px] tnum font-medium">{(r.gap * 100).toFixed(0)} pt</span>
          </div>
        );
      })}
    </div>
  );
}
