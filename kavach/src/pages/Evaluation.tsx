import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { OfflineRun } from '../api/types';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Card, CardHeader, CardBody, Segmented, Notice, Badge, Table, Th, Td, EmptyState, Spinner, pct, cn } from '../components/ui/kit';
import { FlaskConical, Maximize2, X } from 'lucide-react';

const RUN_LABEL: Record<string, string> = {
  results_freespeech: 'Free speech',
  results: 'Scripted reading',
};

const FIGURE_LABEL: Record<string, string> = {
  det: 'DET curves per configuration',
  csbg_heatmap: 'Language choice per class and speaker',
  stability: 'CSBG error vs enrolment speech',
  ablation: 'Ablations',
};

export function Evaluation() {
  const [tab, setTab] = useState<'offline' | 'live'>('offline');
  return (
    <>
      <PageHeader
        eyebrow="Research"
        title="Evaluation"
        description="Offline experiments are produced by `python -m kavach.experiments` with a dev/test split and bootstrap intervals. The live tab only scores logins clicked through this UI."
        actions={<Segmented<'offline' | 'live'> value={tab} onChange={setTab} options={[{ value: 'offline', label: 'Offline experiments' }, { value: 'live', label: 'This session’s logins' }]} />}
      />
      <PageBody>{tab === 'offline' ? <Offline /> : <Live />}</PageBody>
    </>
  );
}

// ---------------------------------------------------------------- offline

function Offline() {
  const { data: runs, isLoading } = useQuery({ queryKey: ['offline-results'], queryFn: apiClient.getOfflineResults });
  const [runId, setRunId] = useState('');
  useEffect(() => {
    if (runs?.length && !runs.some(r => r.id === runId)) {
      setRunId((runs.find(r => r.id.includes('free')) ?? runs[0]).id);
    }
  }, [runs, runId]);

  if (isLoading) return <Spinner label="Loading experiment runs" />;
  if (!runs?.length) return <Card><EmptyState icon={<FlaskConical className="w-5 h-5" />} title="No experiment runs found">Run <code className="mono">python -m kavach.experiments --out paper/results/</code>.</EmptyState></Card>;

  const run = runs.find(r => r.id === runId) ?? runs[0];
  return (
    <>
      <div className="flex items-center gap-4">
        <Segmented value={run.id} onChange={setRunId} options={runs.map(r => ({
          value: r.id, label: `${RUN_LABEL[r.id] ?? r.id} · ${r.results.corpus.n_speakers} speakers`,
        }))} />
        <span className="text-[12.5px] text-app-text-subtle">
          generated {new Date(run.results.environment.generated_utc).toLocaleDateString()} · commit <span className="mono">{run.results.environment.git_commit.slice(0, 7)}</span>
        </span>
      </div>
      <RunView run={run} />
    </>
  );
}

function RunView({ run }: { run: OfflineRun }) {
  const r = run.results;
  const [zoom, setZoom] = useState<string | null>(null);

  return (
    <>
      {!r.reportable && (
        <Notice tone="warning" title={`Not reportable yet — ${r.blockers.length} reason${r.blockers.length === 1 ? '' : 's'}`}>
          <ul className="mt-1 flex flex-col gap-1 list-disc pl-4">
            {r.blockers.map((b, i) => <li key={i}>{b}</li>)}
          </ul>
        </Notice>
      )}

      <Card className="grid grid-cols-2 md:grid-cols-5 divide-x divide-app-border">
        {[
          ['Corpus', `${r.corpus.provenance === 'RECORDED' ? 'Free speech' : 'Scripted'}`],
          ['Speakers', `${r.corpus.n_speakers} (${r.split.dev_speakers.length} dev / ${r.split.test_speakers.length} test)`],
          ['Utterances', `${r.corpus.n_utterances}`],
          ['Test trials', `${r.split.n_test_trials}`],
          ['Sessions', r.corpus.cross_session ? 'cross-session' : 'one per speaker'],
        ].map(([k, v]) => (
          <div key={k} className="px-5 py-4">
            <div className="text-[12px] text-app-text-muted">{k}</div>
            <div className="text-[15px] font-semibold mt-0.5">{v}</div>
          </div>
        ))}
      </Card>

      <Card>
        <CardHeader
          title="Verification error by configuration"
          subtitle="Equal error rate on the held-out test speakers, with 95% bootstrap interval. The dashed line is chance (50%)."
        />
        <Table>
          <thead>
            <tr>
              <Th>Configuration</Th>
              <Th className="w-[38%]">EER</Th>
              <Th align="right">95% CI</Th>
              <Th align="right">minDCF</Th>
              <Th align="right">AUC</Th>
              <Th align="right">Genuine / impostor</Th>
            </tr>
          </thead>
          <tbody>
            {r.configurations.map(c => (
              <tr key={c.name}>
                <Td className="font-medium">
                  {c.name}
                  {!c.is_reliable && <Badge tone="warning" className="ml-2">too few trials</Badge>}
                </Td>
                <Td>
                  <div className="flex items-center gap-3">
                    <div className="relative flex-1 h-2 rounded-full bg-app-surface-muted">
                      <div className="absolute inset-y-0 left-0 rounded-full bg-app-text/15" style={{ left: `${c.eer_ci[0] * 100}%`, width: `${Math.max(0.5, (c.eer_ci[1] - c.eer_ci[0]) * 100)}%` }} />
                      <div className={cn('absolute top-1/2 w-2.5 h-2.5 -translate-y-1/2 -translate-x-1/2 rounded-full', c.eer >= 0.4 ? 'bg-app-reject' : c.eer >= 0.15 ? 'bg-app-warning' : 'bg-app-accept')} style={{ left: `${Math.max(0.6, c.eer * 100)}%` }} />
                      <div className="absolute -top-1 -bottom-1 left-1/2 border-l border-dashed border-app-text-subtle" />
                    </div>
                    <span className="tnum font-semibold w-14 text-right">{pct(c.eer)}</span>
                  </div>
                </Td>
                <Td align="right" className="tnum text-app-text-muted">{pct(c.eer_ci[0])}–{pct(c.eer_ci[1])}</Td>
                <Td align="right" className="tnum">{c.min_dcf.toFixed(3)}</Td>
                <Td align="right" className="tnum">{c.auc.toFixed(2)}</Td>
                <Td align="right" className="tnum text-app-text-muted">{c.n_genuine} / {c.n_impostor}{c.n_vetoed ? ` · ${c.n_vetoed} vetoed` : ''}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
        <div className="px-5 py-3 text-[12.5px] text-app-text-muted border-t border-app-border">
          Fitted fusion weights ({r.fitted.weights_source}):{' '}
          {Object.entries(r.fitted.weights).map(([k, v]) => `${k.replace('_', ' ')} ${v.toFixed(2)}`).join(' · ')}
          {' '}· threshold {r.fitted.threshold.toFixed(3)} · veto {r.fitted.veto_floor === null ? 'none' : r.fitted.veto_floor.toFixed(2)}
        </div>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <Card>
          <CardHeader title="Ablations" subtitle="Δ is against the baseline of the same scope; rows in different scopes are not comparable." />
          <Table>
            <thead><tr><Th>Removed</Th><Th>Scope</Th><Th align="right">EER</Th><Th align="right">Δ</Th></tr></thead>
            <tbody>
              {r.ablations.map(a => (
                <tr key={a.name} title={a.note}>
                  <Td>{a.name}</Td>
                  <Td><Badge>{a.scope}</Badge></Td>
                  <Td align="right" className="tnum">{pct(a.eer)}</Td>
                  <Td align="right" className={cn('tnum font-medium', a.delta > 0.001 ? 'text-app-reject' : a.delta < -0.001 ? 'text-app-accept' : 'text-app-text-subtle')}>
                    {a.delta > 0 ? '+' : ''}{(a.delta * 100).toFixed(1)}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader title="CSBG vs enrolment budget" subtitle="How much speech a defender needs — equally, how much an attacker must overhear to copy the habit." />
            <Table>
              <thead><tr><Th>Enrolment clips</Th><Th align="right">≈ speech</Th><Th align="right">EER</Th><Th align="right">95% CI</Th></tr></thead>
              <tbody>
                {r.stability.map(s => (
                  <tr key={s.n_utterances}>
                    <Td className="tnum">{s.n_utterances}</Td>
                    <Td align="right" className="tnum">{Math.round(s.approx_seconds)} s</Td>
                    <Td align="right" className="tnum font-medium">{pct(s.eer)}</Td>
                    <Td align="right" className="tnum text-app-text-muted">{pct(s.ci_low)}–{pct(s.ci_high)}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          </Card>
          <Card>
            <CardHeader title="Branch coverage" subtitle="A branch that scored nothing and one that found no signal look the same in a table. This is the difference." />
            <CardBody className="flex flex-col gap-3">
              {r.corpus.branch_coverage.map(b => (
                <div key={b.name} className="flex flex-col gap-1">
                  <div className="flex justify-between text-[13px]"><span>{b.name}</span><span className="tnum text-app-text-muted">{b.measured} / {b.measured + b.unavailable} trials</span></div>
                  <div className="h-2 rounded-full bg-app-surface-muted overflow-hidden">
                    <div className={cn('h-full rounded-full', b.coverage > 0.9 ? 'bg-app-accept' : 'bg-app-warning')} style={{ width: `${b.coverage * 100}%` }} />
                  </div>
                </div>
              ))}
            </CardBody>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader title="Figures" subtitle="Generated by the experiment runner (greyscale-safe, publication format). Click to enlarge." />
        <CardBody className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {run.figures.map(f => (
            <button key={f} onClick={() => setZoom(f)} className="group text-left rounded-lg border border-app-border bg-white overflow-hidden hover:border-app-border-strong transition-colors">
              <img src={apiClient.offlineFigureUrl(run.id, f)} alt={FIGURE_LABEL[f] ?? f} className="w-full h-64 object-contain p-3" loading="lazy" />
              <div className="flex items-center justify-between px-4 py-2.5 border-t border-app-border bg-app-surface text-[13px]">
                <span className="font-medium text-app-text">{FIGURE_LABEL[f] ?? f}</span>
                <Maximize2 className="w-3.5 h-3.5 text-app-text-subtle group-hover:text-app-text" />
              </div>
            </button>
          ))}
        </CardBody>
      </Card>

      {r.caveats.length > 0 && (
        <Card>
          <CardHeader title="Caveats the runner attached" />
          <CardBody>
            <ul className="flex flex-col gap-2">
              {r.caveats.map((c, i) => <li key={i} className="text-[13px] text-app-text-muted leading-relaxed">• {c}</li>)}
            </ul>
          </CardBody>
        </Card>
      )}

      {zoom && (
        <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-10 animate-in" onClick={() => setZoom(null)}>
          <div className="relative bg-white rounded-lg max-w-5xl w-full p-6" onClick={e => e.stopPropagation()}>
            <button onClick={() => setZoom(null)} className="absolute top-3 right-3 p-1.5 rounded-md text-neutral-500 hover:bg-neutral-100"><X className="w-4 h-4" /></button>
            <img src={apiClient.offlineFigureUrl(run.id, zoom)} alt={zoom} className="w-full max-h-[78vh] object-contain" />
          </div>
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------- live

function Live() {
  const { data, isLoading } = useQuery({ queryKey: ['evaluation'], queryFn: apiClient.getEvaluation });
  if (isLoading) return <Spinner />;
  if (!data?.configurations.length) {
    return (
      <Card>
        <EmptyState icon={<FlaskConical className="w-5 h-5" />} title="Not enough logins to score yet">
          This tab computes error rates from logins made through the Authenticate page, so it needs both genuine and impostor attempts. It has no dev/test split and is a demo, not a result.
        </EmptyState>
      </Card>
    );
  }
  return (
    <>
      <Notice tone="info">Computed from this database’s login history: no trial design, no held-out speakers. Use the offline tab for anything you would quote.</Notice>
      <Card>
        <CardHeader title="Error rates from login history" />
        <Table>
          <thead><tr><Th>Configuration</Th><Th align="right">EER</Th><Th align="right">minDCF</Th><Th align="right">FAR @ 1% FRR</Th><Th align="right">FRR @ 1% FAR</Th></tr></thead>
          <tbody>
            {data.configurations.map(c => (
              <tr key={c.name}>
                <Td className="font-medium">{c.name}</Td>
                <Td align="right" className="tnum">{pct(c.eer)}</Td>
                <Td align="right" className="tnum">{Number.isFinite(c.minDcf) ? c.minDcf.toFixed(3) : 'n/a'}</Td>
                <Td align="right" className="tnum">{pct(c.farAtFrr1)}</Td>
                <Td align="right" className="tnum">{pct(c.frrAtFar1)}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </>
  );
}
