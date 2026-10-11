import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Card, CardHeader, CardBody, Notice } from '../components/ui/kit';

const KIND_LABEL: Record<string, string> = {
  words: 'Ten-word login phrases',
  read: 'One sentence read repeatedly',
  read_prompt: 'Prompts read aloud',
};

const person = (name: string) => name.replace(/^S\d+\s*·\s*/, '');

function Tile({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-app-border bg-app-surface px-5 py-4">
      <div className="font-serif text-[40px] leading-none tnum">{value.toLocaleString()}</div>
      <div className="text-[13px] text-app-text-muted mt-2">{label}</div>
    </div>
  );
}

export function Dataset() {
  const { data, error } = useQuery({ queryKey: ['dataset'], queryFn: apiClient.dataset });
  const rows = [...(data?.speakers ?? [])].sort((a, b) => b.pilotClips + b.studioClips - (a.pilotClips + a.studioClips));
  const max = Math.max(1, ...rows.map(r => r.pilotClips + r.studioClips));
  const pilot = rows.reduce((n, r) => n + r.pilotClips, 0);
  const studio = rows.reduce((n, r) => n + r.studioClips, 0);
  const recorded = rows.filter(r => r.studioClips > 0);

  return (
    <>
      <PageHeader eyebrow="Data" title="Dataset" description="How many recordings each person contributed." />
      <PageBody>
        {error && <Notice tone="reject" title="Could not load the dataset">{(error as Error).message}</Notice>}
        {data && (
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <Tile label="People" value={rows.length} />
              <Tile label="Clips in total" value={data.totalClips} />
              <Tile label="Pilot corpus clips" value={pilot} />
              <Tile label="Recording Studio clips" value={studio} />
            </div>

            <Card>
              <CardHeader title="Clips per person" subtitle="Pilot corpus (code-switched speech) and Recording Studio sittings" />
              <CardBody className="flex flex-col gap-3">
                {rows.map(r => {
                  const total = r.pilotClips + r.studioClips;
                  return (
                    <div key={r.id} className="grid grid-cols-[150px_minmax(0,1fr)_64px] items-center gap-4 text-[14px]">
                      <span className="font-medium truncate" title={r.name}>{person(r.name)}</span>
                      <div className="flex h-6 rounded overflow-hidden bg-app-surface-muted" style={{ width: `${(total / max) * 100}%`, minWidth: total ? 4 : 0 }}>
                        <div className="bg-app-accent" style={{ width: `${total ? (r.pilotClips / total) * 100 : 0}%` }} title={`${r.pilotClips} pilot clips`} />
                        <div className="bg-app-accept" style={{ flex: 1 }} title={`${r.studioClips} Studio clips`} />
                      </div>
                      <span className="tnum text-right font-semibold">{total}</span>
                    </div>
                  );
                })}
                <div className="flex gap-5 pt-2 text-[12.5px] text-app-text-muted">
                  <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm bg-app-accent" /> Pilot corpus</span>
                  <span className="flex items-center gap-2"><span className="w-3 h-3 rounded-sm bg-app-accept" /> Recording Studio</span>
                </div>
              </CardBody>
            </Card>

            {recorded.map(r => (
              <Card key={r.id}>
                <CardHeader title={`${person(r.name)}: Recording Studio`} subtitle={`${r.studioClips} clips across ${Object.keys(r.studioBySession).length} sittings`} />
                <CardBody className="grid grid-cols-1 md:grid-cols-2 gap-8 text-[14px]">
                  <div className="flex flex-col gap-2">
                    <div className="text-app-text-subtle text-[12.5px]">By sitting</div>
                    {Object.entries(r.studioBySession).map(([s, n]) => (
                      <div key={s} className="flex justify-between"><span>{s}</span><span className="tnum font-semibold">{n} clips</span></div>
                    ))}
                  </div>
                  <div className="flex flex-col gap-2">
                    <div className="text-app-text-subtle text-[12.5px]">By kind of recording</div>
                    {Object.entries(r.studioByKind).map(([k, n]) => (
                      <div key={k} className="flex justify-between"><span>{KIND_LABEL[k] ?? k}</span><span className="tnum font-semibold">{n} clips</span></div>
                    ))}
                  </div>
                </CardBody>
              </Card>
            ))}
          </div>
        )}
      </PageBody>
    </>
  );
}
