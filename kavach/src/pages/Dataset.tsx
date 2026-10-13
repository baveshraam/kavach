import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import type { DatasetSpeaker } from '../api/types';
import { PageHeader, PageBody } from '../components/layout/PageHeader';
import { Card, CardHeader, CardBody, Notice } from '../components/ui/kit';

const KIND_LABEL: Record<string, string> = {
  words: 'Ten-word login phrases',
  read: 'One sentence read repeatedly',
  read_prompt: 'Prompts read aloud',
};

const person = (r: { pseudonym: string; id: string }) => r.pseudonym || r.id;
const total = (r: DatasetSpeaker) => r.pilotClips + r.studioClips;

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
  // The dataset API sends pseudonyms only (ruling R18). The candidate, the one person with Studio sittings, is named from the
  // speaker list the Speakers page already shows; testers stay pseudonyms.
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const nameOf = (r: DatasetSpeaker) => speakers?.find(s => s.id === r.id)?.displayName.split(' · ')[1] ?? person(r);

  const rows = [...(data?.speakers ?? [])].sort((a, b) => total(b) - total(a));
  const candidates = rows.filter(r => r.studioClips > 0);
  const testers = rows.filter(r => r.studioClips === 0);
  const max = Math.max(1, ...testers.map(total));
  const pilot = rows.reduce((n, r) => n + r.pilotClips, 0);
  const studio = rows.reduce((n, r) => n + r.studioClips, 0);
  const lo = Math.min(...testers.map(total));
  const each = lo === max ? `${max} clips each` : `${lo}–${max} clips each`;

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

            {candidates.map(r => (
              <Card key={r.id}>
                <CardHeader title="Candidate recordings" subtitle="Pilot corpus plus Recording Studio sittings" />
                <CardBody className="flex flex-col gap-6">
                  <div className="flex items-end justify-between gap-6">
                    <div>
                      <div className="font-serif text-[32px] leading-none">{nameOf(r)}</div>
                      <div className="text-[13px] text-app-text-muted mt-2">{person(r)} · {r.pilotClips} pilot + {r.studioClips} Studio clips</div>
                    </div>
                    <div className="text-right">
                      <div className="font-serif text-[68px] leading-none tnum text-app-accent">{total(r)}</div>
                      <div className="text-[13px] text-app-text-muted mt-2">clips</div>
                    </div>
                  </div>
                  <div className="flex h-3 rounded overflow-hidden bg-app-surface-muted">
                    <div className="bg-app-accent" style={{ width: `${(r.pilotClips / total(r)) * 100}%` }} title={`${r.pilotClips} pilot clips`} />
                    <div className="bg-app-accept" style={{ flex: 1 }} title={`${r.studioClips} Studio clips`} />
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-[14px]">
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
                  </div>
                </CardBody>
              </Card>
            ))}

            {testers.length > 0 && (
              <Card>
                <CardHeader title="Tester recordings" subtitle={`${each} · pilot corpus, code-switched speech`} />
                <CardBody className="flex flex-col gap-3">
                  {testers.map(r => (
                    <div key={r.id} className="grid grid-cols-[150px_minmax(0,1fr)_64px] items-center gap-4 text-[14px]">
                      <span className="font-medium truncate">{person(r)}</span>
                      <div className="flex h-6 rounded overflow-hidden bg-app-surface-muted" style={{ width: `${(total(r) / max) * 100}%`, minWidth: 4 }}>
                        <div className="bg-app-accent" style={{ flex: 1 }} title={`${r.pilotClips} pilot clips`} />
                      </div>
                      <span className="tnum text-right font-semibold">{total(r)}</span>
                    </div>
                  ))}
                </CardBody>
              </Card>
            )}
          </div>
        )}
      </PageBody>
    </>
  );
}
