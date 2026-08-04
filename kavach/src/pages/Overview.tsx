import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine } from 'recharts';
import { cn } from '../components/layout/AppLayout';

function StatTile({ label, value, delta, deltaClass = 'opacity-40' }: { label: string, value: string | number, delta: string, deltaClass?: string }) {
  return (
    <div className="p-3 border-r border-app-border last:border-r-0">
      <p className="text-[10px] uppercase tracking-wider opacity-60 mb-1 truncate">{label}</p>
      <p className="text-xl font-mono tabular-nums">{value}</p>
      <p className={cn("text-[10px] font-mono truncate", deltaClass)}>{delta}</p>
    </div>
  );
}

export function Overview() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: utterances } = useQuery({ queryKey: ['utterances'], queryFn: apiClient.getUtterances });
  const { data: authHistory } = useQuery({ queryKey: ['authHistory'], queryFn: apiClient.getAuthHistory });
  const { data: evalMetrics } = useQuery({ queryKey: ['evalMetrics'], queryFn: apiClient.getEvaluation });

  const enrolledCount = speakers?.length || 0;
  const utteranceCount = utterances?.length || 0;
  const corpusDurationSec = utterances?.reduce((acc, u) => acc + u.durationSec, 0) || 0;
  const formatDuration = (sec: number) => {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    return `${h}h ${m}m`;
  };

  const fusedHist = evalMetrics?.scoreDistributions.find(d => d.branch === 'Fused');
  
  // Transform histogram data for Recharts
  const histData = [];
  if (fusedHist) {
    // Basic binning for demo purposes, 0 to 1 with 0.1 step
    for (let i = 0; i <= 10; i++) {
      const binMin = i / 10;
      const binMax = (i + 1) / 10;
      histData.push({
        bin: binMin.toFixed(1),
        genuine: fusedHist.genuine.filter(v => v >= binMin && v < binMax).length,
        impostor: fusedHist.impostor.filter(v => v >= binMin && v < binMax).length
      });
    }
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader 
        title="System Overview" 
        actions={
          <>
            <button className="text-[11px] font-mono uppercase border border-app-border px-3 py-1 hover:bg-app-bg transition-colors">Export CSV</button>
            <button className="text-[11px] font-mono uppercase bg-app-accent text-white px-3 py-1 hover:bg-app-accent-hover transition-colors">Refresh Stats</button>
          </>
        }
      />
      <div className="p-6 max-w-[1400px] w-full mx-auto flex-1 overflow-auto flex flex-col gap-6">
        
        {/* Stat Tiles */}
        <div className="grid grid-cols-6 border border-app-border bg-app-surface">
          <StatTile label="Enrolled Speakers" value={enrolledCount} delta="+12 this week" deltaClass="text-app-accept" />
          <StatTile label="Total Utterances" value={utteranceCount} delta="85.2GB total" />
          <StatTile label="Corpus Duration" value={formatDuration(corpusDurationSec)} delta="Avg 13.1s/utt" />
          <StatTile label="Auth Trials (7d)" value={authHistory?.length || 0} delta="380 success" />
          <StatTile label="System EER" value="3.12%" delta="+0.04% vs v0.0.9" deltaClass="text-app-reject" />
          <StatTile label="Attack Rejection" value="98.4%" delta="Strong (A1-A3)" deltaClass="text-app-accept" />
        </div>

        <div className="flex gap-6 h-[500px]">
          {/* Recent Auth Attempts */}
          <div className="w-[60%] flex flex-col border border-app-border bg-app-surface">
            <div className="px-4 py-2 border-b border-app-border bg-app-bg">
              <h3 className="text-[13px] font-semibold">Recent Authentication Attempts</h3>
            </div>
            <div className="overflow-auto">
              <table className="w-full text-left border-collapse">
                <thead className="bg-app-surface-muted border-b border-app-border">
                  <tr>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60">Time</th>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60">Speaker</th>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60 text-right">S_spk</th>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60 text-right">S_csbg</th>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60 text-right">Fused</th>
                    <th className="p-2 text-[10px] uppercase font-mono tracking-widest opacity-60 text-center">Decision</th>
                  </tr>
                </thead>
                <tbody className="text-[12px]">
                  {authHistory?.map(auth => (
                    <tr key={auth.id} className="border-b border-app-border hover:bg-app-bg cursor-pointer transition-colors">
                      <td className="p-2 font-mono truncate">{new Date(auth.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</td>
                      <td className="p-2 truncate">{auth.speakerId}</td>
                      <td className="p-2 font-mono text-right">{auth.branches.find(b => b.name === 'speaker_embedding')?.score.toFixed(3)}</td>
                      <td className="p-2 font-mono text-right">{auth.branches.find(b => b.name === 'csbg')?.score.toFixed(3)}</td>
                      <td className="p-2 font-mono text-right">{auth.fusedScore.toFixed(3)}</td>
                      <td className="p-2 text-center">
                        <span className={cn("inline-flex items-center gap-1.5",
                          auth.decision === 'ACCEPT' ? 'text-app-accept' : 
                          auth.decision === 'REJECT' ? 'text-app-reject' : 'text-app-warning'
                        )}>
                          <span className={cn("w-1.5 h-1.5", 
                            auth.decision === 'ACCEPT' ? 'bg-app-accept' : 
                            auth.decision === 'REJECT' ? 'bg-app-reject' : 'bg-app-warning'
                          )}></span>
                          {auth.decision}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Score Distributions */}
          <div className="w-[40%] flex flex-col border border-app-border bg-app-surface">
            <div className="px-4 py-2 border-b border-app-border bg-app-bg">
              <h3 className="text-[13px] font-semibold">Score Distributions</h3>
            </div>
            <div className="flex-1 p-6 flex flex-col min-h-0">
              <ResponsiveContainer width="100%" height="100%" className="border-b border-l border-app-border">
                <BarChart data={histData} margin={{ top: 0, right: 0, left: -30, bottom: -15 }}>
                  <XAxis dataKey="bin" tick={false} axisLine={false} />
                  <YAxis tick={false} axisLine={false} />
                  <ReferenceLine x="0.6" stroke="var(--app-accent)" strokeDasharray="3 3" opacity={0.4} />
                  <Bar dataKey="impostor" fill="var(--app-reject)" fillOpacity={0.2} barSize={16} />
                  <Bar dataKey="genuine" fill="var(--app-accept)" fillOpacity={0.2} barSize={16} />
                </BarChart>
              </ResponsiveContainer>
              <div className="flex justify-between text-[10px] font-mono uppercase opacity-50 mt-2">
                <span>0.00 (Impostor)</span>
                <span>0.50</span>
                <span>1.00 (Genuine)</span>
              </div>
              <div className="mt-4 flex gap-4">
                 <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-app-reject opacity-40"></span>
                    <span className="text-[10px] uppercase tracking-wider">Non-target</span>
                 </div>
                 <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-app-accept opacity-40"></span>
                    <span className="text-[10px] uppercase tracking-wider">Target</span>
                 </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

