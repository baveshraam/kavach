import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, BarChart, Bar, Legend, Tooltip } from 'recharts';
import { Download } from 'lucide-react';
import { cn } from '../components/layout/AppLayout';

export function Evaluation() {
  const { data } = useQuery({ queryKey: ['evaluation'], queryFn: apiClient.getEvaluation });

  return (
    <div className="flex flex-col h-full">
      <PageHeader 
        title="Evaluation" 
        actions={
          <>
            <button className="h-8 px-3 border border-app-border text-[11px] uppercase tracking-wider text-app-text flex items-center gap-2 hover:bg-app-bg transition-colors">
              <Download className="w-3.5 h-3.5" /> Export SVG
            </button>
            <button className="h-8 px-3 border border-app-border text-[11px] uppercase tracking-wider text-app-text flex items-center gap-2 hover:bg-app-bg transition-colors">
              <Download className="w-3.5 h-3.5" /> Export CSV
            </button>
          </>
        }
      />
      <div className="p-6 max-w-[1400px] w-full mx-auto flex-1 overflow-auto flex flex-col gap-8">
        
        {/* Top Row: DET Curve & Ablation */}
        <div className="grid grid-cols-2 gap-8 h-[400px]">
          <div className="border border-app-border bg-app-surface flex flex-col">
             <div className="px-4 py-3 border-b border-app-border">
               <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted">DET Curves (Log Scale)</h3>
             </div>
             <div className="flex-1 p-6 flex">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--app-border)" />
                    <XAxis type="number" dataKey="far" name="FAR" domain={['dataMin', 'dataMax']} scale="log" tickFormatter={v => (v*100).toFixed(1)+'%'} tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={{ stroke: 'var(--app-border)' }} />
                    <YAxis type="number" dataKey="frr" name="FRR" domain={['dataMin', 'dataMax']} scale="log" tickFormatter={v => (v*100).toFixed(1)+'%'} tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={{ stroke: 'var(--app-border)' }} />
                    {data?.configurations.map((c, i) => (
                      <Line 
                        key={c.name} 
                        data={c.detCurve} 
                        type="monotone" 
                        dataKey="frr" 
                        stroke={i === 3 ? 'var(--app-accent)' : i === 0 ? 'var(--app-text-muted)' : i === 1 ? 'var(--app-warning)' : 'var(--app-text)'} 
                        strokeWidth={i === 3 ? 2 : 1}
                        dot={<rect width="4" height="4" fill="currentColor" />}
                        activeDot={{ r: 4 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <div className="w-48 pl-4 flex flex-col justify-center gap-3 border-l border-app-border">
                   {data?.configurations.map((c, i) => (
                     <div key={c.name} className="flex items-center gap-2">
                       <div className="w-3 h-[2px]" style={{ backgroundColor: i === 3 ? 'var(--app-accent)' : i === 0 ? 'var(--app-text-muted)' : i === 1 ? 'var(--app-warning)' : 'var(--app-text)' }} />
                       <span className={cn("text-[11px]", i === 3 ? "font-medium text-app-text" : "text-app-text-muted")}>{c.name}</span>
                     </div>
                   ))}
                </div>
             </div>
          </div>

          <div className="border border-app-border bg-app-surface flex flex-col">
             <div className="px-4 py-3 border-b border-app-border">
               <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted">System Ablation Results</h3>
             </div>
             <div className="flex-1 overflow-auto">
               <table className="w-full text-left border-collapse">
                 <thead className="bg-app-bg">
                   <tr>
                     <th className="px-4 py-3 text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border">Configuration</th>
                     <th className="px-4 py-3 text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border text-right">EER %</th>
                     <th className="px-4 py-3 text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border text-right">minDCF</th>
                     <th className="px-4 py-3 text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border text-right">FAR@FRR=1%</th>
                     <th className="px-4 py-3 text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border text-right">FRR@FAR=1%</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-app-border">
                   {data?.configurations.map(c => (
                     <tr key={c.name}>
                       <td className="px-4 py-3 text-[13px]">{c.name}</td>
                       <td className={cn("px-4 py-3 mono text-right", c.eer === Math.min(...data.configurations.map(x => x.eer)) && "font-semibold text-app-accent")}>{c.eer.toFixed(2)}</td>
                       <td className={cn("px-4 py-3 mono text-right", c.minDcf === Math.min(...data.configurations.map(x => x.minDcf)) && "font-semibold text-app-accent")}>{c.minDcf.toFixed(3)}</td>
                       <td className={cn("px-4 py-3 mono text-right", c.farAtFrr1 === Math.min(...data.configurations.map(x => x.farAtFrr1)) && "font-semibold text-app-accent")}>{c.farAtFrr1.toFixed(1)}</td>
                       <td className={cn("px-4 py-3 mono text-right", c.frrAtFar1 === Math.min(...data.configurations.map(x => x.frrAtFar1)) && "font-semibold text-app-accent")}>{c.frrAtFar1.toFixed(1)}</td>
                     </tr>
                   ))}
                 </tbody>
               </table>
             </div>
          </div>
        </div>

        {/* Bottom Row: Stability & Fairness */}
        <div className="grid grid-cols-2 gap-8 h-[300px]">
          <div className="border border-app-border bg-app-surface flex flex-col">
             <div className="px-4 py-3 border-b border-app-border">
               <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted">CSBG Stability Curve</h3>
             </div>
             <div className="flex-1 p-6 pb-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data?.stabilityCurve} margin={{ top: 5, right: 30, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--app-border)" />
                    <XAxis dataKey="durationSec" tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={{ stroke: 'var(--app-border)' }} tickFormatter={v => v + 's'} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={false} tickFormatter={v => v + '%'} />
                    {/* Fake confidence interval by drawing two very light lines, Recharts doesn't have Area natively without AreaChart but we can mix */}
                    <Line type="monotone" dataKey="ciHigh" stroke="var(--app-border-strong)" strokeWidth={1} dot={false} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="ciLow" stroke="var(--app-border-strong)" strokeWidth={1} dot={false} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="eer" stroke="var(--app-accent)" strokeWidth={2} dot={{ r: 3, fill: 'var(--app-accent)' }} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="text-center text-[10px] uppercase text-app-text-muted mt-2">Enrolment Duration (seconds)</div>
             </div>
          </div>

          <div className="border border-app-border bg-app-surface flex flex-col">
             <div className="px-4 py-3 border-b border-app-border">
               <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted">Fairness Breakdown (EER %)</h3>
             </div>
             <div className="flex-1 p-6 pb-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={
                    // Group data for BarChart
                    ['Monolingual TA', 'Monolingual EN', 'Code-Mixed'].map(cond => {
                      const m = data?.fairness.find(f => f.condition === cond && f.group === 'Male');
                      const f = data?.fairness.find(f => f.condition === cond && f.group === 'Female');
                      return { name: cond, Male: m?.eer, Female: f?.eer, nM: m?.sampleCount, nF: f?.sampleCount };
                    })
                  } margin={{ top: 20, right: 0, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--app-border)" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={{ stroke: 'var(--app-border)' }} />
                    <YAxis tick={{ fontSize: 11, fill: 'var(--app-text-muted)' }} tickLine={false} axisLine={false} />
                    <Bar dataKey="Male" fill="var(--app-accent)" radius={0}>
                      {/* Would use customized label to show n=... but skipping for strict adherence to no-clutter */}
                    </Bar>
                    <Bar dataKey="Female" fill="var(--app-text-muted)" radius={0} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-2">
                 <div className="flex items-center gap-1.5 text-[11px] text-app-text-muted">
                    <div className="w-2 h-2 bg-app-accent"></div> Male
                 </div>
                 <div className="flex items-center gap-1.5 text-[11px] text-app-text-muted">
                    <div className="w-2 h-2 bg-app-text-muted"></div> Female
                 </div>
              </div>
             </div>
          </div>
        </div>

      </div>
    </div>
  );
}

