import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { Speaker } from '../api/types';
import { cn } from '../components/layout/AppLayout';
import { X, Trash2 } from 'lucide-react';

export function Speakers() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
  const [search, setSearch] = useState('');

  const filtered = speakers?.filter(s => 
    s.id.toLowerCase().includes(search.toLowerCase()) || 
    s.displayName.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full relative overflow-hidden">
      <PageHeader 
        title="Speakers" 
        actions={
          <input 
            type="text" 
            placeholder="Filter speakers..." 
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="h-8 border border-app-border px-2 text-[12px] focus:outline-none focus:border-app-accent w-64"
          />
        }
      />
      <div className="flex-1 overflow-auto p-6 max-w-[1400px] w-full mx-auto">
        <table className="w-full text-left border-collapse border border-app-border bg-app-surface">
          <thead className="bg-app-bg sticky top-0 border-b border-app-border z-10">
            <tr>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">ID</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Name</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Dominant Lang</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">Utterances</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">Duration</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">CMI</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">I-index</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">CSBG Dens</th>
              <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Enrolled</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-app-border">
            {filtered?.map(s => (
              <tr 
                key={s.id} 
                onClick={() => setSelectedSpeaker(s)}
                className={cn(
                  "hover:bg-app-bg/50 cursor-pointer transition-colors duration-120",
                  selectedSpeaker?.id === s.id && "bg-app-accent/5"
                )}
              >
                <td className="px-4 py-2 mono text-app-text-muted">{s.id}</td>
                <td className="px-4 py-2 font-medium">{s.displayName}</td>
                <td className="px-4 py-2">{s.dominantLanguage}</td>
                <td className="px-4 py-2 mono text-right">{s.utteranceCount}</td>
                <td className="px-4 py-2 mono text-right">{(s.totalDurationSec / 60).toFixed(1)}m</td>
                <td className="px-4 py-2 mono text-right">{s.cmi.toFixed(1)}</td>
                <td className="px-4 py-2 mono text-right">{s.iIndex.toFixed(2)}</td>
                <td className="px-4 py-2 mono text-right">{s.csbgDensity.toFixed(2)}</td>
                <td className="px-4 py-2 mono text-app-text-muted truncate max-w-[100px]">{s.enrolledAt.split('T')[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Drawer */}
      {selectedSpeaker && (
        <div className="absolute top-0 right-0 bottom-0 w-[500px] bg-app-surface border-l border-app-border shadow-[-10px_0_30px_rgba(0,0,0,0.05)] flex flex-col z-20 transition-transform">
          <div className="h-[48px] px-6 border-b border-app-border flex items-center justify-between shrink-0 bg-app-bg">
             <h3 className="text-[14px] font-semibold flex items-center gap-2">
               {selectedSpeaker.displayName} <span className="mono text-[11px] text-app-text-muted font-normal">{selectedSpeaker.id}</span>
             </h3>
             <button onClick={() => setSelectedSpeaker(null)} className="p-1 hover:bg-app-border rounded-sm transition-colors text-app-text-muted"><X className="w-4 h-4" /></button>
          </div>
          
          <div className="flex-1 overflow-auto p-6 flex flex-col gap-8">
             <section>
               <h4 className="text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border pb-1 mb-3">Profile Data</h4>
               <div className="grid grid-cols-2 gap-y-3 gap-x-4">
                 <div><div className="text-[10px] text-app-text-muted">Age Range</div><div>{selectedSpeaker.ageRange}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Gender</div><div>{selectedSpeaker.gender}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Dominant Lang</div><div>{selectedSpeaker.dominantLanguage}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Other Langs</div><div>{selectedSpeaker.otherLanguages.join(', ') || '-'}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Device</div><div>{selectedSpeaker.device}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Environment</div><div>{selectedSpeaker.environment}</div></div>
               </div>
             </section>

             <section>
               <h4 className="text-[11px] uppercase tracking-wider text-app-text-muted border-b border-app-border pb-1 mb-3">Code-Switch Statistics</h4>
               <div className="grid grid-cols-2 gap-y-3 gap-x-4">
                 <div><div className="text-[10px] text-app-text-muted">CMI</div><div className="mono">{selectedSpeaker.cmi.toFixed(1)}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">I-Index</div><div className="mono">{selectedSpeaker.iIndex.toFixed(2)}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">Matrix Ratio</div><div className="mono">{selectedSpeaker.matrixLanguageRatio.toFixed(2)}</div></div>
                 <div><div className="text-[10px] text-app-text-muted">CSBG Density</div><div className="mono">{selectedSpeaker.csbgDensity.toFixed(2)}</div></div>
               </div>
             </section>

             <section className="mt-auto pt-6">
                <button 
                  className="flex items-center gap-2 text-app-reject hover:text-white hover:bg-app-reject px-3 py-1.5 rounded-sm transition-colors duration-120 text-[12px] font-medium"
                  onClick={() => {
                    if (confirm('Delete speaker & all data?')) {
                      apiClient.deleteSpeaker(selectedSpeaker.id).then(() => {
                        setSelectedSpeaker(null);
                        // In a real app we'd invalidate the query here
                        window.location.reload();
                      });
                    }
                  }}
                >
                   <Trash2 className="w-3.5 h-3.5" /> Delete speaker & all data
                </button>
             </section>
          </div>
        </div>
      )}
    </div>
  );
}

