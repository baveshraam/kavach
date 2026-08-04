import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { AttackType } from '../api/types';

const ATTACKS: { id: AttackType, code: string, name: string, desc: string, input: string }[] = [
  { id: 'A1_REPLAY', code: 'A1', name: 'Replay', desc: 'Replay a previously recorded utterance', input: 'Auth record' },
  { id: 'A2_SPLICE', code: 'A2', name: 'Splice', desc: 'Concatenate recorded word segments', input: 'Corpus audio' },
  { id: 'A3_CLONE_NAIVE', code: 'A3', name: 'Clone (naive)', desc: 'TTS voice clone, blind to graph', input: 'Text (imposter)' },
  { id: 'A4_CLONE_KNOWLEDGE', code: 'A4', name: 'Clone + Knowledge', desc: 'TTS clone speaking correct answer', input: 'Text (target)' },
  { id: 'A5_CLONE_ADAPTIVE', code: 'A5', name: 'Style-Adaptive Clone', desc: 'Clone imitating CS pattern', input: 'Text (target + CSBG)' }
];

export function AttackLab() {
  const queryClient = useQueryClient();
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const { data: attacks } = useQuery({ queryKey: ['attacks'], queryFn: apiClient.getAttacks });
  
  const [selectedSpeakerId, setSelectedSpeakerId] = useState('');

  const generateMutation = useMutation({
    mutationFn: ({ type }: { type: AttackType }) => apiClient.generateAttack(type, selectedSpeakerId, 100),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['attacks'] })
  });

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Attack Lab" />
      <div className="p-6 max-w-[1400px] w-full mx-auto flex-1 overflow-auto flex flex-col gap-8">
        
        <div className="border border-app-border p-3 bg-app-bg text-[12px] text-app-text-muted border-l-4 border-l-app-warning">
          Ethics Notice: Generated attack audio is used strictly for evaluating system defences. It is stored securely, in isolation, and is never included in the released corpus.
        </div>

        <section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted">Generate Attack</h3>
            <label className="flex items-center gap-2 text-[12px]">
              <span className="text-app-text-muted">Target Speaker:</span>
              <select value={selectedSpeakerId} onChange={e => setSelectedSpeakerId(e.target.value)} className="h-8 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-surface w-48">
                 <option value="" disabled>Select speaker...</option>
                 {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-5 gap-4">
            {ATTACKS.map(atk => (
              <div key={atk.id} className="border border-app-border bg-app-surface p-4 flex flex-col gap-3">
                <div className="flex items-center gap-2">
                  <span className="mono text-[11px] px-1.5 py-0.5 border border-app-border-strong text-app-text-muted">{atk.code}</span>
                  <span className="font-semibold text-[13px]">{atk.name}</span>
                </div>
                <div className="text-[11px] text-app-text-muted leading-relaxed flex-1">
                  {atk.desc}
                </div>
                <div className="text-[10px] uppercase tracking-wider text-app-text-muted mb-2">Input: {atk.input}</div>
                <button
                  onClick={() => generateMutation.mutate({ type: atk.id })}
                  disabled={!selectedSpeakerId || generateMutation.isPending}
                  className="h-8 w-full border border-app-accent text-app-accent font-medium text-[12px] hover:bg-app-accent hover:text-white transition-colors duration-120 disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-app-accent"
                >
                  Generate (100 trials)
                </button>
              </div>
            ))}
          </div>
        </section>

        <section className="flex-1 min-h-0 flex flex-col">
           <h3 className="text-[13px] font-semibold uppercase tracking-wider text-app-text-muted mb-4">Attack Results</h3>
           <div className="flex-1 overflow-auto border border-app-border bg-app-surface">
              <table className="w-full text-left border-collapse">
                <thead className="bg-app-bg sticky top-0 border-b border-app-border z-10">
                  <tr>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted w-32">Run ID</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted w-32">Type</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted w-32">Target</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted border-l border-app-border text-center">ECAPA only</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted border-l border-app-border text-center">+ Knowledge</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted border-l border-app-border text-center">+ CSBG</th>
                    <th className="px-4 py-2 text-[11px] uppercase tracking-wider text-app-text-muted border-l border-app-border text-center bg-app-bg/50">Full Fusion</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-app-border">
                  {attacks?.map(a => {
                    const renderCell = (rate: number, bgClass: string = '') => {
                      // single-hue tint: transparent to light red
                      const alpha = rate * 0.4; // max 0.4 opacity red
                      return (
                        <td className={`px-4 py-2 mono text-center border-l border-app-border ${bgClass}`} style={{ backgroundColor: `rgba(155, 50, 50, ${alpha})` }}>
                          {(rate * 100).toFixed(1)}%
                        </td>
                      );
                    };

                    return (
                      <tr key={a.id} className="hover:bg-app-bg/50 transition-colors">
                        <td className="px-4 py-2 mono text-app-text-muted">{a.id}</td>
                        <td className="px-4 py-2">{ATTACKS.find(x => x.id === a.attackType)?.code}</td>
                        <td className="px-4 py-2 mono">{a.targetSpeakerId}</td>
                        {renderCell(a.successRateByConfig.ecapa_only)}
                        {renderCell(a.successRateByConfig.plus_knowledge)}
                        {renderCell(a.successRateByConfig.plus_csbg)}
                        {renderCell(a.successRateByConfig.full_fusion, 'font-medium')}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
           </div>
        </section>

      </div>
    </div>
  );
}

