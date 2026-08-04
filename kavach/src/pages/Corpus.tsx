import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { Download, Trash2, CheckCircle, Circle } from 'lucide-react';
import { cn } from '../components/layout/AppLayout';

export function Corpus() {
  const { data: utterances } = useQuery({ queryKey: ['utterances'], queryFn: apiClient.getUtterances });
  
  const [filterSpeaker, setFilterSpeaker] = useState('');
  const [filterType, setFilterType] = useState('');
  const [filterStatus, setFilterStatus] = useState('');

  const filtered = utterances?.filter(u => {
    if (filterSpeaker && u.speakerId !== filterSpeaker) return false;
    if (filterType && u.type !== filterType) return false;
    if (filterStatus === 'annotated' && !u.annotated) return false;
    if (filterStatus === 'pending' && u.annotated) return false;
    return true;
  });

  const totalDuration = utterances?.reduce((acc, u) => acc + u.durationSec, 0) || 0;
  const uniqueSpeakers = new Set(utterances?.map(u => u.speakerId)).size;
  
  // Calculate token split
  let taTokens = 0;
  let enTokens = 0;
  let neutralTokens = 0;
  utterances?.forEach(u => {
    u.tokens.forEach(t => {
      if (t.language === 'TA') taTokens++;
      else if (t.language === 'EN') enTokens++;
      else neutralTokens++;
    });
  });
  const totalTokens = taTokens + enTokens + neutralTokens || 1;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <PageHeader 
        title="Corpus" 
        actions={
          <>
            <button className="h-8 px-3 border border-app-border text-[11px] uppercase tracking-wider text-app-text flex items-center gap-2 hover:bg-app-bg transition-colors">
              <Download className="w-3.5 h-3.5" /> Export JSON
            </button>
            <button className="h-8 px-3 border border-app-border text-[11px] uppercase tracking-wider text-app-text flex items-center gap-2 hover:bg-app-bg transition-colors">
              <Download className="w-3.5 h-3.5" /> Export CSV
            </button>
          </>
        }
      />
      
      {/* Summary Strip */}
      <div className="h-[48px] px-6 border-b border-app-border bg-app-surface flex items-center gap-8 shrink-0 text-[12px]">
        <div className="flex items-center gap-2">
          <span className="text-app-text-muted">Total Duration:</span>
          <span className="mono">{(totalDuration/3600).toFixed(2)}h</span>
        </div>
        <div className="w-[1px] h-4 bg-app-border" />
        <div className="flex items-center gap-2">
          <span className="text-app-text-muted">Speakers:</span>
          <span className="mono">{uniqueSpeakers}</span>
        </div>
        <div className="w-[1px] h-4 bg-app-border" />
        <div className="flex items-center gap-2">
          <span className="text-app-text-muted">Total Tokens:</span>
          <span className="mono">{totalTokens}</span>
        </div>
        <div className="w-[1px] h-4 bg-app-border" />
        <div className="flex items-center gap-3 flex-1 max-w-[300px]">
          <span className="text-app-text-muted">Lang Split:</span>
          <div className="flex-1 h-[6px] flex bg-app-bg overflow-hidden border border-app-border">
            <div className="bg-app-text" style={{width: `${(taTokens/totalTokens)*100}%`}} title={`TA: ${((taTokens/totalTokens)*100).toFixed(1)}%`} />
            <div className="bg-app-accent" style={{width: `${(enTokens/totalTokens)*100}%`}} title={`EN: ${((enTokens/totalTokens)*100).toFixed(1)}%`} />
            <div className="bg-app-text-muted" style={{width: `${(neutralTokens/totalTokens)*100}%`}} title={`Neutral: ${((neutralTokens/totalTokens)*100).toFixed(1)}%`} />
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="h-[48px] px-6 border-b border-app-border bg-app-surface flex items-center gap-6 shrink-0 text-[12px]">
        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Speaker:</span>
          <input 
            type="text" 
            placeholder="All" 
            value={filterSpeaker} 
            onChange={e => setFilterSpeaker(e.target.value)} 
            className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg w-32 mono" 
          />
        </label>
        
        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Type:</span>
          <select value={filterType} onChange={e => setFilterType(e.target.value)} className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg w-32">
             <option value="">All Types</option>
             <option value="monolingual-ta">Monolingual TA</option>
             <option value="monolingual-en">Monolingual EN</option>
             <option value="code-mixed">Code-Mixed</option>
             <option value="free-speech">Free Speech</option>
             <option value="auth-response">Auth Response</option>
          </select>
        </label>

        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Status:</span>
          <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg w-32">
             <option value="">All</option>
             <option value="annotated">Annotated</option>
             <option value="pending">Pending</option>
          </select>
        </label>
        
        <div className="flex-1" />
        
        <div className="text-app-text-muted mono text-[11px]">
          Showing {filtered?.length || 0} / {utterances?.length || 0}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6 max-w-[1600px] w-full mx-auto">
        <table className="w-full text-left border-collapse border border-app-border bg-app-surface">
          <thead className="bg-app-bg sticky top-0 border-b border-app-border z-10">
            <tr>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">File ID</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Speaker</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Type</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">Dur</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-right">Hz</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Transcript</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted">Tokens (LID)</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted text-center w-24">Annotated</th>
              <th className="px-3 py-2 text-[11px] uppercase tracking-wider text-app-text-muted w-16"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-app-border">
            {filtered?.map(u => (
              <tr key={u.id} className="hover:bg-app-bg/50 transition-colors">
                <td className="px-3 py-2 mono text-app-text-muted">{u.id}</td>
                <td className="px-3 py-2 mono text-app-accent hover:underline cursor-pointer">{u.speakerId}</td>
                <td className="px-3 py-2 mono text-app-text-muted">{u.type}</td>
                <td className="px-3 py-2 mono text-right">{u.durationSec.toFixed(1)}s</td>
                <td className="px-3 py-2 mono text-right">{u.sampleRate/1000}k</td>
                <td className="px-3 py-2 max-w-md truncate" title={u.transcript}>{u.transcript}</td>
                <td className="px-3 py-2 mono text-app-text-muted">{u.tokens.length}</td>
                <td className="px-3 py-2 text-center">
                  <div className="flex justify-center">
                    {u.annotated ? <CheckCircle className="w-4 h-4 text-app-accept" /> : <Circle className="w-4 h-4 text-app-border-strong" />}
                  </div>
                </td>
                <td className="px-3 py-2">
                  <button className="p-1 hover:bg-app-border rounded-sm transition-colors text-app-text-muted hover:text-app-reject">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

