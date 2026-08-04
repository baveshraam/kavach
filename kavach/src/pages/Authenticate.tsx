import { useState, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { cn } from '../components/layout/AppLayout';
import { AudioRecorder } from '../components/ui/AudioRecorder';
import { Challenge, AuthResult } from '../api/types';

export function Authenticate() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  const [selectedSpeakerId, setSelectedSpeakerId] = useState('');
  const [challenge, setChallenge] = useState<Challenge | null>(null);
  const [authResult, setAuthResult] = useState<AuthResult | null>(null);
  const [countdown, setCountdown] = useState(0);

  const issueMutation = useMutation({
    mutationFn: (spkId: string) => apiClient.issueChallenge(spkId),
    onSuccess: (data) => {
      setChallenge(data);
      setAuthResult(null);
      setCountdown(Math.floor((new Date(data.expiresAt).getTime() - Date.now()) / 1000));
    }
  });

  const authMutation = useMutation({
    mutationFn: (blob: Blob) => apiClient.authenticate(challenge!.id, blob),
    onSuccess: (data) => setAuthResult(data)
  });

  useEffect(() => {
    let timer: number;
    if (countdown > 0 && !authResult) {
      timer = window.setInterval(() => setCountdown(c => c - 1), 1000);
    }
    return () => clearInterval(timer);
  }, [countdown, authResult]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Authenticate" />
      <div className="p-6 max-w-[1400px] w-full mx-auto flex-1 overflow-auto">
        <div className="grid grid-cols-2 gap-8 h-full">
          
          {/* Left Column - Flow */}
          <div className="flex flex-col gap-6">
            <div className="flex items-end gap-4">
              <label className="flex flex-col gap-1.5 flex-1">
                <span className="text-[11px] uppercase tracking-wider text-app-text-muted">Target Speaker</span>
                <select 
                  value={selectedSpeakerId} 
                  onChange={e => setSelectedSpeakerId(e.target.value)} 
                  className="h-8 border border-app-border px-2 focus:border-app-accent focus:outline-none w-full"
                >
                  <option value="" disabled>Select enrolled speaker...</option>
                  {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName} ({s.id})</option>)}
                </select>
              </label>
              <button 
                onClick={() => issueMutation.mutate(selectedSpeakerId)}
                disabled={!selectedSpeakerId || issueMutation.isPending}
                className="h-8 px-4 bg-app-accent text-white font-medium text-[12px] disabled:opacity-50"
              >
                Issue Challenge
              </button>
            </div>

            {challenge && (
              <div className="border border-app-border bg-app-surface p-6 flex flex-col gap-4">
                <div className="text-[18px] text-app-text font-medium leading-relaxed">
                  {challenge.questionText}
                </div>
                <div className="mono text-[11px] text-app-text-muted">
                  challenge_id: {challenge.id} · target_class: {challenge.targetClass} · 
                  issued: {new Date(challenge.issuedAt).toLocaleTimeString()} · 
                  expires in {Math.floor(countdown / 60)}:{(countdown % 60).toString().padStart(2, '0')}
                </div>
              </div>
            )}

            {challenge && !authResult && (
              <div className="flex flex-col gap-2">
                <span className="text-[11px] uppercase tracking-wider text-app-text-muted">Record Response</span>
                <AudioRecorder onAccept={(blob) => authMutation.mutate(blob)} />
              </div>
            )}

            {authResult && (
              <div className="flex flex-col gap-2 mt-4">
                <span className="text-[11px] uppercase tracking-wider text-app-text-muted">Transcribed Response</span>
                <div className="border border-app-border bg-app-bg p-4 text-[14px] leading-relaxed">
                  {authResult.tokens.map((t, i) => (
                    <span 
                      key={i} 
                      title={`${t.semanticClass} (${(t.lidConfidence*100).toFixed(0)}%)`}
                      className={cn(
                        "mr-1 cursor-help",
                        t.language === 'EN' && "text-app-accent",
                        t.language === 'TA' && "text-app-text",
                        t.language === 'NAMED_ENTITY' && "border-b border-dotted border-app-text",
                        t.language === 'NEUTRAL' && "text-app-text-muted"
                      )}
                    >
                      {t.text}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Scores */}
          <div className="flex flex-col min-h-0 border-l border-app-border pl-8">
            {authResult ? (
              <div className="flex flex-col h-full gap-6">
                <h3 className="text-[13px] uppercase tracking-wider text-app-text-muted font-medium">Live Score Breakdown</h3>
                
                <div className="flex flex-col gap-4">
                  {authResult.branches.map(b => (
                    <div key={b.name} className="flex flex-col gap-1.5">
                      <div className="flex justify-between items-end">
                        <span className="text-[13px]">{b.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                        <div className="flex items-center gap-3">
                           <span className="mono text-[10px] text-app-text-muted">w={b.weight.toFixed(2)}</span>
                           <span className="mono text-[14px]">{b.score.toFixed(3)}</span>
                        </div>
                      </div>
                      <div className="relative w-full h-[4px] bg-app-bg">
                        <div className={cn("h-full", b.passed ? "bg-app-text" : "bg-app-reject")} style={{ width: `${b.score * 100}%` }} />
                        <div className="absolute top-0 bottom-0 w-[1px] bg-app-border-strong h-full z-10" style={{ left: `${b.threshold * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="h-[1px] bg-app-border my-2" />

                <div className="flex justify-between items-end">
                  <span className="text-[13px] uppercase tracking-wider text-app-text-muted">Fused Score</span>
                  <span className="mono text-[24px]">{authResult.fusedScore.toFixed(3)}</span>
                </div>

                <div className="flex items-center gap-2 mt-2">
                  <div className={cn("w-3 h-3", 
                    authResult.decision === 'ACCEPT' ? 'bg-app-accept' : 
                    authResult.decision === 'REJECT' ? 'bg-app-reject' : 'bg-app-warning'
                  )} />
                  <span className={cn("text-[14px] font-bold tracking-widest",
                    authResult.decision === 'ACCEPT' ? 'text-app-accept' : 
                    authResult.decision === 'REJECT' ? 'text-app-reject' : 'text-app-warning'
                  )}>{authResult.decision}</span>
                </div>

                <div className="mt-6 flex flex-col gap-2">
                   <span className="text-[11px] uppercase tracking-wider text-app-text-muted">Explanation</span>
                   <div className="mono text-[11px] text-app-text bg-app-bg p-4 border border-app-border whitespace-pre-wrap leading-relaxed">
                     {authResult.explanation.join('\n')}
                   </div>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-full text-app-text-muted text-[13px]">
                Waiting for authentication trial...
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

