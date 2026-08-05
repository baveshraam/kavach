import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { PageHeader } from '../components/layout/PageHeader';
import { GraphViz } from '../components/ui/GraphViz';
import { SKGViz } from '../components/ui/SKGViz';
import { Download } from 'lucide-react';

export function GraphExplorer() {
  const { data: speakers } = useQuery({ queryKey: ['speakers'], queryFn: apiClient.getSpeakers });
  // Starts empty and adopts the first real speaker once they load. It used to
  // default to the mock id 'spk_001', which no real backend issues: the CSBG
  // fetch 404'd, GraphViz got null and never initialised cytoscape, and the
  // page rendered an empty canvas -- while the <select> showed a name, because
  // a value matching no option makes the browser display the first one. A
  // blank graph under an apparently-chosen speaker is the demo failure this
  // page would have had.
  const [selectedSpeakerId, setSelectedSpeakerId] = useState('');

  useEffect(() => {
    if (!speakers?.length) return;
    if (!speakers.some(s => s.id === selectedSpeakerId)) {
      setSelectedSpeakerId(speakers[0].id);
    }
  }, [speakers, selectedSpeakerId]);

  const [graphType, setGraphType] = useState('csbg');
  const [layout, setLayout] = useState('concentric');
  const [threshold, setThreshold] = useState(0.05);

  const { data: csbgData } = useQuery({
    queryKey: ['csbg', selectedSpeakerId],
    queryFn: () => apiClient.getSpeakerCSBG(selectedSpeakerId),
    enabled: !!selectedSpeakerId && graphType === 'csbg'
  });

  const { data: skgData } = useQuery({
    queryKey: ['skg', selectedSpeakerId],
    queryFn: () => apiClient.getSpeakerSKG(selectedSpeakerId),
    enabled: !!selectedSpeakerId && graphType === 'skg'
  });

  const selectedSpeaker = speakers?.find(s => s.id === selectedSpeakerId);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <PageHeader 
        title="Graph Explorer" 
        actions={
          <button className="h-8 px-3 border border-app-border text-[11px] uppercase tracking-wider text-app-text flex items-center gap-2 hover:bg-app-bg transition-colors">
            <Download className="w-3.5 h-3.5" /> Export SVG
          </button>
        }
      />
      
      {/* Toolbar */}
      <div className="h-[48px] px-6 border-b border-app-border bg-app-surface flex items-center gap-6 shrink-0 text-[12px]">
        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Speaker:</span>
          <select value={selectedSpeakerId} onChange={e => setSelectedSpeakerId(e.target.value)} className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg">
             {speakers?.map(s => <option key={s.id} value={s.id}>{s.displayName}</option>)}
          </select>
        </label>
        
        <div className="w-[1px] h-4 bg-app-border" />

        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Graph Type:</span>
          <select value={graphType} onChange={e => setGraphType(e.target.value)} className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg">
             <option value="csbg">Code-Switch Behaviour Graph</option>
             <option value="skg">Speaker Knowledge Graph</option>
          </select>
        </label>

        <div className="w-[1px] h-4 bg-app-border" />

        <label className="flex items-center gap-2">
          <span className="text-app-text-muted">Layout:</span>
          <select value={layout} onChange={e => setLayout(e.target.value)} className="h-7 border border-app-border px-2 focus:border-app-accent focus:outline-none bg-app-bg">
             <option value="concentric">Concentric</option>
             <option value="circle">Circle</option>
             <option value="grid">Grid</option>
             <option value="cose">Force Directed</option>
          </select>
        </label>

        {/* SKG edges are enrolled facts, not estimated probabilities, so there
            is nothing to threshold. A slider that moves and changes nothing is
            worse than an absent one. */}
        {graphType === 'csbg' && (
          <>
            <div className="w-[1px] h-4 bg-app-border" />
            <label className="flex items-center gap-2">
              <span className="text-app-text-muted">Edge Thr (P):</span>
              <input type="range" min="0" max="0.5" step="0.01" value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} className="w-24 accent-app-accent" />
              <span className="mono text-[11px] w-8">{threshold.toFixed(2)}</span>
            </label>
          </>
        )}
      </div>

      <div className="flex-1 relative">
        {graphType === 'csbg' ? (
          <GraphViz data={csbgData || null} layout={layout} threshold={threshold} />
        ) : (
          <SKGViz
            triples={skgData ?? null}
            speakerName={selectedSpeaker?.displayName ?? ''}
            layout={layout}
          />
        )}
      </div>
    </div>
  );
}

