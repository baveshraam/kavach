import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import { CSBG } from '../../api/types';

interface GraphVizProps {
  data: CSBG | null;
  layout: string;
  threshold: number;
}

export function GraphViz({ data, layout, threshold }: GraphVizProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || !data) return;

    const elements: cytoscape.ElementDefinition[] = [
      ...data.nodes.map(n => ({
        data: { id: n.id, label: n.label, kind: n.kind, weight: n.tokenCount }
      })),
      ...data.edges
        .filter(e => e.probability >= threshold)
        .map(e => ({
          data: { 
            id: `${e.source}-${e.target}`, 
            source: e.source, 
            target: e.target, 
            weight: e.probability,
            observations: e.observationCount,
            type: e.edgeType
          }
      }))
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node[kind="class"]',
          style: {
            shape: 'rectangle',
            width: 12,
            height: 12,
            'background-color': '#FFFFFF',
            'border-width': 1,
            'border-color': '#1C1917',
            label: 'data(label)',
            'font-family': 'IBM Plex Mono, monospace',
            'font-size': '10px',
            'text-valign': 'top',
            'text-halign': 'center',
            'text-margin-y': -4,
            color: '#78716C',
          }
        },
        {
          selector: 'node[kind="language"]',
          style: {
            shape: 'rectangle',
            width: 16,
            height: 16,
            'background-color': '#2C4A6E',
            'border-width': 0,
            label: 'data(label)',
            'font-family': 'Inter, sans-serif',
            'font-weight': 'bold',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            color: '#FFFFFF',
          }
        },
        {
          selector: 'edge',
          style: {
            width: 'mapData(weight, 0, 1, 0.5, 4)',
            'line-color': '#D1D0CE',
            'target-arrow-color': '#D1D0CE',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.8,
            'curve-style': 'bezier',
            opacity: 'mapData(observations, 0, 100, 0.2, 1)' as any
          }
        },
        {
          selector: 'edge[type="switch_transition"]',
          style: {
            'line-style': 'dashed',
            'line-dash-pattern': [4, 4]
          }
        }
      ],
      layout: { name: layout, padding: 30 } as any
    });

    cy.on('mouseover', 'edge', (e) => {
      const edge = e.target;
      // In a real app, we'd render a custom tooltip outside cytoscape
      // For now, we can rely on standard UI
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [data, layout, threshold]);

  return <div ref={containerRef} className="w-full h-full bg-app-bg" />;
}
