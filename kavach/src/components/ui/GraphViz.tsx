import { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import { CSBG } from '../../api/types';

interface GraphVizProps {
  data: CSBG | null;
  layout: string;
  threshold: number;
  onReady?: (cy: cytoscape.Core | null) => void;
}

/** Re-render when the theme flips: cytoscape paints to canvas and cannot follow CSS variables. */
export function useThemeKey() {
  const [key, setKey] = useState(() => document.documentElement.className);
  useEffect(() => {
    const obs = new MutationObserver(() => setKey(document.documentElement.className));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  }, []);
  return key;
}

export function themeToken(name: string, fallback: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

/** P(EN | class) for every class node, from the lexical-choice edges. */
export function englishShare(data: CSBG) {
  const out = new Map<string, number>();
  for (const e of data.edges) {
    if (e.edgeType !== 'lexical_choice') continue;
    if (e.target === 'lang:EN') out.set(e.source, e.probability);
    else if (e.target === 'lang:TA' && !out.has(e.source)) out.set(e.source, 1 - e.probability);
  }
  return out;
}

export function GraphViz({ data, layout, threshold, onReady }: GraphVizProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const themeKey = useThemeKey();

  useEffect(() => {
    if (!containerRef.current || !data) return;

    const ta = themeToken('--app-ta', '#B5562F');
    const en = themeToken('--app-en', '#2B4C7E');
    const text = themeToken('--app-text', '#1C1B19');
    const muted = themeToken('--app-text-muted', '#66625A');
    const surface = themeToken('--app-surface', '#FFFFFF');
    const border = themeToken('--app-border-strong', '#CDC9BE');

    const share = englishShare(data);
    const sparse = new Set(data.sparseClasses.map(c => `class:${c}`));
    const maxCount = Math.max(1, ...data.nodes.filter(n => n.kind === 'class').map(n => n.tokenCount));

    const elements: cytoscape.ElementDefinition[] = [
      ...data.nodes.map(n => {
        const pEn = share.get(n.id);
        return {
          data: {
            id: n.id,
            label: n.kind === 'language' ? (n.label === 'TA' ? 'Tamil' : 'English') : n.label.replace(/_/g, ' ').toLowerCase(),
            kind: n.kind,
            size: n.kind === 'class' ? 10 + (layout === 'axis' ? 12 : 22) * Math.sqrt(n.tokenCount / maxCount) : 54,
            ring: pEn === undefined ? border : pEn >= 0.5 ? en : ta,
            fill: n.kind === 'language' ? (n.label === 'TA' ? ta : en) : surface,
            sparse: sparse.has(n.id) ? 1 : 0,
            pEn: pEn ?? 0.5,
          },
        };
      }),
      ...data.edges
        .filter(e => e.probability >= threshold)
        .map(e => ({
          data: {
            id: `${e.source}-${e.target}-${e.edgeType}`,
            source: e.source,
            target: e.target,
            weight: e.probability,
            color: e.target === 'lang:TA' ? ta : e.target === 'lang:EN' ? en : muted,
            type: e.edgeType,
          },
        })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      wheelSensitivity: 0.2,
      style: [
        {
          selector: 'node[kind="class"]',
          style: {
            shape: 'ellipse',
            width: 'data(size)',
            height: 'data(size)',
            'background-color': 'data(fill)',
            'border-width': 2.5,
            'border-color': 'data(ring)',
            label: 'data(label)',
            'font-family': 'Inter, sans-serif',
            'font-size': '11px',
            'font-weight': 500,
            'text-valign': layout === 'axis' ? 'center' : 'bottom',
            'text-halign': layout === 'axis' ? 'right' : 'center',
            'text-margin-y': layout === 'axis' ? 0 : 5,
            'text-margin-x': layout === 'axis' ? 6 : 0,
            color: text,
            'text-background-color': surface,
            'text-background-opacity': 0.85,
            'text-background-padding': '2px',
            'text-background-shape': 'roundrectangle',
          } as any,
        },
        { selector: 'node[sparse = 1]', style: { opacity: 0.45, 'border-style': 'dashed' } as any },
        {
          selector: 'node[kind="language"]',
          style: {
            shape: 'ellipse',
            width: 'data(size)',
            height: 'data(size)',
            'background-color': 'data(fill)',
            'border-width': 0,
            label: 'data(label)',
            'font-family': 'Inter, sans-serif',
            'font-weight': 600,
            'font-size': '12px',
            'text-valign': 'center',
            'text-halign': 'center',
            color: '#FFFFFF',
          } as any,
        },
        {
          selector: 'edge',
          style: {
            width: 'mapData(weight, 0, 1, 0.5, 6)',
            'line-color': 'data(color)',
            'curve-style': 'unbundled-bezier',
            'control-point-distances': [20],
            opacity: 'mapData(weight, 0, 1, 0.12, 0.75)',
          } as any,
        },
        { selector: 'edge[type="switch_transition"]', style: { 'line-style': 'dashed', 'line-dash-pattern': [4, 4] } as any },
        { selector: 'node:selected', style: { 'overlay-opacity': 0.08, 'overlay-color': text } as any },
      ],
      layout: layoutOptions(layout, data, share),
    });

    onReady?.(cy);
    return () => {
      onReady?.(null);
      cy.destroy();
    };
  }, [data, layout, threshold, themeKey]);

  return <div ref={containerRef} className="w-full h-full" />;
}

/**
 * "axis" is the default and the one worth screenshotting: Tamil sits at the
 * left, English at the right, and each class is placed at x = P(English |
 * class). The speaker's habit is then the picture itself -- where each kind of
 * word lands between the two languages -- rather than something to be read
 * off edge widths. Rows are ordered by that same share so labels never stack.
 *
 * The other layouts are kept for exploration. Concentric ranks the language
 * nodes into the middle explicitly; its default (node degree) put them there
 * most of the time and silently reordered whenever the threshold moved.
 */
function layoutOptions(layout: string, data: CSBG, share: Map<string, number>): cytoscape.LayoutOptions {
  if (layout === 'axis') {
    const classes = data.nodes.filter(n => n.kind === 'class').sort((a, b) => (share.get(a.id) ?? 0.5) - (share.get(b.id) ?? 0.5));
    const W = 620, rowH = 27, top = 10;
    const H = top * 2 + rowH * Math.max(1, classes.length - 1);
    const pos: Record<string, { x: number; y: number }> = {
      'lang:TA': { x: -60, y: H / 2 },
      'lang:EN': { x: W + 60, y: H / 2 },
    };
    classes.forEach((n, i) => {
      pos[n.id] = { x: 30 + (share.get(n.id) ?? 0.5) * (W - 60), y: top + i * rowH };
    });
    return { name: 'preset', positions: (node: any) => pos[node.id()] ?? { x: W / 2, y: H / 2 }, padding: 24, fit: true } as any;
  }
  if (layout === 'concentric') {
    return {
      name: 'concentric', padding: 40, minNodeSpacing: 55, avoidOverlap: true,
      concentric: (node: any) => (node.data('kind') === 'language' ? 10 : 1), levelWidth: () => 1,
    } as cytoscape.LayoutOptions;
  }
  if (layout === 'circle') return { name: 'circle', padding: 40, avoidOverlap: true, spacingFactor: 1.4 } as cytoscape.LayoutOptions;
  if (layout === 'grid') return { name: 'grid', padding: 40, avoidOverlap: true, spacingFactor: 1.3 } as cytoscape.LayoutOptions;
  return { name: layout, padding: 40, animate: false } as cytoscape.LayoutOptions;
}
