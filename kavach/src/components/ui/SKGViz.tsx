import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import { Triple } from '../../api/types';

interface SKGVizProps {
  triples: Triple[] | null;
  speakerName: string;
  layout: string;
}

/**
 * The Speaker Knowledge Graph: one speaker at the centre, one edge per
 * enrolled fact, the value at the leaf.
 *
 * This renders exactly what `GET /api/speakers/{id}/skg` returns and nothing
 * derived. The RDF graph behind it carries more -- an `rdfs:label` and a
 * `kavach:semanticClass` on each entity -- but the wire format is flat
 * triples, and inferring the class here would mean duplicating the
 * `FACT_TYPES` table from `skg.py` in TypeScript, where it would drift the
 * first time a fact type is added on the backend. A wrong class on a node is
 * worse than no class: this page is what an operator checks a speaker's
 * enrolment against.
 */
export function SKGViz({ triples, speakerName, layout }: SKGVizProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || !triples || triples.length === 0) return;

    const styles = getComputedStyle(document.documentElement);
    const token = (name: string, fallback: string) =>
      styles.getPropertyValue(name).trim() || fallback;

    // Read the theme rather than hardcoding hexes: this page has a dark mode
    // and cytoscape paints to a canvas, so it cannot inherit CSS variables the
    // way the rest of the UI does.
    const surface = token('--app-surface', '#FFFFFF');
    const text = token('--app-text', '#1A1A19');
    const muted = token('--app-text-muted', '#78716C');
    const border = token('--app-border-strong', '#D1D0CE');
    const accent = token('--app-accent', '#2C4A6E');

    const elements: cytoscape.ElementDefinition[] = [
      { data: { id: '__speaker__', label: speakerName || 'Speaker', kind: 'speaker' } },
    ];

    triples.forEach((t, i) => {
      // Index in the id, not the value: two facts can share a value ("amma"
      // as both familyRole and comfortFood is not impossible), and colliding
      // ids would silently drop the second edge.
      const nodeId = `fact_${i}`;
      elements.push({
        data: { id: nodeId, label: t.object, kind: 'fact', predicate: t.predicate },
      });
      elements.push({
        data: {
          id: `edge_${i}`,
          source: '__speaker__',
          target: nodeId,
          label: humanisePredicate(t.predicate),
        },
      });
    });

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node[kind="speaker"]',
          style: {
            shape: 'rectangle',
            width: 'label',
            height: 22,
            padding: '8px',
            'background-color': accent,
            'border-width': 0,
            label: 'data(label)',
            'font-family': 'Inter, sans-serif',
            'font-weight': 'bold',
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'center',
            color: surface,
          },
        },
        {
          selector: 'node[kind="fact"]',
          style: {
            shape: 'rectangle',
            width: 'label',
            height: 20,
            padding: '6px',
            'background-color': surface,
            'border-width': 1,
            'border-color': border,
            label: 'data(label)',
            'font-family': 'IBM Plex Mono, monospace',
            'font-size': '10px',
            'text-valign': 'center',
            'text-halign': 'center',
            'text-max-width': '140px',
            'text-wrap': 'ellipsis',
            color: text,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': border,
            'target-arrow-color': border,
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.7,
            'curve-style': 'bezier',
            label: 'data(label)',
            'font-family': 'IBM Plex Mono, monospace',
            'font-size': '9px',
            color: muted,
            'text-rotation': 'autorotate',
            'text-background-color': surface,
            'text-background-opacity': 1,
            'text-background-padding': '2px',
          },
        },
      ],
      layout: layoutOptions(layout),
    });

    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [triples, speakerName, layout]);

  if (!triples) {
    return (
      <div className="flex items-center justify-center h-full text-app-text-muted text-[13px]">
        Loading knowledge graph…
      </div>
    );
  }

  // An empty canvas under a chosen speaker is the failure this page already
  // had once with the CSBG (see the comment in GraphExplorer). A speaker with
  // no enrolment interview is a normal state and has to read as one.
  if (triples.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-2 px-8 text-center">
        <span className="text-[13px] text-app-text">No facts enrolled for this speaker.</span>
        <span className="text-[12px] text-app-text-muted max-w-[420px]">
          The knowledge graph is filled in during the enrolment interview. Until
          it has facts, this speaker cannot be issued a challenge and the
          knowledge branch has nothing to verify against.
        </span>
      </div>
    );
  }

  return <div ref={containerRef} className="w-full h-full bg-app-bg" />;
}

/** `favouriteFood` -> `favourite food`. The predicates are the wire format, so
 * they stay camelCase in the data and are only softened for display. */
function humanisePredicate(predicate: string): string {
  return predicate.replace(/([a-z0-9])([A-Z])/g, '$1 $2').toLowerCase();
}

function layoutOptions(layout: string): cytoscape.LayoutOptions {
  // Concentric is the toolbar default and it is wrong here: with one hub and
  // N leaves it ranks every leaf identically and stacks them in one ring at
  // the same radius as each other, overlapping the labels. A star is what this
  // graph is, so lay it out as one.
  if (layout === 'concentric') {
    return { name: 'concentric', padding: 40, minNodeSpacing: 40,
             concentric: (node: any) => (node.data('kind') === 'speaker' ? 10 : 1),
             levelWidth: () => 1 } as cytoscape.LayoutOptions;
  }
  return { name: layout, padding: 40 } as cytoscape.LayoutOptions;
}
