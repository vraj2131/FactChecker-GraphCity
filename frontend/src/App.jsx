import { useState, useCallback } from 'react';
import GraphCanvas from './components/GraphCanvas';
import NodeTooltip from './components/NodeTooltip';
import SideInfoPanel from './components/SideInfoPanel';
import LegendPanel from './components/LegendPanel';
import ClaimInputPanel from './components/ClaimInputPanel';
import InfoTabsPanel from './components/InfoTabsPanel';
import sampleGraph from './mock/sampleGraph.json';
import { VERDICT_CONFIG } from './utils/colorMap';
import { GitBranch, Cpu, Layers } from 'lucide-react';

const SIDE_PANEL_WIDTH = 360;

export default function App() {
  const [hoveredNode, setHoveredNode]   = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mousePos, setMousePos]         = useState({ x: 0, y: 0 });

  const handleNodeHover  = useCallback((node) => setHoveredNode(node),  []);
  const handleNodeSelect = useCallback((node) => setSelectedNode(node), []);
  const handleMouseMove  = useCallback((pos)  => setMousePos(pos),      []);
  const handlePanelClose = useCallback(()     => setSelectedNode(null), []);

  const panelOpen  = !!selectedNode;
  const panelWidth = panelOpen ? SIDE_PANEL_WIDTH : 0;

  const meta    = sampleGraph.metadata;
  const verdict = VERDICT_CONFIG[meta.overall_verdict] ?? VERDICT_CONFIG.not_enough_info;

  return (
    <div className="app">
      {/* ── Full-screen 3D graph — shrinks when side panel is open ──────── */}
      <GraphCanvas
        graphJson={sampleGraph}
        onNodeHover={handleNodeHover}
        onNodeSelect={handleNodeSelect}
        onMouseMove={handleMouseMove}
        panelWidth={panelWidth}
        isNodeSelected={panelOpen}
      />

      {/* ── UI overlays ──────────────────────────────────────────────────── */}
      <div className="ui-layer">

        {/* Top bar */}
        <header className="top-bar">
          <div className="top-bar-brand">
            <div className="brand-icon">
              <GitBranch size={16} />
            </div>
            <span className="brand-name">FactGraph City</span>
            <span className="brand-badge">Phase 12</span>
          </div>

          <div className="top-bar-center">
            <ClaimInputPanel />
          </div>

          <div className="top-bar-right">
            <div className="pipeline-chips">
              <span className="pipeline-chip"><Cpu size={11} />DeBERTa NLI</span>
              <span className="pipeline-chip"><Layers size={11} />Llama 3.1</span>
            </div>
          </div>
        </header>

        {/* Verdict strip */}
        <div className="verdict-strip">
          <div
            className="verdict-pill"
            style={{ color: verdict.color, background: verdict.bg, borderColor: verdict.border }}
          >
            <span className="verdict-dot" style={{ background: verdict.color }} />
            <span className="verdict-pill-label">{verdict.label}</span>
            <span className="verdict-pill-conf">{Math.round(meta.overall_confidence * 100)}% confidence</span>
          </div>
        </div>

        {/* Bottom-left: legend */}
        <div className="legend-anchor">
          <LegendPanel metadata={meta} />
        </div>

        {/* Bottom-right: sources + schema tabs */}
        <div className="itabs-anchor">
          <InfoTabsPanel graphJson={sampleGraph} />
        </div>

        {/* Right: side panel — graph shrinks to make room, no overlap */}
        <div
          className="side-panel-anchor"
          style={{ width: SIDE_PANEL_WIDTH }}
          aria-hidden={!panelOpen}
        >
          <div className={`side-panel-slider ${panelOpen ? 'side-panel-slider--open' : ''}`}>
            <SideInfoPanel
              node={selectedNode}
              graphJson={sampleGraph}
              onClose={handlePanelClose}
            />
          </div>
        </div>
      </div>

      {/* ── Tooltip ──────────────────────────────────────────────────────── */}
      {hoveredNode && <NodeTooltip node={hoveredNode} pos={mousePos} />}
    </div>
  );
}
