import { useState, useCallback, useRef } from 'react';
import GraphCanvas from './components/GraphCanvas';
import NodeTooltip from './components/NodeTooltip';
import SideInfoPanel from './components/SideInfoPanel';
import LegendPanel from './components/LegendPanel';
import ClaimInputPanel from './components/ClaimInputPanel';
import InfoTabsPanel from './components/InfoTabsPanel';
import FilterPanel from './components/FilterPanel';
import LandingPage from './components/LandingPage';
import ContextChainBanner from './components/ContextChainBanner';
import { VERDICT_CONFIG } from './utils/colorMap';
import { verifyClaim } from './api/client';
import { GitBranch, Cpu, Layers, X, Camera } from 'lucide-react';

const SIDE_PANEL_WIDTH = 360;

export default function App() {
  const [graphData, setGraphData]   = useState(null);   // null = landing page
  const [currentClaim, setCurrentClaim] = useState('');
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [filterVerdict, setFilter]  = useState(null);
  const [verificationHistory, setVerificationHistory] = useState([]);  // Feature 5

  const graphCanvasRef = useRef(null);
  const handleSnapshot = useCallback(() => graphCanvasRef.current?.snapshot(), []);

  const [hoveredNode, setHoveredNode]   = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mousePos, setMousePos]         = useState({ x: 0, y: 0 });

  const handleNodeHover  = useCallback((node) => setHoveredNode(node),  []);
  const handleNodeSelect = useCallback((node) => setSelectedNode(node), []);
  const handleMouseMove  = useCallback((pos)  => setMousePos(pos),      []);
  const handlePanelClose = useCallback(()     => setSelectedNode(null), []);

  const handleVerify = useCallback(async (claimText) => {
    setCurrentClaim(claimText);
    setLoading(true);
    setError(null);
    setSelectedNode(null);
    setFilter(null);
    try {
      const graph = await verifyClaim(claimText, verificationHistory);
      setGraphData(graph);

      // Build history entry from top evidence nodes (Feature 5)
      const topSnippets = (graph.nodes || [])
        .filter(n => !n.is_main_claim && n.top_sources?.length > 0)
        .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
        .slice(0, 3)
        .flatMap(n => n.top_sources?.slice(0, 1).map(s => s.snippet || '') || [])
        .filter(Boolean);

      setVerificationHistory(prev => [
        ...prev.slice(-2),
        {
          claim_text: claimText,
          verdict: graph.metadata.overall_verdict,
          confidence: graph.metadata.overall_confidence,
          top_snippets: topSnippets.slice(0, 3),
        },
      ]);
    } catch (err) {
      setError(err.message ?? 'Pipeline error — check backend logs.');
    } finally {
      setLoading(false);
    }
  }, [verificationHistory]);

  const handleClear = useCallback(() => {
    setGraphData(null);
    setSelectedNode(null);
    setFilter(null);
    setError(null);
    setVerificationHistory([]);
  }, []);

  // ── Landing page (no result yet) ────────────────────────────────────────
  if (!graphData && !loading) {
    return (
      <div className="app">
        {error && (
          <div className="error-toast" onClick={() => setError(null)}>
            <span className="error-toast-icon">!</span>
            {error}
            <span className="error-toast-dismiss">×</span>
          </div>
        )}
        <LandingPage onVerify={handleVerify} loading={loading} />
      </div>
    );
  }

  // Loading state — show starfield with spinner, no graph yet
  if (loading && !graphData) {
    return (
      <div className="app">
        <div className="loading-fullscreen">
          <div className="loading-spinner-ring" />
          <p className="loading-fullscreen-text">Analysing claim…</p>
          <p className="loading-fullscreen-sub">Retrieving evidence · Running NLI · Consulting LLM</p>
        </div>
      </div>
    );
  }

  const panelOpen  = !!selectedNode;
  const panelWidth = panelOpen ? SIDE_PANEL_WIDTH : 0;
  const meta       = graphData.metadata;
  const verdict    = VERDICT_CONFIG[meta.overall_verdict] ?? VERDICT_CONFIG.not_enough_info;

  return (
    <div className="app">
      {/* ── Full-screen 3D graph ─────────────────────────────────────────── */}
      <GraphCanvas
        ref={graphCanvasRef}
        graphJson={graphData}
        onNodeHover={handleNodeHover}
        onNodeSelect={handleNodeSelect}
        onMouseMove={handleMouseMove}
        panelWidth={panelWidth}
        isNodeSelected={panelOpen}
        filterVerdict={filterVerdict}
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
          </div>

          <div className="top-bar-center">
            <ClaimInputPanel onVerify={handleVerify} loading={loading} claimText={currentClaim} />
          </div>

          <div className="top-bar-right">
            <div className="pipeline-chips">
              <span className="pipeline-chip"><Cpu size={11} />DeBERTa NLI</span>
              <span className="pipeline-chip"><Layers size={11} />Llama 3.1</span>
            </div>
            <button className="snapshot-btn" onClick={handleSnapshot} title="Save snapshot as PNG">
              <Camera size={14} />
              Snapshot
            </button>
            <button className="clear-btn" onClick={handleClear} title="Clear results">
              <X size={14} />
              Clear
            </button>
          </div>
        </header>

        {/* Verdict strip */}
        <div className="verdict-strip">
          {loading ? (
            <div className="verdict-loading">
              <span className="verdict-loading-dot" />
              <span className="verdict-loading-text">Analysing claim…</span>
            </div>
          ) : (
            <div
              className="verdict-pill"
              style={{ color: verdict.color, background: verdict.bg, borderColor: verdict.border }}
            >
              <span className="verdict-dot" style={{ background: verdict.color }} />
              <span className="verdict-pill-label">{verdict.label}</span>
              <span className="verdict-pill-conf">{Math.round(meta.overall_confidence * 100)}% confidence</span>
            </div>
          )}
        </div>

        {/* Context chain banner — shows prior claims in the chain */}
        {verificationHistory.length > 0 && (
          <ContextChainBanner
            history={verificationHistory}
            onClear={() => setVerificationHistory([])}
          />
        )}

        {/* Error toast */}
        {error && (
          <div className="error-toast" onClick={() => setError(null)}>
            <span className="error-toast-icon">!</span>
            {error}
            <span className="error-toast-dismiss">×</span>
          </div>
        )}

        {/* Bottom-left: legend + filter */}
        <div className="legend-anchor">
          <LegendPanel metadata={meta} />
          <FilterPanel activeFilter={filterVerdict} onFilter={setFilter} />
        </div>

        {/* Bottom-right: sources + schema tabs */}
        <div className="itabs-anchor">
          <InfoTabsPanel graphJson={graphData} />
        </div>

        {/* Right: side panel */}
        <div
          className="side-panel-anchor"
          style={{ width: SIDE_PANEL_WIDTH }}
          aria-hidden={!panelOpen}
        >
          <div className={`side-panel-slider ${panelOpen ? 'side-panel-slider--open' : ''}`}>
            <SideInfoPanel
              node={selectedNode}
              graphJson={graphData}
              onClose={handlePanelClose}
              onVerify={handleVerify}
            />
          </div>
        </div>
      </div>

      {/* ── Tooltip ──────────────────────────────────────────────────────── */}
      {hoveredNode && !loading && <NodeTooltip node={hoveredNode} pos={mousePos} />}
    </div>
  );
}
