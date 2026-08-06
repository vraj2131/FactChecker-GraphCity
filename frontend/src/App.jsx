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
import VerifyProgressOverlay from './components/VerifyProgressOverlay';
import { VERDICT_CONFIG } from './utils/colorMap';
import { verifyClaim } from './api/client';
import { pushClaimHistory, decayContextForLLM, buildHistoryEntry } from './utils/claimChain';
import { GitBranch, Cpu, Layers, X, Camera } from 'lucide-react';

const SIDE_PANEL_WIDTH = 360;

export default function App() {
  const [graphData, setGraphData]   = useState(null);   // null = landing page
  const [currentClaim, setCurrentClaim] = useState('');
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [filterVerdict, setFilter]  = useState(null);
  const [verificationHistory, setVerificationHistory] = useState([]);  // Feature 5
  const graphCacheRef = useRef(new Map());  // Feature 15: claim_text → graph, for instant chain navigation

  const graphCanvasRef = useRef(null);
  const handleSnapshot = useCallback(() => graphCanvasRef.current?.snapshot(), []);

  const [hoveredNode, setHoveredNode]   = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mousePos, setMousePos]         = useState({ x: 0, y: 0 });

  const handleNodeHover  = useCallback((node) => setHoveredNode(node),  []);
  const handleNodeSelect = useCallback((node) => setSelectedNode(node), []);
  const handleMouseMove  = useCallback((pos)  => setMousePos(pos),      []);
  const handlePanelClose = useCallback(()     => setSelectedNode(null), []);

  const handleVerify = useCallback(async (claimText, enabledSourceGroups = null) => {
    setCurrentClaim(claimText);
    setError(null);
    setSelectedNode(null);
    setFilter(null);

    // Feature 15: revisiting a claim already verified this session (e.g.
    // clicking it in the claim-chain banner) renders instantly from cache
    // instead of re-running the pipeline. Skipped when the user explicitly
    // customized source groups for this run, since that could change results.
    if (!enabledSourceGroups) {
      const cached = graphCacheRef.current.get(claimText);
      if (cached) {
        setGraphData(cached);
        setVerificationHistory(prev => pushClaimHistory(prev, buildHistoryEntry(claimText, cached)));
        return;
      }
    }

    setLoading(true);
    try {
      const graph = await verifyClaim(claimText, decayContextForLLM(verificationHistory), enabledSourceGroups);
      setGraphData(graph);
      graphCacheRef.current.set(claimText, graph);
      setVerificationHistory(prev => pushClaimHistory(prev, buildHistoryEntry(claimText, graph)));
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
    graphCacheRef.current.clear();
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

  // Loading state — 3-step progress wheel, no graph yet
  if (loading && !graphData) {
    return (
      <div className="app">
        <VerifyProgressOverlay claimText={currentClaim} />
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

        {/* Claim chain banner — prior claims in the chain, clickable to revisit (Feature 15) */}
        {verificationHistory.length > 0 && (
          <ContextChainBanner
            history={verificationHistory}
            currentClaim={currentClaim}
            onSelect={handleVerify}
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
          <FilterPanel
            activeFilter={filterVerdict}
            onFilter={setFilter}
          />
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

      {/* ── Re-verify progress wheel over the existing graph ─────────────── */}
      {loading && <VerifyProgressOverlay claimText={currentClaim} />}

      {/* ── Tooltip ──────────────────────────────────────────────────────── */}
      {hoveredNode && !loading && <NodeTooltip node={hoveredNode} pos={mousePos} />}
    </div>
  );
}
