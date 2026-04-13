import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { NODE_COLORS, EDGE_COLORS, NODE_TYPE_LABELS, EDGE_TYPE_LABELS, VERDICT_CONFIG } from '../utils/colorMap';

const NODE_LEGEND = [
  { key: 'main_claim_verified',   label: 'Main Claim — Verified',   color: NODE_COLORS.main_claim_verified },
  { key: 'main_claim_rejected',   label: 'Main Claim — Rejected',   color: NODE_COLORS.main_claim_rejected },
  { key: 'main_claim_nei',        label: 'Main Claim — Inconclusive',color: NODE_COLORS.main_claim_nei },
  { key: 'direct_support',        label: NODE_TYPE_LABELS.direct_support,        color: NODE_COLORS.direct_support },
  { key: 'direct_refute',         label: NODE_TYPE_LABELS.direct_refute,         color: NODE_COLORS.direct_refute },
  { key: 'factcheck_review',      label: NODE_TYPE_LABELS.factcheck_review,      color: NODE_COLORS.factcheck_review },
  { key: 'context_signal',        label: NODE_TYPE_LABELS.context_signal,        color: NODE_COLORS.context_signal },
  { key: 'insufficient_evidence', label: NODE_TYPE_LABELS.insufficient_evidence, color: NODE_COLORS.insufficient_evidence },
];

const EDGE_LEGEND = [
  { key: 'supports',    label: EDGE_TYPE_LABELS.supports,    color: EDGE_COLORS.supports,    dashed: false },
  { key: 'refutes',     label: EDGE_TYPE_LABELS.refutes,     color: EDGE_COLORS.refutes,     dashed: false },
  { key: 'correlated',  label: EDGE_TYPE_LABELS.correlated,  color: EDGE_COLORS.correlated,  dashed: true  },
  { key: 'insufficient',label: EDGE_TYPE_LABELS.insufficient,color: EDGE_COLORS.insufficient,dashed: true  },
];

export default function LegendPanel({ metadata }) {
  const [collapsed, setCollapsed] = useState(false);

  const verdict = VERDICT_CONFIG[metadata?.overall_verdict] ?? VERDICT_CONFIG.not_enough_info;
  const conf = metadata?.overall_confidence ?? 0;

  return (
    <div className={`legend-panel ${collapsed ? 'legend-panel--collapsed' : ''}`}>
      {/* Header row */}
      <div className="legend-header" onClick={() => setCollapsed((v) => !v)}>
        <span className="legend-title">Evidence Map</span>
        <button className="legend-toggle" aria-label="Toggle legend">
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </div>

      {!collapsed && (
        <>
          {/* Overall verdict summary */}
          {metadata && (
            <div className="legend-verdict-row">
              <span
                className="legend-verdict-badge"
                style={{ color: verdict.color, background: verdict.bg, borderColor: verdict.border }}
              >
                <span className="verdict-dot" style={{ background: verdict.color }} />
                {verdict.label}
              </span>
              <span className="legend-conf">{Math.round(conf * 100)}% confidence</span>
            </div>
          )}

          {metadata && (
            <div className="legend-stats">
              <span className="legend-stat">
                <span className="legend-stat-dot" style={{ background: NODE_COLORS.direct_support }} />
                {metadata.support_node_count} support
              </span>
              <span className="legend-stat">
                <span className="legend-stat-dot" style={{ background: NODE_COLORS.direct_refute }} />
                {metadata.refute_node_count} refute
              </span>
              <span className="legend-stat">
                <span className="legend-stat-dot" style={{ background: NODE_COLORS.factcheck_review }} />
                {metadata.factcheck_node_count} fact-check
              </span>
              <span className="legend-stat">
                <span className="legend-stat-dot" style={{ background: NODE_COLORS.context_signal }} />
                {metadata.context_node_count} context
              </span>
            </div>
          )}

          <div className="legend-divider" />

          {/* Node types */}
          <div className="legend-section-label">Node Types</div>
          <div className="legend-items">
            {NODE_LEGEND.map(({ key, label, color }) => (
              <div key={key} className="legend-item">
                <span className="legend-node-dot" style={{ background: color, boxShadow: `0 0 6px ${color}88` }} />
                <span className="legend-item-label">{label}</span>
              </div>
            ))}
          </div>

          <div className="legend-divider" />

          {/* Edge types */}
          <div className="legend-section-label">Edge Types</div>
          <div className="legend-items">
            {EDGE_LEGEND.map(({ key, label, color, dashed }) => (
              <div key={key} className="legend-item">
                <svg width="28" height="6" className="legend-edge-line">
                  <line
                    x1="0" y1="3" x2="28" y2="3"
                    stroke={color}
                    strokeWidth="2"
                    strokeDasharray={dashed ? '4 3' : 'none'}
                  />
                </svg>
                <span className="legend-item-label">{label}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
