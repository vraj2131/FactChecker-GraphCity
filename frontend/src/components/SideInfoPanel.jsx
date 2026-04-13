import { X, ExternalLink, Shield, Zap, BarChart3, Globe, MousePointer, Network } from 'lucide-react';
import {
  NODE_TYPE_LABELS,
  VERDICT_CONFIG,
  SOURCE_TYPE_LABELS,
  SOURCE_TYPE_COLORS,
} from '../utils/colorMap';

/* ─────────────────────────────────────────────────────────────────────────
   Shared sub-components
───────────────────────────────────────────────────────────────────────── */

function ConfidenceBar({ value, color }) {
  return (
    <div className="conf-bar-track">
      <div
        className="conf-bar-fill"
        style={{ width: `${Math.round(value * 100)}%`, background: color }}
      />
      <span className="conf-bar-label">{Math.round(value * 100)}%</span>
    </div>
  );
}

function SourceCard({ source }) {
  const typeLabel  = SOURCE_TYPE_LABELS[source.source_type] ?? source.source_type;
  const typeColor  = SOURCE_TYPE_COLORS[source.source_type] ?? '#94a3b8';
  const stanceConf = VERDICT_CONFIG[source.stance_hint] ?? VERDICT_CONFIG.neutral;

  return (
    <div className="source-card">
      <div className="source-card-header">
        <span className="source-type-tag" style={{ color: typeColor, borderColor: typeColor + '44' }}>
          {typeLabel}
        </span>
        {source.publisher && (
          <span className="source-publisher">{source.publisher}</span>
        )}
        {source.stance_hint && (
          <span
            className="source-stance-badge"
            style={{ color: stanceConf.color, background: stanceConf.bg, borderColor: stanceConf.border }}
          >
            {stanceConf.label}
          </span>
        )}
      </div>

      <p className="source-title">{source.title}</p>

      {source.snippet && (
        <p className="source-snippet">"{source.snippet}"</p>
      )}

      <div className="source-scores">
        <div className="score-item">
          <Shield size={11} />
          <span>Trust</span>
          <div className="score-bar-mini">
            <div style={{ width: `${source.trust_score * 100}%`, background: '#3b82f6' }} />
          </div>
          <span className="score-val">{Math.round(source.trust_score * 100)}%</span>
        </div>
        <div className="score-item">
          <Zap size={11} />
          <span>Relevance</span>
          <div className="score-bar-mini">
            <div style={{ width: `${source.relevance_score * 100}%`, background: '#8b5cf6' }} />
          </div>
          <span className="score-val">{Math.round(source.relevance_score * 100)}%</span>
        </div>
      </div>

      {source.url && (
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="source-url-btn"
        >
          <Globe size={12} />
          Open Source
          <ExternalLink size={11} />
        </a>
      )}

      {source.published_at && (
        <span className="source-date">
          {new Date(source.published_at).toLocaleDateString('en-US', {
            year: 'numeric', month: 'short', day: 'numeric',
          })}
        </span>
      )}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   Main claim view — shows evidence overview, NOT raw sources
   The graph IS the source explorer. This panel explains the verdict.
───────────────────────────────────────────────────────────────────────── */

function EvidenceNodeCard({ evNode }) {
  const verdict    = VERDICT_CONFIG[evNode.verdict] ?? VERDICT_CONFIG.neutral;
  const typeLabel  = NODE_TYPE_LABELS[evNode.node_type] ?? evNode.node_type;
  const topSource  = evNode.top_sources?.[0];
  const typeColor  = SOURCE_TYPE_COLORS[topSource?.source_type] ?? '#94a3b8';
  const typeLabel2 = SOURCE_TYPE_LABELS[topSource?.source_type] ?? topSource?.source_type;

  return (
    <div className="ev-node-card" style={{ borderLeftColor: evNode.color }}>
      <div className="ev-node-card-header">
        <span
          className="ev-node-type-badge"
          style={{ color: evNode.color, background: evNode.color + '18', borderColor: evNode.color + '44' }}
        >
          {typeLabel}
        </span>
        <span
          className="ev-node-verdict-badge"
          style={{ color: verdict.color }}
        >
          {verdict.label}
        </span>
        <span className="ev-node-conf">{Math.round(evNode.confidence * 100)}%</span>
      </div>

      <p className="ev-node-text">
        {evNode.text.length > 110 ? evNode.text.slice(0, 110) + '…' : evNode.text}
      </p>

      {topSource && (
        <div className="ev-node-source-row">
          <span className="ev-node-source-tag" style={{ color: typeColor }}>
            {typeLabel2}
          </span>
          {topSource.publisher && (
            <span className="ev-node-publisher">{topSource.publisher}</span>
          )}
          {topSource.url && (
            <a
              href={topSource.url}
              target="_blank"
              rel="noopener noreferrer"
              className="ev-node-url"
              title={topSource.url}
            >
              <ExternalLink size={10} />
            </a>
          )}
        </div>
      )}
    </div>
  );
}

function MainClaimPanel({ node, graphJson, onClose }) {
  const verdict = VERDICT_CONFIG[node.verdict] ?? VERDICT_CONFIG.not_enough_info;

  // Derive evidence nodes from graph (exclude main claim)
  const evidenceNodes = graphJson?.nodes?.filter((n) => !n.is_main_claim) ?? [];

  // Group by verdict direction
  const refuting  = evidenceNodes.filter((n) => n.verdict === 'refutes');
  const supporting = evidenceNodes.filter((n) => n.verdict === 'supports');
  const contextual = evidenceNodes.filter((n) => ['correlated', 'insufficient'].includes(n.verdict));

  const meta = graphJson?.metadata;

  return (
    <>
      <div className="side-panel-header">
        <div className="side-panel-title-row">
          <div className="side-panel-main-label">
            <Network size={13} />
            Main Claim
          </div>
          <button className="side-panel-close" onClick={onClose} aria-label="Close panel">
            <X size={16} />
          </button>
        </div>

        <div
          className="side-panel-verdict"
          style={{ color: verdict.color, background: verdict.bg, borderColor: verdict.border }}
        >
          <span className="verdict-dot" style={{ background: verdict.color }} />
          {verdict.label}
          {meta && (
            <span className="verdict-conf-inline">{Math.round(meta.overall_confidence * 100)}% confidence</span>
          )}
        </div>
      </div>

      <div className="side-panel-body">
        <p className="side-panel-claim-text">{node.text}</p>

        {/* Confidence bar */}
        <div className="side-panel-section">
          <div className="section-label">
            <BarChart3 size={13} />
            Overall Confidence
          </div>
          <ConfidenceBar value={meta?.overall_confidence ?? node.confidence} color={node.color} />
        </div>

        {/* Analysis */}
        {node.short_explanation && (
          <div className="side-panel-section">
            <div className="section-label">Verdict Reasoning</div>
            <p className="side-panel-explanation">{node.short_explanation}</p>
          </div>
        )}

        {/* Evidence summary counts */}
        {meta && (
          <div className="side-panel-section">
            <div className="section-label">Evidence Summary</div>
            <div className="ev-summary-chips">
              {meta.refute_node_count > 0 && (
                <div className="ev-chip" style={{ color: '#f97316', background: 'rgba(249,115,22,0.1)', borderColor: 'rgba(249,115,22,0.3)' }}>
                  <span className="ev-chip-count">{meta.refute_node_count}</span>
                  Refuting
                </div>
              )}
              {meta.support_node_count > 0 && (
                <div className="ev-chip" style={{ color: '#3b82f6', background: 'rgba(59,130,246,0.1)', borderColor: 'rgba(59,130,246,0.3)' }}>
                  <span className="ev-chip-count">{meta.support_node_count}</span>
                  Supporting
                </div>
              )}
              {meta.factcheck_node_count > 0 && (
                <div className="ev-chip" style={{ color: '#a855f7', background: 'rgba(168,85,247,0.1)', borderColor: 'rgba(168,85,247,0.3)' }}>
                  <span className="ev-chip-count">{meta.factcheck_node_count}</span>
                  Fact-Checked
                </div>
              )}
              {meta.context_node_count > 0 && (
                <div className="ev-chip" style={{ color: '#818cf8', background: 'rgba(129,140,248,0.1)', borderColor: 'rgba(129,140,248,0.3)' }}>
                  <span className="ev-chip-count">{meta.context_node_count}</span>
                  Context
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hint to explore nodes */}
        <div className="explore-hint">
          <MousePointer size={12} />
          <span>Click any node in the graph to inspect its source and evidence details.</span>
        </div>

        {/* Evidence node cards — refuting first (most impactful) */}
        {refuting.length > 0 && (
          <div className="side-panel-section">
            <div className="section-label" style={{ color: '#f97316' }}>Refuting Evidence</div>
            <div className="ev-node-list">
              {refuting.map((n) => <EvidenceNodeCard key={n.node_id} evNode={n} />)}
            </div>
          </div>
        )}

        {supporting.length > 0 && (
          <div className="side-panel-section">
            <div className="section-label" style={{ color: '#3b82f6' }}>Supporting Evidence</div>
            <div className="ev-node-list">
              {supporting.map((n) => <EvidenceNodeCard key={n.node_id} evNode={n} />)}
            </div>
          </div>
        )}

        {contextual.length > 0 && (
          <div className="side-panel-section">
            <div className="section-label" style={{ color: '#818cf8' }}>Context Signals</div>
            <div className="ev-node-list">
              {contextual.map((n) => <EvidenceNodeCard key={n.node_id} evNode={n} />)}
            </div>
          </div>
        )}

        {/* Open best source */}
        {node.best_source_url && (
          <a
            href={node.best_source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="open-source-btn"
            style={{ borderColor: node.color + '55', color: node.color }}
          >
            <ExternalLink size={14} />
            Open Best Source
          </a>
        )}

        {/* Retrieval notes */}
        {meta?.retrieval_notes && (
          <p className="retrieval-notes">{meta.retrieval_notes}</p>
        )}
      </div>
    </>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   Evidence node view — full source details
───────────────────────────────────────────────────────────────────────── */

function EvidencePanel({ node, onClose }) {
  const verdict   = VERDICT_CONFIG[node.verdict] ?? VERDICT_CONFIG.neutral;
  const typeLabel = NODE_TYPE_LABELS[node.node_type] ?? node.node_type;

  return (
    <>
      <div className="side-panel-header">
        <div className="side-panel-title-row">
          <span
            className="side-panel-type-badge"
            style={{ color: node.color, background: node.color + '18', borderColor: node.color + '44' }}
          >
            {typeLabel}
          </span>
          <button className="side-panel-close" onClick={onClose} aria-label="Close panel">
            <X size={16} />
          </button>
        </div>

        <div
          className="side-panel-verdict"
          style={{ color: verdict.color, background: verdict.bg, borderColor: verdict.border }}
        >
          <span className="verdict-dot" style={{ background: verdict.color }} />
          {verdict.label}
        </div>
      </div>

      <div className="side-panel-body">
        <p className="side-panel-claim-text">{node.text}</p>

        <div className="side-panel-section">
          <div className="section-label">
            <BarChart3 size={13} />
            Confidence Score
          </div>
          <ConfidenceBar value={node.confidence} color={node.color} />
        </div>

        {node.short_explanation && (
          <div className="side-panel-section">
            <div className="section-label">Analysis</div>
            <p className="side-panel-explanation">{node.short_explanation}</p>
          </div>
        )}

        {node.top_sources?.length > 0 && (
          <div className="side-panel-section">
            <div className="section-label">
              <Globe size={13} />
              Evidence Source{node.source_count > 1 ? `s (${node.source_count} total)` : ''}
            </div>
            <div className="sources-list">
              {node.top_sources.map((src) => (
                <SourceCard key={src.source_id} source={src} />
              ))}
            </div>
          </div>
        )}

        {node.best_source_url && (
          <a
            href={node.best_source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="open-source-btn"
            style={{ borderColor: node.color + '55', color: node.color }}
          >
            <ExternalLink size={14} />
            Open Best Source
          </a>
        )}
      </div>
    </>
  );
}

/* ─────────────────────────────────────────────────────────────────────────
   Export
───────────────────────────────────────────────────────────────────────── */

export default function SideInfoPanel({ node, graphJson, onClose }) {
  if (!node) return null;

  return (
    <aside className="side-panel">
      {node.is_main_claim
        ? <MainClaimPanel node={node} graphJson={graphJson} onClose={onClose} />
        : <EvidencePanel  node={node} onClose={onClose} />
      }
    </aside>
  );
}
