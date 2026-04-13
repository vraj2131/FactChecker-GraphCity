import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import {
  NODE_COLORS, EDGE_COLORS,
  NODE_TYPE_LABELS, EDGE_TYPE_LABELS,
  SOURCE_TYPE_COLORS,
  VERDICT_CONFIG,
} from '../utils/colorMap';

// All available retriever sources in the pipeline (fixed list)
const ALL_RETRIEVERS = [
  { key: 'wikipedia', label: 'Wikipedia',    desc: 'FEVER offline FAISS index — encyclopaedic facts' },
  { key: 'livewiki',  label: 'Live Wiki',     desc: 'Live Wikipedia API — real-time article lookup' },
  { key: 'factcheck', label: 'Fact Check',   desc: 'Google Fact Check API — professional debunks' },
  { key: 'guardian',  label: 'The Guardian', desc: 'Guardian news API — trusted journalism' },
  { key: 'newsapi',   label: 'NewsAPI',      desc: 'Aggregated news — broad current coverage' },
  { key: 'gdelt',     label: 'GDELT',        desc: 'Global event database — broad geopolitical signals' },
];

// ── Sources tab ──────────────────────────────────────────────────────────────

function SourcesTab({ graphJson }) {
  // Count how many results each source_type contributed to this graph
  const counts = {};
  for (const node of graphJson.nodes) {
    for (const src of node.top_sources ?? []) {
      counts[src.source_type] = (counts[src.source_type] ?? 0) + 1;
    }
  }

  const totalResults = Object.values(counts).reduce((s, v) => s + v, 0);

  return (
    <div className="itab-body">
      {ALL_RETRIEVERS.map((r) => {
        const color    = SOURCE_TYPE_COLORS[r.key] ?? '#94a3b8';
        const count    = counts[r.key] ?? 0;
        const active   = count > 0;

        return (
          <div
            key={r.key}
            className={`itab-retriever-row ${active ? 'itab-retriever-row--active' : 'itab-retriever-row--inactive'}`}
          >
            <span
              className="itab-retriever-dot"
              style={active
                ? { background: color, boxShadow: `0 0 6px ${color}99` }
                : { background: 'rgba(255,255,255,0.08)' }
              }
            />
            <div className="itab-retriever-info">
              <div className="itab-retriever-name" style={{ color: active ? color : 'var(--text-muted)' }}>
                {r.label}
                {active && <span className="itab-retriever-count">{count}</span>}
              </div>
              <div className="itab-retriever-desc">{r.desc}</div>
            </div>
            {active && (
              <div className="itab-retriever-bar-track">
                <div
                  className="itab-retriever-bar-fill"
                  style={{ width: `${(count / totalResults) * 100}%`, background: color }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ── Schema tab ───────────────────────────────────────────────────────────────

function SchemaTab({ graphJson }) {
  const meta = graphJson.metadata;

  // Count node types
  const nodeTypeCounts = {};
  for (const n of graphJson.nodes) {
    nodeTypeCounts[n.node_type] = (nodeTypeCounts[n.node_type] ?? 0) + 1;
  }

  // Count edge types
  const edgeTypeCounts = {};
  for (const e of graphJson.edges) {
    edgeTypeCounts[e.edge_type] = (edgeTypeCounts[e.edge_type] ?? 0) + 1;
  }

  const totalNodes = graphJson.nodes.length;
  const totalEdges = graphJson.edges.length;

  return (
    <div className="itab-body">

      {/* Verdict row */}
      {meta && (() => {
        const v = VERDICT_CONFIG[meta.overall_verdict] ?? VERDICT_CONFIG.not_enough_info;
        return (
          <div className="itab-schema-verdict">
            <span
              className="itab-schema-verdict-badge"
              style={{ color: v.color, background: v.bg, borderColor: v.border }}
            >
              <span className="verdict-dot" style={{ background: v.color }} />
              {v.label}
            </span>
            <span className="itab-schema-conf" style={{ color: v.color }}>
              {Math.round(meta.overall_confidence * 100)}%
            </span>
          </div>
        );
      })()}

      {/* Node schema */}
      <div className="itab-schema-section">
        <div className="itab-schema-heading">
          Nodes
          <span className="itab-schema-total">{totalNodes} total</span>
        </div>
        {Object.entries(nodeTypeCounts).map(([type, count]) => {
          const color = NODE_COLORS[type] ?? '#424242';
          const label = NODE_TYPE_LABELS[type] ?? type;
          return (
            <div key={type} className="itab-schema-row">
              <span className="itab-schema-dot" style={{ background: color, boxShadow: `0 0 5px ${color}88` }} />
              <span className="itab-schema-label">{label}</span>
              <div className="itab-schema-bar-track">
                <div
                  className="itab-schema-bar-fill"
                  style={{ width: `${(count / totalNodes) * 100}%`, background: color }}
                />
              </div>
              <span className="itab-schema-count">{count}</span>
            </div>
          );
        })}
      </div>

      {/* Edge schema */}
      <div className="itab-schema-section">
        <div className="itab-schema-heading">
          Edges
          <span className="itab-schema-total">{totalEdges} total</span>
        </div>
        {Object.entries(edgeTypeCounts).map(([type, count]) => {
          const color = EDGE_COLORS[type] ?? '#bdbdbd';
          const label = EDGE_TYPE_LABELS[type] ?? type;
          return (
            <div key={type} className="itab-schema-row">
              <svg width="22" height="6" style={{ flexShrink: 0 }}>
                <line x1="0" y1="3" x2="22" y2="3"
                  stroke={color} strokeWidth="2"
                  strokeDasharray={type === 'correlated' || type === 'insufficient' ? '4 3' : 'none'}
                />
              </svg>
              <span className="itab-schema-label">{label}</span>
              <div className="itab-schema-bar-track">
                <div
                  className="itab-schema-bar-fill"
                  style={{ width: `${(count / totalEdges) * 100}%`, background: color }}
                />
              </div>
              <span className="itab-schema-count">{count}</span>
            </div>
          );
        })}
      </div>

      {/* Pipeline info */}
      <div className="itab-schema-section">
        <div className="itab-schema-heading">Pipeline</div>
        <div className="itab-pipeline-rows">
          <div className="itab-pipeline-row">
            <span className="itab-pipeline-key">Retrieval</span>
            <span className="itab-pipeline-val">
              {meta?.retrieval_notes?.split('from:')[1]?.trim() ?? '—'}
            </span>
          </div>
          <div className="itab-pipeline-row">
            <span className="itab-pipeline-key">NLI Model</span>
            <span className="itab-pipeline-val">DeBERTa v3</span>
          </div>
          <div className="itab-pipeline-row">
            <span className="itab-pipeline-key">LLM</span>
            <span className="itab-pipeline-val">Llama 3.1 8B (Groq)</span>
          </div>
          <div className="itab-pipeline-row">
            <span className="itab-pipeline-key">Confidence</span>
            <span className="itab-pipeline-val">Piecewise calibrated</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

const TABS = ['Sources', 'Schema'];

export default function InfoTabsPanel({ graphJson }) {
  const [activeTab, setActiveTab] = useState('Sources');
  const [collapsed, setCollapsed] = useState(false);

  const sourceCount = new Set(
    graphJson.nodes.flatMap((n) => (n.top_sources ?? []).map((s) => s.source_id))
  ).size;

  return (
    <div className={`itabs-panel ${collapsed ? 'itabs-panel--collapsed' : ''}`}>
      {/* Tab bar */}
      <div className="itabs-tabbar">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`itabs-tab ${activeTab === tab && !collapsed ? 'itabs-tab--active' : ''}`}
            onClick={() => {
              if (collapsed) {
                setCollapsed(false);
                setActiveTab(tab);
              } else if (activeTab === tab) {
                setCollapsed(true);
              } else {
                setActiveTab(tab);
              }
            }}
          >
            {tab}
            {tab === 'Sources' && (
              <span className="itabs-tab-count">{sourceCount}</span>
            )}
          </button>
        ))}
        <button
          className="itabs-collapse-btn"
          onClick={() => setCollapsed((v) => !v)}
          aria-label="Toggle panel"
        >
          {collapsed ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
      </div>

      {/* Content */}
      {!collapsed && (
        <div className="itabs-content">
          {activeTab === 'Sources' && <SourcesTab graphJson={graphJson} />}
          {activeTab === 'Schema'  && <SchemaTab  graphJson={graphJson} />}
        </div>
      )}
    </div>
  );
}
