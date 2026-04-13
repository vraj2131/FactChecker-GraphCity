import { NODE_TYPE_LABELS, VERDICT_CONFIG, SOURCE_TYPE_LABELS, SOURCE_TYPE_COLORS } from '../utils/colorMap';

export default function NodeTooltip({ node, pos }) {
  if (!node) return null;

  const verdict = VERDICT_CONFIG[node.verdict] ?? VERDICT_CONFIG.neutral;
  const typeLabel = NODE_TYPE_LABELS[node.node_type] ?? node.node_type;
  const topSource = node.top_sources?.[0];
  const sourceTypeLabel = SOURCE_TYPE_LABELS[topSource?.source_type] ?? topSource?.source_type;
  const sourceTypeColor = SOURCE_TYPE_COLORS[topSource?.source_type] ?? '#94a3b8';

  const OFFSET_X = 18;
  const OFFSET_Y = -12;

  // Keep tooltip inside viewport
  const left = Math.min(pos.x + OFFSET_X, window.innerWidth - 320);
  const top  = Math.max(pos.y + OFFSET_Y, 8);

  return (
    <div
      className="node-tooltip"
      style={{ left, top }}
    >
      {/* Type badge */}
      <div className="tooltip-type-row">
        <span
          className="tooltip-type-badge"
          style={{ background: node.color + '22', color: node.color, borderColor: node.color + '55' }}
        >
          {typeLabel}
        </span>
        {topSource && (
          <span
            className="tooltip-source-badge"
            style={{ color: sourceTypeColor, borderColor: sourceTypeColor + '44' }}
          >
            {sourceTypeLabel}
          </span>
        )}
      </div>

      {/* Title / snippet */}
      <p className="tooltip-text">
        {topSource?.title ?? node.text.slice(0, 90) + (node.text.length > 90 ? '…' : '')}
      </p>

      {topSource?.snippet && (
        <p className="tooltip-snippet">
          "{topSource.snippet.slice(0, 120)}{topSource.snippet.length > 120 ? '…' : ''}"
        </p>
      )}

      {/* Bottom row: verdict + confidence */}
      <div className="tooltip-bottom">
        <span
          className="tooltip-verdict"
          style={{
            color: verdict.color,
            background: verdict.bg,
            border: `1px solid ${verdict.border}`,
          }}
        >
          {verdict.label}
        </span>
        <span className="tooltip-conf">
          {Math.round(node.confidence * 100)}% confidence
        </span>
      </div>

      {node.best_source_url && (
        <div className="tooltip-hint">Click to open source ↗</div>
      )}
    </div>
  );
}
