// Maps edge types to visual properties for the 3D graph.

export const EDGE_STYLES = {
  supports: {
    particleCount: 3,
    particleSpeed: 0.006,
    opacity: 0.85,
    arrowLength: 5,
  },
  refutes: {
    particleCount: 3,
    particleSpeed: 0.006,
    opacity: 0.85,
    arrowLength: 5,
  },
  correlated: {
    particleCount: 1,
    particleSpeed: 0.003,
    opacity: 0.5,
    arrowLength: 3,
  },
  insufficient: {
    particleCount: 0,
    particleSpeed: 0,
    opacity: 0.3,
    arrowLength: 2,
  },
  // fallback
  default: {
    particleCount: 1,
    particleSpeed: 0.004,
    opacity: 0.6,
    arrowLength: 3,
  },
};

export function getEdgeStyle(edgeType) {
  return EDGE_STYLES[edgeType] ?? EDGE_STYLES.default;
}

// Feature 14: relation labels that are mutual/symmetric between two nodes —
// an arrow on these would imply a one-directional relationship that
// doesn't exist (A corroborates B is equally "B corroborates A").
// `edge_type` alone can't distinguish these (e.g. both `corroborates` and
// `provides_context` map to edge_type "correlated"), so this is keyed on
// the more specific `label` field instead.
const SYMMETRIC_RELATION_LABELS = new Set([
  'corroborates',
  'contradicts',
  'shared_topic',
]);

/**
 * Arrow length for a link — 0 (no arrowhead) for symmetric/mutual
 * relations, otherwise the directional length from EDGE_STYLES.
 */
export function getArrowLength(link) {
  if (SYMMETRIC_RELATION_LABELS.has(link.label)) return 0;
  return getEdgeStyle(link.edge_type).arrowLength;
}
