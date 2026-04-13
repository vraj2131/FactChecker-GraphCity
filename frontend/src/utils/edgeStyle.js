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
