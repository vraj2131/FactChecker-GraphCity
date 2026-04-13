// Mirrors backend/app/utils/constants.py color definitions exactly.

export const NODE_COLORS = {
  main_claim_verified:   '#2E7D32',
  main_claim_rejected:   '#C62828',
  main_claim_nei:        '#757575',
  direct_support:        '#1565C0',
  direct_refute:         '#BF360C',
  factcheck_review:      '#6A1B9A',
  context_signal:        '#4A148C',
  insufficient_evidence: '#424242',
};

export const EDGE_COLORS = {
  supports:    '#1E88E5',
  refutes:     '#FB8C00',
  correlated:  '#8E24AA',
  insufficient:'#BDBDBD',
};

// Human-readable display labels
export const NODE_TYPE_LABELS = {
  main_claim:            'Main Claim',
  direct_support:        'Direct Support',
  direct_refute:         'Direct Refute',
  factcheck_review:      'Fact-Check Review',
  context_signal:        'Context Signal',
  insufficient_evidence: 'Insufficient Evidence',
};

export const EDGE_TYPE_LABELS = {
  supports:    'Supports',
  refutes:     'Refutes',
  correlated:  'Correlated Context',
  insufficient:'Insufficient',
};

// Verdict → display config
export const VERDICT_CONFIG = {
  verified: {
    label: 'VERIFIED',
    color: '#22c55e',
    bg: 'rgba(34,197,94,0.12)',
    border: 'rgba(34,197,94,0.35)',
  },
  rejected: {
    label: 'REJECTED',
    color: '#ef4444',
    bg: 'rgba(239,68,68,0.12)',
    border: 'rgba(239,68,68,0.35)',
  },
  not_enough_info: {
    label: 'INCONCLUSIVE',
    color: '#94a3b8',
    bg: 'rgba(148,163,184,0.10)',
    border: 'rgba(148,163,184,0.25)',
  },
  supports: {
    label: 'SUPPORTS',
    color: '#3b82f6',
    bg: 'rgba(59,130,246,0.12)',
    border: 'rgba(59,130,246,0.35)',
  },
  refutes: {
    label: 'REFUTES',
    color: '#f97316',
    bg: 'rgba(249,115,22,0.12)',
    border: 'rgba(249,115,22,0.35)',
  },
  correlated: {
    label: 'CORRELATED',
    color: '#a855f7',
    bg: 'rgba(168,85,247,0.12)',
    border: 'rgba(168,85,247,0.30)',
  },
  insufficient: {
    label: 'INSUFFICIENT',
    color: '#6b7280',
    bg: 'rgba(107,114,128,0.10)',
    border: 'rgba(107,114,128,0.25)',
  },
  neutral: {
    label: 'NEUTRAL',
    color: '#94a3b8',
    bg: 'rgba(148,163,184,0.10)',
    border: 'rgba(148,163,184,0.25)',
  },
};

export const SOURCE_TYPE_LABELS = {
  wikipedia: 'Wikipedia',
  livewiki:  'Live Wiki',
  factcheck: 'FactCheck',
  guardian:  'The Guardian',
  newsapi:   'NewsAPI',
  gdelt:     'GDELT',
  other:     'Other',
};

export const SOURCE_TYPE_COLORS = {
  wikipedia: '#94a3b8',
  livewiki:  '#38bdf8',
  factcheck: '#c084fc',
  guardian:  '#34d399',
  newsapi:   '#fb923c',
  gdelt:     '#facc15',
  other:     '#64748b',
};

/** Return the correct color for a node based on type + verdict. */
export function getNodeColor(node) {
  if (node.is_main_claim) {
    const v = node.verdict?.toLowerCase();
    if (v === 'verified') return NODE_COLORS.main_claim_verified;
    if (v === 'rejected') return NODE_COLORS.main_claim_rejected;
    return NODE_COLORS.main_claim_nei;
  }
  return NODE_COLORS[node.node_type] ?? '#424242';
}
