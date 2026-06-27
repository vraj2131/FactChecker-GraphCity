import { X } from 'lucide-react';
import { VERDICT_CONFIG } from '../utils/colorMap';

// Feature 15: each pill is a clickable claim — revisits that claim in the
// chain (instant if already cached in this session, see App.jsx).
export default function ContextChainBanner({ history, currentClaim, onSelect, onClear }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="context-chain-banner">
      <span className="context-chain-label">Claim chain:</span>
      <div className="context-chain-pills">
        {history.map((item) => {
          const cfg = VERDICT_CONFIG[item.verdict] ?? VERDICT_CONFIG.not_enough_info;
          const shortClaim = item.claim_text.length > 40
            ? item.claim_text.slice(0, 40) + '…'
            : item.claim_text;
          const isCurrent = item.claim_text === currentClaim;
          return (
            <button
              key={item.claim_text}
              type="button"
              className={`context-chain-pill${isCurrent ? ' context-chain-pill--active' : ''}`}
              style={{ borderColor: cfg.border, color: cfg.color, background: cfg.bg }}
              title={isCurrent ? `${item.claim_text} (current)` : `Revisit: ${item.claim_text}`}
              onClick={() => !isCurrent && onSelect?.(item.claim_text)}
              disabled={isCurrent}
            >
              <span
                className="context-chain-dot"
                style={{ background: cfg.color }}
              />
              {shortClaim}
              <span className="context-chain-conf">
                {Math.round(item.confidence * 100)}%
              </span>
            </button>
          );
        })}
      </div>
      <button className="context-chain-clear" onClick={onClear} title="Clear claim chain">
        <X size={12} />
        Clear
      </button>
    </div>
  );
}
