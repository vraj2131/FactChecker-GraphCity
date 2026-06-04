import { X } from 'lucide-react';
import { VERDICT_CONFIG } from '../utils/colorMap';

export default function ContextChainBanner({ history, onClear }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="context-chain-banner">
      <span className="context-chain-label">Context chain:</span>
      <div className="context-chain-pills">
        {history.map((item, i) => {
          const cfg = VERDICT_CONFIG[item.verdict] ?? VERDICT_CONFIG.not_enough_info;
          const shortClaim = item.claim_text.length > 40
            ? item.claim_text.slice(0, 40) + '…'
            : item.claim_text;
          return (
            <span
              key={i}
              className="context-chain-pill"
              style={{ borderColor: cfg.border, color: cfg.color, background: cfg.bg }}
              title={item.claim_text}
            >
              <span
                className="context-chain-dot"
                style={{ background: cfg.color }}
              />
              {shortClaim}
              <span className="context-chain-conf">
                {Math.round(item.confidence * 100)}%
              </span>
            </span>
          );
        })}
      </div>
      <button className="context-chain-clear" onClick={onClear} title="Clear context chain">
        <X size={12} />
        Clear
      </button>
    </div>
  );
}
