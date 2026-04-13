import { useState } from 'react';
import { Search, Sparkles } from 'lucide-react';

const DEMO_CLAIMS = [
  'The Great Wall of China is visible from space.',
  'Vaccines cause autism.',
  'Barack Obama was the 44th President of the USA.',
  'Humans only use 10% of their brains.',
  'Amazon stock rose by 5% today.',
];

export default function ClaimInputPanel() {
  const [value, setValue] = useState(DEMO_CLAIMS[0]);
  const [focused, setFocused] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="claim-input-panel">
      <div className={`claim-input-wrapper ${focused ? 'claim-input-wrapper--focused' : ''}`}>
        <Search className="claim-input-icon" size={16} />
        <input
          type="text"
          className="claim-input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Enter a claim to fact-check…"
          spellCheck={false}
        />
        <div className="claim-btn-wrapper">
          <button
            className="claim-verify-btn"
            onMouseEnter={() => setShowTooltip(true)}
            onMouseLeave={() => setShowTooltip(false)}
            onClick={() => setShowTooltip(true)}
          >
            <Sparkles size={14} />
            Verify
          </button>
          {showTooltip && (
            <div className="claim-coming-tooltip">
              API integration in Phase 13
            </div>
          )}
        </div>
      </div>

      {/* Sample claims quick-select */}
      <div className="sample-claims">
        {DEMO_CLAIMS.slice(1).map((claim) => (
          <button
            key={claim}
            className="sample-claim-chip"
            onClick={() => setValue(claim)}
          >
            {claim.length > 40 ? claim.slice(0, 40) + '…' : claim}
          </button>
        ))}
      </div>
    </div>
  );
}
