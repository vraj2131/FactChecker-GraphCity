import { useState } from 'react';
import { Search, Sparkles, Loader } from 'lucide-react';

const DEMO_CLAIMS = [
  'The Great Wall of China is visible from space.',
  'Vaccines cause autism.',
  'Barack Obama was the 44th President of the USA.',
  'Humans only use 10% of their brains.',
  'Amazon stock rose by 5% today.',
];

export default function ClaimInputPanel({ onVerify, loading = false }) {
  const [value, setValue] = useState(DEMO_CLAIMS[0]);
  const [focused, setFocused] = useState(false);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (trimmed && !loading) onVerify?.(trimmed);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSubmit();
  };

  return (
    <div className="claim-input-panel">
      <div className={`claim-input-wrapper ${focused ? 'claim-input-wrapper--focused' : ''} ${loading ? 'claim-input-wrapper--loading' : ''}`}>
        {loading
          ? <Loader className="claim-input-icon claim-input-icon--spin" size={16} />
          : <Search className="claim-input-icon" size={16} />
        }
        <input
          type="text"
          className="claim-input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder="Enter a claim to fact-check…"
          spellCheck={false}
          disabled={loading}
        />
        <button
          className={`claim-verify-btn ${loading ? 'claim-verify-btn--loading' : ''}`}
          onClick={handleSubmit}
          disabled={loading || !value.trim()}
        >
          {loading
            ? <><Loader size={14} className="btn-spin" />Verifying…</>
            : <><Sparkles size={14} />Verify</>
          }
        </button>
      </div>

      {/* Sample claims quick-select */}
      <div className="sample-claims">
        {DEMO_CLAIMS.slice(1).map((claim) => (
          <button
            key={claim}
            className="sample-claim-chip"
            onClick={() => { setValue(claim); }}
            disabled={loading}
          >
            {claim.length > 40 ? claim.slice(0, 40) + '…' : claim}
          </button>
        ))}
      </div>
    </div>
  );
}
