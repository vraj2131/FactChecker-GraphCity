import { useState, useEffect } from 'react';
import { Search, Sparkles, Loader, ChevronDown, ChevronUp } from 'lucide-react';

const DEMO_CLAIMS = [
  'Vaccines cause autism.',
  'Barack Obama was the 44th President of the USA.',
  'Humans only use 10% of their brains.',
  'Amazon stock rose by 5% today.',
];

const SOURCE_GROUPS = [
  { key: 'wikipedia',  label: 'Wikipedia',          desc: 'FAISS index + live Wikipedia',      alwaysOn: true  },
  { key: 'live_news',  label: 'Live News',           desc: 'Guardian · NewsAPI · GDELT'                         },
  { key: 'factcheck',  label: 'Fact-Checkers',       desc: 'Professional fact-check sources'                    },
  { key: 'scientific', label: 'Scientific',          desc: 'OpenAlex · PubMed · arXiv'                         },
  { key: 'financial',  label: 'Financial / Crypto',  desc: 'FRED · SEC · CoinGecko · World Bank'               },
  { key: 'web_search', label: 'Web Search',          desc: 'DuckDuckGo — may slow results'                      },
  { key: 'social',     label: 'Social Media',        desc: 'Reddit + Bluesky'                                   },
];

const DEFAULT_GROUPS = ['wikipedia', 'live_news', 'factcheck', 'scientific', 'financial'];

export default function ClaimInputPanel({ onVerify, loading = false, claimText = '' }) {
  const [value, setValue] = useState(claimText);
  const [focused, setFocused] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [enabledGroups, setEnabledGroups] = useState(new Set(DEFAULT_GROUPS));

  useEffect(() => {
    if (claimText) setValue(claimText);
  }, [claimText]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (trimmed && !loading) onVerify?.(trimmed, [...enabledGroups]);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSubmit();
  };

  const toggleGroup = (key) => {
    setEnabledGroups(prev => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
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

      {/* Sources toggle */}
      <div className="sources-row">
        <button
          className="sources-toggle-btn"
          onClick={() => setSourcesOpen(v => !v)}
          disabled={loading}
        >
          {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          Sources
          <span className="sources-active-count">{enabledGroups.size} / {SOURCE_GROUPS.length}</span>
        </button>

        {sourcesOpen && (
          <div className="sources-panel">
            {SOURCE_GROUPS.map(({ key, label, desc, alwaysOn }) => {
              const on = alwaysOn || enabledGroups.has(key);
              return (
                <label
                  key={key}
                  className={`source-group-row ${alwaysOn ? 'source-group-row--locked' : ''}`}
                  title={alwaysOn ? 'Always enabled' : desc}
                >
                  <input
                    type="checkbox"
                    className="source-group-check"
                    checked={on}
                    disabled={alwaysOn}
                    onChange={() => !alwaysOn && toggleGroup(key)}
                  />
                  <span className="source-group-label">{label}</span>
                  <span className="source-group-desc">{desc}</span>
                </label>
              );
            })}
          </div>
        )}
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
