import { useState } from 'react';
import { Search, Sparkles, Loader, GitBranch } from 'lucide-react';

const SUGGESTIONS = [
  'The Great Wall of China is visible from space.',
  'Vaccines cause autism.',
  'Barack Obama was the 44th President of the USA.',
  'Humans only use 10% of their brains.',
  'Amazon stock rose by 5% today.',
  'Einstein failed math in school.',
];

export default function LandingPage({ onVerify, loading = false }) {
  const [value, setValue] = useState('');

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (trimmed && !loading) onVerify(trimmed);
  };

  return (
    <div className="landing-page">
      <div className="landing-card">
        {/* Logo */}
        <div className="landing-logo">
          <div className="brand-icon landing-brand-icon">
            <GitBranch size={22} />
          </div>
          <span className="landing-title">FactGraph City</span>
        </div>

        <p className="landing-subtitle">
          Enter any claim and our AI pipeline will retrieve evidence,
          run NLI analysis, and build a 3D knowledge graph in seconds.
        </p>

        {/* Input */}
        <div className={`landing-input-wrapper ${loading ? 'landing-input-wrapper--loading' : ''}`}>
          {loading
            ? <Loader className="landing-input-icon landing-input-icon--spin" size={18} />
            : <Search className="landing-input-icon" size={18} />
          }
          <input
            className="landing-input"
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Enter a claim to fact-check…"
            spellCheck={false}
            autoFocus
            disabled={loading}
          />
          <button
            className={`landing-verify-btn ${loading ? 'landing-verify-btn--loading' : ''}`}
            onClick={handleSubmit}
            disabled={loading || !value.trim()}
          >
            {loading
              ? <><Loader size={15} className="btn-spin" />Verifying…</>
              : <><Sparkles size={15} />Verify Claim</>
            }
          </button>
        </div>

        {/* Suggestions */}
        <div className="landing-suggestions-label">Try these examples</div>
        <div className="landing-suggestions">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              className="landing-suggestion-chip"
              onClick={() => { setValue(s); }}
              disabled={loading}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Pipeline badges */}
        <div className="landing-pipeline">
          <span className="landing-pipeline-badge">DeBERTa NLI</span>
          <span className="landing-pipeline-sep">·</span>
          <span className="landing-pipeline-badge">Llama 3.1 (Groq)</span>
          <span className="landing-pipeline-sep">·</span>
          <span className="landing-pipeline-badge">6 Retrievers</span>
          <span className="landing-pipeline-sep">·</span>
          <span className="landing-pipeline-badge">3D Knowledge Graph</span>
        </div>
      </div>
    </div>
  );
}
