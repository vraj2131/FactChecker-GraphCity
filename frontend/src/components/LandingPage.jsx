import { useState } from 'react';
import { Search, Sparkles, Loader, GitBranch, ChevronDown, ChevronUp } from 'lucide-react';
import { SOURCE_GROUPS, DEFAULT_GROUPS, isDefaultSelection } from '../utils/sourceGroups';

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
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [enabledGroups, setEnabledGroups] = useState(new Set(DEFAULT_GROUPS));
  const [deepNli, setDeepNli] = useState(true);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || loading) return;
    // null when defaults are untouched, so App.jsx's cache fast-path works
    onVerify(trimmed, isDefaultSelection(enabledGroups) ? null : [...enabledGroups], deepNli);
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

        {/* Source group selector — same groups as the top-bar panel */}
        <div className="sources-row landing-sources-row">
          <div className="sources-controls-line">
            <button
              className="sources-toggle-btn"
              onClick={() => setSourcesOpen(v => !v)}
              disabled={loading}
            >
              {sourcesOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              Sources
              <span className="sources-active-count">{enabledGroups.size} / {SOURCE_GROUPS.length}</span>
            </button>
            <label
              className="deep-nli-toggle"
              title="Re-check borderline stance labels with a stronger NLI model (more accurate, a few seconds slower)"
            >
              <input
                type="checkbox"
                className="source-group-check"
                checked={deepNli}
                disabled={loading}
                onChange={() => setDeepNli(v => !v)}
              />
              Deep NLI
            </label>
          </div>

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
          <span className="landing-pipeline-badge">14+ Retrievers</span>
          <span className="landing-pipeline-sep">·</span>
          <span className="landing-pipeline-badge">3D Knowledge Graph</span>
        </div>
      </div>
    </div>
  );
}
