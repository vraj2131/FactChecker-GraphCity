import { Filter } from 'lucide-react';

const FILTERS = [
  { key: null,          label: 'All',      color: '#7e8fa8' },
  { key: 'supports',    label: 'Support',  color: '#3b82f6' },
  { key: 'refutes',     label: 'Refute',   color: '#f97316' },
  { key: 'correlated',  label: 'Context',  color: '#818cf8' },
];

export default function FilterPanel({ activeFilter, onFilter, includeSocial, onToggleSocial }) {
  return (
    <div className="filter-panel">
      <span className="filter-label">
        <Filter size={11} />
        Filter
      </span>
      {FILTERS.map(({ key, label, color }) => {
        const active = activeFilter === key;
        return (
          <button
            key={label}
            className={`filter-btn ${active ? 'filter-btn--active' : ''}`}
            style={active
              ? { color, borderColor: color, background: color + '20' }
              : {}
            }
            onClick={() => onFilter(key)}
          >
            {active && <span className="filter-btn-dot" style={{ background: color }} />}
            {label}
          </button>
        );
      })}

      <div className="social-toggle" title="Include Reddit + Bluesky as evidence sources">
        <span className="social-toggle-label">Social</span>
        <button
          className={`social-toggle-btn ${includeSocial ? 'social-toggle-btn--on' : ''}`}
          onClick={onToggleSocial}
          aria-pressed={includeSocial}
        >
          <span className="social-toggle-thumb" />
        </button>
      </div>
    </div>
  );
}
