import { Filter } from 'lucide-react';

const FILTERS = [
  { key: null,          label: 'All',      color: '#7e8fa8' },
  { key: 'supports',    label: 'Support',  color: '#3b82f6' },
  { key: 'refutes',     label: 'Refute',   color: '#f97316' },
  { key: 'correlated',  label: 'Context',  color: '#818cf8' },
];

export default function FilterPanel({ activeFilter, onFilter }) {
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
    </div>
  );
}
