import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function ErrorState({ message = 'Something went wrong.', onRetry }) {
  return (
    <div className="error-overlay">
      <div className="error-card">
        <div className="error-icon">
          <AlertTriangle size={32} />
        </div>
        <h3 className="error-title">Pipeline Error</h3>
        <p className="error-message">{message}</p>
        {onRetry && (
          <button className="error-retry-btn" onClick={onRetry}>
            <RefreshCw size={14} />
            Retry
          </button>
        )}
      </div>
    </div>
  );
}
