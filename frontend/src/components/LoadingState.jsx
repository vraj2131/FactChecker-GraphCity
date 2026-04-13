export default function LoadingState({ message = 'Analysing claim…' }) {
  return (
    <div className="loading-overlay">
      <div className="loading-card">
        <div className="loading-spinner">
          <div className="spinner-ring spinner-ring--outer" />
          <div className="spinner-ring spinner-ring--inner" />
          <div className="spinner-core" />
        </div>
        <p className="loading-message">{message}</p>
        <div className="loading-steps">
          <span className="loading-step loading-step--active">Retrieving sources</span>
          <span className="loading-step">NLI classification</span>
          <span className="loading-step">LLM reasoning</span>
          <span className="loading-step">Building graph</span>
        </div>
      </div>
    </div>
  );
}
