import { useEffect, useRef, useState } from 'react';
import { Search, BrainCircuit, Network, Check } from 'lucide-react';

const STEPS = [
  { icon: Search,       label: 'Retrieving Evidence', sub: 'Wikipedia · news · fact-check APIs' },
  { icon: BrainCircuit, label: 'Analysing Stance',    sub: 'DeBERTa NLI + Llama 3.1 classification' },
  { icon: Network,      label: 'Building Graph',      sub: 'Confidence scoring · evidence map' },
];

// The backend is a single request with no progress stream, so steps advance
// on timings that mirror the real pipeline phases (retrieval dominates).
// The final step holds until the response actually arrives.
const STEP_ADVANCE_MS = [7000, 6000];

const RING_R = 62;
const RING_C = 2 * Math.PI * RING_R;

export default function VerifyProgressOverlay({ claimText = '' }) {
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0); // 0..1 ring fill
  const stepRef = useRef(0);

  // Advance through steps on phase timers
  useEffect(() => {
    const timers = STEP_ADVANCE_MS.map((_, i) =>
      setTimeout(() => {
        stepRef.current = i + 1;
        setStep(i + 1);
      }, STEP_ADVANCE_MS.slice(0, i + 1).reduce((a, b) => a + b, 0))
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  // Ring eases toward the current step's ceiling so it never stalls dead
  // or falsely hits 100% before the response lands.
  useEffect(() => {
    const id = setInterval(() => {
      setProgress((p) => {
        const ceiling = (stepRef.current + 1) / STEPS.length - 0.04;
        return p + (ceiling - p) * 0.045;
      });
    }, 120);
    return () => clearInterval(id);
  }, []);

  const pct = Math.round(progress * 100);
  const dashOffset = RING_C * (1 - progress);

  return (
    <div className="verify-overlay">
      <div className="verify-overlay-card">
        {/* Big progress wheel */}
        <div className="verify-wheel">
          <svg viewBox="0 0 160 160" className="verify-wheel-svg">
            <circle cx="80" cy="80" r={RING_R} fill="none"
              stroke="rgba(255,255,255,0.08)" strokeWidth="9" />
            <circle cx="80" cy="80" r={RING_R} fill="none"
              stroke="url(#verifyGrad)" strokeWidth="9" strokeLinecap="round"
              strokeDasharray={RING_C} strokeDashoffset={dashOffset}
              transform="rotate(-90 80 80)"
              style={{ transition: 'stroke-dashoffset 0.25s linear' }} />
            <defs>
              <linearGradient id="verifyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#818cf8" />
                <stop offset="100%" stopColor="#6d28d9" />
              </linearGradient>
            </defs>
          </svg>
          <div className="verify-wheel-center">
            <span className="verify-wheel-pct">{pct}%</span>
            <span className="verify-wheel-step">Step {Math.min(step + 1, 3)} / 3</span>
          </div>
          <div className="verify-wheel-spinner" />
        </div>

        {claimText && (
          <p className="verify-overlay-claim">"{claimText.length > 90 ? claimText.slice(0, 90) + '…' : claimText}"</p>
        )}

        {/* Step list */}
        <div className="verify-steps">
          {STEPS.map(({ icon: Icon, label, sub }, i) => {
            const state = i < step ? 'done' : i === step ? 'active' : 'pending';
            return (
              <div key={label} className={`verify-step verify-step--${state}`}>
                <div className="verify-step-icon">
                  {state === 'done' ? <Check size={15} /> : <Icon size={15} />}
                </div>
                <div className="verify-step-text">
                  <span className="verify-step-label">{label}</span>
                  <span className="verify-step-sub">{sub}</span>
                </div>
                {state === 'active' && <span className="verify-step-pulse" />}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
