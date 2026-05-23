const CX = 100, CY = 90, R = 72, SW = 16, NEEDLE_R = 56;

function pt(r, deg) {
  const rad = (deg * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY - r * Math.sin(rad)];
}

// Arc path going counterclockwise (over the top) from fromDeg → toDeg
function arcD(fromDeg, toDeg) {
  const [x1, y1] = pt(R, fromDeg);
  const [x2, y2] = pt(R, toDeg);
  const span = fromDeg - toDeg;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 ${span >= 180 ? 1 : 0} 0 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export default function GaugePanel({ value, label = 'Confidence', bars = [] }) {
  const clamped = Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(clamped * 100);

  // 0% = 180°, 100% = 0°
  const needleDeg = (1 - clamped) * 180;
  const [nx, ny] = pt(NEEDLE_R, needleDeg);

  const zoneColor =
    clamped < 0.4 ? '#ef4444' :
    clamped < 0.65 ? '#f59e0b' : '#22c55e';

  return (
    <div className="gauge-panel">
      <svg viewBox="0 0 200 130" className="gauge-svg">
        {/* Background track */}
        <path d={arcD(180, 0)} fill="none" stroke="rgba(255,255,255,0.13)" strokeWidth={SW} strokeLinecap="round" />

        {/* Zone boundary ticks at 40% (108°) and 65% (63°) */}
        {[108, 63].map((deg) => {
          const [ix, iy] = pt(R - SW / 2 - 1, deg);
          const [ox, oy] = pt(R + SW / 2 + 1, deg);
          return <line key={deg} x1={ix.toFixed(2)} y1={iy.toFixed(2)}
            x2={ox.toFixed(2)} y2={oy.toFixed(2)}
            stroke="rgba(255,255,255,0.2)" strokeWidth={1.5} />;
        })}

        {/* Filled arc up to value */}
        {clamped > 0.005 && (
          <path
            d={arcD(180, Math.max(needleDeg, 0.5))}
            fill="none"
            stroke={zoneColor}
            strokeWidth={SW}
            strokeLinecap="round"
          />
        )}

        {/* Needle */}
        <line x1={CX} y1={CY} x2={nx.toFixed(2)} y2={ny.toFixed(2)}
          stroke="white" strokeWidth={2.5} strokeLinecap="round" />
        <circle cx={CX} cy={CY} r={5} fill="white" />
        <circle cx={CX} cy={CY} r={2.5} fill={zoneColor} />

        {/* Percentage text only — label moved outside SVG */}
        <text x={CX} y={CY + 22} textAnchor="middle" fontSize="22"
          fontWeight="700" fill="white" fontFamily="inherit">{pct}%</text>

        {/* Zone corner labels */}
        <text x="16" y="108" fontSize="8" fill="rgba(239,68,68,0.55)"  textAnchor="middle">Low</text>
        <text x="184" y="108" fontSize="8" fill="rgba(34,197,94,0.55)" textAnchor="middle">High</text>
      </svg>

      {/* Label outside SVG — always clearly readable */}
      <p className="gauge-label">{label}</p>

      {bars.length > 0 && (
        <div className="gauge-bars">
          {bars.map(({ label: bl, value: bv, color }) => (
            <div key={bl} className="gauge-bar-row">
              <span className="gauge-bar-label">{bl}</span>
              <div className="gauge-bar-track">
                <div className="gauge-bar-fill" style={{ width: `${Math.round(bv * 100)}%`, background: color }} />
              </div>
              <span className="gauge-bar-pct">{Math.round(bv * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
