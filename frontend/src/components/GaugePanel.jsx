const CX = 100, CY = 90, R = 72, SW = 14, NEEDLE_R = R - 6;

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

export default function GaugePanel({ value, label = 'Confidence', bars = [], color }) {
  const clamped = Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(clamped * 100);

  // 0% = 180°, 100% = 0°
  const needleDeg = (1 - clamped) * 180;
  const [nx, ny] = pt(NEEDLE_R, needleDeg);

  // Prefer the caller's verdict/edge-type color (matches the badge shown
  // above the gauge) so a high-confidence REJECTED claim reads as red, not
  // green — green/amber/red by raw magnitude alone contradicted the
  // verdict badge whenever confidence was high but the verdict was
  // "rejected". Falls back to a magnitude-based scale only when no
  // verdict color is given.
  const zoneColor = color ?? (
    clamped < 0.4 ? '#ef4444' :
    clamped < 0.65 ? '#f59e0b' : '#22c55e'
  );

  return (
    <div className="gauge-panel">
      <svg viewBox="0 0 200 130" className="gauge-svg">
        {/* Background track */}
        <path d={arcD(180, 0)} fill="none" stroke="rgba(255,255,255,0.13)" strokeWidth={SW} strokeLinecap="round" />

        {/* Filled arc up to value — butt caps avoid the "blob" artifact at low/high values */}
        {clamped > 0.005 && (
          <path
            d={arcD(180, needleDeg)}
            fill="none"
            stroke={zoneColor}
            strokeWidth={SW}
            strokeLinecap="butt"
          />
        )}

        {/* Needle — reaches almost to the track for a connected look */}
        <line x1={CX} y1={CY} x2={nx.toFixed(2)} y2={ny.toFixed(2)}
          stroke="white" strokeWidth={2} strokeLinecap="round" opacity={0.9} />
        <circle cx={CX} cy={CY} r={4.5} fill="white" />
        <circle cx={CX} cy={CY} r={2.2} fill={zoneColor} />

        {/* Percentage text */}
        <text x={CX} y={CY + 26} textAnchor="middle" fontSize="24"
          fontWeight="700" fill="white" fontFamily="inherit">{pct}%</text>

        {/* Zone corner labels */}
        <text x="14" y="106" fontSize="8" fill="rgba(239,68,68,0.6)"  textAnchor="middle">Low</text>
        <text x="186" y="106" fontSize="8" fill="rgba(34,197,94,0.6)" textAnchor="middle">High</text>
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
