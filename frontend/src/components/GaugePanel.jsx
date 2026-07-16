const CX = 100, CY = 82, R = 64, SW = 13;

function pt(r, deg) {
  const rad = (deg * Math.PI) / 180;
  return [CX + r * Math.cos(rad), CY - r * Math.sin(rad)];
}

// Arc from fromDeg → toDeg over the top of the gauge.
// Sweep flag MUST be 1 (screen-clockwise): with 0, the full 180° track
// renders as the BOTTOM semicircle and partial fills bulge off-center,
// so the grey track and colored fill never coincide.
function arcD(fromDeg, toDeg) {
  const [x1, y1] = pt(R, fromDeg);
  const [x2, y2] = pt(R, toDeg);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export default function GaugePanel({ value, label = 'Confidence', bars = [], color }) {
  const clamped = Math.max(0, Math.min(1, value ?? 0));
  const pct = Math.round(clamped * 100);

  // 0% = 180° (left), 100% = 0° (right)
  const needleDeg = (1 - clamped) * 180;

  // Diamond needle: tip → b1 → tail → b2
  // Widest at center pivot so the hub cap sits cleanly on top.
  const rad = (needleDeg * Math.PI) / 180;
  const dir  = [ Math.cos(rad), -Math.sin(rad)];
  const perp = [ Math.sin(rad),  Math.cos(rad)];
  const TIP_R  = R - SW / 2 - 2;
  const TAIL_R = 9;
  const HALF_W = 3;
  const tip  = [CX + TIP_R  * dir[0],  CY + TIP_R  * dir[1]];
  const tail = [CX - TAIL_R * dir[0],  CY - TAIL_R * dir[1]];
  const b1   = [CX + HALF_W * perp[0], CY + HALF_W * perp[1]];
  const b2   = [CX - HALF_W * perp[0], CY - HALF_W * perp[1]];
  const needlePts = [tip, b1, tail, b2].map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(' ');

  const zoneColor = color ?? (
    clamped < 0.4 ? '#ef4444' :
    clamped < 0.65 ? '#f59e0b' : '#22c55e'
  );

  return (
    <div className="gauge-panel">
      <svg viewBox="0 0 200 126" className="gauge-svg">
        {/*
          Both arcs use strokeLinecap="butt" so their endpoints coincide
          exactly. "round" caps add a 6.5px half-circle beyond each endpoint
          on the background track, causing the grey to bleed past the green.
        */}
        <path
          d={arcD(180, 0)}
          fill="none"
          stroke="rgba(255,255,255,0.12)"
          strokeWidth={SW}
          strokeLinecap="butt"
        />

        {clamped > 0.005 && (
          <path
            d={arcD(180, needleDeg)}
            fill="none"
            stroke={zoneColor}
            strokeWidth={SW}
            strokeLinecap="butt"
          />
        )}

        {/* Needle drawn first; hub circles cap it on top */}
        <polygon points={needlePts} fill="white" opacity={0.92} />
        <circle cx={CX} cy={CY} r={4.5} fill="white" />
        <circle cx={CX} cy={CY} r={2.5} fill={zoneColor} />

        {/* Percentage readout */}
        <text
          x={CX} y={CY + 28}
          textAnchor="middle"
          fontSize="24" fontWeight="700"
          fill="white" fontFamily="inherit"
        >{pct}%</text>

        {/* Low / High corner labels */}
        <text x="12"  y="98" fontSize="8" fill="rgba(239,68,68,0.55)"  textAnchor="middle">Low</text>
        <text x="188" y="98" fontSize="8" fill="rgba(34,197,94,0.55)"  textAnchor="middle">High</text>
      </svg>

      <p className="gauge-label">{label}</p>

      {bars.length > 0 && (
        <div className="gauge-bars">
          {bars.map(({ label: bl, value: bv, color: bc }) => (
            <div key={bl} className="gauge-bar-row">
              <span className="gauge-bar-label">{bl}</span>
              <div className="gauge-bar-track">
                <div className="gauge-bar-fill" style={{ width: `${Math.round(bv * 100)}%`, background: bc }} />
              </div>
              <span className="gauge-bar-pct">{Math.round(bv * 100)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
