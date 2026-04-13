// Maps backend node size values to Three.js sphere radii and physics vals.
// Backend constants: NODE_SIZE_MAIN_CLAIM=24, NODE_SIZE_DIRECT_EVIDENCE=10, NODE_SIZE_WEAK_EVIDENCE=7

/** Three.js sphere radius for a given backend size value. */
export function getNodeRadius(backendSize) {
  if (backendSize >= 20) return 6.5;  // main claim
  if (backendSize >= 9)  return 3.2;  // direct evidence
  return 2.2;                          // context / weak
}

/** D3 physics collision radius (val prop in react-force-graph-3d). */
export function getNodePhysicsVal(backendSize) {
  if (backendSize >= 20) return 60;
  if (backendSize >= 9)  return 20;
  return 10;
}

/** Glow halo multiplier relative to core radius. */
export function getGlowMultiplier(isMain) {
  return isMain ? 2.4 : 2.0;
}
