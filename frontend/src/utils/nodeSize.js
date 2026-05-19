// Maps backend node size values to Three.js sphere radii and physics vals.
// Backend constants: NODE_SIZE_MAIN_CLAIM=24, NODE_SIZE_DIRECT_EVIDENCE=10, NODE_SIZE_WEAK_EVIDENCE=7

/** Three.js sphere radius for a given backend size value. */
export function getNodeRadius(backendSize) {
  if (backendSize >= 20) return 5.0;  // main claim
  if (backendSize >= 9)  return 2.4;  // direct evidence
  return 1.6;                          // context / weak
}

/** D3 physics collision radius (val prop in react-force-graph-3d). */
export function getNodePhysicsVal(backendSize) {
  if (backendSize >= 20) return 40;
  if (backendSize >= 9)  return 14;
  return 7;
}

/** Glow halo multiplier relative to core radius. */
export function getGlowMultiplier(isMain) {
  return isMain ? 2.2 : 1.8;
}
