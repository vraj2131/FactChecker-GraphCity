import { getNodePhysicsVal } from './nodeSize';

function hexToHsl(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s;
  const l = (max + min) / 2;
  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4;
    }
    h /= 6;
  }
  return [h * 360, s * 100, l * 100];
}

function hslToHex(h, s, l) {
  h /= 360; s /= 100; l /= 100;
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return '#' + [r, g, b].map(x => Math.round(x * 255).toString(16).padStart(2, '0')).join('');
}

function rankAdjustedColor(hex, rank) {
  if (!rank || !hex || !hex.startsWith('#') || hex.length < 7) return hex;
  const [h, s, l] = hexToHsl(hex);
  if (rank <= 5) {
    // Top tier: noticeably brighter and more saturated
    return hslToHex(h, Math.min(s * 1.5, 100), Math.min(l * 1.6, 90));
  }
  if (rank >= 11) {
    // Lower tier: clearly washed out
    return hslToHex(h, s * 0.5, l);
  }
  return hex;
}

/**
 * Converts a GraphResponse JSON (backend schema) into the format
 * expected by react-force-graph-3d:
 *   { nodes: [{ id, ...rest }], links: [{ source, target, ...rest }] }
 *
 * The original node and edge data is preserved so components can access
 * all fields (color, size, confidence, top_sources, etc.) directly.
 */
export function transformToForceGraph(graphJson) {
  if (!graphJson?.nodes || !graphJson?.edges) {
    return { nodes: [], links: [] };
  }

  const nodes = graphJson.nodes.map((node) => ({
    ...node,
    id: node.node_id,                          // required by react-force-graph-3d
    val: getNodePhysicsVal(node.size),          // physics collision size
    // __size and __isMain used by nodeThreeObject in GraphCanvas
    __size: node.size,
    __isMain: node.is_main_claim,
    color: node.is_main_claim ? node.color : rankAdjustedColor(node.color, node.rank),
  }));

  const links = graphJson.edges.map((edge) => ({
    ...edge,
    // source/target already match node_id values
  }));

  return { nodes, links };
}
