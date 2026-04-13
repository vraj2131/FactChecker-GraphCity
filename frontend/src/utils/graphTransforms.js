import { getNodePhysicsVal } from './nodeSize';

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
  }));

  const links = graphJson.edges.map((edge) => ({
    ...edge,
    // source/target already match node_id values
  }));

  return { nodes, links };
}
