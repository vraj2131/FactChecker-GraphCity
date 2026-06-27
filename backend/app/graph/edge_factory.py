"""
Phase 10: Edge Factory.

Builds Edge objects connecting the main claim node to each evidence node.
Edge type, color, width, and dashed style are all derived from the LLM
classification and edge confidence score.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

from backend.app.models.llm_model import LLMResult, NodeLink, SourceClassification
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.edge_schema import Edge
from backend.app.schemas.node_schema import Node
from backend.app.schemas.source_schema import Source
from backend.app.services.confidence_service import ConfidenceService
from backend.app.utils.constants import (
    EDGE_COLOR_CORRELATED,
    EDGE_COLOR_INSUFFICIENT,
    EDGE_COLOR_REFUTES,
    EDGE_COLOR_SUPPORTS,
    EDGE_WIDTH_MAX,
    EDGE_WIDTH_MIN,
    NODE_LINK_MAX_PAIRS,
    NODE_SIMILARITY_JACCARD_THRESHOLD,
    NODE_SIMILARITY_MAX_EDGES,
)

logger = logging.getLogger(__name__)

# LLM classification → edge_type
_CLASSIFICATION_TO_EDGE_TYPE: Dict[str, str] = {
    "direct_support":     "supports",
    "direct_refute":      "refutes",
    "correlated_context": "correlated",
    "insufficient":       "insufficient",
}

# edge_type → hex color
_EDGE_TYPE_TO_COLOR: Dict[str, str] = {
    "supports":     EDGE_COLOR_SUPPORTS,
    "refutes":      EDGE_COLOR_REFUTES,
    "correlated":   EDGE_COLOR_CORRELATED,
    "insufficient": EDGE_COLOR_INSUFFICIENT,
}

# edge_type → dashed (correlated and insufficient are rendered dashed)
_EDGE_TYPE_DASHED: Dict[str, bool] = {
    "supports":     False,
    "refutes":      False,
    "correlated":   True,
    "insufficient": True,
}


def _edge_width(edge_confidence: float, relevance: float = 1.0) -> float:
    """Scale combined confidence+relevance signal to width [EDGE_WIDTH_MIN, EDGE_WIDTH_MAX]."""
    combined = max(0.0, min(1.0, edge_confidence * 0.6 + relevance * 0.4))
    return round(EDGE_WIDTH_MIN + combined * (EDGE_WIDTH_MAX - EDGE_WIDTH_MIN), 2)


def build_edges(
    sources: List[Source],
    llm_result: LLMResult,
    nli_results: Dict[int, NLIResult],
    confidence_svc: ConfidenceService,
) -> List[Edge]:
    """
    Build one edge per source, connecting node_ev_XX → node_main.

    Direction is evidence → main claim (not main → evidence) so the arrow
    drawn on the graph reads "this evidence feeds into the verdict",
    matching how a user reads the side panel (click evidence to see how
    it supports/refutes the claim) — see Feature 14 in plan.md.

    Edge properties:
    - edge_type  : from LLM classification (factcheck sources use their NLI label)
    - color      : edge_type → color map
    - weight     : edge confidence score [0, 1]
    - width      : scaled from weight for visual thickness
    - dashed     : True for correlated/insufficient (weaker evidence)
    - label      : short human-readable type string
    - explanation: LLM rationale for this source
    """
    class_by_idx: Dict[int, SourceClassification] = {
        sc.index: sc for sc in llm_result.sources
    }

    edges: List[Edge] = []
    for i, source in enumerate(sources, start=1):
        sc: Optional[SourceClassification] = class_by_idx.get(i)
        llm_class = sc.classification if sc else "insufficient"
        rationale = (sc.rationale or "") if sc else ""

        edge_type = _CLASSIFICATION_TO_EDGE_TYPE.get(llm_class, "insufficient")

        nli = nli_results.get(i - 1)
        edge_conf = confidence_svc.compute_edge_confidence(source, llm_class, nli)

        edges.append(Edge(
            source=f"node_ev_{i:02d}",
            target="node_main",
            edge_type=edge_type,
            weight=round(edge_conf, 3),
            color=_EDGE_TYPE_TO_COLOR.get(edge_type, EDGE_COLOR_INSUFFICIENT),
            width=_edge_width(edge_conf, source.relevance_score),
            dashed=_EDGE_TYPE_DASHED.get(edge_type, True),
            label=edge_type,
            explanation=rationale[:300] if rationale else None,
        ))

    return edges


# ---------------------------------------------------------------------------
# Inter-node edge helpers
# ---------------------------------------------------------------------------

# LLM relation → edge_type (reuse existing frontend-visible types)
_RELATION_TO_EDGE_TYPE: Dict[str, str] = {
    "corroborates":     "correlated",
    "contradicts":      "refutes",
    "provides_context": "correlated",
}

# LLM classification → simple group for Jaccard heuristic
_CLASS_GROUP: Dict[str, str] = {
    "direct_support":     "support",
    "direct_refute":      "refute",
    "correlated_context": "context",
    "insufficient":       "weak",
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "if", "then", "than", "it", "its", "this", "that",
    "have", "has", "had", "not", "no", "so", "about", "into", "over",
}


def _tokenize(text: str) -> Set[str]:
    """Simple lowercase word tokenizer — removes stopwords and short tokens."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    return {t for t in cleaned.split() if len(t) >= 3 and t not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_inter_node_edges(
    sources: List[Source],
    llm_result: LLMResult,
    confidence_svc: ConfidenceService,
) -> List[Edge]:
    """
    Build edges between evidence nodes (not main claim).

    Two passes:
    1. LLM node_links — high-quality, typed relationships (corroborates/contradicts/provides_context)
    2. Jaccard heuristic — snippet similarity → shared_topic correlated edges
       for pairs not already linked by LLM.

    Returns a list of Edge objects with source/target as node_ev_XX ids.
    """
    if not sources:
        return []

    class_by_idx: Dict[int, str] = {
        sc.index: sc.classification for sc in llm_result.sources
    }

    # Track which (min, max) pairs already have an edge to prevent duplicates
    linked_pairs: Set[Tuple[int, int]] = set()
    edges: List[Edge] = []

    # ── Pass 1: LLM node_links ──────────────────────────────────────────────
    valid_links = [
        nl for nl in (llm_result.node_links or [])
        if 1 <= nl.from_index <= len(sources) and 1 <= nl.to_index <= len(sources)
        and nl.from_index != nl.to_index
    ][:NODE_LINK_MAX_PAIRS]

    for nl in valid_links:
        pair = (min(nl.from_index, nl.to_index), max(nl.from_index, nl.to_index))
        if pair in linked_pairs:
            continue
        linked_pairs.add(pair)

        edge_type = _RELATION_TO_EDGE_TYPE.get(nl.relation, "correlated")
        color = _EDGE_TYPE_TO_COLOR.get(edge_type, EDGE_COLOR_CORRELATED)
        dashed = edge_type in ("correlated", "insufficient")

        # Edge confidence: average of the two nodes' source quality signals
        src_a = sources[nl.from_index - 1]
        src_b = sources[nl.to_index - 1]
        avg_relevance = (src_a.relevance_score + src_b.relevance_score) / 2
        weight = round(avg_relevance * 0.7, 3)

        edges.append(Edge(
            source=f"node_ev_{nl.from_index:02d}",
            target=f"node_ev_{nl.to_index:02d}",
            edge_type=edge_type,
            weight=weight,
            color=color,
            width=_edge_width(weight, avg_relevance) * 0.7,   # inter-node edges slightly thinner
            dashed=dashed,
            label=nl.relation,
            explanation=None,
        ))
        logger.debug(
            "Inter-node edge (LLM): ev_%02d → ev_%02d  [%s]",
            nl.from_index, nl.to_index, nl.relation,
        )

    # ── Pass 2: Jaccard heuristic for remaining similar pairs ──────────────
    tokens = [_tokenize((s.snippet or "") + " " + s.title) for s in sources]
    heuristic_count = 0

    for i in range(1, len(sources) + 1):
        if heuristic_count >= NODE_SIMILARITY_MAX_EDGES:
            break
        for j in range(i + 1, len(sources) + 1):
            if heuristic_count >= NODE_SIMILARITY_MAX_EDGES:
                break
            pair = (i, j)
            if pair in linked_pairs:
                continue

            sim = _jaccard(tokens[i - 1], tokens[j - 1])
            if sim < NODE_SIMILARITY_JACCARD_THRESHOLD:
                continue

            linked_pairs.add(pair)
            heuristic_count += 1

            # Determine edge type from classification groups
            group_i = _CLASS_GROUP.get(class_by_idx.get(i, "insufficient"), "weak")
            group_j = _CLASS_GROUP.get(class_by_idx.get(j, "insufficient"), "weak")

            if group_i == "support" and group_j == "refute" or \
               group_i == "refute" and group_j == "support":
                edge_type = "refutes"
            else:
                edge_type = "correlated"

            color = _EDGE_TYPE_TO_COLOR.get(edge_type, EDGE_COLOR_CORRELATED)
            weight = round(sim * 0.6, 3)   # scale down — heuristic edges are weaker
            avg_rel = (sources[i - 1].relevance_score + sources[j - 1].relevance_score) / 2

            edges.append(Edge(
                source=f"node_ev_{i:02d}",
                target=f"node_ev_{j:02d}",
                edge_type=edge_type,
                weight=weight,
                color=color,
                width=max(_edge_width(weight, avg_rel) * 0.55, 0.3),
                dashed=True,
                label="shared_topic",
                explanation=None,
            ))

    logger.info(
        "Inter-node edges: %d LLM + %d heuristic = %d total",
        len(valid_links), heuristic_count, len(edges),
    )
    return edges


def build_extended_edges(ext_pairs: List[Tuple[Node, str]]) -> List[Edge]:
    """
    Build edges connecting each Tier 2 extended node to its Tier 1 parent.

    Args:
        ext_pairs: List of (extended_node, parent_node_id) from build_extended_nodes().

    Returns:
        List of Edge objects (child → parent, thinner than Tier 1 edges).
        Direction matches build_edges(): the child feeds evidence into its
        parent, same as evidence → main claim (Feature 14).
    """
    edges: List[Edge] = []
    for ext_node, parent_node_id in ext_pairs:
        edge_type = "supports" if ext_node.node_type == "direct_support" else "refutes"
        color = EDGE_COLOR_SUPPORTS if edge_type == "supports" else EDGE_COLOR_REFUTES

        edges.append(Edge(
            source=ext_node.node_id,
            target=parent_node_id,
            edge_type=edge_type,
            weight=round(ext_node.confidence, 3),
            color=color,
            width=max(_edge_width(ext_node.confidence) * 0.65, EDGE_WIDTH_MIN),
            dashed=False,
            label=edge_type,
            explanation=None,
        ))

    logger.info("Extended edges: %d Tier-2 branch edges built", len(edges))
    return edges
