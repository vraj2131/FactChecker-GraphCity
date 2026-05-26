"""
Phase 10: Node Factory.

Builds Node objects from pipeline outputs (sources, LLM classifications,
confidence scores). Called by graph_builder_service.
"""

from typing import Dict, List, Optional, Set, Tuple

from backend.app.models.llm_model import LLMResult, SourceClassification
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.node_schema import Node
from backend.app.schemas.source_schema import Source
from backend.app.services.confidence_service import ConfidenceOutput, ConfidenceService
from backend.app.utils.constants import (
    EXTENDED_NODE_JACCARD_THRESHOLD,
    EXTENDED_NODES_PER_PARENT,
    GRAPH_MAX_TOP_SOURCES,
    NODE_COLOR_CONTEXT_SIGNAL,
    NODE_COLOR_DIRECT_REFUTE,
    NODE_COLOR_DIRECT_SUPPORT,
    NODE_COLOR_FACTCHECK_REVIEW,
    NODE_COLOR_INSUFFICIENT,
    NODE_COLOR_MAIN_NEI,
    NODE_COLOR_MAIN_REJECTED,
    NODE_COLOR_MAIN_VERIFIED,
    NODE_SIZE_DIRECT_EVIDENCE,
    NODE_SIZE_EXTENDED_EVIDENCE,
    NODE_SIZE_MAIN_CLAIM,
    NODE_SIZE_WEAK_EVIDENCE,
)

# LLM classification → node_type (factcheck source_type overrides below)
_CLASSIFICATION_TO_NODE_TYPE: Dict[str, str] = {
    "direct_support":     "direct_support",
    "direct_refute":      "direct_refute",
    "correlated_context": "context_signal",
    "insufficient":       "insufficient_evidence",
}

# node_type → hex color
_NODE_TYPE_TO_COLOR: Dict[str, str] = {
    "direct_support":        NODE_COLOR_DIRECT_SUPPORT,
    "direct_refute":         NODE_COLOR_DIRECT_REFUTE,
    "factcheck_review":      NODE_COLOR_FACTCHECK_REVIEW,
    "context_signal":        NODE_COLOR_CONTEXT_SIGNAL,
    "insufficient_evidence": NODE_COLOR_INSUFFICIENT,
}

# node_type → verdict label for evidence nodes
_NODE_TYPE_TO_VERDICT: Dict[str, str] = {
    "direct_support":        "supports",
    "direct_refute":         "refutes",
    "factcheck_review":      "refutes",
    "context_signal":        "correlated",
    "insufficient_evidence": "insufficient",
}

# node_type → visual size
_NODE_TYPE_TO_SIZE: Dict[str, float] = {
    "direct_support":        NODE_SIZE_DIRECT_EVIDENCE,
    "direct_refute":         NODE_SIZE_DIRECT_EVIDENCE,
    "factcheck_review":      NODE_SIZE_DIRECT_EVIDENCE,
    "context_signal":        NODE_SIZE_WEAK_EVIDENCE,
    "insufficient_evidence": NODE_SIZE_WEAK_EVIDENCE,
}


def _main_node_color(verdict: str) -> str:
    return {
        "verified":        NODE_COLOR_MAIN_VERIFIED,
        "rejected":        NODE_COLOR_MAIN_REJECTED,
        "not_enough_info": NODE_COLOR_MAIN_NEI,
    }.get(verdict, NODE_COLOR_MAIN_NEI)


def _nli_verdict(nli: Optional[NLIResult]) -> Optional[str]:
    if nli is None:
        return None
    return {
        "supports":        "supports",
        "refutes":         "refutes",
        "not_enough_info": "insufficient",
    }.get(nli.label)


def build_main_node(
    claim_text: str,
    confidence_output: ConfidenceOutput,
    llm_result: LLMResult,
    llm_input_sources: List[Source],
) -> Node:
    """
    Build the central main claim node.

    Attaches:
    - verdict + confidence from ConfidenceOutput
    - top_sources (up to 5) for hover display
    - best_source_url from LLM's best_source_index for click-to-redirect
    - short_explanation from LLM for hover display
    """
    verdict = confidence_output.overall_verdict
    top_sources = llm_input_sources[:GRAPH_MAX_TOP_SOURCES]

    # Resolve best source URL — LLM picks best_source_index (1-based)
    best_url: Optional[str] = None
    if llm_result.best_source_index is not None:
        idx = llm_result.best_source_index - 1
        if 0 <= idx < len(llm_input_sources):
            best_url = str(llm_input_sources[idx].url)
    if best_url is None and top_sources:
        best_url = str(top_sources[0].url)

    short_exp = llm_result.short_explanation or f"Verdict: {verdict}."

    return Node(
        node_id="node_main",
        node_type="main_claim",
        text=claim_text,
        verdict=verdict,
        confidence=round(confidence_output.overall_confidence, 3),
        size=NODE_SIZE_MAIN_CLAIM,
        color=_main_node_color(verdict),
        best_source_url=best_url,
        top_sources=top_sources,
        short_explanation=short_exp,
        source_count=len(llm_input_sources),
        is_main_claim=True,
    )


def build_evidence_nodes(
    sources: List[Source],
    llm_result: LLMResult,
    nli_results: Dict[int, NLIResult],
    confidence_svc: ConfidenceService,
) -> List[Node]:
    """
    Build one evidence node per source.

    node_type is driven by LLM classification.
    factcheck sources always become factcheck_review regardless of LLM classification.
    NLI verdict refines the verdict label where available.
    Edge confidence is reused as node confidence.
    """
    class_by_idx: Dict[int, SourceClassification] = {
        sc.index: sc for sc in llm_result.sources
    }

    nodes: List[Node] = []
    for i, source in enumerate(sources, start=1):
        sc: Optional[SourceClassification] = class_by_idx.get(i)
        llm_class = sc.classification if sc else "insufficient"
        rationale = (sc.rationale or "") if sc else ""

        # factcheck source_type always wins over LLM classification for node_type
        if source.source_type == "factcheck":
            node_type = "factcheck_review"
        else:
            node_type = _CLASSIFICATION_TO_NODE_TYPE.get(llm_class, "insufficient_evidence")

        # NLI verdict is more fine-grained than type default
        nli = nli_results.get(i - 1)
        verdict = _nli_verdict(nli) or _NODE_TYPE_TO_VERDICT.get(node_type, "insufficient")

        edge_conf = confidence_svc.compute_edge_confidence(source, llm_class, nli)
        short_exp = rationale[:300] if rationale else f"{source.source_type}: {llm_class}"

        nodes.append(Node(
            node_id=f"node_ev_{i:02d}",
            node_type=node_type,
            text=source.snippet or source.title,
            verdict=verdict,
            confidence=round(edge_conf, 3),
            size=_NODE_TYPE_TO_SIZE.get(node_type, NODE_SIZE_WEAK_EVIDENCE),
            color=_NODE_TYPE_TO_COLOR.get(node_type, NODE_COLOR_INSUFFICIENT),
            best_source_url=str(source.url),
            top_sources=[source],
            short_explanation=short_exp,
            source_count=1,
            is_main_claim=False,
        ))

    return nodes


# ---------------------------------------------------------------------------
# Tier 2 tokenizer helpers (intentionally local — avoids cross-module import)
# ---------------------------------------------------------------------------

_EXT_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "if", "then", "than", "it", "its", "this", "that",
    "have", "has", "had", "not", "no", "so", "about", "into", "over",
}


def _ext_tokenize(text: str) -> Set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or ""))
    return {t for t in cleaned.split() if len(t) >= 3 and t not in _EXT_STOPWORDS}


def _ext_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_extended_nodes(
    extended_pool: List[Source],
    tier1_nodes: List[Node],
    tier1_sources: List[Source],
    confidence_svc: ConfidenceService,
) -> List[Tuple[Node, str]]:
    """
    Build Tier 2 branch nodes from the unused source pool (sources beyond LLM input).

    For each Tier 1 direct_support / direct_refute node, attach up to
    EXTENDED_NODES_PER_PARENT child nodes whose:
      - stance_hint matches the parent direction (supports / refutes)
      - Jaccard similarity with the parent snippet >= EXTENDED_NODE_JACCARD_THRESHOLD
      - Same source_type as parent gets a small similarity bonus

    Returns a list of (Node, parent_node_id) tuples consumed by build_extended_edges().
    """
    if not extended_pool or not tier1_nodes:
        return []

    results: List[Tuple[Node, str]] = []
    used_ids: Set[str] = set()

    for i, (t1_node, t1_source) in enumerate(zip(tier1_nodes, tier1_sources), start=1):
        if t1_node.node_type not in ("direct_support", "direct_refute"):
            continue

        required_stance = "supports" if t1_node.node_type == "direct_support" else "refutes"
        llm_class = "direct_support" if t1_node.node_type == "direct_support" else "direct_refute"
        parent_tokens = _ext_tokenize((t1_source.snippet or "") + " " + t1_source.title)

        candidates: List[Tuple[float, Source]] = []
        for ext_src in extended_pool:
            if ext_src.source_id in used_ids:
                continue
            if ext_src.stance_hint != required_stance:
                continue

            child_tokens = _ext_tokenize((ext_src.snippet or "") + " " + ext_src.title)
            sim = _ext_jaccard(parent_tokens, child_tokens)
            if sim < EXTENDED_NODE_JACCARD_THRESHOLD:
                continue

            # Small bonus for same source_type (encourages topical coherence)
            type_bonus = 0.08 if ext_src.source_type == t1_source.source_type else 0.0
            candidates.append((sim + type_bonus, ext_src))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for child_rank, (_, child_src) in enumerate(candidates[:EXTENDED_NODES_PER_PARENT], start=1):
            used_ids.add(child_src.source_id)

            node_id = f"node_ext_{i:02d}_{child_rank}"
            child_verdict = _NODE_TYPE_TO_VERDICT.get(t1_node.node_type, "insufficient")
            edge_conf = confidence_svc.compute_edge_confidence(child_src, llm_class, None)

            results.append((
                Node(
                    node_id=node_id,
                    node_type=t1_node.node_type,
                    text=child_src.snippet or child_src.title,
                    verdict=child_verdict,
                    confidence=round(edge_conf, 3),
                    size=NODE_SIZE_EXTENDED_EVIDENCE,
                    color=_NODE_TYPE_TO_COLOR.get(t1_node.node_type, NODE_COLOR_INSUFFICIENT),
                    best_source_url=str(child_src.url),
                    top_sources=[child_src],
                    short_explanation=f"Additional {child_src.source_type} source corroborating this evidence node.",
                    source_count=1,
                    is_main_claim=False,
                ),
                t1_node.node_id,
            ))

    return results
