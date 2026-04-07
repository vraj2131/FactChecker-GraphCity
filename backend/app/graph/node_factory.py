"""
Phase 10: Node Factory.

Builds Node objects from pipeline outputs (sources, LLM classifications,
confidence scores). Called by graph_builder_service.
"""

from typing import Dict, List, Optional

from backend.app.models.llm_model import LLMResult, SourceClassification
from backend.app.models.nli_model import NLIResult
from backend.app.schemas.node_schema import Node
from backend.app.schemas.source_schema import Source
from backend.app.services.confidence_service import ConfidenceOutput, ConfidenceService
from backend.app.utils.constants import (
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
