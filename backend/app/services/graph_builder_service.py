"""
Phase 10: Graph Builder Service.

Converts a VerifyClaimResult (full pipeline output) into a GraphResponse
(nodes + edges + metadata) ready for the frontend 3D graph renderer.

Input:  VerifyClaimResult  — from verify_claim_service.py
Output: GraphResponse      — validated Pydantic model (node_schema / edge_schema)

The GraphResponse JSON is what the React frontend consumes directly.
Each node carries all hover data (verdict, confidence, top_sources,
short_explanation) and a best_source_url for click-to-redirect.
"""

import logging
from typing import Optional

from backend.app.graph.edge_factory import build_edges
from backend.app.graph.node_factory import build_evidence_nodes, build_main_node
from backend.app.schemas.response_schema import GraphMetadata, GraphResponse
from backend.app.services.confidence_service import ConfidenceService
from backend.app.services.verify_claim_service import VerifyClaimResult

logger = logging.getLogger(__name__)


class GraphBuilderService:
    """
    Converts a VerifyClaimResult into a GraphResponse.

    Usage:
        builder = GraphBuilderService()
        graph = builder.build(result)
        print(graph.model_dump_json(indent=2))
    """

    def __init__(self, confidence_svc: Optional[ConfidenceService] = None) -> None:
        self._confidence_svc = confidence_svc or ConfidenceService()

    def build(self, result: VerifyClaimResult) -> GraphResponse:
        """
        Build the full GraphResponse from a VerifyClaimResult.

        Steps:
        1. Build main claim node (verdict, confidence, top_sources, best_source_url)
        2. Build evidence nodes (one per llm_input_source)
        3. Build edges (main → each evidence node)
        4. Assemble GraphMetadata
        5. Return validated GraphResponse
        """
        sources = result.llm_input_sources
        llm = result.llm_result
        nli = result.nli_results
        conf = result.confidence_output

        # 1. Main claim node
        main_node = build_main_node(
            claim_text=result.claim_text,
            confidence_output=conf,
            llm_result=llm,
            llm_input_sources=sources,
        )

        # 2. Evidence nodes
        evidence_nodes = build_evidence_nodes(
            sources=sources,
            llm_result=llm,
            nli_results=nli,
            confidence_svc=self._confidence_svc,
        )

        # 3. Edges
        edges = build_edges(
            sources=sources,
            llm_result=llm,
            nli_results=nli,
            confidence_svc=self._confidence_svc,
        )

        all_nodes = [main_node] + evidence_nodes

        # 4. Metadata — count nodes by type
        support_count  = sum(1 for n in evidence_nodes if n.node_type == "direct_support")
        refute_count   = sum(1 for n in evidence_nodes if n.node_type == "direct_refute")
        insuff_count   = sum(1 for n in evidence_nodes if n.node_type == "insufficient_evidence")
        context_count  = sum(1 for n in evidence_nodes if n.node_type == "context_signal")
        fc_count       = sum(1 for n in evidence_nodes if n.node_type == "factcheck_review")

        top_support = max(
            (n.confidence for n in evidence_nodes if n.node_type in ("direct_support",)),
            default=None,
        )
        top_refute = max(
            (n.confidence for n in evidence_nodes if n.node_type in ("direct_refute", "factcheck_review")),
            default=None,
        )

        retrieval_note = _build_retrieval_note(result)

        metadata = GraphMetadata(
            claim_text=result.claim_text,
            overall_verdict=conf.overall_verdict,
            overall_confidence=round(conf.overall_confidence, 3),
            total_nodes=len(all_nodes),
            total_edges=len(edges),
            support_node_count=support_count,
            refute_node_count=refute_count,
            insufficient_node_count=insuff_count,
            context_node_count=context_count,
            factcheck_node_count=fc_count,
            top_support_score=round(top_support, 3) if top_support is not None else None,
            top_refute_score=round(top_refute, 3) if top_refute is not None else None,
            retrieval_notes=retrieval_note,
        )

        graph = GraphResponse(metadata=metadata, nodes=all_nodes, edges=edges)

        logger.info(
            "GraphBuilderService: built graph — nodes=%d edges=%d verdict=%s conf=%.2f",
            len(all_nodes), len(edges), conf.overall_verdict, conf.overall_confidence,
        )

        return graph


def _build_retrieval_note(result: VerifyClaimResult) -> Optional[str]:
    """Generate a short human-readable note about retrieval conditions."""
    n = len(result.sources)
    if n == 0:
        return "No sources retrieved. Verdict is based on absence of evidence."
    types = {s.source_type for s in result.sources}
    return f"{n} sources retrieved from: {', '.join(sorted(types))}."
