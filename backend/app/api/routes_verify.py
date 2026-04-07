"""
Phase 11: POST /api/v1/verify-claim route.

Accepts a VerifyClaimRequest, runs the full fact-checking pipeline via
VerifyClaimService, builds a GraphResponse via GraphBuilderService,
and returns the JSON graph to the frontend.

Error handling:
- 422  Unprocessable Entity — Pydantic validation failure (FastAPI auto)
- 400  Bad Request         — claim rejected by domain validation
- 503  Service Unavailable — pipeline failed (LLM/retrieval error)
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies import get_graph_service, get_verify_service
from backend.app.schemas.request_schema import VerifyClaimRequest
from backend.app.schemas.response_schema import GraphResponse
from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.verify_claim_service import VerifyClaimService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["verify"])


@router.post(
    "/verify-claim",
    response_model=GraphResponse,
    summary="Verify a claim and return a graph",
    response_description="Graph nodes, edges, and metadata for the verified claim.",
)
async def verify_claim(
    request: VerifyClaimRequest,
    verify_svc: VerifyClaimService = Depends(get_verify_service),
    graph_svc: GraphBuilderService = Depends(get_graph_service),
) -> GraphResponse:
    """
    Run the full fact-checking pipeline for `claim_text` and return a graph.

    Pipeline steps (inside VerifyClaimService.verify):
    1. Retrieve candidates from all sources (Wikipedia FAISS, LiveWiki,
       FactCheck, Guardian, NewsAPI, GDELT)
    2. Extract snippets
    3. Expand via FAISS if needed (EvidenceExpansionService)
    4. NLI score each snippet
    5. LLM refine + classify sources
    6. Confidence aggregation + calibration
    7. Graph build (node_factory + edge_factory)
    8. Return JSON

    Returns:
        GraphResponse with nodes, edges, and metadata.

    Raises:
        HTTPException 400: if claim_text is empty after stripping.
        HTTPException 503: if an internal pipeline error occurs.
    """
    claim = request.claim_text.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="claim_text cannot be empty.")

    logger.info("POST /verify-claim — claim='%s'", claim[:80])
    t0 = time.perf_counter()

    try:
        result = verify_svc.verify(claim, use_cache=True)
    except Exception as exc:
        logger.exception("VerifyClaimService.verify() failed for claim='%s'", claim[:80])
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline error: {exc}",
        ) from exc

    try:
        graph = graph_svc.build(result)
    except Exception as exc:
        logger.exception("GraphBuilderService.build() failed for claim='%s'", claim[:80])
        raise HTTPException(
            status_code=503,
            detail=f"Graph build error: {exc}",
        ) from exc

    elapsed = time.perf_counter() - t0
    logger.info(
        "POST /verify-claim — done in %.2fs | verdict=%s conf=%.2f nodes=%d edges=%d",
        elapsed,
        graph.metadata.overall_verdict,
        graph.metadata.overall_confidence,
        graph.metadata.total_nodes,
        graph.metadata.total_edges,
    )

    return graph
