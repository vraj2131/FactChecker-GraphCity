"""
Phase 13: GET /api/v1/sources — retrieval-only debug endpoint.

Runs just the retrieval pipeline (no NLI, no LLM) and returns the
raw ranked sources for a claim.  Useful for:
- Inspecting which retrievers fired and what they returned
- Frontend "Sources" tab live counts
- Debugging retrieval quality without a full pipeline run
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.dependencies import get_retrieval_service
from backend.app.schemas.source_schema import Source
from backend.app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sources"])


class SourcesResponse(BaseModel):
    """Response from the retrieval-only sources endpoint."""

    claim_text: str = Field(..., description="The claim that was retrieved for.")
    total: int = Field(..., description="Number of sources returned.")
    retriever_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of sources contributed by each retriever type.",
    )
    sources: List[Source] = Field(default_factory=list)


@router.get(
    "/sources",
    response_model=SourcesResponse,
    summary="Retrieve raw sources for a claim (no NLI / LLM)",
    response_description="Ranked, deduplicated sources from all retrievers.",
)
async def get_sources(
    claim: str = Query(..., min_length=3, max_length=500, description="Claim text to retrieve evidence for."),
    max_results: int = Query(default=10, ge=1, le=30, description="Max sources to return."),
    retrieval_svc: RetrievalService = Depends(get_retrieval_service),
) -> SourcesResponse:
    """
    Run retrieval only (Wikipedia FAISS, LiveWiki, FactCheck, Guardian, NewsAPI, GDELT)
    and return ranked, deduplicated sources.

    Does NOT run NLI or LLM — purely the retrieval + ranking pipeline.
    Results are cached on disk so repeated calls for the same claim are fast.

    Raises:
        HTTPException 400: if claim is empty after stripping.
        HTTPException 503: if all retrievers fail.
    """
    claim = claim.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="claim cannot be empty.")

    logger.info("GET /sources — claim='%s' max_results=%d", claim[:80], max_results)

    try:
        sources = retrieval_svc.retrieve(
            query=claim,
            max_results=max_results,
            use_cache=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("RetrievalService.retrieve() failed for claim='%s'", claim[:80])
        raise HTTPException(status_code=503, detail=f"Retrieval error: {exc}") from exc

    # Count per retriever type
    retriever_counts: Dict[str, int] = {}
    for src in sources:
        retriever_counts[src.source_type] = retriever_counts.get(src.source_type, 0) + 1

    logger.info(
        "GET /sources — returned %d sources | counts=%s",
        len(sources),
        retriever_counts,
    )

    return SourcesResponse(
        claim_text=claim,
        total=len(sources),
        retriever_counts=retriever_counts,
        sources=sources,
    )
