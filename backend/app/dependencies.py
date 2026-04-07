"""
Phase 11: FastAPI dependency injection.

Provides singleton instances of VerifyClaimService and GraphBuilderService
via functools.lru_cache so models are loaded once at startup, not per-request.

Usage in route handlers:
    @router.post("/verify-claim")
    async def verify(
        request: VerifyClaimRequest,
        verify_svc: VerifyClaimService = Depends(get_verify_service),
        graph_svc: GraphBuilderService = Depends(get_graph_service),
    ): ...
"""

import logging
from functools import lru_cache

from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.verify_claim_service import VerifyClaimService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_verify_service() -> VerifyClaimService:
    """
    Build and cache a VerifyClaimService singleton.

    Called once on first request; subsequent calls return the cached instance.
    lru_cache(maxsize=1) is safe here because VerifyClaimService is stateless
    after construction (all state lives in caches on disk).
    """
    logger.info("Initialising VerifyClaimService (first request)...")
    svc = VerifyClaimService.build_default()
    logger.info("VerifyClaimService ready.")
    return svc


@lru_cache(maxsize=1)
def get_graph_service() -> GraphBuilderService:
    """
    Build and cache a GraphBuilderService singleton.

    GraphBuilderService is lightweight (no models), but cached for consistency.
    """
    return GraphBuilderService()
