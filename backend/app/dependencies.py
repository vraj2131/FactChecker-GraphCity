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
from pathlib import Path

from backend.app.services.graph_builder_service import GraphBuilderService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.verify_claim_service import VerifyClaimService
from backend.app.utils.constants import DEFAULT_RETRIEVAL_CACHE_DIR

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


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """
    Build and cache a standalone RetrievalService singleton for the sources endpoint.

    Uses the same retriever registry and cache directory as VerifyClaimService
    but skips NLI / LLM — retrieval only.
    """
    from backend.app.retrieval.factcheck_retriever import FactCheckRetriever
    from backend.app.retrieval.gdelt_retriever import GDELTRetriever
    from backend.app.retrieval.guardian_retriever import GuardianRetriever
    from backend.app.retrieval.livewiki_retriever import LiveWikiRetriever
    from backend.app.retrieval.newsapi_retriever import NewsApiRetriever
    from backend.app.retrieval.wikipedia_retriever import WikipediaRetriever
    from backend.app.retrieval.retriever_registry import RetrieverRegistry
    from backend.app.services.cache_service import CacheService
    from backend.app.services.evidence_expansion_service import EvidenceExpansionService
    from backend.app.services.ranking_service import RankingService

    registry = RetrieverRegistry()
    registry.register(WikipediaRetriever())
    registry.register(LiveWikiRetriever())
    registry.register(FactCheckRetriever())
    registry.register(GuardianRetriever())
    registry.register(NewsApiRetriever())
    registry.register(GDELTRetriever())

    logger.info("Initialising RetrievalService singleton...")
    svc = RetrievalService(
        registry=registry,
        cache=CacheService(DEFAULT_RETRIEVAL_CACHE_DIR),
        ranking=RankingService(),
        expansion=EvidenceExpansionService(),
    )
    logger.info("RetrievalService ready.")
    return svc
