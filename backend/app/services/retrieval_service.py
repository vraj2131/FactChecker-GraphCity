import logging
from typing import List, Optional

from backend.app.preprocessing.deduplicate import deduplicate_sources
from backend.app.preprocessing.normalize_text import normalize_claim_text
from backend.app.retrieval.retriever_registry import RetrieverRegistry
from backend.app.schemas.source_schema import Source
from backend.app.services.cache_service import CacheService
from backend.app.services.evidence_expansion_service import EvidenceExpansionService
from backend.app.services.ranking_service import RankingService
from backend.app.utils.constants import (
    DEFAULT_PER_RETRIEVER_MAX_RESULTS,
    MAX_RESULTS_PER_SOURCE_TYPE,
    RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE,
    SUPPORTED_RETRIEVER_SOURCES,
)
from backend.app.utils.hashing import build_cache_key

logger = logging.getLogger(__name__)

# Retrievers that benefit from adversarial/debunking query variants.
# Offline retrievers (wikipedia) use embedding similarity and don't benefit from
# keyword negations; live-wiki already searches live Wikipedia.
_ADVERSARIAL_RETRIEVER_SOURCES = {"factcheck", "guardian", "gdelt"}

# Max results fetched per adversarial query per retriever (kept low to avoid flooding)
_ADVERSARIAL_MAX_RESULTS = 3


def _adversarial_queries(claim: str) -> List[str]:
    """
    Generate debunking/counter-evidence search variants for a claim.

    These extra queries help surface refuting evidence (myth-busting articles,
    debunks) that a literal search for the claim text might miss.
    """
    base = claim.strip().rstrip(".")
    return [
        f"{base} myth debunked",
        f"{base} false",
    ]


def _apply_source_diversity(sources: List[Source], max_per_type: int) -> List[Source]:
    """
    Keep at most max_per_type results per source_type, preserving rank order.

    This prevents a single high-scoring retriever (e.g. Guardian) from
    filling all output slots and crowding out other source types.
    """
    counts: dict = {}
    result: List[Source] = []
    for source in sources:
        t = source.source_type
        if counts.get(t, 0) < max_per_type:
            result.append(source)
            counts[t] = counts.get(t, 0) + 1
    return result


class RetrievalService:
    """
    Orchestrates retrieval across all registered sources for a single claim query.

    Flow:
    1. Normalize query
    2. Build a stable cache key
    3. Return cached result if available (and use_cache=True)
    4. Query each requested retriever independently (failures are tolerated)
    5. Merge all results, deduplicate, rank
    6. Expand snippets via EvidenceExpansionService
    7. Slice to max_results
    8. Persist to cache
    9. Return final List[Source]

    A single failing retriever never aborts the whole call.
    If ALL retrievers fail, a RuntimeError is raised.
    """

    def __init__(
        self,
        registry: RetrieverRegistry,
        cache: CacheService,
        ranking: RankingService,
        expansion: EvidenceExpansionService,
    ) -> None:
        self._registry = registry
        self._cache = cache
        self._ranking = ranking
        self._expansion = expansion

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        use_cache: bool = True,
        per_retriever_max: int = DEFAULT_PER_RETRIEVER_MAX_RESULTS,
        expand_queries: bool = True,
    ) -> List[Source]:
        """
        Retrieve and return ranked, deduplicated evidence sources for a claim.

        Args:
            query:            The claim text to retrieve evidence for.
            max_results:      Maximum number of sources to return after merge/dedup/rank.
            sources:          List of retriever names to query. Defaults to all registered.
            use_cache:        If True, check cache before querying and save result after.
            per_retriever_max: Max results requested from each individual retriever.
            expand_queries:   If True, also query adversarial variants ("{claim} myth debunked",
                              "{claim} false") against fact-check-focused retrievers to surface
                              counter-evidence for potentially false claims.

        Returns:
            List[Source] sorted by descending relevance/trust score, length <= max_results.

        Raises:
            ValueError:   If query is empty.
            RuntimeError: If every retriever fails with an exception.
        """
        # --- Validate and normalize query ---
        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        normalized_query = normalize_claim_text(query)

        # --- Determine which retrievers to use ---
        requested_sources = sources if sources else SUPPORTED_RETRIEVER_SOURCES
        available_sources = [
            s for s in requested_sources if self._registry.is_registered(s)
        ]

        if not available_sources:
            logger.warning(
                "No requested retrievers are registered. Requested: %s. Registered: %s",
                requested_sources,
                self._registry.list_names(),
            )
            return []

        # --- Cache key ---
        cache_key = build_cache_key(
            source_name=RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE,
            query=normalized_query,
            sources=sorted(available_sources),
            max_results=max_results,
            per_retriever_max=per_retriever_max,
        )

        # --- Cache lookup ---
        if use_cache and self._cache.exists(
            RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE, cache_key
        ):
            cached_data = self._cache.load(
                RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE, cache_key
            )
            if cached_data is not None:
                logger.info(
                    "Cache hit for query='%s' (key=%s)", normalized_query, cache_key[:12]
                )
                return [Source(**item) for item in cached_data]

        # --- Query each retriever ---
        all_results: List[Source] = []
        failed_retrievers: List[str] = []
        source_counts: dict = {}

        for source_name in available_sources:
            try:
                retriever = self._registry.get(source_name)
                results = retriever.retrieve(
                    query=normalized_query,
                    max_results=per_retriever_max,
                )
                source_counts[source_name] = len(results)
                all_results.extend(results)
                logger.info(
                    "Retriever '%s' returned %d results for query='%s'",
                    source_name,
                    len(results),
                    normalized_query,
                )
            except Exception as exc:
                failed_retrievers.append(source_name)
                logger.warning(
                    "Retriever '%s' failed for query='%s': %s",
                    source_name,
                    normalized_query,
                    exc,
                )

        # --- All retrievers failed ---
        if not all_results and len(failed_retrievers) == len(available_sources):
            raise RuntimeError(
                f"All retrievers failed for query='{normalized_query}'. "
                f"Failed: {failed_retrievers}"
            )

        if failed_retrievers:
            logger.warning(
                "Partial retrieval: %d retrievers failed (%s). "
                "Proceeding with %d raw results from %d sources.",
                len(failed_retrievers),
                failed_retrievers,
                len(all_results),
                len(available_sources) - len(failed_retrievers),
            )

        # --- Adversarial query expansion (debunking variants) ---
        if expand_queries:
            adv_queries = _adversarial_queries(normalized_query)
            adv_sources = [
                s for s in available_sources
                if s in _ADVERSARIAL_RETRIEVER_SOURCES and self._registry.is_registered(s)
            ]
            for adv_query in adv_queries:
                for source_name in adv_sources:
                    try:
                        retriever = self._registry.get(source_name)
                        adv_results = retriever.retrieve(
                            query=adv_query,
                            max_results=_ADVERSARIAL_MAX_RESULTS,
                        )
                        all_results.extend(adv_results)
                        logger.debug(
                            "Adversarial query '%s' via '%s': %d results",
                            adv_query[:60],
                            source_name,
                            len(adv_results),
                        )
                    except Exception as exc:
                        logger.debug(
                            "Adversarial retriever '%s' failed for query='%s': %s",
                            source_name,
                            adv_query[:60],
                            exc,
                        )

        logger.info(
            "Raw results before dedup/rank: %d | per-source: %s",
            len(all_results),
            source_counts,
        )

        # --- Deduplicate ---
        deduped = deduplicate_sources(all_results)
        logger.info(
            "After dedup: %d results (removed %d duplicates)",
            len(deduped),
            len(all_results) - len(deduped),
        )

        # --- Rank ---
        ranked = self._ranking.rank(deduped)

        # --- Enforce source diversity (cap per source type) ---
        diverse = _apply_source_diversity(ranked, MAX_RESULTS_PER_SOURCE_TYPE)
        logger.info(
            "After diversity cap (%d per type): %d results (was %d)",
            MAX_RESULTS_PER_SOURCE_TYPE,
            len(diverse),
            len(ranked),
        )

        # --- Expand snippets ---
        expanded = self._expansion.expand(normalized_query, diverse)

        # --- Slice ---
        final = expanded[:max_results]

        # --- Persist to cache ---
        if use_cache:
            try:
                serializable = [item.model_dump(mode="json") for item in final]
                self._cache.save(
                    RETRIEVAL_ORCHESTRATION_CACHE_NAMESPACE, cache_key, serializable
                )
                logger.info(
                    "Saved %d results to cache (key=%s)", len(final), cache_key[:12]
                )
            except Exception as exc:
                logger.warning("Failed to save retrieval results to cache: %s", exc)

        return final
