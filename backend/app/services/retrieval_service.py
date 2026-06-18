import logging
from datetime import datetime, timezone
from typing import List, Optional

from backend.app.preprocessing.deduplicate import deduplicate_sources
from backend.app.preprocessing.entity_extractor import extract_claim_entities, anchor_present
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

# Feature 4a: query decomposition — long claims only
_DECOMP_MIN_WORDS = 8
_DECOMP_MAX_RESULTS = 3
_DECOMP_RETRIEVER_SOURCES = {"guardian", "newsapi", "livewiki", "duckduckgo"}
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
    "and", "or", "but", "that", "this", "it", "its", "not", "no",
}

# Feature 4b: date-weighted re-ranking
_DATE_RECENT_DAYS = 90
_DATE_OLD_DAYS = 730   # 2 years
_DATE_RECENT_BONUS = 0.05
_DATE_OLD_PENALTY = 0.05


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


def _decompose_query(claim: str) -> List[str]:
    """
    For claims longer than _DECOMP_MIN_WORDS, produce up to 2 entity-focused
    sub-queries by extracting proper nouns, numbers, and content words.
    Returns [] for short claims.
    """
    words = claim.strip().rstrip(".").split()
    if len(words) <= _DECOMP_MIN_WORDS:
        return []

    # Prefer proper nouns (capitalized mid-sentence) and tokens with digits
    entities = [
        w.strip(".,!?;:\"'()[]")
        for i, w in enumerate(words)
        if (i > 0 and w[:1].isupper()) or any(c.isdigit() for c in w)
    ]

    # Fall back to content words if not enough proper nouns
    if len(entities) < 3:
        entities = [
            w.strip(".,!?;:\"'()[]")
            for w in words
            if w.lower().strip(".,!?") not in _STOPWORDS and len(w) > 3
        ]

    entities = [e for e in entities if e]
    if not entities:
        return []

    sub_queries: List[str] = [" ".join(entities[:3])]
    if len(entities) >= 4:
        sub_queries.append(" ".join(entities[-3:]))

    return sub_queries[:2]


def _parse_date(date_str: str) -> datetime | None:
    """Parse a published_at string into a timezone-aware datetime, or None on failure."""
    if not date_str:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d",
    ):
        try:
            dt = datetime.strptime(date_str[:len(fmt) + 5].strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _apply_date_weighting(sources: List[Source]) -> List[Source]:
    """
    Adjust trust_score based on publication recency before ranking:
      - Published within 90 days  → +0.05
      - Published 2+ years ago    → -0.05
    Sources without a parseable date are unchanged.
    """
    now = datetime.now(timezone.utc)
    result: List[Source] = []
    boosted = penalised = 0
    for source in sources:
        delta = 0.0
        if source.published_at:
            pub = _parse_date(str(source.published_at))
            if pub is not None:
                age_days = (now - pub).days
                if age_days <= _DATE_RECENT_DAYS:
                    delta = _DATE_RECENT_BONUS
                    boosted += 1
                elif age_days >= _DATE_OLD_DAYS:
                    delta = -_DATE_OLD_PENALTY
                    penalised += 1

        if delta != 0.0:
            new_trust = max(0.0, min(1.0, source.trust_score + delta))
            result.append(source.model_copy(update={"trust_score": new_trust}))
        else:
            result.append(source)

    logger.info(
        "Date weighting: %d boosted (≤90d), %d penalised (≥2yr), %d unchanged",
        boosted, penalised, len(sources) - boosted - penalised,
    )
    return result


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

        # --- Sub-query decomposition for long claims (Feature 4a) ---
        if expand_queries:
            sub_queries = _decompose_query(normalized_query)
            decomp_sources = [
                s for s in available_sources
                if s in _DECOMP_RETRIEVER_SOURCES and self._registry.is_registered(s)
            ]
            for sub_query in sub_queries:
                for source_name in decomp_sources:
                    try:
                        retriever = self._registry.get(source_name)
                        sub_results = retriever.retrieve(
                            query=sub_query,
                            max_results=_DECOMP_MAX_RESULTS,
                        )
                        all_results.extend(sub_results)
                        logger.info(
                            "Sub-query '%s' via '%s': %d results",
                            sub_query[:60], source_name, len(sub_results),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Sub-query retriever '%s' failed for query='%s': %s",
                            source_name, sub_query[:60], exc,
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

        # --- Low-yield fallback: if too few sources, retry with entity-focused query ---
        _FALLBACK_MIN_SOURCES = 3
        _FALLBACK_RETRIEVER_SOURCES = {"wikipedia", "livewiki", "duckduckgo"}
        if len(deduped) < _FALLBACK_MIN_SOURCES:
            fallback_words = [
                w.strip(".,!?;:'\"()[]")
                for w in normalized_query.split()
                if w.lower().strip(".,!?") not in _STOPWORDS and len(w) > 3
            ]
            if fallback_words:
                fallback_query = " ".join(fallback_words[:5]) + " facts"
                fb_sources = [
                    s for s in available_sources
                    if s in _FALLBACK_RETRIEVER_SOURCES and self._registry.is_registered(s)
                ]
                for source_name in fb_sources:
                    try:
                        retriever = self._registry.get(source_name)
                        fb_results = retriever.retrieve(query=fallback_query, max_results=3)
                        deduped_ids = {s.source_id for s in deduped}
                        new = [s for s in fb_results if s.source_id not in deduped_ids]
                        deduped.extend(new)
                        logger.info(
                            "Low-yield fallback '%s' via '%s': +%d sources",
                            fallback_query[:60], source_name, len(new),
                        )
                    except Exception as exc:
                        logger.debug("Fallback retriever '%s' failed: %s", source_name, exc)

        # --- Date-weighted trust adjustment before ranking (Feature 4b) ---
        date_weighted = _apply_date_weighting(deduped)

        # --- Rank (with entity-overlap penalty) ---
        ranked = self._ranking.rank(date_weighted, claim=normalized_query)

        # --- Hard entity filter: remove clearly off-topic sources ---
        # Sources missing ALL anchor entities from the claim are almost certainly
        # unrelated. We remove them after ranking so only topically irrelevant
        # sources are dropped. A floor of 3 prevents over-filtering sparse claims.
        _HARD_FILTER_MIN = 3
        anchors, _ = extract_claim_entities(normalized_query)
        if anchors:
            filtered = [
                s for s in ranked
                if anchor_present(anchors, f"{s.title or ''} {s.snippet or ''}")
            ]
            if len(filtered) >= _HARD_FILTER_MIN:
                logger.info(
                    "Hard entity filter: removed %d off-topic sources (kept %d, anchors=%s)",
                    len(ranked) - len(filtered), len(filtered), anchors,
                )
                ranked = filtered
            else:
                logger.info(
                    "Hard entity filter skipped: only %d sources would remain (floor=%d)",
                    len(filtered), _HARD_FILTER_MIN,
                )

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
