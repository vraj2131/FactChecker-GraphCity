"""
Phase 10 prerequisite: VerifyClaimService.

Orchestrates the full fact-checking pipeline for a single claim:
  retrieval → NLI stance → LLM classification → confidence scoring

Returns a VerifyClaimResult bundle that graph_builder_service consumes
to build the GraphResponse.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.models.llm_model import LLMResult, GroqLLMModel, get_groq_llm_model
from backend.app.models.nli_model import NLIModel, NLIResult
from backend.app.preprocessing.entity_extractor import extract_claim_entities, anchor_present
from backend.app.schemas.source_schema import Source
from backend.app.services.cache_service import CacheService
from backend.app.services.confidence_service import ConfidenceOutput, ConfidenceService
from backend.app.services.evidence_expansion_service import EvidenceExpansionService
from backend.app.services.ranking_service import RankingService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.stance_service import StanceService
from backend.app.utils.constants import (
    CONTEXT_EXPANSION_ENABLED,
    DEFAULT_RETRIEVAL_CACHE_DIR,
    GROQ_MODEL_NAME,
    LLM_MAX_INPUT_SOURCES,
    NLI_CONFIRM_MODEL_NAME,
)
from backend.app.retrieval.retriever_registry import RetrieverRegistry

logger = logging.getLogger(__name__)


@dataclass
class VerifyClaimResult:
    """Full pipeline output for a single claim — input to graph_builder_service."""

    claim_text: str
    sources: List[Source]                    # retrieved + NLI-classified sources
    nli_results: Dict[int, NLIResult]        # keyed by 0-based source index
    llm_result: LLMResult                    # LLM verdict + per-source classifications
    confidence_output: ConfidenceOutput      # calibrated confidence + verdict
    llm_input_sources: List[Source] = field(default_factory=list)  # top-N sent to LLM


class VerifyClaimService:
    """
    Wires together RetrievalService, StanceService, LLM, and ConfidenceService
    into a single `.verify(claim_text)` call.

    Designed for direct use from:
    - FastAPI endpoint (Phase 11)
    - graph_builder_service (Phase 10)
    - scripts / notebooks

    Usage:
        svc = VerifyClaimService.build_default()
        result = svc.verify("The Eiffel Tower is in Paris.")
    """

    def __init__(
        self,
        retrieval_svc: RetrievalService,
        stance_svc: StanceService,
        llm_model: GroqLLMModel,
        confidence_svc: ConfidenceService,
        max_retrieval_results: int = 60,
        llm_input_sources: int = LLM_MAX_INPUT_SOURCES,
        context_expansion_svc=None,   # Optional[ContextExpansionService]
    ) -> None:
        self._retrieval = retrieval_svc
        self._stance = stance_svc
        self._llm = llm_model
        self._confidence = confidence_svc
        self._max_retrieval_results = max_retrieval_results
        self._llm_input_sources = llm_input_sources
        self._context_expansion = context_expansion_svc

    # Maps source group names → registered retriever source names
    _GROUP_TO_SOURCES: dict = {
        "wikipedia":  ["wikipedia"],
        "live_news":  ["livewiki", "guardian", "newsapi", "gdelt"],
        "factcheck":  ["factcheck"],
        "web_search": ["duckduckgo"],
    }
    # Maps source group names → domain tags used by domain router (Feature 11)
    _GROUP_TO_DOMAINS: dict = {
        "scientific": {"science", "health"},
        "financial":  {"economics", "crypto"},
    }

    def verify(
        self,
        claim_text: str,
        use_cache: bool = True,
        context_claims: Optional[List] = None,
        include_social: bool = False,
        enabled_source_groups: Optional[List[str]] = None,
    ) -> VerifyClaimResult:
        """
        Run the full pipeline for a single claim.

        Args:
            claim_text:  The claim to fact-check.
            use_cache:   Whether to use cached retrieval + LLM results.

        Returns:
            VerifyClaimResult with all pipeline outputs populated.
        """
        claim_text = claim_text.strip()
        logger.info("VerifyClaimService: verifying claim='%s'", claim_text[:80])

        # Resolve active source groups
        _DEFAULT_GROUPS = {"wikipedia", "live_news", "factcheck", "scientific", "financial"}
        active_groups: set = set(enabled_source_groups) if enabled_source_groups is not None else _DEFAULT_GROUPS
        # "social" in enabled groups takes priority over the separate include_social flag
        _include_social = include_social or ("social" in active_groups)

        # Build retriever source list for main retrieval from active non-domain groups
        _active_retriever_sources: list = []
        for grp, srcs in self._GROUP_TO_SOURCES.items():
            if grp in active_groups:
                _active_retriever_sources.extend(srcs)

        # 1. Retrieve direct evidence from selected sources
        sources = self._retrieval.retrieve(
            claim_text,
            max_results=self._max_retrieval_results,
            use_cache=use_cache,
            sources=_active_retriever_sources if _active_retriever_sources else None,
        )
        logger.info("VerifyClaimService: retrieved %d direct sources", len(sources))

        # 1a. Social media retrieval (gated by 'social' group or include_social flag)
        if _include_social:
            from backend.app.retrieval.reddit_retriever import RedditRetriever
            from backend.app.retrieval.bluesky_retriever import BlueskyRetriever
            existing_ids = {s.source_id for s in sources}
            for retriever in [RedditRetriever(), BlueskyRetriever()]:
                try:
                    for src in retriever.retrieve(claim_text, max_results=5):
                        if src.source_id not in existing_ids:
                            sources.append(src)
                            existing_ids.add(src.source_id)
                except Exception as exc:
                    logger.warning("Social retriever %s failed: %s", retriever.source_name, exc)
            logger.info("VerifyClaimService: %d sources after social media retrieval", len(sources))

        # 1b. Domain-specific retrieval (Feature 11) — gated by 'scientific'/'financial' groups
        from backend.app.utils.domain_router import detect_domains
        _allowed_domains: set = set()
        for grp, domain_tags in self._GROUP_TO_DOMAINS.items():
            if grp in active_groups:
                _allowed_domains.update(domain_tags)
        domains = detect_domains(claim_text) & _allowed_domains
        if domains:
            _DOMAIN_RETRIEVERS = {
                "science": ["openalex", "arxiv"],
                "health": ["pubmed", "openalex"],
                "economics": ["fred", "worldbank", "sec_edgar"],
                "crypto": ["coingecko"],
            }
            _RETRIEVER_MAP = {
                "openalex": lambda: __import__(
                    "backend.app.retrieval.openalex_retriever",
                    fromlist=["OpenAlexRetriever"],
                ).OpenAlexRetriever(),
                "arxiv": lambda: __import__(
                    "backend.app.retrieval.arxiv_retriever",
                    fromlist=["ArxivRetriever"],
                ).ArxivRetriever(),
                "pubmed": lambda: __import__(
                    "backend.app.retrieval.pubmed_retriever",
                    fromlist=["PubMedRetriever"],
                ).PubMedRetriever(),
                "sec_edgar": lambda: __import__(
                    "backend.app.retrieval.sec_edgar_retriever",
                    fromlist=["SecEdgarRetriever"],
                ).SecEdgarRetriever(),
                "fred": lambda: __import__(
                    "backend.app.retrieval.fred_retriever",
                    fromlist=["FredRetriever"],
                ).FredRetriever(),
                "coingecko": lambda: __import__(
                    "backend.app.retrieval.coingecko_retriever",
                    fromlist=["CoinGeckoRetriever"],
                ).CoinGeckoRetriever(),
                "worldbank": lambda: __import__(
                    "backend.app.retrieval.worldbank_retriever",
                    fromlist=["WorldBankRetriever"],
                ).WorldBankRetriever(),
            }
            existing_ids = {s.source_id for s in sources}
            seen_domain_retrievers: set = set()
            for domain in sorted(domains):
                for retriever_key in _DOMAIN_RETRIEVERS.get(domain, []):
                    if retriever_key in seen_domain_retrievers:
                        continue
                    seen_domain_retrievers.add(retriever_key)
                    try:
                        retriever = _RETRIEVER_MAP[retriever_key]()
                        for src in retriever.retrieve(claim_text, max_results=5):
                            if src.source_id not in existing_ids:
                                sources.append(src)
                                existing_ids.add(src.source_id)
                    except Exception as exc:
                        logger.warning(
                            "Domain retriever %s failed: %s", retriever_key, exc
                        )
            logger.info(
                "VerifyClaimService: %d sources after domain retrieval (domains=%s)",
                len(sources),
                domains,
            )

        # 1c. Context expansion — contributing-factor sources (optional)
        if self._context_expansion is not None and CONTEXT_EXPANSION_ENABLED:
            context_sources = self._context_expansion.retrieve_context_sources(
                claim_text, self._retrieval, use_cache=use_cache
            )
            if context_sources:
                existing_ids = {s.source_id for s in sources}
                added = 0
                for cs in context_sources:
                    if cs.source_id not in existing_ids:
                        sources.append(cs)
                        existing_ids.add(cs.source_id)
                        added += 1
                logger.info(
                    "VerifyClaimService: added %d context sources (%d already present)",
                    added, len(context_sources) - added,
                )
                # Re-rank the merged pool so high-trust context sources
                # (Guardian/NewsAPI relevance_score=1.0, trust_score=0.82-0.88)
                # aren't buried behind low-relevance FAISS sources at the top.
                sources.sort(
                    key=lambda s: s.trust_score * s.relevance_score,
                    reverse=True,
                )
                # Re-apply entity filter after context expansion merge —
                # context sources bypass retrieval_service's hard filter.
                _anchors, _ = extract_claim_entities(claim_text)
                if _anchors and len(sources) > 3:
                    _filtered = [
                        s for s in sources
                        if anchor_present(_anchors, f"{s.title or ''} {s.snippet or ''}")
                    ]
                    if len(_filtered) >= 3:
                        logger.info(
                            "Post-expansion entity filter: removed %d off-topic sources",
                            len(sources) - len(_filtered),
                        )
                        sources = _filtered

        if not sources:
            logger.warning("VerifyClaimService: no sources retrieved for claim='%s'", claim_text[:80])
            from backend.app.models.llm_model import LLMResult, SourceClassification
            empty_llm = LLMResult(
                overall_verdict="insufficient",
                confidence=0.0,
                best_source_index=None,
                short_explanation="No sources retrieved.",
                sources=[],
            )
            from backend.app.services.confidence_service import ConfidenceOutput
            empty_conf = ConfidenceOutput(
                overall_confidence=0.05,
                overall_verdict="not_enough_info",
                support_score=0.0,
                refute_score=0.0,
                evidence_quality=0.0,
                corroboration=0.0,
                coverage=0.0,
                raw_confidence=0.0,
            )
            return VerifyClaimResult(
                claim_text=claim_text,
                sources=[],
                nli_results={},
                llm_result=empty_llm,
                confidence_output=empty_conf,
                llm_input_sources=[],
            )

        # 2. NLI stance classification
        classified_sources, nli_results = self._stance.classify(claim_text, sources)
        logger.info("VerifyClaimService: NLI classified %d sources", len(classified_sources))

        # 3. LLM classification (top N sources only)
        llm_input = classified_sources[: self._llm_input_sources]
        llm_result = self._llm.classify(claim_text, llm_input, use_cache=use_cache, context_claims=context_claims)
        logger.info(
            "VerifyClaimService: LLM verdict=%s conf=%.2f",
            llm_result.overall_verdict,
            llm_result.confidence,
        )

        # 4. Confidence scoring
        confidence_output = self._confidence.compute_main_confidence(
            llm_result=llm_result,
            sources=llm_input,
            nli_results=nli_results,
        )
        logger.info(
            "VerifyClaimService: verdict=%s confidence=%.2f",
            confidence_output.overall_verdict,
            confidence_output.overall_confidence,
        )

        return VerifyClaimResult(
            claim_text=claim_text,
            sources=classified_sources,
            nli_results=nli_results,
            llm_result=llm_result,
            confidence_output=confidence_output,
            llm_input_sources=llm_input,
        )

    @classmethod
    def build_default(
        cls,
        cache_dir: Path = DEFAULT_RETRIEVAL_CACHE_DIR,
        groq_model: str = GROQ_MODEL_NAME,
        use_nli_cascade: bool = True,
    ) -> "VerifyClaimService":
        """
        Build a fully wired VerifyClaimService with default settings.

        Registers all retrievers (wikipedia FAISS, livewiki, factcheck,
        guardian, newsapi, gdelt) and uses Groq as the LLM backend.

        Args:
            cache_dir:        Directory for retrieval + NLI cache files.
            groq_model:       Groq model ID to use.
            use_nli_cascade:  If True, run a second stronger NLI model on
                              borderline supports/refutes classifications.
        """
        from backend.app.retrieval.duckduckgo_retriever import DuckDuckGoRetriever
        from backend.app.retrieval.factcheck_retriever import FactCheckRetriever
        from backend.app.retrieval.gdelt_retriever import GDELTRetriever as GdeltRetriever
        from backend.app.retrieval.guardian_retriever import GuardianRetriever
        from backend.app.retrieval.livewiki_retriever import LiveWikiRetriever
        from backend.app.retrieval.newsapi_retriever import NewsApiRetriever
        from backend.app.retrieval.wikipedia_retriever import WikipediaRetriever

        registry = RetrieverRegistry()
        registry.register(WikipediaRetriever())
        registry.register(LiveWikiRetriever())
        registry.register(FactCheckRetriever())
        registry.register(GuardianRetriever())
        registry.register(NewsApiRetriever())
        registry.register(GdeltRetriever())
        registry.register(DuckDuckGoRetriever())

        cache = CacheService(cache_dir)
        retrieval_svc = RetrievalService(
            registry=registry,
            cache=cache,
            ranking=RankingService(),
            expansion=EvidenceExpansionService(),
        )

        nli_model = NLIModel()
        confirm_model = NLIModel(model_name=NLI_CONFIRM_MODEL_NAME) if use_nli_cascade else None
        stance_svc = StanceService(
            model=nli_model,
            cache=cache,
            confirm_model=confirm_model,
        )

        llm_model = get_groq_llm_model(model_name=groq_model)
        confidence_svc = ConfidenceService()

        # Context expansion — build lazily so missing GROQ_API_KEY doesn't block
        from backend.app.services.context_expansion_service import ContextExpansionService
        try:
            context_expansion_svc = ContextExpansionService.build_default(
                model_name=groq_model,
            )
        except Exception as exc:
            logger.warning("ContextExpansionService unavailable: %s", exc)
            context_expansion_svc = None

        return cls(
            retrieval_svc=retrieval_svc,
            stance_svc=stance_svc,
            llm_model=llm_model,
            confidence_svc=confidence_svc,
            context_expansion_svc=context_expansion_svc,
        )
