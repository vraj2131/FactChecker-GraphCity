"""
Phase 10 prerequisite: VerifyClaimService.

Orchestrates the full fact-checking pipeline for a single claim:
  retrieval → NLI stance → LLM classification → confidence scoring

Returns a VerifyClaimResult bundle that graph_builder_service consumes
to build the GraphResponse.
"""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
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
        deep_nli: bool = True,
    ) -> VerifyClaimResult:
        """
        Run the full pipeline for a single claim.

        Args:
            claim_text:  The claim to fact-check.
            use_cache:   Whether to use cached retrieval + LLM results.
            deep_nli:    If True, borderline NLI supports/refutes labels are
                         re-checked by the stronger confirm model. If False,
                         only the fast NLI model runs (faster, less accurate).

        Returns:
            VerifyClaimResult with all pipeline outputs populated.
        """
        claim_text = claim_text.strip()
        t_start = time.perf_counter()
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

        # ── Launch ALL retrieval work concurrently ──────────────────────────
        # Main retrieval, social, domain, and context expansion only need
        # claim_text, so nothing has to wait for anything else. Previously
        # these ran as sequential waves (main → social → domain → context),
        # stacking their timeout ceilings.
        executor = ThreadPoolExecutor(max_workers=10)
        _EXTRAS_DEADLINE_S = 14.0   # absolute budget for the side retrievals
        extra_futures: Dict[str, Future] = {}

        fut_main = executor.submit(
            self._retrieval.retrieve,
            claim_text,
            max_results=self._max_retrieval_results,
            use_cache=use_cache,
            sources=_active_retriever_sources if _active_retriever_sources else None,
        )

        if _include_social:
            from backend.app.retrieval.reddit_retriever import RedditRetriever
            from backend.app.retrieval.bluesky_retriever import BlueskyRetriever
            for retriever in (RedditRetriever(), BlueskyRetriever()):
                extra_futures[retriever.source_name] = executor.submit(
                    retriever.retrieve, claim_text, max_results=5
                )

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
            retriever_keys: set = set()
            for domain in sorted(domains):
                retriever_keys.update(_DOMAIN_RETRIEVERS.get(domain, []))
            for key in retriever_keys:
                extra_futures[key] = executor.submit(
                    lambda k=key: _RETRIEVER_MAP[k]().retrieve(claim_text, max_results=5)
                )

        if self._context_expansion is not None and CONTEXT_EXPANSION_ENABLED:
            extra_futures["context_expansion"] = executor.submit(
                self._context_expansion.retrieve_context_sources,
                claim_text,
                self._retrieval,
                use_cache=use_cache,
            )

        # ── Main retrieval result ───────────────────────────────────────────
        try:
            sources = fut_main.result(timeout=30.0)
        except Exception as exc:
            executor.shutdown(wait=False)
            raise RuntimeError(f"Main retrieval failed: {exc}") from exc
        logger.info(
            "VerifyClaimService: retrieved %d direct sources (%.1fs)",
            len(sources), time.perf_counter() - t_start,
        )

        # ── NLI overlap: classify main sources NOW, while the side
        # retrievals are still running in the pool. Results land in the
        # per-(claim,snippet) NLI cache, so the final classify pass over the
        # merged pool re-uses them for free.
        if sources:
            self._stance.classify(claim_text, sources, deep=deep_nli)
            logger.info(
                "VerifyClaimService: pre-classified %d main sources (%.1fs)",
                len(sources), time.perf_counter() - t_start,
            )

        # ── Collect side retrievals (most finished during NLI above) ────────
        context_added = False
        existing_ids = {s.source_id for s in sources}
        for label, fut in extra_futures.items():
            remaining = max(0.5, _EXTRAS_DEADLINE_S - (time.perf_counter() - t_start))
            try:
                results = fut.result(timeout=remaining)
            except Exception as exc:
                logger.warning("VerifyClaimService: '%s' retrieval skipped: %s", label, exc)
                continue
            added = 0
            for src in results or []:
                if src.source_id not in existing_ids:
                    sources.append(src)
                    existing_ids.add(src.source_id)
                    added += 1
            if added and label == "context_expansion":
                context_added = True
            logger.info("VerifyClaimService: '%s' added %d sources", label, added)
        executor.shutdown(wait=False)

        # Context sources bypass retrieval_service's ranking + hard entity
        # filter, so re-sort and re-filter the merged pool when they arrived.
        if context_added:
            sources.sort(
                key=lambda s: s.trust_score * s.relevance_score,
                reverse=True,
            )
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

        # 2. NLI stance classification over the final merged pool. Main
        # sources were pre-classified above and hit the NLI cache here; only
        # sources added by the side retrievals need fresh inference.
        classified_sources, nli_results = self._stance.classify(
            claim_text, sources, deep=deep_nli
        )
        logger.info(
            "VerifyClaimService: NLI classified %d sources (%.1fs)",
            len(classified_sources), time.perf_counter() - t_start,
        )

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
