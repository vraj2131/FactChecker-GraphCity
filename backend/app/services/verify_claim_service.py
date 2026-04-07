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
from typing import Dict, List, Optional

from backend.app.models.llm_model import LLMResult, GroqLLMModel, get_groq_llm_model
from backend.app.models.nli_model import NLIModel, NLIResult
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
        max_retrieval_results: int = 20,
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

    def verify(self, claim_text: str, use_cache: bool = True) -> VerifyClaimResult:
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

        # 1. Retrieve direct evidence from all sources
        sources = self._retrieval.retrieve(
            claim_text,
            max_results=self._max_retrieval_results,
            use_cache=use_cache,
        )
        logger.info("VerifyClaimService: retrieved %d direct sources", len(sources))

        # 1b. Context expansion — contributing-factor sources (optional)
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
        llm_result = self._llm.classify(claim_text, llm_input, use_cache=use_cache)
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
        use_nli_cascade: bool = False,
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
