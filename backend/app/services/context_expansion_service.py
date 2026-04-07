"""
Context Expansion Service.

For any claim, uses Groq to automatically generate 2-3 "contributing factor"
search queries — background context, causal factors, or enabling conditions
that could explain why the claim is true or false.

Examples:
  Claim : "Amazon stock rose by 5%."
  Generated:
    - "Amazon AWS cloud revenue earnings 2025"
    - "US tech sector Wall Street rally"
    - "Federal Reserve interest rates technology stocks"

  Claim : "Vaccines cause autism."
  Generated:
    - "CDC vaccine safety clinical trials research"
    - "autism spectrum disorder genetic environmental causes"
    - "RFK Jr health policy misinformation"

These context sources are retrieved through the live retrievers (Guardian,
NewsAPI, LiveWiki, GDELT), merged with the direct evidence pool, and flow
through NLI + LLM classification normally. The LLM correctly classifies them
as `correlated_context`, which maps to `context_signal` nodes in the graph.

No hardcoding — the LLM generates domain-appropriate queries for any input.
"""

import json
import logging
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

from backend.app.schemas.source_schema import Source
from backend.app.services.cache_service import CacheService
from backend.app.utils.constants import (
    CONTEXT_EXPANSION_CACHE_NAMESPACE,
    CONTEXT_EXPANSION_ENABLED,
    CONTEXT_EXPANSION_MAX_QUERIES,
    CONTEXT_EXPANSION_MAX_RESULTS_PER_QUERY,
    CONTEXT_EXPANSION_PROMPT_VERSION,
    CONTEXT_EXPANSION_RETRIEVER_SOURCES,
    DEFAULT_CACHE_DIR,
    GROQ_MODEL_NAME,
)
from backend.app.utils.hashing import build_cache_key

logger = logging.getLogger(__name__)

# Prompt asking Groq to generate contributing-factor queries.
# Deliberately short to minimise tokens and latency (~0.3s on Groq free tier).
_SYSTEM_PROMPT = """\
You are a research assistant that identifies background causes and contextual factors behind factual claims.
Given a claim, generate short search queries for news/articles about CONTRIBUTING FACTORS — not about the claim itself.
Contributing factors include: policies, economic trends, scientific research, events, or conditions that could cause or explain the claim.
Respond with ONLY a valid JSON array of strings. No explanation. No markdown. No extra keys.\
"""

_USER_TEMPLATE = """\
Claim: "{claim}"

Generate exactly {n} short search queries (5-10 words each) for contributing-factor articles.
They must be about background context, causes, or enabling conditions — NOT the claim itself.

Respond with ONLY a JSON array, e.g.: ["query one", "query two", "query three"]\
"""


def _extract_json_array(text: str) -> List[str]:
    """
    Extract a JSON array from raw LLM output.
    Handles markdown fences and leading/trailing noise.
    """
    # Strip markdown fences
    text = re.sub(r"```[a-z]*", "", text).strip()
    # Find first [ ... ] block
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM output: {text[:200]!r}")
    parsed = json.loads(match.group())
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)}")
    return [str(q).strip() for q in parsed if str(q).strip()]


class ContextExpansionService:
    """
    Generates and retrieves contributing-factor sources for a claim.

    Usage (standalone):
        svc = ContextExpansionService.build_default()
        queries = svc.generate_context_queries("Amazon stock rose by 5%.")
        # → ["Amazon AWS earnings 2025", "US tech sector rally", ...]

    Usage (inside VerifyClaimService):
        context_sources = svc.retrieve_context_sources(claim, retrieval_svc)
    """

    def __init__(
        self,
        groq_client: Groq,
        cache: CacheService,
        model_name: str = GROQ_MODEL_NAME,
        max_queries: int = CONTEXT_EXPANSION_MAX_QUERIES,
        max_results_per_query: int = CONTEXT_EXPANSION_MAX_RESULTS_PER_QUERY,
        retriever_sources: Optional[List[str]] = None,
        enabled: bool = CONTEXT_EXPANSION_ENABLED,
    ) -> None:
        self._client = groq_client
        self._cache = cache
        self._model = model_name
        self._max_queries = max_queries
        self._max_results = max_results_per_query
        self._retriever_sources = retriever_sources or CONTEXT_EXPANSION_RETRIEVER_SOURCES
        self._enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_context_queries(self, claim: str, use_cache: bool = True) -> List[str]:
        """
        Ask Groq to generate contributing-factor search queries for `claim`.

        Returns up to `max_queries` short query strings, or [] on failure.
        Results are cached so repeated calls for the same claim are free.
        """
        if not self._enabled:
            return []

        claim = claim.strip()
        if not claim:
            return []

        cache_key = build_cache_key(
            source_name=CONTEXT_EXPANSION_CACHE_NAMESPACE,
            query=claim,
            n=self._max_queries,
            prompt_version=CONTEXT_EXPANSION_PROMPT_VERSION,
        )

        if use_cache:
            cached = self._cache.load(CONTEXT_EXPANSION_CACHE_NAMESPACE, cache_key)
            if cached is not None:
                logger.info(
                    "ContextExpansionService: cache hit for claim='%s'", claim[:60]
                )
                return cached if isinstance(cached, list) else []

        user_msg = _USER_TEMPLATE.format(claim=claim, n=self._max_queries)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.3,   # low temperature → consistent, focused queries
                max_tokens=150,    # array of 3 short strings fits in ~80 tokens
            )
            raw = response.choices[0].message.content or ""
            queries = _extract_json_array(raw)[: self._max_queries]
        except Exception as exc:
            logger.warning(
                "ContextExpansionService: query generation failed for claim='%s': %s",
                claim[:60], exc,
            )
            return []

        if use_cache and queries:
            try:
                self._cache.save(CONTEXT_EXPANSION_CACHE_NAMESPACE, cache_key, queries)
            except Exception as exc:
                logger.warning("ContextExpansionService: cache save failed: %s", exc)

        logger.info(
            "ContextExpansionService: generated %d context queries for claim='%s': %s",
            len(queries), claim[:60], queries,
        )
        return queries

    def retrieve_context_sources(
        self,
        claim: str,
        retrieval_svc,   # RetrievalService — avoid circular import
        use_cache: bool = True,
    ) -> List[Source]:
        """
        Generate context queries and retrieve sources for each.

        Returns a flat, deduplicated list of Source objects from all context
        queries. Sources that were already retrieved as direct evidence are
        deduped by the caller (VerifyClaimService) using source_id.

        Args:
            claim:         The original claim text.
            retrieval_svc: A RetrievalService instance (injected to avoid
                           circular import).
            use_cache:     Passed to both query generation and retrieval.
        """
        if not self._enabled:
            return []

        queries = self.generate_context_queries(claim, use_cache=use_cache)
        if not queries:
            return []

        all_sources: List[Source] = []
        seen_ids: set = set()

        for query in queries:
            try:
                sources = retrieval_svc.retrieve(
                    query=query,
                    max_results=self._max_results,
                    sources=self._retriever_sources,
                    use_cache=use_cache,
                    expand_queries=False,   # no adversarial variants for context queries
                )
                for s in sources:
                    if s.source_id not in seen_ids:
                        seen_ids.add(s.source_id)
                        all_sources.append(s)
            except Exception as exc:
                logger.warning(
                    "ContextExpansionService: retrieval failed for query='%s': %s",
                    query[:60], exc,
                )

        logger.info(
            "ContextExpansionService: retrieved %d context sources for claim='%s'",
            len(all_sources), claim[:60],
        )
        return all_sources

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_default(
        cls,
        api_key: Optional[str] = None,
        model_name: str = GROQ_MODEL_NAME,
        cache_dir=DEFAULT_CACHE_DIR,
    ) -> "ContextExpansionService":
        """Build a ContextExpansionService with default settings."""
        load_dotenv()
        resolved_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Context expansion requires Groq."
            )
        return cls(
            groq_client=Groq(api_key=resolved_key),
            cache=CacheService(cache_dir),
            model_name=model_name,
        )
