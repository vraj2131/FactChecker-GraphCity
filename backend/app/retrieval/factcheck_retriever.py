from typing import Any, Dict, List, Optional

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_RETRIEVER_MAX_RESULTS,
    DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
    SOURCE_NAME_FACTCHECK,
)


class FactCheckRetriever(BaseRetriever):
    """
    Google Fact Check Tools API retriever.

    Notes:
    - Uses the claims:search endpoint
    - Returns normalized Source objects
    - Gracefully returns [] if no API key is provided
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "en-US",
    ) -> None:
        super().__init__(source_name=SOURCE_NAME_FACTCHECK)
        self.api_key = api_key
        self.language_code = language_code

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetch raw claim search results from Google Fact Check Tools API.

        Sends multiple query variants (original + entity-only + debunk variant)
        and merges unique claims. This significantly improves coverage for science
        myths and viral claims where the exact phrasing differs from indexed claims.
        """
        if not self.api_key:
            return {"claims": []}

        page_size = kwargs.get("page_size", max_results)
        review_publisher_site_filter = kwargs.get("review_publisher_site_filter")

        query_variants = self._build_query_variants(query)
        merged_claims: List[Dict[str, Any]] = []
        seen_urls: set = set()

        for variant in query_variants:
            params = {
                "query": variant,
                "key": self.api_key,
                "languageCode": self.language_code,
                "pageSize": page_size,
            }
            if review_publisher_site_filter:
                params["reviewPublisherSiteFilter"] = review_publisher_site_filter

            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                data = response.json()
            except Exception:
                continue

            for claim in data.get("claims", []) or []:
                for review in claim.get("claimReview", []) or []:
                    url = (review.get("url") or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        # Only add claim once (first review url as dedup key)
                        break
                else:
                    continue
                merged_claims.append(claim)

        return {"claims": merged_claims}

    @staticmethod
    def _build_query_variants(query: str) -> List[str]:
        """
        Build up to 4 query variants for the Fact Check Tools API.

        1. Original query (full sentence)
        2. Key entities only (drops stopwords / verbs / numbers)
        3. Debunk variant: key entities + "false" — surfaces myth-busting checks
        4. Short entity variant: first 2-3 meaningful tokens only
        """
        _STOPS = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
            "and", "or", "but", "if", "then", "than", "into", "over", "under",
            "after", "before", "during", "about", "between", "through",
            "this", "that", "these", "those", "it", "its", "only", "just",
        }
        raw_tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in query.split()]
        entity_tokens = [
            tok for tok in raw_tokens
            if tok and tok.lower() not in _STOPS and not tok.isdigit() and len(tok) >= 2
        ]

        variants: List[str] = []
        seen: set = set()

        def add(v: str) -> None:
            v = v.strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                variants.append(v)

        add(query)
        if entity_tokens:
            add(" ".join(entity_tokens))
            add(" ".join(entity_tokens[:3]))
            # Debunk variant: surfaces "X is false / myth / debunked" fact-checks
            add(" ".join(entity_tokens[:4]) + " false")

        return variants[:4]

    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Source]:
        """
        Convert raw Google Fact Check Tools API response into normalized Source objects.
        """
        if not raw_data or "claims" not in raw_data:
            return []

        claims = raw_data.get("claims", [])
        if not claims:
            return []

        sources: List[Source] = []

        for claim in claims:
            claim_text = self._safe_strip(claim.get("text"))
            claimant = self._safe_strip(claim.get("claimant"))
            claim_date = self._safe_strip(claim.get("claimDate"))

            claim_reviews = claim.get("claimReview", []) or []
            if not claim_reviews:
                continue

            for review in claim_reviews:
                publisher_obj = review.get("publisher", {}) or {}
                publisher_name = self._safe_strip(publisher_obj.get("name"))
                publisher_site = self._safe_strip(publisher_obj.get("site"))
                review_url = self._safe_strip(review.get("url"))
                review_title = self._safe_strip(review.get("title"))
                textual_rating = self._safe_strip(review.get("textualRating"))
                review_date = self._safe_strip(review.get("reviewDate"))
                language_code = self._safe_strip(review.get("languageCode"))

                if not review_url or not claim_text:
                    continue

                title = review_title or f"Fact Check Review: {claim_text[:120]}"
                snippet = self._build_snippet(
                    claim_text=claim_text,
                    claimant=claimant,
                    textual_rating=textual_rating,
                    publisher_name=publisher_name,
                    claim_date=claim_date,
                    review_date=review_date,
                    language_code=language_code,
                )

                stance_hint = self._infer_stance_hint(textual_rating)
                trust_score = self._estimate_trust_score(
                    publisher_name=publisher_name,
                    publisher_site=publisher_site,
                    textual_rating=textual_rating,
                )
                relevance_score = self._estimate_relevance_score(
                    query=query,
                    claim_text=claim_text,
                    review_title=review_title,
                )

                source_id = self._build_source_id(
                    review_url=review_url,
                    claim_text=claim_text,
                )

                source = Source(
                    source_id=source_id,
                    source_type="factcheck",
                    title=title,
                    url=review_url,
                    publisher=publisher_name or publisher_site,
                    snippet=snippet,
                    published_at=review_date or claim_date,
                    trust_score=trust_score,
                    relevance_score=relevance_score,
                    stance_hint=stance_hint,
                )
                sources.append(source)

        return self.postprocess(sources, max_results=max_results)

    @staticmethod
    def _safe_strip(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _build_source_id(review_url: str, claim_text: str) -> str:
        compact_claim = "".join(ch.lower() if ch.isalnum() else "_" for ch in claim_text)[:60]
        return f"factcheck::{compact_claim}::{abs(hash(review_url))}"

    @staticmethod
    def _build_snippet(
        claim_text: Optional[str],
        claimant: Optional[str],
        textual_rating: Optional[str],
        publisher_name: Optional[str],
        claim_date: Optional[str],
        review_date: Optional[str],
        language_code: Optional[str],
    ) -> str:
        # Build a natural-language sentence so NLI can derive entailment/contradiction.
        # "Claim reviewed: X | Rating: Y" is structured metadata — NLI models can't
        # reliably infer supports/refutes from it.
        if not claim_text:
            return ""

        parts = [claim_text.strip()]

        if textual_rating:
            parts.append(f"This claim was rated {textual_rating}.")

        if publisher_name:
            parts.append(f"Fact-checked by {publisher_name}.")

        return " ".join(parts)

    @staticmethod
    def _infer_stance_hint(textual_rating: Optional[str]) -> Optional[str]:
        if not textual_rating:
            return None

        rating = textual_rating.lower()

        refute_terms = [
            "false",
            "mostly false",
            "pants on fire",
            "incorrect",
            "misleading",
            "fake",
            "wrong",
            "not true",
        ]
        support_terms = [
            "true",
            "mostly true",
            "correct",
            "accurate",
        ]
        insufficient_terms = [
            "unproven",
            "unverified",
            "mixture",
            "mixed",
            "half true",
            "partly true",
            "needs context",
            "missing context",
        ]

        if any(term in rating for term in refute_terms):
            return "refutes"
        if any(term in rating for term in support_terms):
            return "supports"
        if any(term in rating for term in insufficient_terms):
            return "insufficient"

        return None

    @staticmethod
    def _estimate_trust_score(
        publisher_name: Optional[str],
        publisher_site: Optional[str],
        textual_rating: Optional[str],
    ) -> float:
        """
        Simple heuristic trust score for Phase 4.
        """
        score = 0.85

        if publisher_name:
            score += 0.05
        if publisher_site:
            score += 0.03
        if textual_rating:
            score += 0.02

        return min(score, 0.98)

    @staticmethod
    def _estimate_relevance_score(
        query: str,
        claim_text: Optional[str],
        review_title: Optional[str],
    ) -> float:
        """
        Rough lexical relevance score for Phase 4 smoke testing.
        """
        query_tokens = FactCheckRetriever._tokenize(query)
        if not query_tokens:
            return 0.0

        claim_tokens = FactCheckRetriever._tokenize(claim_text or "")
        title_tokens = FactCheckRetriever._tokenize(review_title or "")

        overlap = len(query_tokens & (claim_tokens | title_tokens))
        if overlap <= 0:
            return 0.0

        return min(overlap / max(len(query_tokens), 1), 1.0)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
        return {token for token in cleaned.split() if len(token) >= 2}
