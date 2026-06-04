from typing import Any, Dict, List, Optional, Set

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_RETRIEVER_MAX_RESULTS,
    SOURCE_NAME_DUCKDUCKGO,
)

TRUST_SCORE_DUCKDUCKGO = 0.60


class DuckDuckGoRetriever(BaseRetriever):
    """Web retriever using the duckduckgo-search library (free, no API key)."""

    STOPWORDS: Set[str] = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
        "and", "or", "but", "if", "then", "than", "into", "over", "under",
        "after", "before", "about", "this", "that", "it", "its",
        "we", "they", "he", "she", "you", "i",
    }

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_DUCKDUCKGO)

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                return []

        try:
            results = list(DDGS().text(query, max_results=max_results * 2))
            return results
        except Exception:
            return []

    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Source]:
        if not raw_data:
            return []

        query_tokens = self._tokenize(query)
        sources: List[Source] = []

        for item in raw_data:
            title = self._safe_strip(item.get("title"))
            url = self._safe_strip(item.get("href"))
            body = self._safe_strip(item.get("body"))

            if not url or not title:
                continue

            snippet = body or title
            relevance = self._compute_relevance(query_tokens, title, snippet)

            source_id = f"duckduckgo::{abs(hash(url))}"
            sources.append(
                Source(
                    source_id=source_id,
                    source_type=SOURCE_NAME_DUCKDUCKGO,
                    title=title,
                    url=url,
                    publisher=self._extract_domain(url),
                    snippet=snippet,
                    published_at=None,
                    trust_score=TRUST_SCORE_DUCKDUCKGO,
                    relevance_score=relevance,
                    stance_hint=None,
                )
            )

        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return self.postprocess(sources, max_results=max_results)

    def _compute_relevance(
        self, query_tokens: Set[str], title: Optional[str], snippet: Optional[str]
    ) -> float:
        title_tokens = self._tokenize(title or "")
        snippet_tokens = self._tokenize(snippet or "")

        title_overlap = len(query_tokens & title_tokens)
        snippet_overlap = len(query_tokens & snippet_tokens)

        if not query_tokens:
            return 0.5

        raw = (3 * title_overlap + snippet_overlap) / (3 * len(query_tokens))
        return min(raw, 1.0)

    @classmethod
    def _tokenize(cls, text: str) -> Set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {
            tok for tok in cleaned.split()
            if len(tok) >= 2 and tok not in cls.STOPWORDS
        }

    @staticmethod
    def _safe_strip(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc or url
        except Exception:
            return url
