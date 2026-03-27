import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from backend.app.preprocessing.snippet_extractor import score_sentence_relevance
from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_LIVEWIKI_MAX_RESULTS,
    DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
    SOURCE_NAME_LIVEWIKI,
)

logger = logging.getLogger(__name__)


class LiveWikiRetriever(BaseRetriever):
    """
    Retrieves evidence from live Wikipedia via the public MediaWiki API.

    No API key required. Two-step process:
    1. Search Wikipedia for pages matching the claim query.
    2. Fetch the plain-text summary (first paragraph) for each result page.

    The summary extract is used directly as the snippet — it is already
    clean, human-readable prose and works well for NLI inference.

    Differs from WikipediaRetriever (FEVER-based FAISS):
    - Not limited to the frozen FEVER evidence set
    - Returns the actual Wikipedia page intro, not individual sentences
    - Better coverage for well-known entities not prominent in FEVER
    """

    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    HEADERS = {
        "User-Agent": "FactCheckerBot/1.0 (educational fact-checking research project)",
        "Accept": "application/json",
    }

    def __init__(self, max_search_results: int = DEFAULT_LIVEWIKI_MAX_RESULTS) -> None:
        super().__init__(source_name=SOURCE_NAME_LIVEWIKI)
        self._max_search_results = max_search_results

    # ------------------------------------------------------------------
    # BaseRetriever interface
    # ------------------------------------------------------------------

    def fetch_raw(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Search Wikipedia and fetch summaries for the top matching pages.

        Returns a list of page dicts:
            {
                "pageid":  int,
                "title":   str,
                "extract": str,   # plain-text first paragraph
                "url":     str,   # desktop page URL
            }

        Returns [] if the search request fails (errors are logged, not raised).
        Individual page summary failures are skipped with a warning.
        """
        search_limit = min(self._max_search_results, max(max_results, 1))
        titles = self._search(query, limit=search_limit)

        if not titles:
            logger.info("LiveWikiRetriever: no search results for query='%s'", query)
            return []

        pages: List[Dict] = []
        for title in titles:
            page = self._fetch_summary(title)
            if page:
                pages.append(page)

        logger.info(
            "LiveWikiRetriever: fetched %d page summaries for query='%s'",
            len(pages),
            query,
        )
        return pages

    def normalize(
        self,
        raw_data: List[Dict],
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> List[Source]:
        """
        Convert page summary dicts into Source objects.

        Each page's plain-text extract is used as the snippet.
        Pages with zero relevance to the query are dropped.
        Results are sorted by relevance_score descending.
        """
        if not raw_data:
            return []

        sources: List[Source] = []

        for page in raw_data:
            title = page.get("title", "").strip()
            extract = (page.get("extract") or "").strip()
            url = page.get("url", "").strip()
            pageid = page.get("pageid")

            if not title or not url:
                continue

            # Use only the first sentence of the extract as the snippet so
            # the NLI model sees a tight, claim-relevant statement.
            snippet = self._first_sentence(extract) if extract else None

            relevance = (
                score_sentence_relevance(query, snippet)
                if snippet
                else 0.0
            )

            # Drop pages with no claim-term overlap at all
            if relevance == 0.0:
                logger.debug(
                    "LiveWikiRetriever: dropping zero-relevance page '%s'", title
                )
                continue

            source_id = f"livewiki::{pageid}" if pageid else f"livewiki::{title[:60]}"

            source = Source(
                source_id=source_id,
                source_type=SOURCE_NAME_LIVEWIKI,
                title=f"Wikipedia: {title}",
                url=url,
                publisher="Wikipedia",
                snippet=snippet,
                trust_score=0.80,
                relevance_score=min(relevance, 1.0),
            )
            sources.append(source)

        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return self.postprocess(sources, max_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search(self, query: str, limit: int) -> List[str]:
        """
        Call the MediaWiki search API and return a list of page titles.
        Returns [] on any network or parse error.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        try:
            response = requests.get(
                self.SEARCH_URL,
                params=params,
                headers=self.HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("query", {}).get("search", [])
            return [r["title"] for r in results if r.get("title")]
        except Exception as exc:
            logger.warning("LiveWikiRetriever: search failed for query='%s': %s", query, exc)
            return []

    def _fetch_summary(self, title: str) -> Optional[Dict]:
        """
        Call the REST summary API for a single page title.
        Returns None on any failure.
        """
        encoded_title = quote(title.replace(" ", "_"), safe="")
        url = self.SUMMARY_URL.format(title=encoded_title)
        try:
            response = requests.get(
                url,
                headers=self.HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

            extract = (data.get("extract") or "").strip()
            desktop_url = (
                data.get("content_urls", {})
                    .get("desktop", {})
                    .get("page", "")
            ) or f"https://en.wikipedia.org/wiki/{encoded_title}"
            pageid = data.get("pageid")

            if not extract:
                logger.debug("LiveWikiRetriever: empty extract for page '%s'", title)
                return None

            return {
                "pageid": pageid,
                "title": data.get("title", title),
                "extract": extract,
                "url": desktop_url,
            }
        except Exception as exc:
            logger.warning(
                "LiveWikiRetriever: summary fetch failed for page '%s': %s", title, exc
            )
            return None

    @staticmethod
    def _first_sentence(text: str) -> Optional[str]:
        """
        Extract the first sentence from a plain-text paragraph.
        Falls back to the full text (truncated to 500 chars) if no sentence
        boundary is found.
        """
        for i, ch in enumerate(text):
            if ch in ".!?" and i + 1 < len(text) and text[i + 1] in (" ", "\n"):
                sentence = text[: i + 1].strip()
                if sentence:
                    return sentence
        # No sentence boundary found — return the full extract, capped at 500 chars
        return text[:500].strip() or None
