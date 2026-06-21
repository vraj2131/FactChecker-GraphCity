import logging
from typing import Any, Dict, List, Optional

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_OPENALEX

logger = logging.getLogger(__name__)

TRUST_SCORE_OPENALEX = 0.90
_BASE_URL = "https://api.openalex.org/works"
_HEADERS = {"User-Agent": "FactCheckerBot/1.0 (mailto:factcheck@research.edu)"}


def _reconstruct_abstract(inv_index: Optional[Dict]) -> str:
    """Reconstruct prose from OpenAlex's inverted-index abstract format."""
    if not inv_index:
        return ""
    words: Dict[int, str] = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


class OpenAlexRetriever(BaseRetriever):
    """Retrieves scholarly works from OpenAlex (free, no API key required)."""

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_OPENALEX)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict]:
        try:
            params = {
                "search": query,
                "per-page": min(max_results, 10),
                "select": (
                    "id,title,doi,publication_year,"
                    "abstract_inverted_index,primary_location,"
                    "cited_by_count,open_access"
                ),
            }
            resp = requests.get(
                _BASE_URL,
                params=params,
                headers=_HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json().get("results", [])
        except Exception as exc:
            logger.warning("OpenAlexRetriever: fetch failed: %s", exc)
            return []

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Source]:
        if not raw_data:
            return []
        sources: List[Source] = []
        for item in raw_data:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            doi = item.get("doi") or ""
            if doi.startswith("http"):
                url = doi
            elif doi:
                url = f"https://doi.org/{doi}"
            else:
                loc = item.get("primary_location") or {}
                url = (
                    loc.get("landing_page_url")
                    or loc.get("pdf_url")
                    or item.get("id", "")
                    or ""
                )
            if not url or not url.startswith("http"):
                continue

            year = item.get("publication_year")
            abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))
            snippet = (
                abstract[:500]
                if abstract
                else f"Title: {title}. Published {year or 'n/a'} (OpenAlex)."
            )
            cites = item.get("cited_by_count") or 0
            relevance = min(1.0, 0.5 + (cites / 1000.0) * 0.3)

            sources.append(
                Source(
                    source_id=f"openalex::{abs(hash(url)) % (10**15)}",
                    source_type=SOURCE_NAME_OPENALEX,
                    title=title[:500],
                    url=url,
                    publisher="OpenAlex",
                    snippet=snippet,
                    published_at=str(year) if year else None,
                    trust_score=TRUST_SCORE_OPENALEX,
                    relevance_score=relevance,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)
