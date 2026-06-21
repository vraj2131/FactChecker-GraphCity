import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, List

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_ARXIV

logger = logging.getLogger(__name__)

TRUST_SCORE_ARXIV = 0.85
_BASE_URL = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivRetriever(BaseRetriever):
    """Retrieves preprints from arXiv via their public Atom feed API (no key required)."""

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_ARXIV)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        try:
            params = {
                "search_query": f"all:{query}",
                "max_results": min(max_results, 10),
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            resp = requests.get(
                _BASE_URL,
                params=params,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            logger.warning("ArxivRetriever: fetch failed: %s", exc)
            return ""

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Source]:
        if not raw_data:
            return []
        sources: List[Source] = []
        try:
            root = ET.fromstring(raw_data)
        except ET.ParseError as exc:
            logger.warning("ArxivRetriever: XML parse error: %s", exc)
            return []

        for entry in root.findall("atom:entry", _NS):
            title = (entry.findtext("atom:title", "", _NS) or "").strip()
            title = re.sub(r"\s+", " ", title)
            if not title:
                continue

            summary = (entry.findtext("atom:summary", "", _NS) or "").strip()
            snippet = re.sub(r"\s+", " ", summary)[:500] if summary else f"arXiv preprint: {title}"

            url = ""
            for link in entry.findall("atom:link", _NS):
                if link.get("type") == "text/html":
                    url = link.get("href", "")
                    break
            if not url:
                url = (entry.findtext("atom:id", "", _NS) or "").strip()
            if not url or not url.startswith("http"):
                continue

            published = (entry.findtext("atom:published", "", _NS) or "")[:10] or None

            sources.append(
                Source(
                    source_id=f"arxiv::{abs(hash(url)) % (10**15)}",
                    source_type=SOURCE_NAME_ARXIV,
                    title=title[:500],
                    url=url,
                    publisher="arXiv",
                    snippet=snippet,
                    published_at=published,
                    trust_score=TRUST_SCORE_ARXIV,
                    relevance_score=0.7,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)
