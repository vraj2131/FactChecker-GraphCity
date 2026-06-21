import logging
from typing import Any, Dict, List
from urllib.parse import quote

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_WORLDBANK

logger = logging.getLogger(__name__)

TRUST_SCORE_WORLDBANK = 0.90
_SEARCH_URL = "https://search.worldbank.org/api/v2/wds"
_HEADERS = {"Accept": "application/json"}


class WorldBankRetriever(BaseRetriever):
    """
    Retrieves documents and reports from the World Bank Open Knowledge Repository.
    Free, no API key required.
    """

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_WORLDBANK)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict]:
        try:
            params = {
                "format": "json",
                "qterm": query,
                "rows": min(max_results, 10),
                "fl": "id,docdt,display_title,abstracts,url,repnme,count",
                "sort": "score desc",
            }
            resp = requests.get(
                _SEARCH_URL,
                params=params,
                headers=_HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents", {})
            # API returns a dict of {id: doc} or a list depending on version
            if isinstance(docs, dict):
                return list(docs.values())[:max_results]
            if isinstance(docs, list):
                return docs[:max_results]
            return []
        except Exception as exc:
            logger.warning("WorldBankRetriever: fetch failed: %s", exc)
            return []

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Source]:
        if not raw_data:
            return []
        sources: List[Source] = []
        for doc in raw_data:
            title = (doc.get("display_title") or doc.get("repnme") or "").strip()
            if not title:
                continue

            abstract_raw = doc.get("abstracts") or {}
            abstract = ""
            if isinstance(abstract_raw, dict):
                abstract = (abstract_raw.get("cdata!") or abstract_raw.get("en") or "").strip()
            elif isinstance(abstract_raw, str):
                abstract = abstract_raw.strip()
            snippet = abstract[:500] if abstract else f"World Bank report: {title}"

            url = doc.get("url") or doc.get("pdfurl") or ""
            if not url or not url.startswith("http"):
                doc_id = doc.get("id") or ""
                url = (
                    f"https://documents.worldbank.org/en/publication/documents-reports/documentdetail/{doc_id}"
                    if doc_id
                    else f"https://search.worldbank.org/api/v2/wds?qterm={quote(query)}"
                )

            pub_date = (doc.get("docdt") or "")[:10] or None

            sources.append(
                Source(
                    source_id=f"worldbank::{abs(hash(url)) % (10**15)}",
                    source_type=SOURCE_NAME_WORLDBANK,
                    title=title[:500],
                    url=url,
                    publisher="World Bank",
                    snippet=snippet,
                    published_at=pub_date,
                    trust_score=TRUST_SCORE_WORLDBANK,
                    relevance_score=0.75,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)
