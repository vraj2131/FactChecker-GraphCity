import logging
from typing import Any, Dict, List
from urllib.parse import quote

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_EDGAR

logger = logging.getLogger(__name__)

TRUST_SCORE_EDGAR = 0.90
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_HEADERS = {
    "User-Agent": "FactCheckerBot/1.0 (mailto:factcheck@research.edu)",
    "Accept": "application/json",
}


class SecEdgarRetriever(BaseRetriever):
    """
    Retrieves SEC filings via EDGAR full-text search (efts.sec.gov).
    Free, no API key required.
    """

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_EDGAR)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict]:
        try:
            params = {"q": query}
            resp = requests.get(
                _SEARCH_URL,
                params=params,
                headers=_HEADERS,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", {}).get("hits", [])[:max_results]
        except Exception as exc:
            logger.warning("SecEdgarRetriever: fetch failed: %s", exc)
            return []

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any
    ) -> List[Source]:
        if not raw_data:
            return []
        sources: List[Source] = []
        for hit in raw_data:
            src = hit.get("_source", {})
            display_names = src.get("display_names") or ["Unknown"]
            entity = src.get("entity_name") or display_names[0]
            file_type = src.get("file_type") or src.get("form_type") or "Filing"
            file_date = src.get("file_date") or src.get("period_of_report") or ""

            # _id is "{accession_with_dashes}:{filename}" — split to get accession
            raw_id = hit.get("_id", "")
            accession_dashed = raw_id.split(":")[0] if ":" in raw_id else raw_id
            filename = raw_id.split(":")[1] if ":" in raw_id else ""

            # CIK is in ciks list
            ciks = src.get("ciks") or []
            cik = ciks[0].lstrip("0") if ciks else ""
            accession_nodash = accession_dashed.replace("-", "")

            title = f"{entity} — {file_type}" + (f" ({file_date[:10]})" if file_date else "")
            snippet = (
                f"SEC {file_type} filing by {entity}"
                + (f" filed {file_date[:10]}" if file_date else "")
                + f". Source: SEC EDGAR. Accession: {accession_dashed}."
            )

            # Build a direct link to the filing document
            if cik and accession_nodash and filename:
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
            elif cik and accession_nodash:
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/"
            else:
                url = f"https://efts.sec.gov/LATEST/search-index?q={quote(query)}"

            sources.append(
                Source(
                    source_id=f"sec_edgar::{abs(hash(hit.get('_id', query))) % (10**15)}",
                    source_type=SOURCE_NAME_EDGAR,
                    title=title[:500],
                    url=url,
                    publisher="SEC EDGAR",
                    snippet=snippet,
                    published_at=file_date[:10] if file_date else None,
                    trust_score=TRUST_SCORE_EDGAR,
                    relevance_score=0.75,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)
