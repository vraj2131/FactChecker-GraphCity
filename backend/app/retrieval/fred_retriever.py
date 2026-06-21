import logging
import os
from typing import Any, Dict, List, Optional

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEFAULT_RETRIEVER_TIMEOUT_SECONDS, SOURCE_NAME_FRED

logger = logging.getLogger(__name__)

TRUST_SCORE_FRED = 0.93
_SERIES_SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"
_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _get_api_key() -> Optional[str]:
    return os.environ.get("FRED_API_KEY") or None


class FredRetriever(BaseRetriever):
    """
    Retrieves macroeconomic data series from the St. Louis Fed FRED API.
    Requires FRED_API_KEY env var (free key: fred.stlouisfed.org/docs/api/api_key.html).
    Skips gracefully if the key is not set.
    """

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_FRED)

    def fetch_raw(self, query: str, max_results: int = 3, **kwargs: Any) -> List[Dict]:
        api_key = _get_api_key()
        if not api_key:
            logger.info("FredRetriever: FRED_API_KEY not set — skipping")
            return []
        try:
            params = {
                "search_text": query,
                "api_key": api_key,
                "file_type": "json",
                "limit": min(max_results, 5),
                "order_by": "search_rank",
                "sort_order": "desc",
            }
            resp = requests.get(
                _SERIES_SEARCH_URL,
                params=params,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            series_list = resp.json().get("seriess", [])
            return series_list[:max_results]
        except Exception as exc:
            logger.warning("FredRetriever: series search failed: %s", exc)
            return []

    def normalize(
        self, raw_data: Any, query: str, max_results: int = 3, **kwargs: Any
    ) -> List[Source]:
        api_key = _get_api_key()
        if not raw_data or not api_key:
            return []
        sources: List[Source] = []
        for series in raw_data:
            series_id = series.get("id") or ""
            title = series.get("title") or series_id
            units = series.get("units_short") or series.get("units") or ""
            freq = series.get("frequency_short") or ""
            notes = (series.get("notes") or "")[:200]

            latest_value, latest_date = self._fetch_latest(series_id, api_key)

            if latest_value is not None:
                snippet = (
                    f"FRED — {title}: {latest_value} {units} as of {latest_date}. "
                    f"Frequency: {freq}."
                )
                if notes:
                    snippet += f" {notes}"
            else:
                snippet = f"FRED series: {title} ({units}, {freq}). {notes}"

            url = f"https://fred.stlouisfed.org/series/{series_id}"
            sources.append(
                Source(
                    source_id=f"fred::{series_id}",
                    source_type=SOURCE_NAME_FRED,
                    title=f"FRED: {title}"[:500],
                    url=url,
                    publisher="St. Louis Fed (FRED)",
                    snippet=snippet[:500],
                    published_at=latest_date,
                    trust_score=TRUST_SCORE_FRED,
                    relevance_score=0.85,
                    stance_hint=None,
                )
            )
        return self.postprocess(sources, max_results)

    @staticmethod
    def _fetch_latest(series_id: str, api_key: str) -> tuple:
        """Fetch the most recent observation for a FRED series. Returns (value, date) or (None, None)."""
        try:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,
            }
            resp = requests.get(
                _OBSERVATIONS_URL,
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if obs:
                return obs[0].get("value", "."), obs[0].get("date", "")
        except Exception:
            pass
        return None, None
