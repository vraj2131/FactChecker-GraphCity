from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.schemas.source_schema import Source


class BaseRetriever(ABC):
    """
    Base interface for all retrievers.

    Every retriever should:
    - accept a query
    - fetch raw source data
    - normalize results into Source objects
    """

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    def fetch_raw(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> Any:
        """
        Fetch raw data from the underlying source.
        """
        raise NotImplementedError

    @abstractmethod
    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> List[Source]:
        """
        Convert raw source data into normalized Source objects.
        """
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> List[Source]:
        """
        End-to-end retrieval:
        - validate query
        - fetch raw data
        - normalize results
        """
        cleaned_query = self._clean_query(query)
        raw_data = self.fetch_raw(
            query=cleaned_query,
            max_results=max_results,
            **kwargs,
        )
        results = self.normalize(
            raw_data=raw_data,
            query=cleaned_query,
            max_results=max_results,
            **kwargs,
        )
        return results

    @staticmethod
    def _clean_query(query: str) -> str:
        """
        Minimal validation/cleanup for retriever input query.
        """
        if query is None:
            raise ValueError("query cannot be None.")

        cleaned = str(query).strip()
        if not cleaned:
            raise ValueError("query cannot be empty or whitespace only.")

        return cleaned

    @staticmethod
    def deduplicate_sources(sources: List[Source]) -> List[Source]:
        """
        Remove duplicate sources using URL as the primary key.
        Keeps first occurrence.
        """
        seen_urls = set()
        deduped: List[Source] = []

        for source in sources:
            url_key = str(source.url).strip()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            deduped.append(source)

        return deduped

    @staticmethod
    def limit_sources(sources: List[Source], max_results: int) -> List[Source]:
        """
        Trim normalized results to max_results.
        """
        if max_results <= 0:
            return []
        return sources[:max_results]

    def postprocess(
        self,
        sources: List[Source],
        max_results: int,
    ) -> List[Source]:
        """
        Common cleanup after normalization.
        """
        sources = self.deduplicate_sources(sources)
        sources = self.limit_sources(sources, max_results)
        return sources

    def get_source_name(self) -> str:
        return self.source_name
