# from typing import Any, Dict, List, Optional

# import requests

# from backend.app.retrieval.base_retriever import BaseRetriever
# from backend.app.schemas.source_schema import Source
# from backend.app.utils.constants import (
#     DEFAULT_RETRIEVER_MAX_RESULTS,
#     DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
#     SOURCE_NAME_GUARDIAN,
# )


# class GuardianRetriever(BaseRetriever):
#     """
#     Guardian Open Platform retriever.

#     Notes:
#     - Uses the Guardian content search endpoint
#     - Requires an API key
#     - Returns normalized Source objects
#     """

#     BASE_URL = "https://content.guardianapis.com/search"

#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         show_fields: str = "headline,trailText,bodyText,byline",
#         section: Optional[str] = None,
#     ) -> None:
#         super().__init__(source_name=SOURCE_NAME_GUARDIAN)
#         self.api_key = api_key
#         self.show_fields = show_fields
#         self.section = section

#     def fetch_raw(
#         self,
#         query: str,
#         max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
#         **kwargs: Any,
#     ) -> Dict[str, Any]:
#         """
#         Fetch raw results from the Guardian Open Platform.
#         """
#         if not self.api_key:
#             return {"response": {"results": []}}

#         page_size = kwargs.get("page_size", max_results)
#         section = kwargs.get("section", self.section)
#         order_by = kwargs.get("order_by", "relevance")
#         from_date = kwargs.get("from_date")
#         to_date = kwargs.get("to_date")

#         params = {
#             "api-key": self.api_key,
#             "q": query,
#             "page-size": page_size,
#             "order-by": order_by,
#             "show-fields": self.show_fields,
#         }

#         if section:
#             params["section"] = section
#         if from_date:
#             params["from-date"] = from_date
#         if to_date:
#             params["to-date"] = to_date

#         response = requests.get(
#             self.BASE_URL,
#             params=params,
#             timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
#         )
#         response.raise_for_status()
#         return response.json()

#     def normalize(
#         self,
#         raw_data: Any,
#         query: str,
#         max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
#         **kwargs: Any,
#     ) -> List[Source]:
#         """
#         Convert raw Guardian API response into normalized Source objects.
#         """
#         if not raw_data:
#             return []

#         response_obj = raw_data.get("response", {})
#         results = response_obj.get("results", []) or []

#         if not results:
#             return []

#         sources: List[Source] = []

#         for item in results:
#             web_url = self._safe_strip(item.get("webUrl"))
#             web_title = self._safe_strip(item.get("webTitle"))
#             web_publication_date = self._safe_strip(item.get("webPublicationDate"))
#             section_name = self._safe_strip(item.get("sectionName"))
#             guardian_id = self._safe_strip(item.get("id"))

#             fields = item.get("fields", {}) or {}
#             headline = self._safe_strip(fields.get("headline"))
#             trail_text = self._safe_strip(fields.get("trailText"))
#             body_text = self._safe_strip(fields.get("bodyText"))
#             byline = self._safe_strip(fields.get("byline"))

#             if not web_url or not (headline or web_title):
#                 continue

#             title = headline or web_title
#             snippet = self._build_snippet(
#                 trail_text=trail_text,
#                 body_text=body_text,
#                 byline=byline,
#                 section_name=section_name,
#             )

#             relevance_score = self._estimate_relevance_score(
#                 query=query,
#                 title=title,
#                 snippet=snippet,
#                 section_name=section_name,
#             )

#             source_id = self._build_source_id(
#                 guardian_id=guardian_id,
#                 url=web_url,
#                 title=title,
#             )

#             source = Source(
#                 source_id=source_id,
#                 source_type="guardian",
#                 title=title,
#                 url=web_url,
#                 publisher="The Guardian",
#                 snippet=snippet,
#                 published_at=web_publication_date,
#                 trust_score=0.88,
#                 relevance_score=relevance_score,
#                 stance_hint=None,
#             )
#             sources.append(source)

#         return self.postprocess(sources, max_results=max_results)

#     @staticmethod
#     def _safe_strip(value: Any) -> Optional[str]:
#         if value is None:
#             return None
#         text = str(value).strip()
#         return text or None

#     @staticmethod
#     def _build_source_id(
#         guardian_id: Optional[str],
#         url: str,
#         title: str,
#     ) -> str:
#         base_text = guardian_id or url or title
#         short_hash = str(abs(hash(base_text)))
#         return f"guardian::{short_hash}"

#     @staticmethod
#     def _build_snippet(
#         trail_text: Optional[str],
#         body_text: Optional[str],
#         byline: Optional[str],
#         section_name: Optional[str],
#     ) -> str:
#         parts = []

#         if trail_text:
#             parts.append(trail_text)

#         if body_text:
#             shortened_body = body_text[:500].strip()
#             if shortened_body:
#                 parts.append(shortened_body)

#         if byline:
#             parts.append(f"Byline: {byline}")

#         if section_name:
#             parts.append(f"Section: {section_name}")

#         return " | ".join(parts)

#     @staticmethod
#     def _estimate_relevance_score(
#         query: str,
#         title: Optional[str],
#         snippet: Optional[str],
#         section_name: Optional[str],
#     ) -> float:
#         query_tokens = GuardianRetriever._tokenize(query)
#         if not query_tokens:
#             return 0.0

#         title_tokens = GuardianRetriever._tokenize(title or "")
#         snippet_tokens = GuardianRetriever._tokenize(snippet or "")
#         section_tokens = GuardianRetriever._tokenize(section_name or "")

#         overlap = len(query_tokens & (title_tokens | snippet_tokens | section_tokens))
#         if overlap <= 0:
#             return 0.0

#         return min(overlap / max(len(query_tokens), 1), 1.0)

#     @staticmethod
#     def _tokenize(text: str) -> set[str]:
#         cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
#         return {token for token in cleaned.split() if len(token) >= 2}

from typing import Any, Dict, List, Optional, Set

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_RETRIEVER_MAX_RESULTS,
    DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
    SOURCE_NAME_GUARDIAN,
)


class GuardianRetriever(BaseRetriever):
    """
    Guardian Open Platform retriever with stronger ranking.

    Improvements:
    - query rewriting
    - title/trail/body weighted scoring
    - entity boosting
    - section boosting
    - weak-match filtering
    """

    BASE_URL = "https://content.guardianapis.com/search"

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
        "and", "or", "but", "if", "then", "than", "into", "over", "under",
        "after", "before", "during", "about", "between", "through", "up",
        "down", "out", "off", "this", "that", "these", "those", "it", "its"
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        show_fields: str = "headline,trailText,bodyText,byline",
        section: Optional[str] = None,
    ) -> None:
        super().__init__(source_name=SOURCE_NAME_GUARDIAN)
        self.api_key = api_key
        self.show_fields = show_fields
        self.section = section

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetch raw results from the Guardian Open Platform.
        Uses multiple rewritten queries and merges results.
        """
        if not self.api_key:
            return {"response": {"results": []}}

        section = kwargs.get("section", self.section)
        order_by = kwargs.get("order_by", "relevance")
        from_date = kwargs.get("from_date")
        to_date = kwargs.get("to_date")

        query_variants = self._build_query_variants(query)
        merged_results: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        # Pull a few more candidates per query variant, then rank/filter later
        per_query_page_size = max(max_results, 10)

        for query_variant in query_variants:
            params = {
                "api-key": self.api_key,
                "q": query_variant,
                "page-size": per_query_page_size,
                "order-by": order_by,
                "show-fields": self.show_fields,
            }

            if section:
                params["section"] = section
            if from_date:
                params["from-date"] = from_date
            if to_date:
                params["to-date"] = to_date

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            payload = response.json()
            results = payload.get("response", {}).get("results", []) or []

            for item in results:
                web_url = self._safe_strip(item.get("webUrl"))
                if not web_url or web_url in seen_urls:
                    continue
                seen_urls.add(web_url)
                merged_results.append(item)

        return {"response": {"results": merged_results}}

    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Source]:
        """
        Convert raw Guardian API response into normalized Source objects.
        """
        if not raw_data:
            return []

        response_obj = raw_data.get("response", {})
        results = response_obj.get("results", []) or []

        if not results:
            return []

        scored_sources: List[tuple[float, Source]] = []

        query_tokens = self._tokenize(query)
        likely_entities = self._extract_likely_entities(query)
        preferred_section = (kwargs.get("section", self.section) or "").strip().lower()

        for item in results:
            web_url = self._safe_strip(item.get("webUrl"))
            web_title = self._safe_strip(item.get("webTitle"))
            web_publication_date = self._safe_strip(item.get("webPublicationDate"))
            section_name = self._safe_strip(item.get("sectionName"))
            guardian_id = self._safe_strip(item.get("id"))

            fields = item.get("fields", {}) or {}
            headline = self._safe_strip(fields.get("headline"))
            trail_text = self._safe_strip(fields.get("trailText"))
            body_text = self._safe_strip(fields.get("bodyText"))
            byline = self._safe_strip(fields.get("byline"))

            if not web_url or not (headline or web_title):
                continue

            title = headline or web_title
            snippet = self._build_snippet(
                trail_text=trail_text,
                body_text=body_text,
                byline=byline,
                section_name=section_name,
            )

            ranking_score = self._compute_ranking_score(
                query=query,
                query_tokens=query_tokens,
                likely_entities=likely_entities,
                title=title,
                trail_text=trail_text,
                body_text=body_text,
                section_name=section_name,
                preferred_section=preferred_section,
            )

            # Filter weak matches aggressively
            if ranking_score <= 0:
                continue

            relevance_score = self._normalize_score(ranking_score)

            source_id = self._build_source_id(
                guardian_id=guardian_id,
                url=web_url,
                title=title,
            )

            source = Source(
                source_id=source_id,
                source_type="guardian",
                title=title,
                url=web_url,
                publisher="The Guardian",
                snippet=snippet,
                published_at=web_publication_date,
                trust_score=0.88,
                relevance_score=relevance_score,
                stance_hint=None,
            )
            scored_sources.append((ranking_score, source))

        scored_sources.sort(key=lambda x: x[0], reverse=True)
        sources = [source for _, source in scored_sources]

        return self.postprocess(sources, max_results=max_results)

    def _compute_ranking_score(
        self,
        query: str,
        query_tokens: Set[str],
        likely_entities: List[str],
        title: Optional[str],
        trail_text: Optional[str],
        body_text: Optional[str],
        section_name: Optional[str],
        preferred_section: str,
    ) -> float:
        title = title or ""
        trail_text = trail_text or ""
        body_text = body_text or ""
        section_name = section_name or ""

        title_tokens = self._tokenize(title)
        trail_tokens = self._tokenize(trail_text)
        body_tokens = self._tokenize(body_text)
        section_tokens = self._tokenize(section_name)

        title_overlap = len(query_tokens & title_tokens)
        trail_overlap = len(query_tokens & trail_tokens)
        body_overlap = len(query_tokens & body_tokens)
        section_overlap = len(query_tokens & section_tokens)

        query_lower = query.lower().strip()
        title_lower = title.lower()
        trail_lower = trail_text.lower()
        body_lower = body_text.lower()
        section_lower = section_name.lower()

        exact_query_in_title = int(bool(query_lower) and query_lower in title_lower)
        exact_query_in_trail = int(bool(query_lower) and query_lower in trail_lower)

        entity_title_hits = sum(1 for ent in likely_entities if ent in title_lower)
        entity_trail_hits = sum(1 for ent in likely_entities if ent in trail_lower)
        entity_body_hits = sum(1 for ent in likely_entities if ent in body_lower)

        preferred_section_boost = int(bool(preferred_section) and preferred_section in section_lower)

        score = (
            5 * title_overlap
            + 3 * trail_overlap
            + 1 * body_overlap
            + 1 * section_overlap
            + 8 * exact_query_in_title
            + 5 * exact_query_in_trail
            + 6 * entity_title_hits
            + 3 * entity_trail_hits
            + 1 * entity_body_hits
            + 3 * preferred_section_boost
        )

        # Weak-match filter:
        # reject if no entity hit and title/trail overlap is too weak
        if entity_title_hits == 0 and entity_trail_hits == 0:
            if title_overlap == 0 and trail_overlap == 0 and exact_query_in_title == 0:
                return 0.0
            if title_overlap + trail_overlap <= 1 and exact_query_in_title == 0:
                return 0.0

        return float(score)

    def _build_query_variants(self, query: str) -> List[str]:
        """
        Build a few Guardian-friendly query variants.
        """
        cleaned = self._safe_strip(query)
        if not cleaned:
            return []

        variants: List[str] = []
        seen: Set[str] = set()

        def add_variant(text: str) -> None:
            text = self._safe_strip(text)
            if not text:
                return
            lowered = text.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            variants.append(text)

        add_variant(cleaned)

        tokens = cleaned.split()
        no_number_tokens = [tok for tok in tokens if not any(ch.isdigit() for ch in tok)]
        add_variant(" ".join(no_number_tokens))

        likely_entities = self._extract_likely_entities(cleaned)
        if likely_entities:
            add_variant(" ".join(likely_entities))

        # common finance wording improvements
        lowered = cleaned.lower()
        if "stock" in lowered:
            add_variant(cleaned.replace("stock", "shares"))
        if "stocks" in lowered:
            add_variant(cleaned.replace("stocks", "shares"))
        if "%" in cleaned:
            add_variant(cleaned.replace("%", " percent"))

        return variants[:4]

    @classmethod
    def _extract_likely_entities(cls, text: str) -> List[str]:
        """
        Very simple entity heuristic:
        keep tokens not in stopwords, length >= 3, preserve original phrases loosely.
        """
        raw_tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in text.split()]
        entity_tokens = [
            tok.lower()
            for tok in raw_tokens
            if len(tok) >= 3 and tok.lower() not in cls.STOPWORDS and not tok.isdigit()
        ]
        return entity_tokens[:5]

    @staticmethod
    def _safe_strip(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _build_source_id(
        guardian_id: Optional[str],
        url: str,
        title: str,
    ) -> str:
        base_text = guardian_id or url or title
        short_hash = str(abs(hash(base_text)))
        return f"guardian::{short_hash}"

    @staticmethod
    def _build_snippet(
        trail_text: Optional[str],
        body_text: Optional[str],
        byline: Optional[str],
        section_name: Optional[str],
    ) -> str:
        parts = []

        if trail_text:
            parts.append(trail_text)

        if body_text:
            shortened_body = body_text[:500].strip()
            if shortened_body:
                parts.append(shortened_body)

        if byline:
            parts.append(f"Byline: {byline}")

        if section_name:
            parts.append(f"Section: {section_name}")

        return " | ".join(parts)

    @classmethod
    def _tokenize(cls, text: str) -> Set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
        tokens = {token for token in cleaned.split() if len(token) >= 2}
        return {token for token in tokens if token not in cls.STOPWORDS}

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score <= 0:
            return 0.0
        return min(score / 25.0, 1.0)