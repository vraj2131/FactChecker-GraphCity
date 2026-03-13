from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
import time

import requests

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_RETRIEVER_MAX_RESULTS,
    DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
    SOURCE_NAME_GDELT,
)


class GDELTRetriever(BaseRetriever):
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
        "and", "or", "but", "if", "then", "than", "into", "over", "under",
        "after", "before", "during", "about", "between", "through", "up",
        "down", "out", "off", "this", "that", "these", "those", "it", "its",
        "their", "his", "her", "our", "your", "my", "we", "they", "he", "she",
        "you", "i"
    }

    GENERIC_QUERY_TERMS = {
        "cause", "causes", "caused",
        "claim", "claims", "claimed",
        "say", "says", "said",
        "report", "reports", "reported",
        "show", "shows", "showed",
        "reveal", "reveals", "revealed",
        "announce", "announces", "announced",
        "update", "updates", "updated",
        "news", "story", "article"
    }

    LOW_SIGNAL_HOST_PATTERNS = {
        "consent.",
        "privacy.",
        "login.",
        "signup.",
    }

    LOW_SIGNAL_URL_PARTS = {
        "collectconsent",
        "consent",
        "privacy",
        "login",
        "signup",
        "register",
        "preferences",
    }

    def __init__(self, timespan: str = "30d") -> None:
        super().__init__(source_name=SOURCE_NAME_GDELT)
        self.timespan = timespan

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cleaned_query = self._safe_strip(query)
        if not cleaned_query:
            return {"articles": []}

        timespan = kwargs.get("timespan", self.timespan)
        query_variants = self._build_query_variants(cleaned_query)

        merged_articles: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        max_records = min(max(max_results * 2, 5), 10)

        for i, query_variant in enumerate(query_variants):
            params = {
                "query": query_variant,
                "mode": "artlist",
                "format": "json",
                "maxrecords": max_records,
                "timespan": timespan,
            }

            try:
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
                )

                if response.status_code == 429:
                    break

                response.raise_for_status()
                payload = response.json()
                articles = payload.get("articles", []) or []

                for article in articles:
                    canonical_url = self._canonicalize_url(article.get("url"))
                    if not canonical_url:
                        continue
                    if self._is_low_signal_url(canonical_url):
                        continue
                    if canonical_url in seen_urls:
                        continue

                    article["url"] = canonical_url
                    seen_urls.add(canonical_url)
                    merged_articles.append(article)

            except requests.RequestException:
                continue

            if i < len(query_variants) - 1:
                time.sleep(1.0)

        return {"articles": merged_articles}

    def normalize(
        self,
        raw_data: Any,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> List[Source]:
        if not raw_data:
            return []

        articles = raw_data.get("articles", []) or []
        if not articles:
            return []

        cleaned_query = self._safe_strip(query)
        if not cleaned_query:
            return []

        query_tokens = self._tokenize(cleaned_query)
        anchors, intents = self._extract_query_roles(cleaned_query)

        scored_sources: List[Tuple[float, Source]] = []

        for article in articles:
            title = self._safe_strip(article.get("title"))
            article_url = self._safe_strip(article.get("url"))
            domain = self._safe_strip(article.get("domain"))
            seendate = self._safe_strip(article.get("seendate"))
            language = self._safe_strip(article.get("language"))
            sourcecountry = self._safe_strip(article.get("sourcecountry"))

            if not article_url or not title:
                continue

            snippet = self._build_snippet(
                title=title,
                article=article,
                domain=domain,
                language=language,
                sourcecountry=sourcecountry,
            )

            ranking_score = self._compute_ranking_score(
                query=cleaned_query,
                query_tokens=query_tokens,
                anchors=anchors,
                intents=intents,
                title=title,
                description=snippet,
                content="",
                publisher_name=domain,
            )

            if ranking_score <= 0:
                continue

            source_id = self._build_source_id(article_url, title)

            source = Source(
                source_id=source_id,
                source_type="gdelt",
                title=title,
                url=article_url,
                publisher=domain,
                snippet=snippet,
                published_at=seendate,
                trust_score=0.78,
                relevance_score=self._normalize_score(ranking_score),
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
        anchors: List[str],
        intents: List[str],
        title: Optional[str],
        description: Optional[str],
        content: Optional[str],
        publisher_name: Optional[str],
    ) -> float:
        title = title or ""
        description = description or ""
        content = content or ""
        publisher_name = publisher_name or ""

        title_lower = title.lower()
        description_lower = description.lower()
        content_lower = content.lower()

        title_tokens = self._tokenize(title)
        description_tokens = self._tokenize(description)
        content_tokens = self._tokenize(content)
        publisher_tokens = self._tokenize(publisher_name)

        title_overlap = len(query_tokens & title_tokens)
        description_overlap = len(query_tokens & description_tokens)
        content_overlap = len(query_tokens & content_tokens)
        publisher_overlap = len(query_tokens & publisher_tokens)

        exact_query_in_title = int(bool(query) and query.lower() in title_lower)
        exact_query_in_description = int(bool(query) and query.lower() in description_lower)

        anchor_title_hits = sum(1 for term in anchors if term in title_lower)
        anchor_description_hits = sum(1 for term in anchors if term in description_lower)
        anchor_content_hits = sum(1 for term in anchors if term in content_lower)

        intent_title_hits = sum(1 for term in intents if term in title_lower)
        intent_description_hits = sum(1 for term in intents if term in description_lower)
        intent_content_hits = sum(1 for term in intents if term in content_lower)

        anchor_total_hits = (
            anchor_title_hits + anchor_description_hits + anchor_content_hits
        )

        if anchors and anchor_total_hits == 0:
            return 0.0

        if exact_query_in_title == 0 and exact_query_in_description == 0:
            if title_overlap == 0 and description_overlap == 0 and anchor_total_hits == 0:
                return 0.0

        score = (
            12 * exact_query_in_title
            + 7 * exact_query_in_description
            + 10 * anchor_title_hits
            + 6 * anchor_description_hits
            + 2 * anchor_content_hits
            + 5 * intent_title_hits
            + 3 * intent_description_hits
            + 1 * intent_content_hits
            + 3 * title_overlap
            + 2 * description_overlap
            + 1 * content_overlap
            + 0.5 * publisher_overlap
        )

        return max(float(score), 0.0)

    def _build_query_variants(self, query: str) -> List[str]:
        cleaned = self._safe_strip(query)
        if not cleaned:
            return []

        variants: List[str] = []
        seen: Set[str] = set()

        def add_variant(text: Optional[str]) -> None:
            text = self._safe_strip(text)
            if not text:
                return
            lowered = text.lower()
            if lowered in seen:
                return
            seen.add(lowered)
            variants.append(text)

        add_variant(cleaned)
        add_variant(f"\"{cleaned}\"")

        return variants[:2]

    @classmethod
    def _extract_query_roles(cls, query: str) -> Tuple[List[str], List[str]]:
        raw_tokens = [tok.strip(".,:;!?()[]{}\"'") for tok in query.split()]

        anchors: List[str] = []
        intents: List[str] = []
        seen_anchor_keys: Set[str] = set()
        seen_intent_keys: Set[str] = set()

        for tok in raw_tokens:
            if not tok:
                continue

            low = tok.lower()
            if low in cls.STOPWORDS:
                continue

            normalized_key = low.replace("-", "")
            if not normalized_key:
                continue

            is_anchor = (
                tok[:1].isupper()
                or tok.isupper()
                or any(ch.isdigit() for ch in tok)
                or "-" in tok
            )

            if is_anchor:
                if normalized_key not in seen_anchor_keys:
                    seen_anchor_keys.add(normalized_key)
                    anchors.append(low)
                continue

            if low in cls.GENERIC_QUERY_TERMS:
                continue

            if len(low) >= 3 and normalized_key not in seen_intent_keys:
                seen_intent_keys.add(normalized_key)
                intents.append(low)

        if not anchors and raw_tokens:
            fallback_tokens = [
                tok.lower()
                for tok in raw_tokens
                if tok
                and tok.lower() not in cls.STOPWORDS
                and tok.lower() not in cls.GENERIC_QUERY_TERMS
            ]
            if fallback_tokens:
                anchors.append(fallback_tokens[0])
                for tok in fallback_tokens[1:]:
                    if len(tok) >= 3:
                        intents.append(tok)

        return anchors[:5], intents[:5]

    @classmethod
    def _tokenize(cls, text: str) -> Set[str]:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text))
        tokens = {token for token in cleaned.split() if len(token) >= 2}
        return {
            token for token in tokens
            if token not in cls.STOPWORDS and token not in cls.GENERIC_QUERY_TERMS
        }

    @staticmethod
    def _safe_strip(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _build_source_id(url: str, title: str) -> str:
        base_text = url or title
        short_hash = str(abs(hash(base_text)))
        return f"gdelt::{short_hash}"

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score <= 0:
            return 0.0
        return min(score / 35.0, 1.0)

    @classmethod
    def _canonicalize_url(cls, url: Any) -> Optional[str]:
        if not url:
            return None

        raw_url = str(url).strip()
        if not raw_url:
            return None

        try:
            parsed = urlparse(raw_url)
        except Exception:
            return None

        if not parsed.scheme or not parsed.netloc:
            return None

        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")
            and k.lower() not in {"guccounter", "fbclid", "gclid", "mc_cid", "mc_eid"}
        ]

        cleaned = parsed._replace(
            fragment="",
            query=urlencode(filtered_query, doseq=True),
        )
        return urlunparse(cleaned)

    @classmethod
    def _is_low_signal_url(cls, url: str) -> bool:
        lowered = url.lower()

        try:
            parsed = urlparse(lowered)
            host = parsed.netloc
            path = parsed.path
        except Exception:
            host = ""
            path = lowered

        for bad_host in cls.LOW_SIGNAL_HOST_PATTERNS:
            if bad_host in host:
                return True

        for bad_part in cls.LOW_SIGNAL_URL_PARTS:
            if bad_part in path or bad_part in lowered:
                return True

        return False

    @staticmethod
    def _build_snippet(
        title: Optional[str],
        article: Dict[str, Any],
        domain: Optional[str],
        language: Optional[str],
        sourcecountry: Optional[str],
    ) -> str:
        parts: List[str] = []

        if title:
            parts.append(title)

        if domain:
            parts.append(f"Publisher: {domain}")

        if language:
            parts.append(f"Language: {language}")

        if sourcecountry:
            parts.append(f"Country: {sourcecountry}")

        return " | ".join(parts)