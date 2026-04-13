import os
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from dotenv import load_dotenv

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import (
    DEFAULT_RETRIEVER_MAX_RESULTS,
    DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
    SOURCE_NAME_NEWSAPI,
)


class NewsApiRetriever(BaseRetriever):
    """
    NewsAPI retriever using the /v2/everything endpoint.

    Main improvements:
    - generic anchor + intent query understanding
    - broader query variants first, exact phrase later
    - stronger filtering for weak matches
    - canonical URL dedupe
    - low-signal URL filtering
    """

    BASE_URL = "https://newsapi.org/v2/everything"

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

    # Verb forms that describe price/value movement.
    # Treated as intents but also trigger the financial-context guard.
    FINANCIAL_MOVEMENT_VERBS = {
        "rose", "rise", "rises", "risen",
        "fell", "fall", "falls", "fallen",
        "climbed", "climb", "surged", "surge",
        "dropped", "drop", "plunged", "plunge",
        "gained", "gain", "lost", "lose",
        "jumped", "jump", "slipped", "slip",
        "rallied", "rally", "tumbled", "tumble",
    }

    # Unambiguous financial keywords — these almost never appear in product/deal listings.
    # Deliberately excludes "stock"/"stocks" (used as "in stock") and "market"/"markets"
    # (used in "supermarket", "flea market"). Used by the financial-context guard.
    FINANCIAL_CONTEXT_KEYWORDS = {
        "nasdaq", "nyse", "s&p 500", "stock market", "stock price",
        "share price", "earnings", "revenue", "quarterly", "fiscal",
        "investor", "investors", "equity", "portfolio", "dividend",
        "ipo", "valuation", "wall street", "hedge fund", "bull market",
        "bear market", "trading volume", "market cap", "short sell",
        "brokerage", "securities", "shareholder",
    }

    # Domain fragments for deal/shopping aggregator sites.
    # These are excluded from results when the query has financial context.
    DEAL_SITE_DOMAINS = {
        "slickdeals.net", "dansdeals.com", "dealnews.com",
        "9to5toys.com", "9to5mac.com", "retailmenot.com",
        "bensbargains.com", "gottadeal.com", "fatwallet.com",
        "bradsdeals.com", "hip2save.com",
    }

    LOW_SIGNAL_HOST_PATTERNS = {
        "consent.yahoo.com",
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

    def __init__(
        self,
        api_key: Optional[str] = None,
        language: str = "en",
        sort_by: str = "relevancy",
    ) -> None:
        super().__init__(source_name=SOURCE_NAME_NEWSAPI)
        load_dotenv()
        self.api_key = api_key or os.getenv("NEWSAPI_KEY", "").strip() or None
        self.language = language
        self.sort_by = sort_by

    def fetch_raw(
        self,
        query: str,
        max_results: int = DEFAULT_RETRIEVER_MAX_RESULTS,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {"articles": []}

        cleaned_query = self._safe_strip(query)
        if not cleaned_query:
            return {"articles": []}

        language = kwargs.get("language", self.language)
        sort_by = kwargs.get("sort_by", self.sort_by)
        search_in = kwargs.get("search_in", "title,description")
        domains = kwargs.get("domains")
        exclude_domains = kwargs.get("exclude_domains")
        sources = kwargs.get("sources")
        from_date = kwargs.get("from_date")
        to_date = kwargs.get("to_date")

        query_variants = self._build_query_variants(cleaned_query)
        merged_articles: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        per_query_page_size = min(max(max_results * 3, 10), 25)

        headers = {
            "X-Api-Key": self.api_key,
        }

        for query_variant in query_variants:
            params = {
                "q": query_variant,
                "language": language,
                "sortBy": sort_by,
                "pageSize": per_query_page_size,
                "page": 1,
                "searchIn": search_in,
            }

            if domains:
                params["domains"] = domains
            if exclude_domains:
                params["excludeDomains"] = exclude_domains
            if sources:
                params["sources"] = sources
            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date

            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=headers,
                timeout=DEFAULT_RETRIEVER_TIMEOUT_SECONDS,
            )
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
            source_obj = article.get("source", {}) or {}
            publisher_name = self._safe_strip(source_obj.get("name"))
            author = self._safe_strip(article.get("author"))
            title = self._safe_strip(article.get("title"))
            description = self._safe_strip(article.get("description"))
            content = self._safe_strip(article.get("content"))
            article_url = self._safe_strip(article.get("url"))
            published_at = self._safe_strip(article.get("publishedAt"))

            if not article_url or not title:
                continue

            ranking_score = self._compute_ranking_score(
                query=cleaned_query,
                query_tokens=query_tokens,
                anchors=anchors,
                intents=intents,
                title=title,
                description=description,
                content=content,
                publisher_name=publisher_name,
            )

            if ranking_score <= 0:
                continue

            snippet = self._build_snippet(
                title=title,
                description=description,
                content=content,
                author=author,
                publisher_name=publisher_name,
            )

            source_id = self._build_source_id(
                url=article_url,
                title=title,
            )

            source = Source(
                source_id=source_id,
                source_type="newsapi",
                title=title,
                url=article_url,
                publisher=publisher_name,
                snippet=snippet,
                published_at=published_at,
                trust_score=0.82,
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
        publisher_lower = publisher_name.lower()

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
        intent_total_hits = (
            intent_title_hits + intent_description_hits + intent_content_hits
        )

        if anchors and anchor_total_hits == 0:
            return 0.0

        meaningful_query_terms = len(anchors) + len(intents)
        if meaningful_query_terms >= 2 and intents and intent_total_hits == 0:
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
            + 8 * intent_title_hits
            + 5 * intent_description_hits
            + 1 * intent_content_hits
            + 3 * title_overlap
            + 2 * description_overlap
            + 1 * content_overlap
            + 0.5 * publisher_overlap
        )

        if meaningful_query_terms >= 2:
            if anchor_title_hits == 0 and anchor_description_hits == 0:
                score -= 8.0
            if intents and intent_title_hits == 0 and intent_description_hits == 0:
                score -= 6.0

        # Financial-context guard: if query contains a percentage (e.g. "5%") or a
        # financial movement verb (rose, fell, surged…):
        # 1. Block known deal/shopping aggregator domains — they match "rose gold" /
        #    "rose petal" product listings and produce false positives.
        # 2. Require that the title contains at least one financial indicator word
        #    ("stock", "shares", "market", "nasdaq", "nyse", "earnings", "ipo", etc.)
        #    OR that the publisher_name is a known finance outlet. This keeps
        #    legitimate articles like "Amazon Stock Climbs..." while rejecting
        #    product listicles that happen to contain "rose" or a percentage.
        query_has_pct = "%" in query
        query_has_movement_verb = any(
            tok in self.FINANCIAL_MOVEMENT_VERBS
            for tok in query.lower().split()
        )
        if query_has_pct or query_has_movement_verb:
            # Step 1: hard-block deal/shopping aggregator sites
            for deal_domain in self.DEAL_SITE_DOMAINS:
                if deal_domain in title_lower or deal_domain in description_lower or deal_domain in content_lower:
                    return 0.0

            # Step 2: title must contain at least one financial indicator
            TITLE_FINANCIAL_WORDS = {
                "stock", "stocks", "share", "shares", "market", "markets",
                "nasdaq", "nyse", "s&p", "index", "earnings", "revenue",
                "ipo", "equity", "dividend", "investor", "investors",
                "trading", "valuation", "quarterly", "fiscal",
            }
            title_words = set(title_lower.replace(",", " ").replace(".", " ").split())
            has_financial_title = bool(title_words & TITLE_FINANCIAL_WORDS)
            if not has_financial_title:
                return 0.0

        return max(float(score), 0.0)

    def _build_query_variants(self, query: str) -> List[str]:
        cleaned = self._safe_strip(query)
        if not cleaned:
            return []

        anchors, intents = self._extract_query_roles(cleaned)

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

        if anchors and intents:
            add_variant(" ".join(anchors + intents[:1]))
            add_variant(" ".join(anchors[:1] + intents))
            add_variant(" ".join(anchors[:2] + intents[:2]))

        if anchors:
            add_variant(" ".join(anchors[:3]))

        if intents:
            add_variant(" ".join(intents[:3]))

        no_number_tokens = [
            tok for tok in cleaned.split()
            if not any(ch.isdigit() for ch in tok)
        ]
        no_number_query = " ".join(no_number_tokens).strip()
        if no_number_query and no_number_query.lower() != cleaned.lower():
            add_variant(no_number_query)

        if anchors and len(anchors) >= 2:
            add_variant(f"\"{' '.join(anchors[:2])}\"")

        add_variant(f"\"{cleaned}\"")

        return variants[:6]

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
        return f"newsapi::{short_hash}"

    @staticmethod
    def _normalize_score(score: float) -> float:
        if score <= 0:
            return 0.0
        return min(score / 40.0, 1.0)

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
        description: Optional[str],
        content: Optional[str],
        author: Optional[str],
        publisher_name: Optional[str],
    ) -> str:
        # Use only natural-language content. Author/publisher metadata appended
        # with " | " breaks sentence splitting and adds noise for NLI inference.
        if description:
            return description
        if content:
            return content[:220].strip()
        return title or ""