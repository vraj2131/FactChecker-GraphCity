import os
from typing import Any, Dict, List

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import SOURCE_NAME_BLUESKY

TRUST_SCORE_BLUESKY = 0.35


class BlueskyRetriever(BaseRetriever):
    """Retrieves public Bluesky posts via atproto (requires BLUESKY_IDENTIFIER + BLUESKY_APP_PASSWORD in env)."""

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_BLUESKY)
        self._identifier = os.getenv("BLUESKY_IDENTIFIER", "")
        self._app_password = os.getenv("BLUESKY_APP_PASSWORD", "")

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict]:
        if not self._identifier or not self._app_password:
            return []
        try:
            from atproto import Client
            client = Client()
            client.login(self._identifier, self._app_password)
            tokens = self._tokenize(query)
            search_query = " ".join(list(tokens)[:6]) if tokens else query
            resp = client.app.bsky.feed.search_posts(
                params={"q": search_query, "limit": max_results * 2}
            )
            posts = resp.posts if hasattr(resp, "posts") else []
            results = []
            for post in posts:
                record = getattr(post, "record", None)
                text = getattr(record, "text", "") if record else ""
                uri = getattr(post, "uri", "")
                handle = getattr(getattr(post, "author", None), "handle", "bluesky")
                results.append({"text": text, "uri": uri, "handle": handle})
            return results
        except Exception:
            return []

    def normalize(self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any) -> List[Source]:
        if not raw_data:
            return []
        query_tokens = self._tokenize(query)
        sources: List[Source] = []
        for item in raw_data:
            text   = (item.get("text") or "").strip()
            uri    = (item.get("uri") or "").strip()
            handle = item.get("handle", "bluesky")
            if not text:
                continue
            url = f"https://bsky.app/profile/{handle}" if uri else ""
            relevance = self._compute_relevance(query_tokens, text)
            sources.append(Source(
                source_id=f"bluesky::{abs(hash(uri or text))}",
                source_type=SOURCE_NAME_BLUESKY,
                title=text[:120],
                url=url or None,
                publisher=f"@{handle}",
                snippet=text[:300],
                published_at=None,
                trust_score=TRUST_SCORE_BLUESKY,
                relevance_score=relevance,
                stance_hint=None,
            ))
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return self.postprocess(sources, max_results=max_results)

    def _compute_relevance(self, query_tokens, text: str) -> float:
        t_tok = self._tokenize(text)
        if not query_tokens:
            return 0.5
        return min(len(query_tokens & t_tok) / len(query_tokens), 1.0)

    @staticmethod
    def _tokenize(text: str):
        stopwords = {"a","an","the","is","are","was","were","of","in","on","at","to","for","and","or","but","it","its"}
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {tok for tok in cleaned.split() if len(tok) >= 2 and tok not in stopwords}
