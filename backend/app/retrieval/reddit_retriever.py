import os
from typing import Any, Dict, List, Optional

from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import SOURCE_NAME_REDDIT

TRUST_SCORE_REDDIT = 0.45


class RedditRetriever(BaseRetriever):
    """Retrieves public Reddit posts via PRAW (read-only, free app credentials)."""

    def __init__(self) -> None:
        super().__init__(source_name=SOURCE_NAME_REDDIT)

    def fetch_raw(self, query: str, max_results: int = 5, **kwargs: Any) -> List[Dict]:
        client_id     = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            return []
        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="FactGraphCity/1.0 (fact-checking research tool)",
            )
            tokens = self._tokenize(query)
            search_query = " ".join(list(tokens)[:6]) if tokens else query
            results = []
            for submission in reddit.subreddit("all").search(
                search_query, sort="relevance", limit=max_results * 2, time_filter="year"
            ):
                results.append({
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext[:500] if submission.selftext else "",
                    "url": f"https://reddit.com{submission.permalink}",
                    "score": submission.score,
                    "subreddit": str(submission.subreddit),
                    "created_utc": submission.created_utc,
                })
            return results
        except Exception:
            return []

    def normalize(self, raw_data: Any, query: str, max_results: int = 5, **kwargs: Any) -> List[Source]:
        if not raw_data:
            return []
        query_tokens = self._tokenize(query)
        sources: List[Source] = []
        for item in raw_data:
            title   = (item.get("title") or "").strip()
            url     = (item.get("url") or "").strip()
            body    = (item.get("selftext") or "").strip()
            sub     = item.get("subreddit", "reddit")
            if not title or not url:
                continue
            snippet = body if len(body) > 30 else title
            relevance = self._compute_relevance(query_tokens, title, snippet)
            sources.append(Source(
                source_id=f"reddit::{item.get('id', abs(hash(url)))}",
                source_type=SOURCE_NAME_REDDIT,
                title=title,
                url=url,
                publisher=f"r/{sub}",
                snippet=snippet,
                published_at=None,
                trust_score=TRUST_SCORE_REDDIT,
                relevance_score=relevance,
                stance_hint=None,
            ))
        sources.sort(key=lambda s: s.relevance_score, reverse=True)
        return self.postprocess(sources, max_results=max_results)

    def _compute_relevance(self, query_tokens, title: str, snippet: str) -> float:
        t_tok = self._tokenize(title)
        s_tok = self._tokenize(snippet)
        if not query_tokens:
            return 0.5
        raw = (3 * len(query_tokens & t_tok) + len(query_tokens & s_tok)) / (3 * len(query_tokens))
        return min(raw, 1.0)

    @staticmethod
    def _tokenize(text: str):
        stopwords = {"a","an","the","is","are","was","were","of","in","on","at","to","for","and","or","but","it","its","this","that"}
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
        return {tok for tok in cleaned.split() if len(tok) >= 2 and tok not in stopwords}
