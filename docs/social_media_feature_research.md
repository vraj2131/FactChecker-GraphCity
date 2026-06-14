# Feature 9 — Social Media Sources: Full Research & Implementation Plan

**Project:** FactGraph City  
**Author:** Research prepared for sprint meeting  
**Date:** 2026-05-28  
**Status:** Research complete — implementation ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Platform Landscape Analysis](#2-platform-landscape-analysis)
3. [Architecture Approaches](#3-architecture-approaches)
4. [Trust Score Framework for UGC](#4-trust-score-framework-for-ugc)
5. [NLP Pipeline for Social Media Text](#5-nlp-pipeline-for-social-media-text)
6. [Implementation: All Retrievers](#6-implementation-all-retrievers)
7. [Graph Integration Options](#7-graph-integration-options)
8. [UI/UX Design Options](#8-uiux-design-options)
9. [Cross-Platform Deduplication](#9-cross-platform-deduplication)
10. [Rate Limiting & Caching Strategy](#10-rate-limiting--caching-strategy)
11. [Privacy & Ethics](#11-privacy--ethics)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Recommended Phased Roadmap](#13-recommended-phased-roadmap)
14. [Files to Change (Complete List)](#14-files-to-change-complete-list)

---

## 1. Executive Summary

The current FactGraph City retrieves from 6 structured sources (Wikipedia, FactCheck, Guardian, NewsAPI, GDELT, DuckDuckGo). Social media represents a fundamentally different signal: **crowd-sourced, real-time, unfiltered, and often first to surface a story** — but also noisy and unreliable.

The original plan (Reddit + Bluesky) is a solid starting point, but this document expands that into a **full social media retrieval architecture** covering:

- **5 platforms** across 3 tiers of reliability
- **3 distinct implementation approaches** (Simple toggle → Ensemble → Per-platform granular)
- **Dynamic trust scoring** based on engagement signals (not just a fixed float)
- **Social-aware NLP** preprocessing before NLI classification
- **Graph enrichment** — social nodes sized by engagement, clustered by platform
- **Per-platform UI controls** replacing the single toggle

**Bottom line recommendation:** Implement Reddit + Hacker News in Phase 6a (both free, no key needed with proper setup), add Bluesky in Phase 6b, and leave Twitter/X as a documented option pending budget.

---

## 2. Platform Landscape Analysis

### 2.1 Full Platform Comparison

| Platform | Library | Auth | Cost | Quality | Volume | Notes |
|----------|---------|------|------|---------|--------|-------|
| **Reddit** | `praw` | Free app reg | Free | High | Very High | Best fact-check communities |
| **Hacker News** | `requests` (Algolia API) | None | Free | High | Medium | Tech-heavy, expert commentary |
| **Bluesky** | `atproto` | None for search | Free | Medium | Growing | Decentralized, full text search |
| **Mastodon** | `Mastodon.py` | None for public | Free | Medium | Niche | Academic/tech instances useful |
| **Twitter/X** | `tweepy` | OAuth2 required | Free: 500 req/mo | Very High | Very High | Too restricted at free tier |
| **YouTube** | `google-api-python-client` | API key (free quota) | Free (10k/day) | Medium | High | Comments on news videos |
| **Lemmy** | `requests` (REST) | None for public | Free | Medium | Low-Med | Reddit alternative, federated |

### 2.2 Platform Deep Dive

#### Reddit
- **Why it matters:** r/skeptic, r/science, r/worldnews, r/factcheck, r/conspiracy (useful as negative signal) are highly relevant to claim verification. Threads often contain citations and community debunking.
- **Search API:** `PRAW` wraps Reddit's public JSON API. `reddit.subreddit("all").search(query, sort="relevance", limit=10)` — no OAuth needed for read-only, just a `user_agent` and app client_id.
- **Signal quality:** Top posts + upvote ratio + comment count as trust proxy. A post with 5000 upvotes and 200 comments discussing a claim carries much more signal than 3 upvotes.
- **Subreddit tier list for fact-checking:**
  - Tier 1 (high quality): r/science, r/skeptic, r/worldnews, r/factcheck, r/politics
  - Tier 2 (medium): r/news, r/technology, r/history
  - Tier 3 (low, use as negative signal only): r/conspiracy, r/Qult_Headquarters

#### Hacker News
- **Why it matters:** Expert commentary. When a scientific claim is made, HN threads often contain corrections from domain experts. The community self-moderates aggressively.
- **API:** Algolia HN Search API — `http://hn.algolia.com/api/v1/search?query={claim}&tags=story` — fully public, no auth, generous rate limits.
- **Signal quality:** HN karma system is strict. Points = approval of knowledgeable community. Top comments ("Ask HN" threads) often debunk false claims with citations.

#### Bluesky
- **Why it matters:** Growing decentralized platform. Journalists and researchers use it heavily. ATProto protocol means full-text search across all public posts.
- **API:** `atproto` Python SDK. `client.app.bsky.feed.search_posts(q=query, limit=10)` returns posts with like count, repost count, reply count.
- **No auth needed** for read-only public search as of the current ATProto spec.

#### Mastodon
- **Why it matters:** Academic Mastodon instances (mastodon.social, scholar.social, fosstodon.org) are used by researchers. Fact-checking organizations have accounts there.
- **API:** `GET https://mastodon.social/api/v2/search?q={query}&type=statuses` — public, no auth for federated search.
- **Caveat:** Per-instance search; no global federated search endpoint. Best to query 3-4 major instances.

#### YouTube Comments
- **Why it matters:** News organization videos (BBC, Reuters, AP) have comment sections with community fact-checking. Top comments on a news video about a claim are relevant signals.
- **API:** YouTube Data API v3 (free, 10,000 units/day). Search for videos related to claim, then fetch top comments.
- **Trust signal:** Likes on a comment + comment thread replies.

### 2.3 Why Twitter/X Is Not Recommended Now

Free tier: 500 requests/month = ~16 requests/day. A single claim verification could burn 5-10 requests easily. At that rate, the system breaks after 1-3 days of usage. Paid Basic tier is $100/month. Documented and skippable for now — architecture supports adding it later without code changes beyond the retriever file.

---

## 3. Architecture Approaches

Three distinct approaches were evaluated. They are not mutually exclusive — Approach 1 can be implemented first, then extended to Approach 2.

### Approach A — Simple Toggle (Original Plan, Enhanced)

**What it is:** Single `include_social: bool` flag. When true, register Reddit + Bluesky retrievers alongside existing ones. All social results flow through the existing retrieval orchestration (dedup → rank → LLM).

**Pros:** Minimal code change, consistent with existing retriever pattern.  
**Cons:** No per-platform control, no engagement-weighted trust, social posts mixed with news in the same rank pool.

```
User flips toggle → include_social=True in POST body
→ verify_claim_service registers RedditRetriever + BlueskyRetriever
→ orchestrator runs all retrievers in parallel
→ dedup → rank → top 20 to LLM
```

**When to use:** MVP, demo.

---

### Approach B — Ensemble with Separate Social Pool

**What it is:** Social retrievers run in a separate pool. Results are processed independently through their own dedup + rank, then a fixed quota (e.g. top 5 social sources) is injected into the main pool before final ranking.

```
Main pool (news/wiki):  retrievers → dedup → rank → top 15
Social pool:            retrievers → dedup → social_rank → top 5
                                                           ↓
                             merged → final rank → top 20 to LLM
```

**Pros:** Prevents social media from crowding out high-quality news sources. Ensures at least some social signal even if social scores are lower. Configurable quota per source tier.  
**Cons:** More complex orchestration. Requires a `social_rank()` function aware of engagement signals.

**New constant:** `SOCIAL_SOURCES_QUOTA: int = 5` in `constants.py`

---

### Approach C — Per-Platform Granular Control

**What it is:** Each social platform is independently toggleable. Frontend sends `social_sources: List[str]` (e.g. `["reddit", "hackernews"]`) instead of a bool. Backend registers only the requested retrievers.

```python
# request_schema.py
class VerifyClaimRequest(BaseModel):
    claim: str
    social_sources: List[str] = []  # e.g. ["reddit", "hackernews", "bluesky"]
```

**UI:** Three separate toggle switches (Reddit, HN, Bluesky) with platform color indicators.

**Pros:** Maximum user control. Research use case — user can isolate "what does Reddit say vs HN vs Bluesky."  
**Cons:** More UI complexity, 3 toggles vs 1.

**Recommendation:** Implement Approach A first, document the migration path to Approach C, implement C in a follow-up sprint.

---

## 4. Trust Score Framework for UGC

Social media is user-generated content (UGC). Fixed trust scores (Reddit=0.45) miss a key insight: **a highly upvoted post in r/science with 500 comments is more trustworthy than a 0-upvote post in r/conspiracy.**

### 4.1 Dynamic Trust Score Formula

```
trust_score = base_trust × engagement_multiplier × subreddit_authority × recency_factor
```

**Clamped to [0.0, 1.0].**

#### Base Trust by Platform

| Platform | Base Trust | Reasoning |
|----------|-----------|-----------|
| HackerNews | 0.62 | Expert community, strong moderation |
| Reddit (Tier 1 subreddits) | 0.50 | Good moderation, citation culture |
| Reddit (Tier 2 subreddits) | 0.40 | Lower bar, but still useful |
| Reddit (Tier 3 / unfiltered) | 0.25 | Treat as weak signal |
| Bluesky | 0.38 | Newer, less moderation history |
| Mastodon | 0.42 | Academic instances raise this |
| YouTube Comments | 0.30 | High noise, low citation rate |

#### Engagement Multiplier (Reddit)

```python
def engagement_multiplier(upvotes: int, upvote_ratio: float, num_comments: int) -> float:
    score = (upvotes * upvote_ratio) + (num_comments * 0.5)
    if score > 1000:   return 1.3
    if score > 500:    return 1.15
    if score > 100:    return 1.05
    if score > 10:     return 1.0
    return 0.85  # low engagement = less trustworthy
```

#### Subreddit Authority Multiplier

```python
SUBREDDIT_AUTHORITY = {
    "science":     1.20,
    "skeptic":     1.15,
    "worldnews":   1.10,
    "factcheck":   1.25,
    "news":        1.05,
    "technology":  1.05,
    "history":     1.05,
    # default subreddits
    "all":         1.0,
    # negative signal subreddits
    "conspiracy":  0.50,
}
```

#### Recency Factor

```python
def recency_factor(published_at: str) -> float:
    days_old = (now - parse(published_at)).days
    if days_old < 7:    return 1.10   # bonus for very fresh
    if days_old < 30:   return 1.0
    if days_old < 180:  return 0.95
    return 0.90                        # older social posts decay faster than news
```

### 4.2 Comparison: Fixed vs Dynamic Trust

| Source | Fixed Trust | Dynamic (example) |
|--------|------------|-------------------|
| Reddit r/science post, 2000 upvotes, 300 comments, posted 2 days ago | 0.45 | 0.50 × 1.3 × 1.20 × 1.10 = **0.858** |
| Reddit r/conspiracy post, 3 upvotes, 0 comments, 1 year old | 0.45 | 0.25 × 0.85 × 0.50 × 0.90 = **0.096** |

Dynamic scoring dramatically separates signal from noise — something a fixed score cannot do.

---

## 5. NLP Pipeline for Social Media Text

Social media text requires preprocessing before NLI classification because the existing DeBERTa NLI model was trained on clean sentences, not raw tweets or Reddit posts.

### 5.1 Problems with Raw Social Text

| Problem | Example | Effect on NLI |
|---------|---------|--------------|
| Hashtags | `#FakeNews #ClimateChange` | Tokenization noise |
| @mentions | `@CNN is wrong about this` | Named entity confusion |
| URLs inline | `https://t.co/xyz was debunked` | OOV tokens |
| Emoji | `This is FALSE 🤥🤥🤥` | Often dropped, loses sentiment |
| Slang/abbreviations | `tbh idk if this is true lol` | Low NLI confidence |
| Sarcasm | `Oh sure, vaccines cause autism, totally` | NLI misclassification |
| Quote-tweets | `"X said Y" — this is wrong` | Double negation |

### 5.2 Social Text Preprocessor

```python
# backend/app/preprocessing/social_text_cleaner.py

import re

def clean_social_text(text: str, platform: str = "generic") -> str:
    """
    Normalize social media text before NLI snippet extraction.
    """
    # Remove URLs (keep domain for context if needed)
    text = re.sub(r'https?://\S+', '[URL]', text)

    # Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Convert hashtags to plain words: #ClimateCrisis → climate crisis
    text = re.sub(r'#(\w+)', lambda m: _camel_to_words(m.group(1)), text)

    # Remove emoji (keep text around them)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Platform-specific: Reddit upvote notation, post flair
    if platform == "reddit":
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)
        text = re.sub(r'Edit\s*\d*:', '', text, flags=re.IGNORECASE)

    return text


def _camel_to_words(s: str) -> str:
    """Convert CamelCase hashtag to spaced words."""
    return re.sub(r'([A-Z])', r' \1', s).lower().strip()


def is_sarcasm_candidate(text: str) -> bool:
    """
    Heuristic sarcasm detection — flag for lower confidence weighting.
    Not a full sarcasm classifier, just a signal.
    """
    sarcasm_markers = [
        r'\boh sure\b', r'\boh yeah\b', r'\btotally\b.*\b(not|wrong|false)\b',
        r'\bright\b.*🙄', r'\bsure\b.*\bwhatever\b', r'/s$', r'/s\b'
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in sarcasm_markers)
```

### 5.3 Snippet Selection from Social Posts

Reddit posts often have long bodies. The best snippet is not always the post title or the full body — it's the sentence most relevant to the claim:

```python
def extract_best_snippet(post_body: str, claim: str, max_chars: int = 300) -> str:
    """
    Split post body into sentences, return the one most lexically
    similar to the claim. Falls back to first 300 chars if no good match.
    """
    sentences = re.split(r'(?<=[.!?])\s+', post_body)
    if not sentences:
        return post_body[:max_chars]

    claim_tokens = set(claim.lower().split())
    best, best_score = sentences[0], 0.0

    for sent in sentences:
        sent_tokens = set(sent.lower().split())
        if not sent_tokens:
            continue
        jaccard = len(claim_tokens & sent_tokens) / len(claim_tokens | sent_tokens)
        if jaccard > best_score:
            best_score = jaccard
            best = sent

    return best[:max_chars]
```

### 5.4 Stance Hint from Social Text

Social posts often contain explicit linguistic stance markers that can pre-populate `stance_hint` before NLI:

```python
SUPPORT_MARKERS = ["confirmed", "true", "correct", "verified", "this is real", "source:", "fact:"]
REFUTE_MARKERS  = ["false", "debunked", "wrong", "misinformation", "fake", "not true", "myth"]

def infer_stance_hint(text: str) -> Optional[str]:
    text_lower = text.lower()
    support_hits = sum(1 for m in SUPPORT_MARKERS if m in text_lower)
    refute_hits  = sum(1 for m in REFUTE_MARKERS  if m in text_lower)

    if support_hits > refute_hits and support_hits >= 2:
        return "supports"
    if refute_hits > support_hits and refute_hits >= 2:
        return "refutes"
    return None  # let NLI model decide
```

---

## 6. Implementation: All Retrievers

All retrievers inherit `BaseRetriever` from `backend/app/retrieval/base_retriever.py`. They implement two methods: `fetch_raw()` and `normalize()`.

### 6.1 Reddit Retriever

**Auth setup:** Create app at reddit.com/prefs/apps → "script" type → get `client_id` + `client_secret`. Store in `.env`.

```python
# backend/app/retrieval/reddit_retriever.py

import os
import praw
from typing import Any, List
from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.preprocessing.social_text_cleaner import (
    clean_social_text, extract_best_snippet, infer_stance_hint, is_sarcasm_candidate
)
from backend.app.utils.constants import (
    REDDIT_BASE_TRUST, SUBREDDIT_AUTHORITY, REDDIT_TIER1_SUBREDDITS
)


class RedditRetriever(BaseRetriever):

    TIER1_SUBREDDITS = {
        "science", "skeptic", "worldnews", "factcheck",
        "news", "technology", "history", "politics"
    }

    def __init__(self):
        super().__init__(source_name="reddit")
        self._reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="FactGraphCity/1.0 by vraj2131",
            check_for_async=False,
        )

    def fetch_raw(self, query: str, max_results: int = 10, **kwargs) -> Any:
        subreddit_filter = kwargs.get("subreddit", "all")
        sort = kwargs.get("sort", "relevance")
        results = []

        try:
            submissions = self._reddit.subreddit(subreddit_filter).search(
                query, sort=sort, limit=max_results * 2  # over-fetch for filtering
            )
            for sub in submissions:
                results.append({
                    "id":           sub.id,
                    "title":        sub.title,
                    "selftext":     sub.selftext,
                    "url":          f"https://reddit.com{sub.permalink}",
                    "subreddit":    sub.subreddit.display_name.lower(),
                    "score":        sub.score,
                    "upvote_ratio": sub.upvote_ratio,
                    "num_comments": sub.num_comments,
                    "created_utc":  sub.created_utc,
                    "author":       str(sub.author) if sub.author else "[deleted]",
                })
        except Exception:
            pass  # rate limit or network — return empty

        return results

    def normalize(self, raw_data: Any, query: str, max_results: int = 10, **kwargs) -> List[Source]:
        sources = []
        for post in raw_data:
            body = post.get("selftext", "")
            cleaned_body = clean_social_text(body, platform="reddit")
            snippet = extract_best_snippet(cleaned_body or post["title"], query)

            if len(snippet) < 30:   # too short to be useful
                continue

            trust = self._compute_trust(post)
            stance = infer_stance_hint(snippet)

            import datetime
            published = datetime.datetime.utcfromtimestamp(
                post["created_utc"]
            ).isoformat() + "Z"

            sources.append(Source(
                source_id=f"reddit_{post['id']}",
                source_type="reddit",
                title=post["title"][:500],
                url=post["url"],
                publisher=f"r/{post['subreddit']}",
                snippet=snippet,
                published_at=published,
                trust_score=trust,
                relevance_score=0.5,   # will be updated by ranking_service
                stance_hint=stance,
            ))

        return sources[:max_results]

    def _compute_trust(self, post: dict) -> float:
        base = REDDIT_BASE_TRUST

        # Engagement multiplier
        score = post["score"] * post["upvote_ratio"] + post["num_comments"] * 0.5
        if score > 1000:    mult = 1.30
        elif score > 500:   mult = 1.15
        elif score > 100:   mult = 1.05
        elif score > 10:    mult = 1.00
        else:               mult = 0.85

        # Subreddit authority
        sub = post["subreddit"]
        auth = SUBREDDIT_AUTHORITY.get(sub, 1.0)

        return min(1.0, base * mult * auth)
```

### 6.2 Hacker News Retriever

**No auth. No library needed. Uses Algolia's public HN search API.**

```python
# backend/app/retrieval/hackernews_retriever.py

import requests
from datetime import datetime
from typing import Any, List
from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.preprocessing.social_text_cleaner import clean_social_text, infer_stance_hint


HN_SEARCH_URL = "http://hn.algolia.com/api/v1/search"


class HackerNewsRetriever(BaseRetriever):

    def __init__(self):
        super().__init__(source_name="hackernews")

    def fetch_raw(self, query: str, max_results: int = 10, **kwargs) -> Any:
        params = {
            "query":    query,
            "tags":     "story",
            "hitsPerPage": max_results * 2,
        }
        try:
            resp = requests.get(HN_SEARCH_URL, params=params, timeout=8)
            resp.raise_for_status()
            return resp.json().get("hits", [])
        except Exception:
            return []

    def normalize(self, raw_data: Any, query: str, max_results: int = 10, **kwargs) -> List[Source]:
        sources = []
        for hit in raw_data:
            title = hit.get("title", "")
            text  = clean_social_text(hit.get("story_text") or "", platform="hackernews")
            snippet = text[:300] if text else title

            if not snippet:
                continue

            points    = hit.get("points", 0) or 0
            comments  = hit.get("num_comments", 0) or 0
            trust     = self._compute_trust(points, comments)
            stance    = infer_stance_hint(snippet)

            hn_id = hit.get("objectID", "")
            url   = hit.get("url") or f"https://news.ycombinator.com/item?id={hn_id}"

            created_iso = hit.get("created_at", "")

            sources.append(Source(
                source_id=f"hn_{hn_id}",
                source_type="hackernews",
                title=title[:500],
                url=url,
                publisher="Hacker News",
                snippet=snippet,
                published_at=created_iso,
                trust_score=trust,
                relevance_score=0.5,
                stance_hint=stance,
            ))

        return sources[:max_results]

    def _compute_trust(self, points: int, comments: int) -> float:
        base = 0.62
        engagement = points + comments * 0.5
        if engagement > 500:   return min(1.0, base * 1.30)
        if engagement > 100:   return min(1.0, base * 1.15)
        if engagement > 30:    return min(1.0, base * 1.05)
        if engagement > 5:     return base
        return base * 0.85
```

### 6.3 Bluesky Retriever

**No key for public search. Uses `atproto` SDK.**

```python
# backend/app/retrieval/bluesky_retriever.py

from atproto import Client
from typing import Any, List
from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.preprocessing.social_text_cleaner import clean_social_text, infer_stance_hint


class BlueskyRetriever(BaseRetriever):

    def __init__(self):
        super().__init__(source_name="bluesky")
        self._client = Client()

    def fetch_raw(self, query: str, max_results: int = 10, **kwargs) -> Any:
        try:
            resp = self._client.app.bsky.feed.search_posts(
                {"q": query, "limit": max_results * 2}
            )
            return resp.posts or []
        except Exception:
            return []

    def normalize(self, raw_data: Any, query: str, max_results: int = 10, **kwargs) -> List[Source]:
        sources = []
        for post in raw_data:
            record = getattr(post, "record", None)
            if not record:
                continue

            text   = clean_social_text(getattr(record, "text", ""), platform="bluesky")
            if len(text) < 20:
                continue

            likes    = getattr(post, "like_count",   0) or 0
            reposts  = getattr(post, "repost_count", 0) or 0
            replies  = getattr(post, "reply_count",  0) or 0
            trust    = self._compute_trust(likes, reposts, replies)
            stance   = infer_stance_hint(text)

            uri = getattr(post, "uri", "")
            # uri format: at://did:plc:xxx/app.bsky.feed.post/RKEY
            rkey   = uri.split("/")[-1] if uri else "unknown"
            handle = getattr(post.author, "handle", "unknown")
            url    = f"https://bsky.app/profile/{handle}/post/{rkey}"

            created = getattr(record, "created_at", "")

            sources.append(Source(
                source_id=f"bsky_{rkey}",
                source_type="bluesky",
                title=text[:120],   # Bluesky posts are short — title = truncated text
                url=url,
                publisher=f"@{handle} on Bluesky",
                snippet=text[:300],
                published_at=created,
                trust_score=trust,
                relevance_score=0.5,
                stance_hint=stance,
            ))

        return sources[:max_results]

    def _compute_trust(self, likes: int, reposts: int, replies: int) -> float:
        base = 0.38
        engagement = likes + reposts * 1.5 + replies * 0.8
        if engagement > 200:  return min(1.0, base * 1.30)
        if engagement > 50:   return min(1.0, base * 1.15)
        if engagement > 10:   return min(1.0, base * 1.05)
        return base
```

### 6.4 Mastodon Retriever (Bonus — No Library)

**Uses public REST API directly. No library needed. Multi-instance search.**

```python
# backend/app/retrieval/mastodon_retriever.py

import requests
from typing import Any, List
from backend.app.retrieval.base_retriever import BaseRetriever
from backend.app.schemas.source_schema import Source
from backend.app.preprocessing.social_text_cleaner import clean_social_text
import re


MASTODON_INSTANCES = [
    "mastodon.social",
    "fosstodon.org",
    "scholar.social",   # academic instance — high quality
]


class MastodonRetriever(BaseRetriever):

    def __init__(self):
        super().__init__(source_name="mastodon")

    def fetch_raw(self, query: str, max_results: int = 10, **kwargs) -> Any:
        results = []
        per_instance = max(2, max_results // len(MASTODON_INSTANCES))

        for instance in MASTODON_INSTANCES:
            try:
                url = f"https://{instance}/api/v2/search"
                resp = requests.get(url, params={
                    "q": query,
                    "type": "statuses",
                    "limit": per_instance,
                    "resolve": "false",
                }, timeout=6)
                if resp.status_code == 200:
                    statuses = resp.json().get("statuses", [])
                    for s in statuses:
                        s["_instance"] = instance
                    results.extend(statuses)
            except Exception:
                continue

        return results

    def normalize(self, raw_data: Any, query: str, max_results: int = 10, **kwargs) -> List[Source]:
        sources = []
        for status in raw_data:
            # Strip HTML tags from Mastodon content
            content_html = status.get("content", "")
            content_text = re.sub(r'<[^>]+>', ' ', content_html)
            text = clean_social_text(content_text, platform="mastodon")

            if len(text) < 20:
                continue

            favourites = status.get("favourites_count", 0) or 0
            reblogs    = status.get("reblogs_count",    0) or 0
            trust      = min(1.0, 0.42 + (favourites + reblogs * 1.5) * 0.0005)

            account   = status.get("account", {})
            handle    = account.get("acct", "unknown")
            url       = status.get("url", "")
            created   = status.get("created_at", "")
            instance  = status.get("_instance", "mastodon.social")

            sources.append(Source(
                source_id=f"mastodon_{status.get('id', 'x')}",
                source_type="mastodon",
                title=text[:120],
                url=url,
                publisher=f"@{handle} on {instance}",
                snippet=text[:300],
                published_at=created,
                trust_score=trust,
                relevance_score=0.5,
            ))

        return sources[:max_results]
```

---

## 7. Graph Integration Options

Social media nodes need special handling in the graph — they are visually distinct from academic/news sources.

### 7.1 Option A — Mixed Nodes (Simple)

Social nodes appear in the graph alongside news nodes, distinguished only by color. No structural change.

- Reddit: `#FF4500` (Reddit orange-red)
- HackerNews: `#FF6600` (HN orange)
- Bluesky: `#0085FF` (sky blue)
- Mastodon: `#6364FF` (Mastodon purple)

**Change:** `frontend/src/utils/colorMap.js` — add 4 new entries.

### 7.2 Option B — Engagement-Sized Nodes

Node sphere radius in ForceGraph3D is proportional to engagement (upvotes/likes). A post with 5000 upvotes = large node; 3 upvotes = tiny node.

```javascript
// graphTransforms.js
function socialNodeSize(node) {
  if (!node.engagement) return 4;  // default
  if (node.engagement > 1000) return 12;
  if (node.engagement > 500)  return 9;
  if (node.engagement > 100)  return 7;
  if (node.engagement > 10)   return 5;
  return 3;
}
```

**New field needed on Node schema:** `engagement: Optional[int]` — sum of upvotes/likes for that source.

### 7.3 Option C — Social Media Cluster

Social nodes are grouped into a sub-cluster. Instead of connecting directly to main claim, they connect to a "Social Media" meta-node, which connects to main claim. Keeps graph visually clean.

```
Main Claim
├── Guardian article
├── Wikipedia
├── FactCheck review
└── [Social Media Hub] ← meta-node
    ├── Reddit post 1
    ├── Reddit post 2
    └── HN post 1
```

**Pros:** Prevents social nodes from cluttering main graph when many social sources are present.  
**Change:** `backend/app/graph/graph_builder_service.py` — inject a `social_hub` node when social sources > 3.

### 7.4 Option D — Community Sentiment Ring

An additional visual element in SideInfoPanel when clicking the main claim node: a "Community Sentiment" ring chart showing percentage breakdown of social posts that support / refute / are neutral, aggregated from all social sources.

```
Community Sentiment (from 8 social sources):
[████████░░░░░░░] 53% Support
[█████░░░░░░░░░░] 33% Refute
[██░░░░░░░░░░░░░] 14% Neutral
```

**Change:** New `CommunitySentimentBar.jsx` component, called from `MainClaimPanel`.

---

## 8. UI/UX Design Options

### 8.1 Option A — Single Toggle (Original)

```
┌─ FilterPanel ───────────────────────────────┐
│  Sources: [✓] News  [✓] Wikipedia  [✓] GDELT │
│                                               │
│  Social Media:  ○──────● ON                  │
│  [Reddit] [Bluesky] active                   │
└───────────────────────────────────────────────┘
```

### 8.2 Option B — Per-Platform Toggles

```
┌─ FilterPanel ────────────────────────────────────┐
│  Social Media Sources                            │
│  ┌──────────────┬──────┐                         │
│  │ 🟠 Reddit    │  ON  │  r/science, r/skeptic   │
│  ├──────────────┼──────┤                         │
│  │ 🟠 HackerNews│  OFF │  Expert commentary       │
│  ├──────────────┼──────┤                         │
│  │ 🔵 Bluesky   │  OFF │  Journalists/researchers │
│  └──────────────┴──────┘                         │
└──────────────────────────────────────────────────┘
```

### 8.3 Option C — Social Media Sidebar Section

When a social node is clicked, SideInfoPanel shows a rich social card:

```
┌─ SideInfoPanel ──────────────────────────────────────┐
│  🟠 REDDIT  ·  r/skeptic  ·  Posted 2 days ago        │
│                                                        │
│  "The claim that X has been debunked. See [URL]..."    │
│                                                        │
│  ▲ 2,341 upvotes  ·  ▼ 4%  ·  💬 127 comments         │
│                                                        │
│  Trust:    ████████░░  0.82                            │
│  Relevance: ██████░░░░  0.64                           │
│  Stance:   REFUTES                                     │
│                                                        │
│  [→ Verify as Claim]  [↗ Open on Reddit]               │
└────────────────────────────────────────────────────────┘
```

### 8.4 Subreddit Filter UI

A dropdown when Reddit toggle is ON: "Filter to subreddit: [All ▼]" with options for Tier 1 subreddits.

```javascript
// FilterPanel.jsx
const SUBREDDIT_OPTIONS = [
  { value: "all",      label: "All subreddits" },
  { value: "science",  label: "r/science (Expert moderated)" },
  { value: "skeptic",  label: "r/skeptic (Skeptic community)" },
  { value: "worldnews",label: "r/worldnews (Global news)" },
  { value: "factcheck",label: "r/factcheck (Fact-checkers)" },
];
```

---

## 9. Cross-Platform Deduplication

The existing `deduplicate.py` uses URL canonicalization + Jaccard similarity. For social media, additional deduplication is needed because:

1. **Same story shared across platforms:** A Reuters article gets shared on Reddit, retweeted on Twitter, posted on Bluesky. All three social posts link to the same canonical news URL. → Deduplicate by **linked URL, not post URL.**

2. **Near-duplicate posts on same platform:** Multiple Reddit posts about the same story. → Existing Jaccard on snippet tokens handles this.

### 9.1 Linked-URL Deduplication

```python
# backend/app/preprocessing/deduplicate.py — extend existing function

def extract_linked_url(source: Source) -> Optional[str]:
    """
    For social media sources, extract the URL being shared in the snippet.
    Returns None if no outbound link found.
    """
    if source.source_type not in {"reddit", "bluesky", "hackernews", "mastodon"}:
        return None
    if not source.snippet:
        return None
    urls = re.findall(r'https?://\S+', source.snippet)
    return canonicalize_url(urls[0]) if urls else None


def deduplicate_cross_platform(sources: List[Source]) -> List[Source]:
    """
    After Jaccard dedup, remove social posts that all link to the same
    canonical article. Keep the one with the highest trust_score.
    """
    seen_linked_urls: dict[str, Source] = {}
    result = []

    for source in sources:
        linked = extract_linked_url(source)
        if linked:
            existing = seen_linked_urls.get(linked)
            if existing:
                # Keep higher trust
                if source.trust_score > existing.trust_score:
                    result.remove(existing)
                    seen_linked_urls[linked] = source
                    result.append(source)
                # else discard current
            else:
                seen_linked_urls[linked] = source
                result.append(source)
        else:
            result.append(source)

    return result
```

---

## 10. Rate Limiting & Caching Strategy

### 10.1 Rate Limit Summary

| Platform | Rate Limit | Strategy |
|----------|-----------|----------|
| Reddit (PRAW) | 100 req/min (read-only) | Respectful — 1 search = 1 req |
| HackerNews Algolia | No documented limit (generous) | Cache aggressively |
| Bluesky ATProto | ~3000 req/hr for unauthenticated | Sufficient for our use |
| Mastodon | Per-instance, typically 300 req/5min | Multi-instance spread reduces pressure |

### 10.2 Caching

The existing `cache_service.py` already caches retrieval results to `data/cache/retrieval_results/`. Social media results are more time-sensitive than Wikipedia but less than live news.

**Recommended TTLs:**

```python
# constants.py
CACHE_TTL_SECONDS = {
    "wikipedia":   86400,   # 24h — rarely changes
    "factcheck":   3600,    # 1h
    "guardian":    1800,    # 30min
    "newsapi":     1800,    # 30min
    "gdelt":       900,     # 15min
    "duckduckgo":  3600,    # 1h
    "reddit":      600,     # 10min — social media moves fast
    "hackernews":  1200,    # 20min
    "bluesky":     300,     # 5min — very real-time
    "mastodon":    600,     # 10min
}
```

---

## 11. Privacy & Ethics

| Concern | Mitigation |
|---------|-----------|
| Storing user posts | Only cache for TTL defined above; never store in persistent DB |
| PII in posts | `clean_social_text()` strips @mentions. Avoid storing author usernames in graph output by default |
| Deleted posts | PRAW returns `[deleted]` for removed content — filter out in `normalize()` |
| Minors' content | Out of scope for public search endpoints; Reddit search returns only public content |
| Rate limit respect | All retrievers use timeout + try/except; PRAW respects Reddit's rate headers automatically |
| Content that might be harmful | Social sources are used as evidence signals, not surfaced verbatim to end user beyond snippet |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Reddit API changes auth requirements | Medium | High | PRAW is widely maintained; monitor their GitHub |
| Social noise degrades LLM classification | High | Medium | Social pool quota (Approach B) caps at 5/20 LLM slots |
| HN Algolia goes down | Low | Low | Graceful except → empty list; other sources still work |
| Bluesky ATProto auth policy changes | Medium | Medium | Already isolated in one file; swap to authenticated calls |
| Sarcasm misclassified by NLI | High | Medium | `is_sarcasm_candidate()` flag → reduce confidence weight for flagged posts |
| Cross-platform viral misinformation | Medium | High | Trust caps at 0.85 for social regardless of engagement |

---

## 13. Recommended Phased Roadmap

### Phase 6a — Core Social (1 sprint, ~3 days)

Priority: **Reddit + HackerNews** (both free, no paid tier, highest quality UGC).

1. `backend/app/preprocessing/social_text_cleaner.py` — new file (preprocessor)
2. `backend/app/retrieval/reddit_retriever.py` — new file
3. `backend/app/retrieval/hackernews_retriever.py` — new file
4. `backend/app/schemas/request_schema.py` — add `include_social: bool = False`
5. `backend/app/services/verify_claim_service.py` — conditional registration
6. `backend/app/utils/constants.py` — trust scores, subreddit authority map, cache TTLs
7. `frontend/src/utils/colorMap.js` — Reddit + HN colors/labels
8. `frontend/src/components/FilterPanel.jsx` — single social toggle
9. `frontend/src/api/client.js` — pass `include_social` in POST body
10. `frontend/src/App.jsx` — track toggle state

**Deliverable:** Social toggle in UI, Reddit + HN results appear in graph.

---

### Phase 6b — Extended Social (1 sprint, ~2 days)

Priority: **Bluesky + Mastodon + Engagement-sized nodes**.

1. `backend/app/retrieval/bluesky_retriever.py` — new file
2. `backend/app/retrieval/mastodon_retriever.py` — new file
3. `backend/app/schemas/node_schema.py` — add `engagement: Optional[int]`
4. `backend/app/graph/graph_builder_service.py` — populate `engagement` from source metadata
5. `frontend/src/utils/graphTransforms.js` — `socialNodeSize()` function
6. `frontend/src/components/FilterPanel.jsx` — per-platform toggles (Approach C)
7. `backend/app/schemas/request_schema.py` — migrate to `social_sources: List[str] = []`
8. `backend/app/preprocessing/deduplicate.py` — `deduplicate_cross_platform()`

**Deliverable:** 4 social platforms, per-platform toggles, engagement-sized nodes.

---

### Phase 6c — UX Polish (0.5 sprint, ~1 day)

1. `frontend/src/components/SideInfoPanel.jsx` — rich social card (upvotes, comments, ratio display)
2. `frontend/src/components/CommunitySentimentBar.jsx` — sentiment aggregation ring
3. `frontend/src/components/FilterPanel.jsx` — subreddit filter dropdown for Reddit
4. Inject `social_hub` meta-node when social sources > 3 (Approach C in graph)

**Deliverable:** Production-quality social media UX.

---

## 14. Files to Change (Complete List)

### New Files

| File | Description |
|------|-------------|
| `backend/app/retrieval/reddit_retriever.py` | Reddit retriever (PRAW) |
| `backend/app/retrieval/hackernews_retriever.py` | HN retriever (Algolia API) |
| `backend/app/retrieval/bluesky_retriever.py` | Bluesky retriever (atproto) |
| `backend/app/retrieval/mastodon_retriever.py` | Mastodon retriever (REST) |
| `backend/app/preprocessing/social_text_cleaner.py` | Social NLP preprocessor |
| `frontend/src/components/CommunitySentimentBar.jsx` | Sentiment aggregation UI |

### Modified Files

| File | Change |
|------|--------|
| `backend/app/schemas/request_schema.py` | Add `include_social` / `social_sources` field |
| `backend/app/schemas/node_schema.py` | Add `engagement: Optional[int]` |
| `backend/app/schemas/source_schema.py` | Add `reddit`, `hackernews`, `bluesky`, `mastodon` to `source_type` enum |
| `backend/app/services/verify_claim_service.py` | Conditional social retriever registration |
| `backend/app/utils/constants.py` | Trust scores, subreddit authority, cache TTLs, source type priorities |
| `backend/app/preprocessing/deduplicate.py` | `deduplicate_cross_platform()` |
| `backend/app/graph/graph_builder_service.py` | Populate `engagement`, optional social hub node |
| `frontend/src/utils/colorMap.js` | 4 new platform colors |
| `frontend/src/utils/graphTransforms.js` | `socialNodeSize()` |
| `frontend/src/components/FilterPanel.jsx` | Social toggle(s) + subreddit dropdown |
| `frontend/src/components/SideInfoPanel.jsx` | Rich social card in EvidencePanel |
| `frontend/src/api/client.js` | Pass social params in POST body |
| `frontend/src/App.jsx` | Track social toggle state |
| `.env` | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |

### New pip Dependencies

```
praw>=7.7.0
atproto>=0.0.50
```

No new dependencies needed for HackerNews (uses `requests`, already installed) or Mastodon (uses `requests`).

---

*End of research document. Total platforms researched: 7. Implementation approaches evaluated: 3. Retrievers fully specced: 4. Code-ready with full schema compatibility.*
