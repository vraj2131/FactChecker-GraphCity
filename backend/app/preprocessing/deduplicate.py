import re
from typing import List
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from backend.app.schemas.source_schema import Source
from backend.app.utils.constants import DEDUP_JACCARD_THRESHOLD

# Query params that carry no content identity — strip them during canonicalization
_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "ref",
        "source",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
    }
)


def canonicalize_url(url: str) -> str:
    """
    Produce a stable, lowercase string key for URL-based deduplication.

    Steps:
    - Parse into components
    - Lowercase scheme + netloc
    - Remove trailing slashes from path
    - Strip tracking query parameters
    - Re-serialize deterministically (sorted remaining params)
    """
    if not url or not url.strip():
        return url

    parsed = urlparse(url.strip())

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")

    # Filter out tracking params
    raw_params = parse_qs(parsed.query, keep_blank_values=False)
    filtered_params = {
        k: v for k, v in raw_params.items() if k.lower() not in _TRACKING_PARAMS
    }
    query_string = urlencode(sorted(filtered_params.items()), doseq=True)

    canonical = urlunparse((scheme, netloc, path, parsed.params, query_string, ""))
    return canonical


def _word_set(text: str) -> frozenset:
    """
    Tokenize text into a lowercase word set for Jaccard comparison.
    Strips punctuation and splits on whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return frozenset(text.split())


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """
    Compute Jaccard similarity between two word sets.
    Returns 0.0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_sources(sources: List[Source]) -> List[Source]:
    """
    Remove duplicate and near-duplicate sources from a list.

    Two-pass deduplication:

    Pass 1 — URL deduplication:
        Group by canonicalize_url(source.url).
        Within each URL group keep the source with the highest trust_score.
        Tie-breaks preserve the first occurrence.

    Pass 2 — Snippet near-duplicate detection:
        For each surviving source that has a non-empty snippet,
        compare against already-accepted snippets using Jaccard word overlap.
        If similarity >= DEDUP_JACCARD_THRESHOLD, discard the duplicate
        (keep the already-accepted version, which had higher trust_score or appeared earlier).

    Sources without snippets are always kept after URL dedup.

    Original relative order (by first occurrence of each URL) is preserved.
    """
    if not sources:
        return []

    # --- Pass 1: URL deduplication ---
    url_seen: dict = {}  # canonical_url -> index in url_survivors
    url_survivors: List[Source] = []

    for source in sources:
        canonical = canonicalize_url(str(source.url))

        if canonical not in url_seen:
            url_seen[canonical] = len(url_survivors)
            url_survivors.append(source)
        else:
            existing_idx = url_seen[canonical]
            existing = url_survivors[existing_idx]
            # Keep the one with higher trust_score
            if source.trust_score > existing.trust_score:
                url_survivors[existing_idx] = source

    # --- Pass 2: Snippet near-duplicate detection ---
    accepted: List[Source] = []
    accepted_snippet_sets: List[frozenset] = []

    for source in url_survivors:
        snippet = source.snippet

        if not snippet or not snippet.strip():
            # No snippet — cannot compare, always accept
            accepted.append(source)
            accepted_snippet_sets.append(frozenset())
            continue

        candidate_words = _word_set(snippet)

        is_near_dup = False
        for existing_words in accepted_snippet_sets:
            if not existing_words:
                continue
            sim = _jaccard(candidate_words, existing_words)
            if sim >= DEDUP_JACCARD_THRESHOLD:
                is_near_dup = True
                break

        if not is_near_dup:
            accepted.append(source)
            accepted_snippet_sets.append(candidate_words)

    return accepted
