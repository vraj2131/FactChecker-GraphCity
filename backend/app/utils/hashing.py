import hashlib
import json
from typing import Any


def stable_hash_text(text: str) -> str:
    """
    Create a stable SHA256 hash for a plain text string.
    """
    if text is None:
        raise ValueError("text cannot be None.")

    normalized = str(text).strip()
    if not normalized:
        raise ValueError("text cannot be empty or whitespace only.")

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_hash_object(obj: Any) -> str:
    """
    Create a stable SHA256 hash for a JSON-serializable Python object.
    Useful for caching query + params payloads.
    """
    try:
        normalized = json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except TypeError as e:
        raise ValueError(f"Object is not JSON serializable: {e}") from e

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_cache_key(source_name: str, query: str, **kwargs: Any) -> str:
    """
    Build a stable cache key from:
    - source name
    - query
    - extra keyword parameters

    Example use:
        build_cache_key(
            source_name="newsapi",
            query="Amazon stock rose 5%",
            max_results=10,
            language="en",
        )
    """
    if source_name is None or not str(source_name).strip():
        raise ValueError("source_name cannot be empty.")

    if query is None or not str(query).strip():
        raise ValueError("query cannot be empty.")

    payload = {
        "source_name": str(source_name).strip().lower(),
        "query": str(query).strip(),
        "params": kwargs,
    }

    return stable_hash_object(payload)
