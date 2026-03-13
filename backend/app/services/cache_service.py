import json
from pathlib import Path
from typing import Any, Optional

from backend.app.utils.file_io import ensure_dir


class CacheService:
    """
    Simple file-based cache service for Phase 4 retrieval work.

    Use cases:
    - cache raw API responses
    - cache normalized retriever outputs
    - avoid repeated external calls for the same query
    """

    def __init__(self, base_cache_dir: Path) -> None:
        self.base_cache_dir = base_cache_dir
        ensure_dir(self.base_cache_dir)

    def _build_path(self, namespace: str, cache_key: str) -> Path:
        if not namespace or not namespace.strip():
            raise ValueError("namespace cannot be empty.")
        if not cache_key or not cache_key.strip():
            raise ValueError("cache_key cannot be empty.")

        namespace_dir = self.base_cache_dir / namespace.strip()
        ensure_dir(namespace_dir)
        return namespace_dir / f"{cache_key}.json"

    def exists(self, namespace: str, cache_key: str) -> bool:
        path = self._build_path(namespace, cache_key)
        return path.exists()

    def load(self, namespace: str, cache_key: str) -> Optional[Any]:
        path = self._build_path(namespace, cache_key)

        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, namespace: str, cache_key: str, payload: Any) -> Path:
        path = self._build_path(namespace, cache_key)

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return path

    def delete(self, namespace: str, cache_key: str) -> bool:
        path = self._build_path(namespace, cache_key)

        if not path.exists():
            return False

        path.unlink()
        return True

    def clear_namespace(self, namespace: str) -> int:
        if not namespace or not namespace.strip():
            raise ValueError("namespace cannot be empty.")

        namespace_dir = self.base_cache_dir / namespace.strip()
        if not namespace_dir.exists():
            return 0

        deleted_count = 0
        for file_path in namespace_dir.glob("*.json"):
            file_path.unlink()
            deleted_count += 1

        return deleted_count
