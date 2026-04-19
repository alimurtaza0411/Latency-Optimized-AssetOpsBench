"""Query-intent cache for pre-planner semantic reuse.

This cache is intentionally lightweight and dependency-free so it can run in
environments without heavy embedding/reranker stacks.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


_WS_RE = re.compile(r"\s+")


@dataclass
class QueryCacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0


class QueryIntentCache:
    """Semantic cache keyed by normalized user intent text."""

    def __init__(
        self,
        semantic_threshold: float = 0.92,
        ttl_seconds: float = 1800.0,
    ) -> None:
        self.semantic_threshold = semantic_threshold
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, dict[str, Any]] = {}
        self.stats = QueryCacheStats()
        self.last_event: dict[str, Any] = {}

    def _normalize(self, query: str) -> str:
        text = (query or "").strip().lower()
        text = _WS_RE.sub(" ", text)
        return text

    def lookup(self, query: str) -> tuple[bool, dict[str, Any] | None]:
        now = time.time()
        key = self._normalize(query)

        # Exact first.
        exact = self._store.get(key)
        if exact and exact["expires_at"] > now:
            self.stats.hits += 1
            self.last_event = {"hit": True, "mode": "exact", "score": 1.0}
            return True, exact["payload"]

        # Semantic fallback.
        best_score = 0.0
        best_payload: dict[str, Any] | None = None
        best_key = None
        for cached_key, entry in self._store.items():
            if entry["expires_at"] <= now:
                continue
            score = SequenceMatcher(None, key, cached_key).ratio()
            if score > best_score:
                best_score = score
                best_payload = entry["payload"]
                best_key = cached_key

        if best_payload is not None and best_score >= self.semantic_threshold:
            self.stats.hits += 1
            self.last_event = {
                "hit": True,
                "mode": "semantic",
                "score": round(best_score, 4),
                "matched_key": best_key,
            }
            return True, best_payload

        self.stats.misses += 1
        self.last_event = {"hit": False, "mode": "miss", "score": round(best_score, 4)}
        return False, None

    def store(self, query: str, payload: dict[str, Any]) -> None:
        key = self._normalize(query)
        self._store[key] = {
            "payload": payload,
            "expires_at": time.time() + self.ttl_seconds,
        }
        self.stats.inserts += 1

    def summary(self) -> dict[str, Any]:
        total = self.stats.hits + self.stats.misses
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": (self.stats.hits / total) if total else 0.0,
            "inserts": self.stats.inserts,
            "entries": len(self._store),
        }

