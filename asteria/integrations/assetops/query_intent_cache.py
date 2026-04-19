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

# Queries that are judgment calls / recommendations — no single correct answer.
_VOLATILE_RE = re.compile(
    r"\b(recommend|suggest|should\s+i|bundle|predict\s+the\s+next|"
    r"what\s+should|prioritize|prioritise|create\s+a\s+work\s+order|"
    r"generate\s+a\s+plan|advise|prescribe|next\s+maintenance\s+window)\b",
    re.IGNORECASE,
)

# Queries with a rolling time window — the answer changes as the window moves.
_RELATIVE_RE = re.compile(
    r"\b(last\s+week|this\s+week|previous\s+week|past\s+week|"
    r"last\s+month|this\s+month|"
    r"last\s+year|this\s+year|"
    r"yesterday|today|current|latest|recent|"
    r"past\s+\d+\s+days?)\b",
    re.IGNORECASE,
)

# Window duration in seconds for each relative phrase.
_WINDOW_S: dict[str, float] = {
    "last week":     7 * 86400,
    "this week":     7 * 86400,
    "previous week": 7 * 86400,
    "past week":     7 * 86400,
    "last month":   30 * 86400,
    "this month":   30 * 86400,
    "last year":   365 * 86400,
    "this year":   365 * 86400,
    "yesterday":         86400,
    "today":             86400,
    "current":            3600,
    "latest":             3600,
    "recent":       24 * 86400,
}

# Queries with a hard absolute date — the underlying data is immutable.
# Matches: ISO dates (2020-06-01), slash dates (6/1/2020), Month+Year, bare 4-digit year.
# Month names alone are excluded ("may" is too ambiguous); a year must accompany them.
_ABSOLUTE_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}(-\d{2})?|"
    r"\d{1,2}/\d{1,2}/\d{4}|"
    r"(january|february|march|april|june|july|august|"
    r"september|october|november|december)\s+\d{4}|"
    r"\b\d{4}\b)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> tuple[str, float | None]:
    """Classify a query into a cache temporal class.

    Returns (temporal_class, window_seconds):
      VOLATILE  — recommendation / judgment call; never cache
      RELATIVE  — rolling time window; window_seconds is the window duration
      ANCHORED  — fixed historical date; safe to cache indefinitely
      STATIC    — no temporal expression; safe to cache indefinitely
    """
    if _VOLATILE_RE.search(query):
        return "VOLATILE", None

    has_relative = _RELATIVE_RE.search(query)
    has_absolute = _ABSOLUTE_DATE_RE.search(query)

    # If both present, the absolute date anchors the relative phrase
    # e.g. "last week of 2020" → the year pins the window → ANCHORED
    if has_relative and has_absolute:
        return "ANCHORED", None

    if has_relative:
        phrase = re.sub(r"\s+", " ", has_relative.group(0).strip().lower())
        past_n = re.match(r"past\s+(\d+)\s+days?", phrase)
        window_s = float(int(past_n.group(1)) * 86400) if past_n else _WINDOW_S.get(phrase, 7 * 86400)
        return "RELATIVE", window_s

    if has_absolute:
        return "ANCHORED", None

    return "STATIC", None


@dataclass
class QueryCacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    volatile_skips: int = 0


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

    def _temporal_valid(self, entry: dict[str, Any], now: float) -> bool:
        """Return False if a RELATIVE entry's window has rolled over."""
        if entry.get("temporal_class") != "RELATIVE":
            return True
        window_s = entry.get("temporal_window_s")
        if window_s is None:
            return True
        return (now - entry["inserted_at"]) <= window_s

    def lookup(self, query: str) -> tuple[bool, str | None]:
        now = time.time()
        key = self._normalize(query)

        # Exact match first.
        exact = self._store.get(key)
        if exact and exact["expires_at"] > now and self._temporal_valid(exact, now):
            self.stats.hits += 1
            self.last_event = {"hit": True, "mode": "exact", "score": 1.0,
                               "temporal_class": exact.get("temporal_class")}
            return True, exact["answer"]

        # Semantic fallback.
        best_score = 0.0
        best_answer: str | None = None
        best_key = None
        for cached_key, entry in self._store.items():
            if entry["expires_at"] <= now:
                continue
            if not self._temporal_valid(entry, now):
                continue
            score = SequenceMatcher(None, key, cached_key).ratio()
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]
                best_key = cached_key

        if best_answer is not None and best_score >= self.semantic_threshold:
            self.stats.hits += 1
            self.last_event = {
                "hit": True,
                "mode": "semantic",
                "score": round(best_score, 4),
                "matched_key": best_key,
            }
            return True, best_answer

        self.stats.misses += 1
        self.last_event = {"hit": False, "mode": "miss", "score": round(best_score, 4)}
        return False, None

    def store(self, query: str, answer: str) -> None:
        tclass, window_s = classify_query(query)

        if tclass == "VOLATILE":
            self.stats.volatile_skips += 1
            self.last_event = {"hit": False, "mode": "volatile_skip"}
            return

        ttl = window_s if (tclass == "RELATIVE" and window_s is not None) else self.ttl_seconds
        now = time.time()
        key = self._normalize(query)
        self._store[key] = {
            "answer": answer,
            "inserted_at": now,
            "expires_at": now + ttl,
            "temporal_class": tclass,
            "temporal_window_s": window_s,
        }
        self.stats.inserts += 1

    def summary(self) -> dict[str, Any]:
        total = self.stats.hits + self.stats.misses
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": (self.stats.hits / total) if total else 0.0,
            "inserts": self.stats.inserts,
            "volatile_skips": self.stats.volatile_skips,
            "entries": len(self._store),
        }
