"""Unit tests for the query-intent semantic cache."""

from __future__ import annotations

import time

from asteria.integrations.assetops import QueryIntentCache


def test_miss_returns_false_payload_none():
    q = QueryIntentCache()
    hit, payload = q.lookup("What assets are at site MAIN?")
    assert hit is False
    assert payload is None
    assert q.last_event["mode"] == "miss"


def test_exact_hit_after_store():
    q = QueryIntentCache()
    q.store("What assets are at site MAIN?", {"planned_steps": 2})
    hit, payload = q.lookup("What assets are at site MAIN?")
    assert hit is True
    assert payload == {"planned_steps": 2}
    assert q.last_event["mode"] == "exact"


def test_normalization_ignores_case_and_whitespace():
    q = QueryIntentCache()
    q.store("What assets are at site MAIN?", {"planned_steps": 2})
    hit, payload = q.lookup("  what   assets are   at site main?  ")
    assert hit is True
    assert payload == {"planned_steps": 2}
    assert q.last_event["mode"] == "exact"


def test_semantic_hit_above_threshold():
    q = QueryIntentCache(semantic_threshold=0.90)
    q.store("What assets are at site MAIN?", {"planned_steps": 2})
    # One trailing character difference — semantic match by ratio.
    hit, payload = q.lookup("What assets are at site MAIN???")
    assert hit is True
    assert q.last_event["mode"] == "semantic"
    assert q.last_event["score"] >= 0.90


def test_semantic_miss_below_threshold():
    q = QueryIntentCache(semantic_threshold=0.95)
    q.store("What assets are at site MAIN?", {"planned_steps": 2})
    hit, _ = q.lookup("Show me the daily chart for CH-1 last week")
    assert hit is False
    assert q.last_event["mode"] == "miss"


def test_ttl_expiry():
    q = QueryIntentCache(ttl_seconds=0.0)
    q.store("ephemeral", {"x": 1})
    # Force expiry.
    for entry in q._store.values():
        entry["expires_at"] = time.time() - 1
    hit, payload = q.lookup("ephemeral")
    assert hit is False and payload is None


def test_summary_hit_rate():
    q = QueryIntentCache()
    q.store("a question", {"k": 1})
    q.lookup("a question")       # hit
    q.lookup("different thing")  # miss
    s = q.summary()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1
    assert 0.49 <= s["hit_rate"] <= 0.51
