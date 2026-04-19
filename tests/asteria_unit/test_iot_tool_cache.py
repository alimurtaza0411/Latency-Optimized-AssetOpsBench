"""Unit tests for the lightweight IoTToolCache and its call-wrapper."""

from __future__ import annotations

import asyncio
import time

import pytest

from asteria.integrations.assetops import IoTToolCache, build_cached_call_tool


# ── key + canonicalization ────────────────────────────────────────────────────


def test_canonical_args_orders_keys_and_lists():
    c = IoTToolCache()
    a = c._make_key("IoT", "assets", {"site_name": "MAIN", "sensors": ["b", "a"]})
    b = c._make_key("iot", "ASSETS", {"sensors": ["a", "b"], "site_name": "MAIN"})
    assert a == b


def test_canonical_args_trims_whitespace_in_strings():
    c = IoTToolCache()
    a = c._make_key("iot", "assets", {"site_name": " MAIN "})
    b = c._make_key("iot", "assets", {"site_name": "MAIN"})
    assert a == b


# ── cacheability gating ───────────────────────────────────────────────────────


def test_non_iot_server_bypassed():
    c = IoTToolCache()
    hit, val = c.lookup("utilities", "current_date_time", {})
    assert hit is False and val is None
    assert c.last_event["cacheable"] is False
    assert c.stats.bypassed == 1


def test_iot_tool_outside_whitelist_bypassed():
    c = IoTToolCache()
    hit, val = c.lookup("iot", "write_anything", {})
    assert hit is False and val is None
    assert c.last_event["cacheable"] is False


def test_store_ignores_non_cacheable():
    c = IoTToolCache()
    c.store("utilities", "current_date_time", {}, "now")
    assert c.summary()["entries"] == 0
    assert c.stats.inserts == 0


# ── exact + TTL ───────────────────────────────────────────────────────────────


def test_exact_hit_after_store():
    c = IoTToolCache()
    args = {"site_name": "MAIN"}
    c.store("iot", "assets", args, "response-1")
    hit, val = c.lookup("iot", "assets", args)
    assert hit is True and val == "response-1"
    assert c.last_event["mode"] == "exact"
    assert c.summary()["hits"] == 1
    assert c.summary()["misses"] == 0


def test_exact_miss_when_expired(monkeypatch):
    c = IoTToolCache(default_ttl_seconds=0.0)
    args = {"site_name": "MAIN"}
    c.store("iot", "assets", args, "response-1")
    # assets TTL is 3600 — monkey-patch the expiry to force immediate expiry.
    for entry in c._store.values():
        entry["expires_at"] = time.time() - 1
    hit, val = c.lookup("iot", "assets", args)
    assert hit is False and val is None


def test_history_recent_window_uses_long_ttl():
    """Historical windows fully in the past get a 24h TTL."""
    c = IoTToolCache()
    args = {
        "site_name": "MAIN",
        "asset": "CH-1",
        "start": "2020-04-27T00:00:00Z",
        "final": "2020-05-04T00:00:00Z",
    }
    c.store("iot", "history", args, "rows")
    # Read expires_at; should be roughly now + 24h
    key = c._make_key("iot", "history", args)
    remaining = c._store[key]["expires_at"] - time.time()
    assert 23 * 3600 < remaining <= 24 * 3600 + 1


def test_history_open_window_uses_short_ttl():
    c = IoTToolCache()
    args = {"site_name": "MAIN", "asset": "CH-1", "start": "2020-04-27T00:00:00Z"}
    c.store("iot", "history", args, "rows")
    key = c._make_key("iot", "history", args)
    remaining = c._store[key]["expires_at"] - time.time()
    assert remaining <= 300.0 + 1


# ── semantic fallback ─────────────────────────────────────────────────────────


def test_semantic_match_for_sensors_paraphrase():
    c = IoTToolCache(enable_semantic=True, semantic_threshold=0.85)
    c.store("iot", "sensors", {"site_name": "MAIN", "asset": "Chiller_6"}, "sensor-list")
    # Slightly different key (whitespace/case differences) should still match.
    hit, val = c.lookup("iot", "sensors", {"site_name": " MAIN", "asset": "Chiller_6 "})
    assert hit is True
    assert val == "sensor-list"


def test_semantic_disabled_only_exact():
    c = IoTToolCache(enable_semantic=False)
    c.store("iot", "sensors", {"site_name": "MAIN", "asset": "Chiller_6"}, "sensor-list")
    hit, val = c.lookup("iot", "sensors", {"site_name": "OTHER", "asset": "Chiller_6"})
    assert hit is False


def test_semantic_never_applies_to_history():
    """history must not use semantic fallback — only exact match."""
    c = IoTToolCache(enable_semantic=True, semantic_threshold=0.80)
    a = {"site_name": "MAIN", "asset": "CH-1", "start": "2020-04-27T00:00:00Z"}
    b = {"site_name": "MAIN", "asset": "CH-2", "start": "2020-04-27T00:00:00Z"}
    c.store("iot", "history", a, "rows-a")
    hit, val = c.lookup("iot", "history", b)
    assert hit is False


# ── summary + stats ───────────────────────────────────────────────────────────


def test_summary_tracks_hits_misses_and_entries():
    c = IoTToolCache()
    c.store("iot", "assets", {"site_name": "MAIN"}, "one")
    c.lookup("iot", "assets", {"site_name": "MAIN"})   # hit
    c.lookup("iot", "assets", {"site_name": "NEW"})    # miss
    s = c.summary()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1
    assert s["inserts"] == 1
    assert 0.49 <= s["hit_rate"] <= 0.51


# ── build_cached_call_tool wiring ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_build_cached_call_tool_round_trip():
    """Second call to same IoT tool/args must reuse cached response."""
    c = IoTToolCache()
    calls = {"n": 0}

    async def base(path, tool, args):
        calls["n"] += 1
        return f"resp-{calls['n']}"

    wrapped = build_cached_call_tool(base, c)

    r1 = await wrapped("/fake/iot-server.py", "assets", {"site_name": "MAIN"})
    r2 = await wrapped("/fake/iot-server.py", "assets", {"site_name": "MAIN"})
    assert r1 == r2 == "resp-1"
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_build_cached_call_tool_non_iot_bypasses_cache():
    c = IoTToolCache()
    calls = {"n": 0}

    async def base(path, tool, args):
        calls["n"] += 1
        return f"resp-{calls['n']}"

    wrapped = build_cached_call_tool(base, c)
    r1 = await wrapped("/fake/utilities-server.py", "current_date_time", {})
    r2 = await wrapped("/fake/utilities-server.py", "current_date_time", {})
    assert r1 != r2
    assert calls["n"] == 2


@pytest.mark.anyio
async def test_infer_server_name_handles_path_object(tmp_path):
    from pathlib import Path

    c = IoTToolCache()

    async def base(path, tool, args):
        return "ok"

    wrapped = build_cached_call_tool(base, c)
    fake_path = Path(tmp_path / "iot-mcp-server.py")
    await wrapped(fake_path, "assets", {"site_name": "MAIN"})
    hit, val = c.lookup("iot", "assets", {"site_name": "MAIN"})
    assert hit is True and val == "ok"


# ── anyio backend fixture (mirrors src/agent/tests/conftest.py pattern) ───────


@pytest.fixture
def anyio_backend():
    return "asyncio"
