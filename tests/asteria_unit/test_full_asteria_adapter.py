"""Unit tests for the full-Asteria adapter (embedding/judger/Sine layer).

These tests stub AsteriaCache with a minimal dict-backed cache so they don't
need torch / sentence-transformers / faiss at test time.
"""

from __future__ import annotations

from typing import Any

import pytest

from asteria.integrations.assetops import (
    AsteriaIoTToolLayer,
    build_asteria_cached_call_tool,
    compose_stored_answer_from_steps,
    iot_tool_cache_key,
)


class StubAsteriaCache:
    """Behaves like AsteriaCache.lookup/insert but with a plain dict."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.inserts: list[tuple[str, str, float, float]] = []

    def lookup(self, key: str) -> tuple[str | None, dict[str, Any]]:
        if key in self.store:
            return self.store[key], {"hit": True, "source": "sine"}
        return None, {"hit": False, "source": None}

    def insert(self, key: str, value: str, cost: float, latency_ms: float) -> None:
        self.store[key] = value
        self.inserts.append((key, value, cost, latency_ms))


# ── key helper ────────────────────────────────────────────────────────────────


def test_iot_tool_cache_key_shape():
    k = iot_tool_cache_key("IoT", "Assets", {"site_name": "MAIN"})
    assert k == 'iot|assets|{"site_name":"MAIN"}'


def test_iot_tool_cache_key_matches_iot_tool_cache_keys():
    """Full-Asteria keys share the same composite shape as IoTToolCache keys."""
    from asteria.integrations.assetops import IoTToolCache

    c = IoTToolCache()
    args = {"site_name": "MAIN"}
    assert iot_tool_cache_key("iot", "assets", args) == c._make_key("iot", "assets", args)


# ── AsteriaIoTToolLayer lookup / store gating ─────────────────────────────────


def test_layer_bypasses_non_iot_tools():
    layer = AsteriaIoTToolLayer(StubAsteriaCache())
    hit, ans = layer.lookup("utilities", "current_date_time", {})
    assert hit is False and ans is None
    assert layer.last_event["cacheable"] is False
    assert layer.last_event["reason"] == "bypassed"


def test_layer_bypasses_non_whitelisted_iot_tool():
    layer = AsteriaIoTToolLayer(StubAsteriaCache())
    hit, ans = layer.lookup("iot", "unknown_tool", {})
    assert hit is False and ans is None
    assert layer.last_event["cacheable"] is False


def test_layer_lookup_miss_records_miss_event():
    layer = AsteriaIoTToolLayer(StubAsteriaCache())
    hit, ans = layer.lookup("iot", "assets", {"site_name": "MAIN"})
    assert hit is False and ans is None
    assert layer.last_event == {"cacheable": True, "hit": False, "mode": "miss"}


def test_layer_store_then_hit():
    cache = StubAsteriaCache()
    layer = AsteriaIoTToolLayer(cache)
    layer.store("iot", "assets", {"site_name": "MAIN"}, "resp", latency_ms=42.0)
    assert cache.inserts[0][1] == "resp"
    hit, ans = layer.lookup("iot", "assets", {"site_name": "MAIN"})
    assert hit is True
    assert ans == "resp"
    assert layer.last_event["hit"] is True
    assert layer.last_event["mode"] == "sine"


def test_layer_store_ignores_non_cacheable():
    cache = StubAsteriaCache()
    layer = AsteriaIoTToolLayer(cache)
    layer.store("utilities", "current_date_time", {}, "now", latency_ms=1.0)
    assert cache.inserts == []


# ── build_asteria_cached_call_tool wiring ─────────────────────────────────────


@pytest.mark.anyio
async def test_wrapped_call_caches_iot_tool():
    cache = StubAsteriaCache()
    layer = AsteriaIoTToolLayer(cache)

    calls = {"n": 0}

    async def base(path, tool, args):
        calls["n"] += 1
        return f"resp-{calls['n']}"

    wrapped = build_asteria_cached_call_tool(base, layer)

    r1 = await wrapped("/fake/iot-server.py", "assets", {"site_name": "MAIN"})
    r2 = await wrapped("/fake/iot-server.py", "assets", {"site_name": "MAIN"})
    assert r1 == r2 == "resp-1"
    assert calls["n"] == 1
    assert layer.last_event["hit"] is True


@pytest.mark.anyio
async def test_wrapped_call_bypasses_utilities():
    cache = StubAsteriaCache()
    layer = AsteriaIoTToolLayer(cache)

    calls = {"n": 0}

    async def base(path, tool, args):
        calls["n"] += 1
        return f"resp-{calls['n']}"

    wrapped = build_asteria_cached_call_tool(base, layer)
    r1 = await wrapped("/fake/utilities-server.py", "current_date_time", {})
    r2 = await wrapped("/fake/utilities-server.py", "current_date_time", {})
    assert r1 != r2
    assert cache.inserts == []


# ── compose_stored_answer_from_steps ──────────────────────────────────────────


class _Step:
    def __init__(self, step_number: int, task: str) -> None:
        self.step_number = step_number
        self.task = task


class _Result:
    def __init__(self, step_number: int, task: str, response: str, success: bool = True, error: str = "") -> None:
        self.step_number = step_number
        self.task = task
        self.response = response
        self.success = success
        self.error = error


def test_compose_successful_steps():
    ordered = [_Step(1, "Get sites"), _Step(2, "Get assets")]
    context = {
        1: _Result(1, "Get sites", "MAIN"),
        2: _Result(2, "Get assets", "CH-1,CH-2"),
    }
    text = compose_stored_answer_from_steps(ordered, context)
    assert "Step 1 — Get sites: MAIN" in text
    assert "Step 2 — Get assets: CH-1,CH-2" in text


def test_compose_failed_step_formatted_as_error():
    ordered = [_Step(1, "Get sites")]
    context = {1: _Result(1, "Get sites", "", success=False, error="timeout")}
    text = compose_stored_answer_from_steps(ordered, context)
    assert "ERROR: timeout" in text


def test_compose_skips_missing_context():
    ordered = [_Step(1, "Step 1"), _Step(2, "Step 2")]
    context = {1: _Result(1, "Step 1", "ok")}
    text = compose_stored_answer_from_steps(ordered, context)
    assert "Step 1" in text
    assert "Step 2" not in text


# ── build_asteria_cache_stack import-error path ───────────────────────────────


def test_build_asteria_cache_stack_raises_importerror_with_hint(monkeypatch):
    """Simulate missing torch/faiss and assert the helpful ImportError."""
    from asteria.integrations.assetops import full_asteria_adapter as fa

    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *args, **kwargs):
        if name in ("asteria.cache", "asteria.embedding_model", "asteria.semantic_judger"):
            raise ImportError(f"mocked missing {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError) as excinfo:
        fa.build_asteria_cache_stack()
    assert "optional" in str(excinfo.value).lower()


# ── anyio backend fixture ─────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"
