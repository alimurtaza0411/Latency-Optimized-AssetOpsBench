"""Full Asteria stack (embeddings + SemanticJudger + Sine / AsteriaCache) for profiling.

Uses the same composite keys as :class:`IoTToolCache` for IoT tool calls so behaviour
is comparable; query-level keys are the raw user question strings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from asteria.config import DEFAULT_CONFIG

_CACHEABLE_IOT_TOOLS = frozenset({"assets", "sensors", "history"})


def _normalize_server(server: str) -> str:
    s = (server or "").strip().lower()
    if "iot" in s:
        return "iot"
    return s


def _normalize_tool(tool: str) -> str:
    return (tool or "").strip().lower()


def _normalize_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_obj(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            return sorted(value, key=lambda x: str(x))
        return [_normalize_obj(v) for v in value]
    if isinstance(value, str):
        return value.strip()
    return value


def _canonical_args(args: dict[str, Any]) -> str:
    norm = _normalize_obj(args or {})
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))


def iot_tool_cache_key(server: str, tool: str, args: dict[str, Any]) -> str:
    """Stable string key for an IoT tool call (shared shape with IoTToolCache)."""
    return f"{_normalize_server(server)}|{_normalize_tool(tool)}|{_canonical_args(args)}"


def _infer_server_name(server_path: Path | str) -> str:
    if isinstance(server_path, Path):
        text = str(server_path).lower()
    else:
        text = (server_path or "").lower()
    if "iot" in text:
        return "iot"
    return text


def _iot_cacheable(server: str, tool: str) -> bool:
    return _normalize_server(server) == "iot" and _normalize_tool(tool) in _CACHEABLE_IOT_TOOLS


class AsteriaIoTToolLayer:
    """Tracks last tool cache event; wraps AsteriaCache lookup/insert for IoT tools."""

    def __init__(self, cache: Any) -> None:
        self._cache = cache
        self.last_event: dict[str, Any] = {}

    def reset_event(self) -> None:
        self.last_event = {}

    def lookup(self, server: str, tool: str, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not _iot_cacheable(server, tool):
            self.last_event = {"cacheable": False, "hit": False, "reason": "bypassed"}
            return False, None
        key = iot_tool_cache_key(server, tool, args)
        ans, dbg = self._cache.lookup(key)
        if ans is not None:
            self.last_event = {
                "cacheable": True,
                "hit": True,
                "source": dbg.get("source"),
                "mode": str(dbg.get("source", "sine")),
            }
            return True, ans
        self.last_event = {"cacheable": True, "hit": False, "mode": "miss"}
        return False, None

    def store(self, server: str, tool: str, args: dict[str, Any], response: str, latency_ms: float) -> None:
        if not _iot_cacheable(server, tool):
            return
        key = iot_tool_cache_key(server, tool, args)
        self._cache.insert(
            key,
            response,
            cost=DEFAULT_CONFIG.remote_cost_per_call,
            latency_ms=latency_ms,
        )


def build_asteria_cache_stack(
    capacity: int | None = None,
    tau_sim: float | None = None,
    tau_lsm: float | None = None,
    enable_prefetch: bool = True,
) -> Any:
    """Construct :class:`asteria.cache.AsteriaCache` with default paper models.

    Raises ImportError with a short message if optional deps (torch, faiss, etc.) are missing.
    """
    try:
        from asteria.cache import AsteriaCache
        from asteria.embedding_model import EmbeddingModel
        from asteria.semantic_judger import SemanticJudger
    except ImportError as e:
        raise ImportError(
            "Full Asteria requires optional dependencies (e.g. torch, sentence-transformers, "
            "faiss-cpu, transformers). Install them in this environment, then retry."
        ) from e

    cap = capacity if capacity is not None else DEFAULT_CONFIG.cache_capacity
    ts = tau_sim if tau_sim is not None else DEFAULT_CONFIG.tau_sim
    tl = tau_lsm if tau_lsm is not None else DEFAULT_CONFIG.tau_lsm
    emb = EmbeddingModel()
    judger = SemanticJudger()
    return AsteriaCache(
        emb,
        judger,
        capacity=cap,
        tau_sim=ts,
        tau_lsm=tl,
        enable_prefetch=enable_prefetch,
    )


def build_asteria_cached_call_tool(
    base_call_tool: Callable[[Path | str, str, dict], Awaitable[str]],
    layer: AsteriaIoTToolLayer,
) -> Callable[[Path | str, str, dict], Awaitable[str]]:
    """Wrap MCP tool calls with AsteriaCache for IoT assets/sensors/history."""

    async def _cached_call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
        import time

        server_name = _infer_server_name(server_path)
        layer.reset_event()
        hit, cached = layer.lookup(server_name, tool_name, args or {})
        if hit and cached is not None:
            return cached
        t0 = time.perf_counter()
        response = await base_call_tool(server_path, tool_name, args or {})
        tool_ms = (time.perf_counter() - t0) * 1000.0
        layer.store(server_name, tool_name, args or {}, response, latency_ms=tool_ms)
        return response

    return _cached_call_tool


def compose_stored_answer_from_steps(ordered: list, context: dict) -> str:
    """Flatten step results into one string for query-level Asteria insert."""
    lines: list[str] = []
    for step in ordered:
        r = context.get(step.step_number)
        if r is None:
            continue
        if r.success:
            lines.append(f"Step {r.step_number} — {r.task}: {r.response}")
        else:
            lines.append(f"Step {r.step_number} — ERROR: {r.error}")
    return "\n".join(lines)
