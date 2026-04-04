"""IoT-focused cache adapter for AssetOpsBench tool calls.

This module intentionally keeps integration lightweight:
- no required edits to `agent.plan_execute.executor`
- cache wrapper can be injected at call-site (e.g. profiler, custom runner)
- IoT tools are cached conservatively to avoid correctness regressions
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable


_CACHEABLE_IOT_TOOLS = {"assets", "sensors", "history"}


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    bypassed: int = 0


class IoTToolCache:
    """Conservative cache for IoT MCP tool responses.

    Exact-match cache is always enabled.
    Optional semantic fallback is enabled only for low-risk tools (`assets`,
    `sensors`) and never for `history`.
    """

    def __init__(
        self,
        enable_semantic: bool = True,
        semantic_threshold: float = 0.94,
        default_ttl_seconds: float = 900.0,
    ) -> None:
        self.enable_semantic = enable_semantic
        self.semantic_threshold = semantic_threshold
        self.default_ttl_seconds = default_ttl_seconds
        self._store: dict[str, dict[str, Any]] = {}
        self.stats = CacheStats()
        self.last_event: dict[str, Any] = {}

    def _normalize_server(self, server: str) -> str:
        s = (server or "").strip().lower()
        if "iot" in s:
            return "iot"
        return s

    def _normalize_tool(self, tool: str) -> str:
        return (tool or "").strip().lower()

    def _normalize_obj(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._normalize_obj(value[k]) for k in sorted(value)}
        if isinstance(value, list):
            if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
                return sorted(value, key=lambda x: str(x))
            return [self._normalize_obj(v) for v in value]
        if isinstance(value, str):
            return value.strip()
        return value

    def _canonical_args(self, args: dict[str, Any]) -> str:
        norm = self._normalize_obj(args or {})
        return json.dumps(norm, sort_keys=True, separators=(",", ":"))

    def _make_key(self, server: str, tool: str, args: dict[str, Any]) -> str:
        return f"{self._normalize_server(server)}|{self._normalize_tool(tool)}|{self._canonical_args(args)}"

    def _expiry_for(self, tool: str, args: dict[str, Any]) -> float:
        t = self._normalize_tool(tool)
        if t in {"assets", "sensors"}:
            return 3600.0
        if t == "history":
            # Conservative heuristic:
            # - if querying historical window fully in the past, allow long TTL
            # - otherwise use short TTL
            start_raw = (args or {}).get("start")
            final_raw = (args or {}).get("final")
            try:
                now = datetime.now(timezone.utc)
                start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                if final_raw:
                    final_dt = datetime.fromisoformat(final_raw.replace("Z", "+00:00"))
                else:
                    final_dt = now
                if final_dt < now:
                    return 24 * 3600.0
            except Exception:  # noqa: BLE001
                pass
            return 300.0
        return self.default_ttl_seconds

    def _cacheable(self, server: str, tool: str) -> bool:
        return (
            self._normalize_server(server) == "iot"
            and self._normalize_tool(tool) in _CACHEABLE_IOT_TOOLS
        )

    def _semantic_match(self, tool: str, args: dict[str, Any], candidate_key: str) -> bool:
        # Limit semantic fallback to low-risk tools.
        if self._normalize_tool(tool) not in {"assets", "sensors"}:
            return False
        probe_key = self._make_key("iot", tool, args)
        score = SequenceMatcher(None, probe_key, candidate_key).ratio()
        return score >= self.semantic_threshold

    def lookup(self, server: str, tool: str, args: dict[str, Any]) -> tuple[bool, str | None]:
        if not self._cacheable(server, tool):
            self.stats.bypassed += 1
            self.last_event = {"cacheable": False, "hit": False, "reason": "bypassed"}
            return False, None

        now = time.time()
        key = self._make_key(server, tool, args)
        entry = self._store.get(key)
        if entry and entry["expires_at"] > now:
            self.stats.hits += 1
            self.last_event = {"cacheable": True, "hit": True, "mode": "exact"}
            return True, entry["response"]

        # Optional semantic probe for assets/sensors.
        if self.enable_semantic and self._normalize_tool(tool) in {"assets", "sensors"}:
            for candidate_key, candidate in self._store.items():
                if candidate["expires_at"] <= now:
                    continue
                if self._semantic_match(tool, args, candidate_key):
                    self.stats.hits += 1
                    self.last_event = {
                        "cacheable": True,
                        "hit": True,
                        "mode": "semantic",
                        "matched_key": candidate_key,
                    }
                    return True, candidate["response"]

        self.stats.misses += 1
        self.last_event = {"cacheable": True, "hit": False, "mode": "miss"}
        return False, None

    def store(self, server: str, tool: str, args: dict[str, Any], response: str) -> None:
        if not self._cacheable(server, tool):
            return
        ttl = self._expiry_for(tool, args)
        key = self._make_key(server, tool, args)
        self._store[key] = {
            "response": response,
            "expires_at": time.time() + ttl,
        }
        self.stats.inserts += 1

    def summary(self) -> dict[str, Any]:
        total = self.stats.hits + self.stats.misses
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": (self.stats.hits / total) if total else 0.0,
            "inserts": self.stats.inserts,
            "bypassed": self.stats.bypassed,
            "entries": len(self._store),
        }


def _infer_server_name(server_path: Path | str) -> str:
    if isinstance(server_path, Path):
        text = str(server_path).lower()
    else:
        text = (server_path or "").lower()
    if "iot" in text:
        return "iot"
    return text


def build_cached_call_tool(
    base_call_tool: Callable[[Path | str, str, dict], Awaitable[str]],
    cache: IoTToolCache,
) -> Callable[[Path | str, str, dict], Awaitable[str]]:
    """Return a call wrapper that applies IoT cache lookup/insert."""

    async def _cached_call_tool(server_path: Path | str, tool_name: str, args: dict) -> str:
        server_name = _infer_server_name(server_path)
        hit, cached = cache.lookup(server_name, tool_name, args or {})
        if hit and cached is not None:
            return cached
        response = await base_call_tool(server_path, tool_name, args or {})
        cache.store(server_name, tool_name, args or {}, response)
        return response

    return _cached_call_tool

