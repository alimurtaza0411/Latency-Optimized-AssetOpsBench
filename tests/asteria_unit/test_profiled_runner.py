"""End-to-end integration tests for timer.ProfiledRunner.

Stubs MCP transport (_list_tools / _call_tool) and the LLM backend so the full
plan → execute → summarize loop can be exercised without network, CouchDB, or
model downloads. Verifies both the lightweight IoT/query caches and the full
Asteria adapter behave correctly under the runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Ensure the repo root (where timer.py lives) is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ensure src/ is on sys.path too (for `agent`, `llm`).
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import timer  # noqa: E402


# ── canned planner + response fixtures ────────────────────────────────────────

_PLAN_ONE_IOT_STEP = (
    "#Task1: List assets at site MAIN\n"
    "#Server1: iot\n"
    "#Tool1: assets\n"
    "#Dependency1: None\n"
    "#ExpectedOutput1: List of asset names\n"
)

_MOCK_IOT_TOOLS = [
    {"name": "assets", "description": "List assets", "parameters": [{"name": "site_name", "type": "string", "required": True}]},
    {"name": "sensors", "description": "List sensors", "parameters": [{"name": "site_name", "type": "string", "required": True}, {"name": "asset", "type": "string", "required": True}]},
    {"name": "history", "description": "Get history", "parameters": []},
]

_TOOL_RESPONSE = json.dumps({"assets": ["Chiller_6", "Chiller_9"]})


class _SequentialLLM:
    """Returns canned responses in order across successive generate() calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return next(self._responses, "{}")


def _make_runner(**kwargs):
    """Build a ProfiledRunner and swap in a deterministic in-memory LLM."""
    runner = timer.ProfiledRunner(
        model_id="stub/model",
        server_paths={"iot": Path("/fake/iot-server.py")},
        summarize=False,
        **kwargs,
    )
    return runner


# ── baseline runner (no cache) ────────────────────────────────────────────────


def _install_llm(runner, responses):
    runner._llm = _SequentialLLM(responses)
    runner._planner._llm = runner._llm
    runner._executor._llm = runner._llm


def _wire_mocks(runner, *, list_tools, call_tool, preserve_wrapper=False):
    """Point runner at async mocks. If preserve_wrapper, wrap the existing
    runner._call_tool (a cache wrapper) by swapping its captured base tool.
    """
    runner._list_tools = list_tools
    if preserve_wrapper:
        # The wrapper closed over the original base _call_tool at __init__.
        # Easiest path: rewrap directly.
        pass
    runner._call_tool = call_tool


@pytest.mark.anyio
async def test_baseline_runner_executes_plan_without_cache():
    runner = _make_runner()
    _install_llm(runner, [_PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}'])

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    _wire_mocks(runner, list_tools=list_tools, call_tool=call_tool)

    timing = await runner.run("List assets at site MAIN")

    assert timing.total_s > 0
    assert len(timing.steps) == 1
    assert timing.steps[0].server == "iot"
    assert timing.steps[0].tool == "assets"
    assert timing.steps[0].success is True
    assert timing.steps[0].cache_hit is None
    assert timing.cache_summary is None
    assert timing.query_cache_summary is None
    assert timing.asteria_summary is None
    # Exactly one tool invocation.
    assert call_tool.await_count == 1


# ── lightweight IoT cache hit on second run ───────────────────────────────────


@pytest.mark.anyio
async def test_iot_cache_hit_on_repeated_run():
    """With --cache, second run must hit the IoT cache and skip the MCP call."""
    from asteria.integrations.assetops import IoTToolCache, build_cached_call_tool

    runner = _make_runner()  # baseline; wire the cache manually
    cache = IoTToolCache()
    runner._cache = cache

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    runner._list_tools = list_tools
    runner._call_tool = build_cached_call_tool(call_tool, cache)

    _install_llm(
        runner,
        [
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
        ],
    )

    t1 = await runner.run("List assets at site MAIN")
    t2 = await runner.run("List assets at site MAIN")

    assert t1.steps[0].cache_hit is False
    assert t2.steps[0].cache_hit is True
    assert call_tool.await_count == 1
    assert t2.cache_summary["hits"] == 1


# ── query-intent cache short-circuit ──────────────────────────────────────────


@pytest.mark.anyio
async def test_query_cache_short_circuits_second_run():
    runner = _make_runner(query_cache_enabled=True)

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    _wire_mocks(runner, list_tools=list_tools, call_tool=call_tool)

    _install_llm(
        runner,
        [
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
            # Second run hits the query cache — planner + arg LLM NOT called.
        ],
    )

    t1 = await runner.run("List assets at site MAIN")
    t2 = await runner.run("List assets at site MAIN")

    assert t1.query_cache_hit is False
    assert len(t1.steps) == 1
    assert t2.query_cache_hit is True
    assert t2.discovery_s == 0.0
    assert t2.planning_s == 0.0
    assert t2.steps == []
    assert call_tool.await_count == 1


# ── full_asteria mode with a stubbed cache ────────────────────────────────────


class _StubAsteriaCache:
    """Dict-backed stand-in for AsteriaCache — no embeddings required."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.inserts: int = 0

    def lookup(self, key):  # noqa: ANN001
        if key in self.store:
            return self.store[key], {"hit": True, "source": "sine"}
        return None, {"hit": False, "source": None}

    def insert(self, key, value, cost=0.0, latency_ms=0.0):  # noqa: ANN001
        self.store[key] = value
        self.inserts += 1

    def stats_summary(self) -> dict:
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate_%": 0.0,
            "api_calls": self.inserts,
            "ses_in_cache": len(self.store),
        }


@pytest.mark.anyio
async def test_full_asteria_iot_layer_hits_on_repeat():
    """Inject a stub AsteriaCache to exercise the full-asteria branch of timer."""
    from asteria.integrations.assetops import (
        AsteriaIoTToolLayer,
        build_asteria_cached_call_tool,
    )

    # Bypass the heavy build_asteria_cache_stack() by constructing the runner
    # without full_asteria then wiring the Asteria layer manually.
    runner = _make_runner()
    stub = _StubAsteriaCache()
    runner._asteria_cache = stub
    runner._asteria_tool_layer = AsteriaIoTToolLayer(stub)

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    runner._list_tools = list_tools
    runner._call_tool = build_asteria_cached_call_tool(call_tool, runner._asteria_tool_layer)

    _install_llm(
        runner,
        [
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
        ],
    )

    t1 = await runner.run("List assets at site MAIN")
    t2 = await runner.run("List assets at site MAIN")

    # Run 1: miss at every layer, MCP called once, result stored both at the
    # IoT tool layer and at the query-level Asteria cache.
    assert t1.steps[0].cache_hit is False
    assert t1.asteria_query_hit is False
    # Run 2: query-level Asteria cache returns the stored answer pre-planner,
    # so the run returns immediately with no steps executed.
    assert t2.asteria_query_hit is True
    assert t2.steps == []
    assert t2.discovery_s == 0.0
    assert t2.planning_s == 0.0
    assert call_tool.await_count == 1
    assert t1.asteria_summary is not None
    assert t2.asteria_summary is not None


@pytest.mark.anyio
async def test_full_asteria_iot_layer_hits_when_query_cache_disabled():
    """With query-level cache disabled, second run hits the IoT tool cache."""
    from asteria.integrations.assetops import (
        AsteriaIoTToolLayer,
        build_asteria_cached_call_tool,
    )

    runner = _make_runner()

    # Stub cache that only hits on IoT tool keys (not on the question string).
    class _IoTOnlyStub(_StubAsteriaCache):
        def __init__(self) -> None:
            super().__init__()

        def lookup(self, key):  # noqa: ANN001
            # Only hit on the composite IoT key, not on free-text questions.
            if isinstance(key, str) and key.startswith("iot|"):
                return super().lookup(key)
            return None, {"hit": False, "source": None}

        def insert(self, key, value, cost=0.0, latency_ms=0.0):  # noqa: ANN001
            if isinstance(key, str) and key.startswith("iot|"):
                super().insert(key, value, cost, latency_ms)
            # else: drop it — question-level entries are not stored in this variant.

    stub = _IoTOnlyStub()
    runner._asteria_cache = stub
    runner._asteria_tool_layer = AsteriaIoTToolLayer(stub)

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    runner._list_tools = list_tools
    runner._call_tool = build_asteria_cached_call_tool(call_tool, runner._asteria_tool_layer)

    _install_llm(
        runner,
        [
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
        ],
    )

    t1 = await runner.run("List assets at site MAIN")
    t2 = await runner.run("List assets at site MAIN")

    assert t1.steps[0].cache_hit is False
    assert t2.steps[0].cache_hit is True
    assert t2.asteria_query_hit is False
    assert call_tool.await_count == 1


# ── mutual-exclusion CLI guard ────────────────────────────────────────────────


def test_cli_rejects_full_asteria_with_lightweight_flags(capsys):
    """timer.main raises SystemExit when --full-asteria + --cache are combined."""
    import asyncio

    parser = timer._build_parser()
    args = parser.parse_args(["--full-asteria", "--cache", "Q"])
    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(timer._main(args))
    assert "full-asteria" in str(excinfo.value).lower()


# ── print_run + print_summary smoke tests ─────────────────────────────────────


def test_print_run_renders_without_cache(capsys):
    t = timer.RunTiming(question="Q")
    t.total_s = 1.0
    t.discovery_s = 0.2
    t.planning_s = 0.3
    t.summarization_s = 0.5
    timer.print_run(t, run_index=1)
    out = capsys.readouterr().out
    assert "Run 1" in out
    assert "TOTAL" in out


def test_print_summary_skips_for_single_run(capsys):
    t = timer.RunTiming(question="Q")
    timer.print_summary([t])
    assert capsys.readouterr().out == ""


def test_print_summary_renders_for_multi_run(capsys):
    t1 = timer.RunTiming(question="Q")
    t1.total_s = 1.0
    t2 = timer.RunTiming(question="Q")
    t2.total_s = 2.0
    timer.print_summary([t1, t2])
    out = capsys.readouterr().out
    assert "Summary across 2 runs" in out
    assert "TOTAL" in out


# ── anyio backend fixture ─────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"
