"""End-to-end integration tests for timer.ProfiledRunner.

Stubs MCP transport (_list_tools / _call_tool) and the LLM backend so the full
plan -> execute -> summarize loop can be exercised without network, CouchDB, or
model downloads. Verifies baseline execution plus Asteria query-level caching
under the runner.
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
    assert timing.asteria_summary is None
    assert timing.asteria_lookup_s == 0.0
    # Exactly one tool invocation.
    assert call_tool.await_count == 1


# ── Asteria mode with a stubbed cache ─────────────────────────────────────────


class _StubAsteriaCache:
    """Dict-backed stand-in for AsteriaCache — no embeddings required."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.inserts: int = 0

    def lookup(self, key, now=None):  # noqa: ANN001
        if key in self.store:
            return self.store[key], {"hit": True, "source": "sine"}
        return None, {"hit": False, "source": None}

    def insert(self, key, value, cost=0.0, latency_ms=0.0, now=None):  # noqa: ANN001
        self.store[key] = value
        self.inserts += 1
        return {
            "inserted": True,
            "skip_reason": None,
            "temporal_bucket": "STATIC",
            "staticity": 9.0,
            "ttl_hours": 720.0,
        }

    def stats_summary(self) -> dict:
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate_%": 0.0,
            "api_calls": self.inserts,
            "ses_in_cache": len(self.store),
        }


@pytest.mark.anyio
async def test_asteria_query_cache_hits_on_repeat():
    """Inject a stub AsteriaCache to exercise the pre-orchestrator Asteria path."""
    runner = _make_runner()
    stub = _StubAsteriaCache()
    runner._asteria_cache = stub

    list_tools = AsyncMock(return_value=_MOCK_IOT_TOOLS)
    call_tool = AsyncMock(return_value=_TOOL_RESPONSE)
    runner._list_tools = list_tools
    runner._call_tool = call_tool

    _install_llm(
        runner,
        [
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
            _PLAN_ONE_IOT_STEP, '{"site_name": "MAIN"}',
        ],
    )

    t1 = await runner.run("List assets at site MAIN")
    t2 = await runner.run("List assets at site MAIN")

    # Run 1: miss at the query layer, execute normally, then store the result.
    assert t1.asteria_hit is False
    assert len(t1.steps) == 1
    assert t1.asteria_lookup_s > 0.0
    assert t1.asteria_insert_s > 0.0
    # Run 2: query-level Asteria cache returns the stored answer pre-planner.
    assert t2.asteria_hit is True
    assert t2.steps == []
    assert t2.discovery_s == 0.0
    assert t2.planning_s == 0.0
    assert t2.asteria_lookup_s > 0.0
    assert t2.asteria_insert_s == 0.0
    assert call_tool.await_count == 1
    assert t1.asteria_summary is not None
    assert t2.asteria_summary is not None


def test_cli_accepts_asteria_alias_flag():
    parser = timer._build_parser()
    args = parser.parse_args(["--full-asteria", "Q"])
    assert args.asteria is True


def test_cli_accepts_compare_and_sampling_flags():
    parser = timer._build_parser()
    args = parser.parse_args(
        [
            "--compare-asteria",
            "--csv",
            "all_utterance.csv",
            "--sample-count",
            "2",
            "--sample-seed",
            "7",
        ]
    )
    assert args.compare_asteria is True
    assert args.csv == Path("all_utterance.csv")
    assert args.sample_count == 2


def test_describe_temporal_policy_distinguishes_query_shapes():
    static_view = timer._describe_temporal_policy("What assets are at site MAIN?")
    anchored_explicit_view = timer._describe_temporal_policy(
        "Show Chiller 6 data from 2020-06-01 00:00 to 2020-06-01 01:00"
    )
    anchored_relative_view = timer._describe_temporal_policy(
        "Show Chiller 6 data from the last 2 hours"
    )
    volatile_view = timer._describe_temporal_policy(
        "What is the current status of Chiller 6?"
    )

    assert static_view["display_tag"] == "STATIC"
    assert anchored_explicit_view["display_tag"] == "ANCHORED"
    assert anchored_explicit_view["asteria_bucket"] == "ANCHORED"
    # Relative phrases are now resolved to ANCHORED with a concrete window.
    assert anchored_relative_view["display_tag"] == "ANCHORED"
    assert anchored_relative_view["asteria_bucket"] == "ANCHORED"
    assert "time_window" in anchored_relative_view
    assert volatile_view["display_tag"] == "VOLATILE"
    assert volatile_view["asteria_bucket"] == "VOLATILE"


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
