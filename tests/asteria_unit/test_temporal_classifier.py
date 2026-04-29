"""Unit tests for the temporal bucketing system.

Covers:
    - STATIC / ANCHORED / VOLATILE classification correctness
    - Relative phrases resolved to ANCHORED windows against an injected `now`
    - Time-window extraction accuracy
    - VOLATILE bypass behaviour (no cache lookup, no cache insert)
    - Unified judger temporal instruction building
    - Edge cases (ambiguous queries, no dates, relative dates)
"""

from __future__ import annotations

import datetime
import time

import pytest

from asteria.temporal_classifier import (
    TemporalBucket,
    TemporalTag,
    TimeWindow,
    classify,
    passes_temporal_gate,
)


# ── STATIC classification ─────────────────────────────────────────────────────


class TestStaticClassification:
    """Queries with no temporal markers should classify as STATIC."""

    @pytest.mark.parametrize(
        "query",
        [
            "What assets are available at site MAIN?",
            "List all assets at the MAIN site.",
            "What sensors does Chiller 6 have in MAIN?",
            "What vibration analysis capabilities are available?",
            "What bearings are available in the built-in database?",
            "How does the ISO 10816 vibration severity classification work?",
            "Calculate the bearing characteristic frequencies for a 6205 bearing running at 1800 RPM.",
            "Show sensor names for Chiller 6 in site MAIN.",
            "Which assets exist in site MAIN?",
        ],
    )
    def test_static_queries(self, query: str):
        tag = classify(query)
        assert tag.bucket == TemporalBucket.STATIC
        assert tag.time_window is None


# ── ANCHORED classification ───────────────────────────────────────────────────


class TestAnchoredClassification:
    """Queries with explicit date anchors or historical context should classify as ANCHORED."""

    @pytest.mark.parametrize(
        "query",
        [
            # ISO dates
            "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN.",
            "Show Chiller 6 observations at MAIN between 2020-06-01T01:00:00 and 2020-06-01T02:00:00.",
            "Fetch historical readings for Chiller 6 in MAIN from 2020-06-01T00:00:00 to 2020-06-01T01:00:00.",
            "Return time-series data for Chiller 6 at MAIN for 2020-06-01 00:00 to 01:00.",
            "Fetch vibration sensor data from Motor_01, sensor Vibration_X, from 2024-01-15 to 2024-01-15T01:00:00 at site PLANT_A.",
            # Natural dates
            "For Chiller 6 at MAIN, give readings from June 1, 2020 00:00 to 01:00.",
            "What was the pressure on January 15, 2024?",
            "Show data for Motor_01 on March 3rd, 2023.",
            "Readings from 1st June 2020 to 3rd June 2020.",
            # Slash / dot dates
            "Get data for Chiller 6 from 2020/06/01 to 2020/06/02.",
            "Show readings between 06/01/2020 and 06/02/2020.",
            "Data for motor from 2020.06.01 to 2020.06.02.",
            # Ordinal dates
            "Show readings from the 1st of June 2020.",
            # Year anchor — relative phrase gets overridden by the year
            "Get last week of 2020 data for Chiller 6.",
            "Show data from 2023.",
            # Historical context keywords (no explicit dates)
            "Show me the history of Chiller 6 at MAIN.",
            "Give me the time-series data for Motor_01.",
            "Display the event log for Chiller 6.",
            "Pull up the maintenance log for Motor_01.",
            "Show the trend data for this sensor.",
            "I need the archived readings for this asset.",
            "What does the audit trail show?",
            "Generate a shift report for operators.",
            "Show the incident report for Motor_01.",
            "Get the daily report.",
            "Pull the weekly report for this site.",
        ],
    )
    def test_anchored_queries(self, query: str):
        tag = classify(query)
        assert tag.bucket == TemporalBucket.ANCHORED

    def test_extracts_time_window_from_iso_dates(self):
        query = "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.ANCHORED
        assert tag.time_window is not None
        assert "2020-06-01" in tag.time_window.start
        assert "2020-06-01" in tag.time_window.end

    def test_two_different_windows_produce_different_tags(self):
        q1 = "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN."
        q2 = "Get history for Chiller 6 from 2020-06-01T01:00:00 to 2020-06-01T02:00:00 at MAIN."
        t1 = classify(q1)
        t2 = classify(q2)
        assert t1.bucket == TemporalBucket.ANCHORED
        assert t2.bucket == TemporalBucket.ANCHORED
        assert t1.time_window is not None
        assert t2.time_window is not None
        assert not t1.time_window.matches(t2.time_window)

    def test_natural_date_classified_as_anchored(self):
        query = "For Chiller 6 at MAIN, give readings from June 1, 2020 00:00 to 01:00."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.ANCHORED

    def test_single_date_without_range_is_anchored_no_window(self):
        query = "Show data for Chiller 6 on 2020-06-01 at MAIN."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.ANCHORED
        assert tag.time_window is None


# ── RELATIVE → ANCHORED resolution ────────────────────────────────────────────


class TestRelativeResolvedToAnchored:
    """Relative-time queries are resolved against `now` and bucketed as ANCHORED.

    Each query containing a relative phrase (e.g. "yesterday", "last week")
    must classify as ANCHORED at lookup time.  When `now` is supplied,
    `time_window` should be populated for the patterns the resolver covers.
    """

    _NOW = datetime.datetime(2026, 4, 27, 14, 30, 0)

    @pytest.mark.parametrize(
        "query",
        [
            # "last/past N units"
            "Show Chiller 6 data from the last 30 minutes.",
            "Get readings from the past 2 hours for Chiller 6.",
            "What changed in the last 10 minutes?",
            "What happened during the last 48 hours?",
            "For the past 12 hours show vibration.",
            "Show data within the last 24 hours.",
            "Get the previous 3 days of readings.",
            # Named relative periods
            "What happened yesterday with Motor_01?",
            "Show data from last week for Chiller 6.",
            "Show data from last year.",
            "Data from last night.",
            "What happened during the previous shift?",
            "Show data from last Friday.",
            # "this" period
            "Show data from this morning.",
            "Readings since this afternoon.",
            "Data from this week.",
            "Show this month's readings.",
            # "today"
            "What are today's readings for Motor_01?",
            "Pull data from earlier today.",
            # "since" expressions
            "Show readings since midnight.",
            "Get data since yesterday.",
            "Vibration data since last restart.",
            # "ago" / recent
            "Show me what happened 5 minutes ago.",
            "Show recent vibration data for Motor_01.",
            "Data from not long ago.",
            "A few minutes ago the sensor spiked - show me.",
            # "over/in the past"
            "What's the trend over the past few hours?",
        ],
    )
    def test_relative_queries_resolve_to_anchored(self, query: str):
        tag = classify(query, now=self._NOW)
        assert tag.bucket == TemporalBucket.ANCHORED

    def test_yesterday_resolves_to_yesterday_window(self):
        tag = classify("What happened yesterday with Motor_01?", now=self._NOW)
        assert tag.bucket == TemporalBucket.ANCHORED
        assert tag.time_window is not None
        assert tag.time_window.start == "2026-04-26T00:00:00"
        assert tag.time_window.end == "2026-04-26T23:59:59"

    def test_last_n_hours_resolves_to_rolling_window(self):
        tag = classify("Get readings from the last 3 hours.", now=self._NOW)
        assert tag.bucket == TemporalBucket.ANCHORED
        assert tag.time_window is not None
        assert tag.time_window.start == "2026-04-27T11:30:00"
        assert tag.time_window.end == "2026-04-27T14:30:00"

    def test_today_resolves_to_today_so_far(self):
        tag = classify("What are today's readings for Motor_01?", now=self._NOW)
        assert tag.bucket == TemporalBucket.ANCHORED
        assert tag.time_window is not None
        assert tag.time_window.start == "2026-04-27T00:00:00"
        assert tag.time_window.end == "2026-04-27T14:30:00"

    def test_year_anchor_overrides_relative_to_anchored(self):
        """A bare year in the query makes it ANCHORED via Rule 2 (explicit
        date) before Rule 3 (relative-resolved)."""
        tag = classify("Get last week of 2020 data for Chiller 6.", now=self._NOW)
        assert tag.bucket == TemporalBucket.ANCHORED


# ── VOLATILE classification ───────────────────────────────────────────────────


class TestVolatileClassification:
    """Queries with live/current keywords should classify as VOLATILE."""

    @pytest.mark.parametrize(
        "query",
        [
            # Explicit live-state keywords
            "What is the current vibration level for Motor_01?",
            "Show me the latest sensor reading for Chiller 6.",
            "Get the live status of Motor_01 at PLANT_A.",
            "What is the real-time temperature of Chiller 6?",
            "Show Chiller 6 data right now.",
            "What is Motor_01 doing at this moment?",
            "Show the most recent reading for this sensor.",
            "Is the system currently active?",
            "What is the present vibration level?",
            "Give me the up-to-date data.",
            "Show the current state of all assets.",
            "What is the current RPM of Motor_01?",
            "Current flow rate for Chiller 6.",
            "What is the current load on the compressor?",
            "Show current power output.",
            # Urgency / immediacy
            "I need the data immediately.",
            "Check the sensor ASAP.",
            "Get readings right away.",
            "Quick check on Motor_01 temperature.",
            "This is urgent - what is the vibration level?",
            # Streaming / monitoring
            "What is the live telemetry feed for Motor_01?",
            "Are there any active alerts for Chiller 6?",
            "Check the dashboard for Motor_01.",
            "Are any alarms active on Chiller 6?",
            "What is the operating status of Motor_01?",
            "Show me the monitoring data.",
            "Track vibration levels for Motor_01.",
            "Is there a threshold breach on Chiller 6?",
            "What is the running status of the pump?",
            # Status polling / health checks
            "Is Motor_01 running?",
            "Health check on Chiller 6.",
            "What's happening with Motor_01?",
            "What's going on with the compressor?",
            "Is the pump up or down?",
            # Implicit "now" (IoT)
            "What is the temperature of Chiller 6?",
            "What is the vibration on Motor_01?",
            "How hot is the motor?",
            "How much vibration is there?",
            "Show me the readings for Motor_01.",
            "Give me the data values for Chiller 6.",
        ],
    )
    def test_volatile_queries(self, query: str):
        tag = classify(query)
        assert tag.bucket == TemporalBucket.VOLATILE
        assert tag.time_window is None

    def test_volatile_keyword_overrides_dates(self):
        """'current' with a date still → VOLATILE (live intent wins)."""
        query = "Show the current status of Chiller 6 as of 2020-06-01."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.VOLATILE


# ── TimeWindow matching ──────────────────────────────────────────────────────


class TestTimeWindowMatching:

    def test_identical_windows_match(self):
        w1 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        w2 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        assert w1.matches(w2)

    def test_different_windows_do_not_match(self):
        w1 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        w2 = TimeWindow(start="2020-06-01T01:00:00", end="2020-06-01T02:00:00")
        assert not w1.matches(w2)

    def test_normalisation_space_vs_T(self):
        w1 = TimeWindow(start="2020-06-01 00:00:00", end="2020-06-01 01:00:00")
        w2 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        assert w1.matches(w2)

    def test_normalisation_missing_seconds(self):
        w1 = TimeWindow(start="2020-06-01T00:00", end="2020-06-01T01:00")
        w2 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        assert w1.matches(w2)


# ── Temporal Gate ─────────────────────────────────────────────────────────────


class TestTemporalGate:

    def test_static_always_passes(self):
        tag = TemporalTag(bucket=TemporalBucket.STATIC)
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="STATIC",
            cached_window_start=None,
            cached_window_end=None,
            cached_created_at=time.time() - 99999,
        )

    def test_resolved_relative_passes_when_window_matches(self):
        """Relative phrases are now resolved to ANCHORED windows; the gate
        accepts the hit only when those windows match the cached entry."""
        tag = TemporalTag(
            bucket=TemporalBucket.ANCHORED,
            time_window=TimeWindow("2026-04-26T00:00:00", "2026-04-26T23:59:59"),
        )
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="ANCHORED",
            cached_window_start="2026-04-26T00:00:00",
            cached_window_end="2026-04-26T23:59:59",
            cached_created_at=time.time(),
        )

    def test_anchored_passes_with_matching_window(self):
        tag = TemporalTag(
            bucket=TemporalBucket.ANCHORED,
            time_window=TimeWindow("2020-06-01T00:00:00", "2020-06-01T01:00:00"),
        )
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="ANCHORED",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=time.time() - 99999,
        )

    def test_anchored_rejects_mismatched_window(self):
        tag = TemporalTag(
            bucket=TemporalBucket.ANCHORED,
            time_window=TimeWindow("2020-06-01T01:00:00", "2020-06-01T02:00:00"),
        )
        assert not passes_temporal_gate(
            query_tag=tag,
            cached_bucket="ANCHORED",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=time.time(),
        )

    def test_anchored_rejects_wrong_cached_bucket(self):
        tag = TemporalTag(
            bucket=TemporalBucket.ANCHORED,
            time_window=TimeWindow("2020-06-01T00:00:00", "2020-06-01T01:00:00"),
        )
        assert not passes_temporal_gate(
            query_tag=tag,
            cached_bucket="STATIC",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=time.time(),
        )

    def test_volatile_never_passes(self):
        tag = TemporalTag(bucket=TemporalBucket.VOLATILE)
        assert not passes_temporal_gate(
            query_tag=tag,
            cached_bucket="VOLATILE",
            cached_window_start=None,
            cached_window_end=None,
            cached_created_at=time.time(),
        )


# ── Judger temporal instruction builder ──────────────────────────────────────


class TestTemporalInstructionBuilder:

    @pytest.fixture(autouse=True)
    def _import_builder(self):
        from asteria.semantic_judger import SemanticJudger, TemporalContext
        self.build = SemanticJudger._build_temporal_instruction
        self.TemporalContext = TemporalContext

    def test_none_context_returns_base_instruction(self):
        result = self.build(None)
        assert "sufficiently answer" in result
        assert "time window" not in result.lower()

    def test_static_instruction_is_neutral(self):
        ctx = self.TemporalContext(query_bucket="STATIC")
        result = self.build(ctx)
        base = self.build(None)
        assert result == base

    def test_relative_instruction_is_neutral(self):
        ctx = self.TemporalContext(query_bucket="RELATIVE")
        result = self.build(ctx)
        base = self.build(None)
        assert result == base

    def test_anchored_instruction_contains_time_windows(self):
        ctx = self.TemporalContext(
            query_bucket="ANCHORED",
            query_window_start="2020-06-01T00:00:00",
            query_window_end="2020-06-01T01:00:00",
            cached_bucket="ANCHORED",
            cached_window_start="2020-06-01T01:00:00",
            cached_window_end="2020-06-01T02:00:00",
            cached_created_at=1717200000.0,
        )
        result = self.build(ctx)
        assert "2020-06-01T00:00:00" in result
        assert "2020-06-01T01:00:00" in result
        assert "time window" in result.lower()
        assert "answer 'yes' only if" in result.lower() or "answer 'no'" in result.lower()

    def test_anchored_instruction_includes_cached_timestamp(self):
        ctx = self.TemporalContext(
            query_bucket="ANCHORED",
            query_window_start="2020-06-01T00:00:00",
            query_window_end="2020-06-01T01:00:00",
            cached_bucket="ANCHORED",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=1717200000.0,
        )
        result = self.build(ctx)
        assert "stored at" in result.lower()

    def test_anchored_instruction_without_windows_still_mentions_time_bounded(self):
        ctx = self.TemporalContext(query_bucket="ANCHORED")
        result = self.build(ctx)
        assert "time-bounded" in result.lower() or "time window" in result.lower()


# ── VOLATILE bypass integration ───────────────────────────────────────────────


class TestVolatileBypassBehaviour:

    def test_volatile_classification_triggers_bypass_signal(self):
        tag = classify("What is the current vibration level for Motor_01?")
        assert tag.bucket == TemporalBucket.VOLATILE

    def test_volatile_queries_produce_no_time_window(self):
        tag = classify("Show me the latest sensor reading for Chiller 6.")
        assert tag.time_window is None

    def test_static_and_anchored_do_not_trigger_bypass(self):
        t_static = classify("What assets are available at site MAIN?")
        assert t_static.bucket != TemporalBucket.VOLATILE

        t_anchored = classify("Get history from 2020-06-01T00:00:00 to 2020-06-01T01:00:00.")
        assert t_anchored.bucket != TemporalBucket.VOLATILE

    def test_resolved_relative_does_not_trigger_bypass(self):
        tag = classify(
            "Show data from the last 30 minutes.",
            now=datetime.datetime(2026, 4, 27, 14, 30, 0),
        )
        assert tag.bucket != TemporalBucket.VOLATILE
        assert tag.bucket == TemporalBucket.ANCHORED


# ── Integration: cache_stress_test scenarios ─────────────────────────────────


class TestCacheStressTestScenarios:

    @pytest.mark.parametrize(
        "query,expected_bucket",
        [
            ("What assets are available at site MAIN?", TemporalBucket.STATIC),
            ("List all assets at the MAIN site.", TemporalBucket.STATIC),
            ("List sensors for Chiller 6 at site MAIN.", TemporalBucket.STATIC),
            ("What sensors does Chiller 6 have in MAIN?", TemporalBucket.STATIC),
            ("Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN.",
             TemporalBucket.ANCHORED),
            ("Get history for Chiller 6 from 2020-06-01T01:00:00 to 2020-06-01T02:00:00 at MAIN.",
             TemporalBucket.ANCHORED),
            ("Get history for Chiller 6 from 2020-06-01T02:00:00 to 2020-06-01T03:00:00 at MAIN.",
             TemporalBucket.ANCHORED),
        ],
    )
    def test_stress_test_scenarios(self, query: str, expected_bucket: TemporalBucket):
        tag = classify(query)
        assert tag.bucket == expected_bucket
