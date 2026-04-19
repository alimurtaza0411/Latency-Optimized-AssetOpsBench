"""Unit tests for the temporal bucketing system.

Covers:
    - T1/T2/T3 classification correctness
    - Time-window extraction accuracy
    - T3 bypass behaviour (no cache lookup, no cache insert)
    - Unified judger temporal instruction building
    - Edge cases (ambiguous queries, no dates, relative dates)
"""

from __future__ import annotations

import time

import pytest

from asteria.temporal_classifier import (
    TemporalBucket,
    TemporalTag,
    TimeWindow,
    classify,
    passes_temporal_gate,
)


# ── T1 (Static / Metadata) classification ────────────────────────────────────


class TestStaticClassification:
    """Queries with no temporal markers should classify as T1."""

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


# ── T2 (Historical / Bounded-Window) classification ─────────────────────────


class TestHistoricalClassification:
    """Queries with explicit date ranges should classify as T2."""

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
            # Slash dates
            "Get data for Chiller 6 from 2020/06/01 to 2020/06/02.",
            "Show readings between 06/01/2020 and 06/02/2020.",
            # Dot dates
            "Data for motor from 2020.06.01 to 2020.06.02.",
            # Ordinal dates
            "Show readings from the 1st of June 2020.",
            # Historical context keywords (no explicit dates)
            "Show me the history of Chiller 6 at MAIN.",
            "Give me the time-series data for Motor_01.",
            "Display the event log for Chiller 6.",
            "Pull up the maintenance log for Motor_01.",
            "Show the trend data for this sensor.",
            "I need the archived readings for this asset.",
            "What does the audit trail show?",
            "Replay the downtime report for last Friday.",
            "Generate a shift report for operators.",
            "Show the incident report for Motor_01.",
            "Get the daily report.",
            "Pull the weekly report for this site.",
        ],
    )
    def test_historical_queries(self, query: str):
        tag = classify(query)
        assert tag.bucket == TemporalBucket.HISTORICAL

    def test_extracts_time_window_from_iso_dates(self):
        query = "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.HISTORICAL
        assert tag.time_window is not None
        assert "2020-06-01" in tag.time_window.start
        assert "2020-06-01" in tag.time_window.end

    def test_two_different_windows_produce_different_tags(self):
        q1 = "Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN."
        q2 = "Get history for Chiller 6 from 2020-06-01T01:00:00 to 2020-06-01T02:00:00 at MAIN."
        t1 = classify(q1)
        t2 = classify(q2)
        assert t1.bucket == TemporalBucket.HISTORICAL
        assert t2.bucket == TemporalBucket.HISTORICAL
        assert t1.time_window is not None
        assert t2.time_window is not None
        # The windows should NOT match each other.
        assert not t1.time_window.matches(t2.time_window)

    def test_natural_date_classified_as_historical(self):
        query = "For Chiller 6 at MAIN, give readings from June 1, 2020 00:00 to 01:00."
        tag = classify(query)
        # Should detect the natural date and classify as T2.
        assert tag.bucket == TemporalBucket.HISTORICAL

    def test_single_date_without_range_is_historical(self):
        query = "Show data for Chiller 6 on 2020-06-01 at MAIN."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.HISTORICAL
        # Single date → no parseable window.
        assert tag.time_window is None


# ── T3 (Live / Real-Time) classification ─────────────────────────────────────


class TestRealtimeClassification:
    """Queries with live/current keywords should classify as T3."""

    @pytest.mark.parametrize(
        "query",
        [
            # Category 1: Explicit live-state keywords
            "What is the current vibration level for Motor_01?",
            "Show me the latest sensor reading for Chiller 6.",
            "Get the live status of Motor_01 at PLANT_A.",
            "What is the real-time temperature of Chiller 6?",
            "Show Chiller 6 data right now.",
            "What are today's readings for Motor_01?",
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
            # Category 3: Urgency / immediacy
            "I need the data immediately.",
            "Check the sensor ASAP.",
            "Get readings right away.",
            "Quick check on Motor_01 temperature.",
            "This is urgent - what is the vibration level?",
            # Category 4: Streaming / monitoring
            "What is the live telemetry feed for Motor_01?",
            "Are there any active alerts for Chiller 6?",
            "Check the dashboard for Motor_01.",
            "Are any alarms active on Chiller 6?",
            "What is the operating status of Motor_01?",
            "Show me the monitoring data.",
            "Track vibration levels for Motor_01.",
            "Is there a threshold breach on Chiller 6?",
            "What is the running status of the pump?",
            # Category 5: Status polling / health checks
            "Is Motor_01 running?",
            "Health check on Chiller 6.",
            "What's happening with Motor_01?",
            "What's going on with the compressor?",
            "Is the pump up or down?",
            # Category 6: Implicit 'now' (IoT)
            "What is the temperature of Chiller 6?",
            "What is the vibration on Motor_01?",
            "How hot is the motor?",
            "How much vibration is there?",
            "Show me the readings for Motor_01.",
            "Give me the data values for Chiller 6.",
        ],
    )
    def test_realtime_queries(self, query: str):
        tag = classify(query)
        assert tag.bucket == TemporalBucket.REALTIME
        assert tag.time_window is None

    @pytest.mark.parametrize(
        "query",
        [
            # "last/past N units"
            "Show Chiller 6 data from the last 30 minutes.",
            "What happened yesterday with Motor_01?",
            "Get readings from the past 2 hours for Chiller 6.",
            "Show recent vibration data for Motor_01.",
            "Show data from last week for Chiller 6.",
            # Extended relative time expressions
            "Data from last night.",
            "What happened during the previous shift?",
            "Show readings since midnight.",
            "Get data since yesterday.",
            "What's the trend over the past few hours?",
            "Show data from this morning.",
            "Readings since this afternoon.",
            "What changed in the last 10 minutes?",
            "Show me what happened 5 minutes ago.",
            "Pull data from earlier today.",
            "Data from this week.",
            "Show this month's readings.",
            "Vibration data since last restart.",
            "Show data within the last 24 hours.",
            "Get the previous 3 days of readings.",
            "Data from not long ago.",
            "A few minutes ago the sensor spiked - show me.",
            "What happened during the last 48 hours?",
            "For the past 12 hours show vibration.",
            "Show data from last year.",
        ],
    )
    def test_relative_time_classified_as_realtime(self, query: str):
        """Relative time expressions should be classified as T3 (safest)."""
        tag = classify(query)
        assert tag.bucket == TemporalBucket.REALTIME

    def test_realtime_keyword_overrides_dates(self):
        """If a query has both 'current' AND a date, T3 wins (priority)."""
        query = "Show the current status of Chiller 6 as of 2020-06-01."
        tag = classify(query)
        assert tag.bucket == TemporalBucket.REALTIME


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
        """'2020-06-01 00:00:00' should match '2020-06-01T00:00:00'."""
        w1 = TimeWindow(start="2020-06-01 00:00:00", end="2020-06-01 01:00:00")
        w2 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        assert w1.matches(w2)

    def test_normalisation_missing_seconds(self):
        """'2020-06-01T00:00' should match '2020-06-01T00:00:00'."""
        w1 = TimeWindow(start="2020-06-01T00:00", end="2020-06-01T01:00")
        w2 = TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")
        assert w1.matches(w2)


# ── Temporal Gate (legacy, still in temporal_classifier.py) ──────────────────


class TestTemporalGateLegacy:
    """Tests for passes_temporal_gate — retained in the module but
    no longer used on the main cache path.  The primary T1/T2
    decision now lives inside the enriched LLM judger."""

    def test_t1_always_passes(self):
        tag = TemporalTag(bucket=TemporalBucket.STATIC)
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="T1",
            cached_window_start=None,
            cached_window_end=None,
            cached_created_at=time.time() - 99999,
        )

    def test_t2_passes_with_matching_window(self):
        tag = TemporalTag(
            bucket=TemporalBucket.HISTORICAL,
            time_window=TimeWindow("2020-06-01T00:00:00", "2020-06-01T01:00:00"),
        )
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="T2",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=time.time() - 99999,
        )

    def test_t2_rejects_mismatched_window(self):
        tag = TemporalTag(
            bucket=TemporalBucket.HISTORICAL,
            time_window=TimeWindow("2020-06-01T01:00:00", "2020-06-01T02:00:00"),
        )
        assert not passes_temporal_gate(
            query_tag=tag,
            cached_bucket="T2",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=time.time(),
        )

    def test_t3_passes_when_fresh(self):
        tag = TemporalTag(bucket=TemporalBucket.REALTIME)
        assert passes_temporal_gate(
            query_tag=tag,
            cached_bucket="T3",
            cached_window_start=None,
            cached_window_end=None,
            cached_created_at=time.time(),
            freshness_threshold_s=60.0,
        )

    def test_t3_rejects_when_stale(self):
        tag = TemporalTag(bucket=TemporalBucket.REALTIME)
        assert not passes_temporal_gate(
            query_tag=tag,
            cached_bucket="T3",
            cached_window_start=None,
            cached_window_end=None,
            cached_created_at=time.time() - 120,
            freshness_threshold_s=60.0,
        )


# ── Judger temporal instruction builder ──────────────────────────────────────


class TestTemporalInstructionBuilder:
    """Verify the enriched judger instructions contain the right context."""

    @pytest.fixture(autouse=True)
    def _import_builder(self):
        """Import the static method so tests stay lightweight (no model load)."""
        from asteria.semantic_judger import SemanticJudger, TemporalContext
        self.build = SemanticJudger._build_temporal_instruction
        self.TemporalContext = TemporalContext

    def test_none_context_returns_base_instruction(self):
        result = self.build(None)
        assert "sufficiently answer" in result
        assert "static" not in result.lower()
        assert "time window" not in result.lower()

    def test_t1_instruction_is_neutral(self):
        """T1 should return the vanilla base prompt — no bias toward acceptance."""
        ctx = self.TemporalContext(query_bucket="T1")
        result = self.build(ctx)
        base = self.build(None)
        assert result == base  # T1 is identical to no-context

    def test_t2_instruction_contains_time_windows(self):
        ctx = self.TemporalContext(
            query_bucket="T2",
            query_window_start="2020-06-01T00:00:00",
            query_window_end="2020-06-01T01:00:00",
            cached_bucket="T2",
            cached_window_start="2020-06-01T01:00:00",
            cached_window_end="2020-06-01T02:00:00",
            cached_created_at=1717200000.0,
        )
        result = self.build(ctx)
        assert "2020-06-01T00:00:00" in result  # query window
        assert "2020-06-01T01:00:00" in result  # cached window start
        assert "2020-06-01T02:00:00" in result  # cached window end
        assert "time window" in result.lower()
        assert "answer 'yes' only if" in result.lower() or "answer 'no'" in result.lower()

    def test_t2_instruction_includes_cached_timestamp(self):
        ctx = self.TemporalContext(
            query_bucket="T2",
            query_window_start="2020-06-01T00:00:00",
            query_window_end="2020-06-01T01:00:00",
            cached_bucket="T2",
            cached_window_start="2020-06-01T00:00:00",
            cached_window_end="2020-06-01T01:00:00",
            cached_created_at=1717200000.0,
        )
        result = self.build(ctx)
        assert "stored at" in result.lower()

    def test_t2_instruction_without_windows_still_mentions_time_bounded(self):
        ctx = self.TemporalContext(query_bucket="T2")
        result = self.build(ctx)
        assert "time-bounded" in result.lower() or "time window" in result.lower()


# ── T3 Bypass integration ───────────────────────────────────────────────────


class TestT3BypassBehaviour:
    """Verify that T3 classify results would cause bypass at cache level.
    These are unit tests on the classifier — the actual cache bypass is
    integration-tested via the profiled runner tests."""

    def test_t3_classification_triggers_bypass_signal(self):
        """T3 queries should be detectable before any cache work."""
        tag = classify("What is the current vibration level for Motor_01?")
        assert tag.bucket == TemporalBucket.REALTIME
        # AsteriaCache checks this and returns immediately.

    def test_t3_queries_produce_no_time_window(self):
        tag = classify("Show me the latest sensor reading for Chiller 6.")
        assert tag.time_window is None

    def test_t1_and_t2_do_not_trigger_bypass(self):
        t1 = classify("What assets are available at site MAIN?")
        assert t1.bucket != TemporalBucket.REALTIME

        t2 = classify("Get history from 2020-06-01T00:00:00 to 2020-06-01T01:00:00.")
        assert t2.bucket != TemporalBucket.REALTIME


# ── Integration: cache_stress_test scenarios ─────────────────────────────────


class TestCacheStressTestScenarios:
    """Ensure queries from cache_stress_test.py classify correctly."""

    @pytest.mark.parametrize(
        "query,expected_bucket",
        [
            ("What assets are available at site MAIN?", TemporalBucket.STATIC),
            ("List all assets at the MAIN site.", TemporalBucket.STATIC),
            ("List sensors for Chiller 6 at site MAIN.", TemporalBucket.STATIC),
            ("What sensors does Chiller 6 have in MAIN?", TemporalBucket.STATIC),
            ("Get history for Chiller 6 from 2020-06-01T00:00:00 to 2020-06-01T01:00:00 at MAIN.",
             TemporalBucket.HISTORICAL),
            ("Get history for Chiller 6 from 2020-06-01T01:00:00 to 2020-06-01T02:00:00 at MAIN.",
             TemporalBucket.HISTORICAL),
            ("Get history for Chiller 6 from 2020-06-01T02:00:00 to 2020-06-01T03:00:00 at MAIN.",
             TemporalBucket.HISTORICAL),
        ],
    )
    def test_stress_test_scenarios(self, query: str, expected_bucket: TemporalBucket):
        tag = classify(query)
        assert tag.bucket == expected_bucket
