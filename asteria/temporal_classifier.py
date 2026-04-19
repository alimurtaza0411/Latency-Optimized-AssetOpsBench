"""
Temporal Bucketing — query-time classification and cache gating.

Classifies incoming queries into three temporal buckets:
    T1 (Static/Metadata):     Schema, config, reference data — always cacheable.
    T2 (Historical/Bounded):  Explicit time-window queries — cacheable only when windows match.
    T3 (Live/Real-Time):      Current-state queries — cacheable only within a freshness threshold.

Usage:
    from asteria.temporal_classifier import classify, passes_temporal_gate

    tag = classify("Get Chiller 6 history from 2020-06-01T00:00 to 2020-06-01T01:00")
    # tag.bucket == TemporalBucket.HISTORICAL
    # tag.time_window == TimeWindow(start="2020-06-01T00:00:00", end="2020-06-01T01:00:00")

    ok = passes_temporal_gate(tag, cached_se, freshness_s=60.0)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── Enums & data classes ─────────────────────────────────────────────────────

class TemporalBucket(Enum):
    """Three temporal regimes for query classification."""
    STATIC = "T1"        # metadata, reference data — always safely cacheable
    HISTORICAL = "T2"    # bounded time-window — cacheable if window matches
    REALTIME = "T3"      # live/current data — cacheable only within freshness


@dataclass
class TimeWindow:
    """An explicit time range parsed from a query."""
    start: str   # ISO 8601 string
    end: str     # ISO 8601 string

    def matches(self, other: "TimeWindow") -> bool:
        """Strict equality: both start and end must match (normalised)."""
        return (
            _normalise_ts(self.start) == _normalise_ts(other.start)
            and _normalise_ts(self.end) == _normalise_ts(other.end)
        )


@dataclass
class TemporalTag:
    """Result of temporal classification for a query."""
    bucket: TemporalBucket
    time_window: Optional[TimeWindow] = None  # populated for T2 only
    classified_at: float = field(default_factory=time.time)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _normalise_ts(ts: str) -> str:
    """Strip whitespace and normalise separators for comparison."""
    ts = ts.strip()
    # Accept "2020-06-01 00:00:00" and "2020-06-01T00:00:00" as equal
    ts = ts.replace(" ", "T")
    # Remove trailing seconds if :00
    if re.match(r".*T\d{2}:\d{2}:\d{2}$", ts):
        pass  # keep full form
    elif re.match(r".*T\d{2}:\d{2}$", ts):
        ts += ":00"
    return ts


# ── T3 (Real-Time) detection ────────────────────────────────────────────────

# Category 1: Explicit live-state keywords
# Queries that explicitly ask for current, live, or real-time data.
_REALTIME_LIVE_STATE: List[str] = [
    r"\bcurrent(ly)?\b",
    r"\bright\s+now\b",
    r"\b(right\s+)?at\s+this\s+(very\s+)?(moment|instant|time)\b",
    r"\bas\s+of\s+now\b",
    r"\bas\s+of\s+right\s+now\b",
    r"\blive\b",
    r"\breal[\-\s]?time\b",
    r"\blatest\b",
    r"\bmost\s+recent\b",
    r"\bjust\s+now\b",
    r"\bpresent(ly)?\b",
    r"\bpresent[\-\s]?day\b",
    r"\bactive(ly)?\b",
    r"\bongoing\b",
    r"\bin[\-\s]?progress\b",
    r"\bup[\-\s]?to[\-\s]?date\b",
    r"\bup[\-\s]?to[\-\s]?the[\-\s]?minute\b",
    r"\bnow\b",
    r"\btoday\b",
    r"\btoday'?s?\b",
    r"\bthis\s+instant\b",
    r"\bat\s+present\b",
    r"\bcurrently\s+reading\b",
    r"\bcurrent\s+state\b",
    r"\bcurrent\s+status\b",
    r"\bcurrent\s+value\b",
    r"\bcurrent\s+level\b",
    r"\bcurrent\s+reading\b",
    r"\bcurrent\s+condition\b",
    r"\bcurrent\s+measurement\b",
    r"\bcurrent\s+temperature\b",
    r"\bcurrent\s+pressure\b",
    r"\bcurrent\s+vibration\b",
    r"\bcurrent\s+rpm\b",
    r"\bcurrent\s+flow\b",
    r"\bcurrent\s+speed\b",
    r"\bcurrent\s+load\b",
    r"\bcurrent\s+output\b",
    r"\bcurrent\s+power\b",
    r"\bcurrent\s+position\b",
]

# Category 2: Relative time expressions
# These SOUND like T2 (historical) but their meaning changes with wall-clock
# time, so we classify them as T3 for safety.
_REALTIME_RELATIVE_TIME: List[str] = [
    # "last N {unit}" / "past N {unit}"
    r"\blast\s+\d+\s+(hour|minute|min|second|sec|day|week|month|year)s?\b",
    r"\bpast\s+\d+\s+(hour|minute|min|second|sec|day|week|month|year)s?\b",
    r"\bprevious\s+\d+\s+(hour|minute|min|second|sec|day|week|month|year)s?\b",
    r"\bprior\s+\d+\s+(hour|minute|min|second|sec|day|week|month|year)s?\b",
    r"\bwithin\s+(the\s+)?(last|past)\s+\d+\s+(hour|minute|min|second|sec|day|week|month|year)s?\b",
    # Named relative periods
    r"\byesterday\b",
    r"\blast\s+hour\b",
    r"\blast\s+week\b",
    r"\blast\s+month\b",
    r"\blast\s+year\b",
    r"\blast\s+night\b",
    r"\blast\s+shift\b",
    r"\bprevious\s+hour\b",
    r"\bprevious\s+day\b",
    r"\bprevious\s+week\b",
    r"\bprevious\s+month\b",
    r"\bprevious\s+shift\b",
    r"\bthis\s+morning\b",
    r"\bthis\s+afternoon\b",
    r"\bthis\s+evening\b",
    r"\bthis\s+week\b",
    r"\bthis\s+month\b",
    r"\bthis\s+year\b",
    r"\bthis\s+hour\b",
    r"\bthis\s+shift\b",
    r"\bearlier\s+today\b",
    r"\bsince\s+(this\s+)?morning\b",
    r"\bsince\s+(this\s+)?afternoon\b",
    r"\bsince\s+midnight\b",
    r"\bsince\s+noon\b",
    r"\bsince\s+yesterday\b",
    r"\bsince\s+last\s+(hour|day|week|month|shift|restart|reboot)\b",
    r"\bover\s+the\s+(last|past)\s+(few|couple(\s+of)?)\s+(hour|minute|day|week|month)s?\b",
    r"\bin\s+the\s+(last|past)\s+(few|couple(\s+of)?)\s+(hour|minute|day|week|month)s?\b",
    r"\bfor\s+the\s+(last|past)\s+\d+\s+(hour|minute|day|week|month)s?\b",
    r"\bduring\s+the\s+(last|past)\s+\d+\s+(hour|minute|day|week|month)s?\b",
    r"\bago\b",  # "5 minutes ago", "an hour ago"
    r"\brecent(ly)?\b",
    r"\bnot\s+long\s+ago\b",
    r"\ba\s+(few|couple)\s+(of\s+)?(minute|hour|day|moment)s?\s+ago\b",
]

# Category 3: Urgency / immediacy signals
# Language suggesting the user needs the answer NOW, implying freshness matters.
_REALTIME_URGENCY: List[str] = [
    r"\bright\s+away\b",
    r"\bimmediately\b",
    r"\binstantly\b",
    r"\binstantaneous(ly)?\b",
    r"\basap\b",
    r"\burgent(ly)?\b",
    r"\bat\s+once\b",
    r"\bwithout\s+delay\b",
    r"\bquick(ly)?\s+(check|look|read|scan|glance|update|snapshot)\b",
    r"\btime[\-\s]?critical\b",
]

# Category 4: Streaming / monitoring queries
# Questions about continuous data feeds or monitoring dashboards.
_REALTIME_STREAMING: List[str] = [
    r"\bstream(ing)?\b",
    r"\bmonitor(ing)?\b",
    r"\bwatch(ing)?\b",
    r"\btrack(ing)?\b",
    r"\bfeed\b",
    r"\bdashboard\b",
    r"\btelemetry\b",
    r"\balert(s|ing)?\b",
    r"\balarm(s|ing)?\b",
    r"\bnotif(y|ication)s?\b",
    r"\bthreshold\s+(breach|violation|exceedance)\b",
    r"\boperating\s+(status|condition|state|mode)\b",
    r"\brunning\s+(status|state|condition)\b",
]

# Category 5: Status polling / health check queries
_REALTIME_STATUS: List[str] = [
    r"\bstatus\s+of\b",
    r"\bhealth\s+(check|status|of)\b",
    r"\bup\s+or\s+down\b",
    r"\bis\s+it\s+(running|online|active|operational|working|functioning)\b",
    r"\bis\s+\w+\s+(running|online|active|operational|working|functioning)\b",
    r"\bare\s+the(y|re)\s+(running|online|active|operational)\b",
    r"\bis\s+\w+\s+(on|off|up|down|idle|busy|faulted|tripped)\b",
    r"\boperational\s+(right\s+now|at\s+the\s+moment|currently)\b",
    r"\bwhat\s+is\s+happening\b",
    r"\bwhat'?s\s+happening\b",
    r"\bwhat\s+is\s+going\s+on\b",
    r"\bwhat'?s\s+going\s+on\b",
]

# Category 6: Implicit "now" phrasing (IoT / industrial context)
_REALTIME_IMPLICIT: List[str] = [
    r"\bshow\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bgive\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bget\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bwhat\s+is\s+the\s+(temperature|pressure|vibration|rpm|flow|speed|level|load)\b",
    r"\bwhat\s+are\s+the\s+(temperature|pressure|vibration|rpm|flow|speed|level|load)s?\b",
    r"\bhow\s+(hot|cold|fast|slow|loud|high|low)\s+is\b",
    r"\bhow\s+much\s+(vibration|pressure|flow|power|load|noise)\b",
]

_REALTIME_RE = re.compile(
    "|".join(
        _REALTIME_LIVE_STATE
        + _REALTIME_RELATIVE_TIME
        + _REALTIME_URGENCY
        + _REALTIME_STREAMING
        + _REALTIME_STATUS
        + _REALTIME_IMPLICIT
    ),
    re.IGNORECASE,
)


def _is_realtime(query: str) -> bool:
    """Check if query asks for live/current data."""
    return bool(_REALTIME_RE.search(query))


# ── T2 (Historical / Bounded-Window) detection ──────────────────────────────

# Format 1: ISO 8601 datetime patterns  (2020-06-01, 2020-06-01T00:00:00)
_ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"                     # date part
    r"(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?",     # optional time part
    re.IGNORECASE,
)

# Format 2: Natural date patterns (month names in many forms)
_MONTH_NAMES = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)
# "June 1", "June 1, 2020", "1 June 2020", "1st June 2020", "June 1st, 2020"
_NATURAL_DATE_RE = re.compile(
    rf"(?:"
    rf"\b{_MONTH_NAMES}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAMES}(?:,?\s*\d{{4}})?\b"
    rf")",
    re.IGNORECASE,
)

# Format 3: Slash-separated dates (2020/06/01, 06/01/2020)
_SLASH_DATE_RE = re.compile(
    r"\b(?:\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4})\b"
)

# Format 4: Dot-separated dates (2020.06.01, 01.06.2020)
_DOT_DATE_RE = re.compile(
    r"\b(?:\d{4}\.\d{2}\.\d{2}|\d{2}\.\d{2}\.\d{4})\b"
)

# Format 5: Written-out dates with ordinals ("the 1st of June 2020")
_ORDINAL_DATE_RE = re.compile(
    rf"\bthe\s+\d{{1,2}}(?:st|nd|rd|th)\s+of\s+{_MONTH_NAMES}"
    rf"(?:,?\s*\d{{4}})?\b",
    re.IGNORECASE,
)

# Format 6: Unix timestamps / epoch references
_EPOCH_RE = re.compile(
    r"\b(?:epoch|unix[\-\s]?timestamp|timestamp)\s*[:=]?\s*\d{10,13}\b",
    re.IGNORECASE,
)

# Format 7: Informal time of day references with dates
_TIME_OF_DAY_RE = re.compile(
    r"\b\d{1,2}\s*(?:am|pm|AM|PM)\b"
)

# Range signal words — expanded to cover many connective styles
_RANGE_RE = re.compile(
    r"\b("
    r"from|between|to|until|till|through|thru|"
    r"ending|starting|beginning|commencing|"
    r"spanning|covering|ranging|"
    r"prior\s+to|up\s+(?:to|until|till)|"
    r"no\s+(?:earlier|later)\s+than|"
    r"on\s+or\s+(?:before|after)|"
    r"before|after|during|within|at|on"
    r")\b",
    re.IGNORECASE,
)

# Historical context keywords — phrases that strongly imply bounded time queries
_HISTORICAL_CONTEXT_RE = re.compile(
    r"\b("
    r"histor(y|ical|ic)|time[\-\s]?series|"
    r"trend(s|ing)?|log(s|ged)?|"
    r"record(s|ed)?|archive(d|s)?|"
    r"past\s+data|old(er)?\s+data|"
    r"previous\s+data|earlier\s+data|"
    r"back\s+in|looking\s+back|"
    r"retrosp(ect|ective)|"
    r"replay|playback|audit\s+trail|"
    r"event\s+log|incident\s+report|"
    r"downtime\s+(report|log|history|record)|"
    r"maintenance\s+(log|history|record)|"
    r"shift\s+report|daily\s+report|weekly\s+report|"
    r"monthly\s+report"
    r")\b",
    re.IGNORECASE,
)


def _find_iso_dates(query: str) -> List[str]:
    """Extract all ISO-style date(time) strings from query text."""
    return _ISO_DATETIME_RE.findall(query)


def _has_time_range_context(query: str) -> bool:
    """Check if query contains range-indicating language."""
    return bool(_RANGE_RE.search(query))


def _is_historical(query: str) -> Tuple[bool, Optional[TimeWindow]]:
    """
    Check if query contains explicit date references or historical-context
    keywords making it a bounded-window historical query.

    Checks all date formats:
        - ISO 8601 (2020-06-01, 2020-06-01T00:00:00)
        - Natural dates (June 1, 2020 / 1st June 2020)
        - Slash dates (2020/06/01, 06/01/2020)
        - Dot dates (2020.06.01)
        - Ordinal dates (the 1st of June 2020)
        - Unix timestamps (epoch: 1717200000)
        - AM/PM time markers (3 PM)
        - Historical context keywords (history, time-series, logs, archive)

    Returns (is_historical, extracted_time_window_or_None).
    """
    iso_dates = _find_iso_dates(query)
    natural_dates = _NATURAL_DATE_RE.findall(query)
    slash_dates = _SLASH_DATE_RE.findall(query)
    dot_dates = _DOT_DATE_RE.findall(query)
    ordinal_dates = _ORDINAL_DATE_RE.findall(query)
    epoch_refs = _EPOCH_RE.findall(query)
    ampm_times = _TIME_OF_DAY_RE.findall(query)

    all_dates = (
        iso_dates + natural_dates + slash_dates
        + dot_dates + ordinal_dates + epoch_refs + ampm_times
    )

    if len(all_dates) > 0:
        # Explicit dates found → historical. Try to extract a window.
        window = _extract_time_window_from_iso(iso_dates)
        return True, window

    # No explicit dates, but check for historical context keywords
    # (e.g. "show me the history", "time-series data", "event log")
    if _HISTORICAL_CONTEXT_RE.search(query):
        return True, None

    return False, None


def _extract_time_window_from_iso(iso_dates: List[str]) -> Optional[TimeWindow]:
    """
    Given a list of ISO date strings found in a query, try to form
    a (start, end) window.  Takes the first two distinct dates as
    start and end.
    """
    if len(iso_dates) < 2:
        # Single date — we know it's historical but can't form a window.
        # Still classify as T2 but without a matchable window.
        return None

    # Normalise and deduplicate while preserving order.
    seen = []
    for d in iso_dates:
        nd = _normalise_ts(d)
        if nd not in seen:
            seen.append(nd)

    if len(seen) >= 2:
        return TimeWindow(start=seen[0], end=seen[1])
    return None


# ── Main classifier ──────────────────────────────────────────────────────────

def classify(query: str) -> TemporalTag:
    """
    Classify a query into one of three temporal buckets.

    Priority order:
        1. T3 (Real-Time) — if live/current keywords or relative time detected
        2. T2 (Historical) — if explicit date references detected
        3. T1 (Static)     — default: metadata / reference queries

    Input:  query (str)
    Output: TemporalTag with bucket and optional time_window
    """
    # Rule 1: Real-time keywords take priority (even if dates are present,
    # "current" or "latest" signals the user wants *now* data).
    if _is_realtime(query):
        return TemporalTag(bucket=TemporalBucket.REALTIME)

    # Rule 2: Explicit dates → historical bounded-window query.
    is_hist, window = _is_historical(query)
    if is_hist:
        return TemporalTag(
            bucket=TemporalBucket.HISTORICAL,
            time_window=window,
        )

    # Rule 3: Default — metadata / knowledge / static.
    return TemporalTag(bucket=TemporalBucket.STATIC)


# ── Temporal Gate ────────────────────────────────────────────────────────────

def passes_temporal_gate(
    query_tag: TemporalTag,
    cached_bucket: str,
    cached_window_start: Optional[str],
    cached_window_end: Optional[str],
    cached_created_at: float,
    freshness_threshold_s: float = 60.0,
) -> bool:
    """
    Decide whether a semantic cache hit should be accepted given the
    temporal characteristics of the incoming query and the cached entry.

    Parameters
    ----------
    query_tag : TemporalTag
        Classification of the incoming query.
    cached_bucket : str
        The temporal bucket string ("T1", "T2", "T3") stored on the cached SE.
    cached_window_start, cached_window_end : str | None
        The time window (ISO strings) stored on the cached SE (T2 only).
    cached_created_at : float
        time.time() when the cached SE was created.
    freshness_threshold_s : float
        Maximum staleness in seconds for T3 hits (default: 60s / 1 min).

    Returns
    -------
    bool
        True if the cache hit should be accepted; False if it should be
        treated as a miss despite semantic similarity.
    """
    qb = query_tag.bucket

    # ── T1 (Static): always accept semantic hits ─────────────────────────
    if qb == TemporalBucket.STATIC:
        return True

    # ── T2 (Historical): accept only if time windows match exactly ───────
    if qb == TemporalBucket.HISTORICAL:
        # If the cached entry is not historical, reject.
        if cached_bucket != TemporalBucket.HISTORICAL.value:
            return False

        # If either side lacks a parseable window, we can't confirm a match
        # → reject to be safe.
        if query_tag.time_window is None:
            return False
        if cached_window_start is None or cached_window_end is None:
            return False

        cached_window = TimeWindow(
            start=cached_window_start,
            end=cached_window_end,
        )
        return query_tag.time_window.matches(cached_window)

    # ── T3 (Real-Time): accept only if cache entry is fresh enough ───────
    if qb == TemporalBucket.REALTIME:
        age_s = time.time() - cached_created_at
        return age_s <= freshness_threshold_s

    # Unknown bucket — reject for safety.
    return False
