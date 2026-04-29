"""
Temporal Bucketing — query-time classification for cache gating.

Classifies incoming queries into three buckets:
    VOLATILE  (V): Live/current-state queries — never cached.
    ANCHORED  (A): Time-bounded queries — cached with long TTL, gated by window match.
                   Includes both explicit date queries ("June 1, 2020") and
                   relative-time queries ("yesterday", "last week") that have
                   been resolved to concrete windows using the current clock.
    STATIC    (S): Metadata/reference queries — cached with staticity-based TTL.

(Legacy) RELATIVE bucket is preserved in the enum for back-compat but is
never returned by classify(): relative phrases now resolve to ANCHORED
windows at classification time using the supplied wall clock.

Usage:
    from asteria.temporal_classifier import classify

    tag = classify("Get Chiller 6 history from 2020-06-01T00:00 to 2020-06-01T01:00")
    # tag.bucket == TemporalBucket.ANCHORED
    # tag.time_window == TimeWindow("2020-06-01T00:00:00", "2020-06-01T01:00:00")

    tag = classify("Show data from yesterday", now=datetime(2026, 4, 27, 14, 0))
    # tag.bucket == TemporalBucket.ANCHORED
    # tag.time_window == TimeWindow("2026-04-26T00:00:00", "2026-04-26T23:59:59")

    tag = classify("What is the current vibration level for Motor_01?")
    # tag.bucket == TemporalBucket.VOLATILE
"""

from __future__ import annotations

import datetime as _dt
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ── Enums & data classes ─────────────────────────────────────────────────────

class TemporalBucket(Enum):
    VOLATILE = "VOLATILE"   # live/current data — never cached
    RELATIVE = "RELATIVE"   # relative-time references — cached with decay TTL
    ANCHORED = "ANCHORED"   # explicit absolute date range — cached with long TTL
    STATIC   = "STATIC"     # metadata/reference — cached with staticity-based TTL


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
    time_window: Optional[TimeWindow] = None  # populated for ANCHORED only
    classified_at: float = field(default_factory=time.time)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _normalise_ts(ts: str) -> str:
    """Strip whitespace and normalise separators for comparison."""
    ts = ts.strip().replace(" ", "T")
    if re.match(r".*T\d{2}:\d{2}:\d{2}$", ts):
        pass
    elif re.match(r".*T\d{2}:\d{2}$", ts):
        ts += ":00"
    return ts


# ── VOLATILE detection ───────────────────────────────────────────────────────
# Live-state, urgency, streaming/monitoring, status polling, implicit-now IoT.
# Relative-time expressions ("last week", "yesterday") are NOT in this list —
# they belong to RELATIVE.

_VOLATILE_LIVE_STATE: List[str] = [
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

_VOLATILE_URGENCY: List[str] = [
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

_VOLATILE_STREAMING: List[str] = [
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

_VOLATILE_STATUS: List[str] = [
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

_VOLATILE_IMPLICIT: List[str] = [
    r"\bshow\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bgive\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bget\s+me\s+(the\s+)?(reading|data|value|measurement|level)s?\b",
    r"\bwhat\s+is\s+the\s+(temperature|pressure|vibration|rpm|flow|speed|level|load)\b",
    r"\bwhat\s+are\s+the\s+(temperature|pressure|vibration|rpm|flow|speed|level|load)s?\b",
    r"\bhow\s+(hot|cold|fast|slow|loud|high|low)\s+is\b",
    r"\bhow\s+much\s+(vibration|pressure|flow|power|load|noise)\b",
]

_VOLATILE_RE = re.compile(
    "|".join(
        _VOLATILE_LIVE_STATE
        + _VOLATILE_URGENCY
        + _VOLATILE_STREAMING
        + _VOLATILE_STATUS
        + _VOLATILE_IMPLICIT
    ),
    re.IGNORECASE,
)


def _is_volatile(query: str) -> bool:
    return bool(_VOLATILE_RE.search(query))


# ── RELATIVE detection ───────────────────────────────────────────────────────
# Relative time expressions whose meaning shifts with wall-clock time.
# These ARE cached (unlike VOLATILE) — TTL handles freshness.

_RELATIVE_PATTERNS: List[str] = [
    # "today" and variants
    r"\btoday\b",
    r"\btoday'?s?\b",
    # "last/past/previous N {unit}"
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
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b",
    r"\bprevious\s+hour\b",
    r"\bprevious\s+day\b",
    r"\bprevious\s+week\b",
    r"\bprevious\s+month\b",
    r"\bprevious\s+shift\b",
    r"\bprevious\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b",
    r"\bthis\s+morning\b",
    r"\bthis\s+afternoon\b",
    r"\bthis\s+evening\b",
    r"\bthis\s+week\b",
    r"\bthis\s+month\b",
    r"\bthis\s+year\b",
    r"\bthis\s+hour\b",
    r"\bthis\s+shift\b",
    r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
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
    r"\bago\b",
    r"\brecent(ly)?\b",
    r"\bnot\s+long\s+ago\b",
    r"\ba\s+(few|couple)\s+(of\s+)?(minute|hour|day|moment)s?\s+ago\b",
]

_RELATIVE_RE = re.compile(
    "|".join(_RELATIVE_PATTERNS),
    re.IGNORECASE,
)


def _is_relative(query: str) -> bool:
    return bool(_RELATIVE_RE.search(query))


# ── ANCHORED (explicit absolute date) detection ───────────────────────────────

# ISO 8601 dates (2020-06-01, 2020-06-01T00:00:00)
_ISO_DATETIME_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?",
    re.IGNORECASE,
)

_MONTH_NAMES = (
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)

# Natural dates: "June 1, 2020", "1st June 2020"
_NATURAL_DATE_RE = re.compile(
    rf"(?:"
    rf"\b{_MONTH_NAMES}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s*\d{{4}})?\b"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAMES}(?:,?\s*\d{{4}})?\b"
    rf")",
    re.IGNORECASE,
)

# Slash and dot dates: 2020/06/01, 06/01/2020, 2020.06.01
_SLASH_DATE_RE = re.compile(r"\b(?:\d{4}/\d{2}/\d{2}|\d{2}/\d{2}/\d{4})\b")
_DOT_DATE_RE = re.compile(r"\b(?:\d{4}\.\d{2}\.\d{2}|\d{2}\.\d{2}\.\d{4})\b")

# Ordinal dates: "the 1st of June 2020"
_ORDINAL_DATE_RE = re.compile(
    rf"\bthe\s+\d{{1,2}}(?:st|nd|rd|th)\s+of\s+{_MONTH_NAMES}"
    rf"(?:,?\s*\d{{4}})?\b",
    re.IGNORECASE,
)

# Unix epoch references
_EPOCH_RE = re.compile(
    r"\b(?:epoch|unix[\-\s]?timestamp|timestamp)\s*[:=]?\s*\d{10,13}\b",
    re.IGNORECASE,
)

# AM/PM time markers (only when explicitly qualified)
_TIME_OF_DAY_RE = re.compile(r"\b\d{1,2}\s*(?:am|pm)\b", re.IGNORECASE)

# Bare year — anchors a relative expression in absolute time ("last week of 2020")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Historical context keywords — anchor queries with no explicit dates
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
    return _ISO_DATETIME_RE.findall(query)


def _has_explicit_date(query: str) -> bool:
    """True if query contains any explicit absolute date or year anchor."""
    return bool(
        _find_iso_dates(query)
        or _NATURAL_DATE_RE.search(query)
        or _SLASH_DATE_RE.search(query)
        or _DOT_DATE_RE.search(query)
        or _ORDINAL_DATE_RE.search(query)
        or _EPOCH_RE.search(query)
        or _TIME_OF_DAY_RE.search(query)
        or _YEAR_RE.search(query)
    )


def _extract_time_window_from_iso(iso_dates: List[str]) -> Optional[TimeWindow]:
    """Extract a (start, end) window from two or more ISO date strings."""
    if len(iso_dates) < 2:
        return None
    seen: List[str] = []
    for d in iso_dates:
        nd = _normalise_ts(d)
        if nd not in seen:
            seen.append(nd)
    if len(seen) >= 2:
        return TimeWindow(start=seen[0], end=seen[1])
    return None


# ── Relative-window resolution ───────────────────────────────────────────────
# Convert relative phrases into concrete (start, end) ISO windows using the
# supplied wall clock. The resolver covers the same surface area as
# _RELATIVE_PATTERNS above; anything unmatched falls back to a None window
# (caller will treat as relative-without-anchor → defaults to STATIC).

_DAY_NAMES = {
    "monday":    0, "tuesday":   1, "wednesday": 2, "thursday":  3,
    "friday":    4, "saturday":  5, "sunday":    6,
}

_UNIT_TO_SECONDS = {
    "second": 1, "sec": 1,
    "minute": 60, "min": 60,
    "hour":   3600,
    "day":    86400,
    "week":   86400 * 7,
    "month":  86400 * 30,
    "year":   86400 * 365,
}


def _iso(dt: _dt.datetime) -> str:
    """Format a datetime as ISO 8601 with second precision, no tz suffix."""
    return dt.replace(microsecond=0).isoformat()


def _start_of_day(dt: _dt.datetime) -> _dt.datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _end_of_day(dt: _dt.datetime) -> _dt.datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def _resolve_relative_window(
    query: str,
    now: _dt.datetime,
) -> Optional[TimeWindow]:
    """Resolve a relative phrase in `query` to a concrete TimeWindow.

    Handles the major patterns from _RELATIVE_PATTERNS. Returns None when
    the phrase shape is recognised relative but not resolvable (rare).
    """
    q = query.lower()

    # "last/past/previous N <unit>" — rolling window ending at `now`
    m = re.search(
        r"\b(?:last|past|previous|prior)\s+(\d+)\s+"
        r"(hour|minute|min|second|sec|day|week|month|year)s?\b",
        q,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = _dt.timedelta(seconds=n * _UNIT_TO_SECONDS[unit])
        return TimeWindow(start=_iso(now - delta), end=_iso(now))

    # "for/during/over the last/past N <unit>"
    m = re.search(
        r"\b(?:for|during|over|in)\s+(?:the\s+)?(?:last|past)\s+(\d+)\s+"
        r"(hour|minute|day|week|month|year)s?\b",
        q,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = _dt.timedelta(seconds=n * _UNIT_TO_SECONDS[unit])
        return TimeWindow(start=_iso(now - delta), end=_iso(now))

    # "over/in the last/past few/couple <unit>"  → treat few=3
    m = re.search(
        r"\b(?:over|in)\s+the\s+(?:last|past)\s+(?:few|couple(?:\s+of)?)\s+"
        r"(hour|minute|day|week|month)s?\b",
        q,
    )
    if m:
        unit = m.group(1)
        delta = _dt.timedelta(seconds=3 * _UNIT_TO_SECONDS[unit])
        return TimeWindow(start=_iso(now - delta), end=_iso(now))

    # "a few/couple <unit>s ago" → 3 units back
    m = re.search(
        r"\ba\s+(?:few|couple)\s+(?:of\s+)?(minute|hour|day)s?\s+ago\b", q
    )
    if m:
        unit = m.group(1)
        delta = _dt.timedelta(seconds=3 * _UNIT_TO_SECONDS[unit])
        return TimeWindow(start=_iso(now - delta), end=_iso(now))

    # Single-unit relative phrases ─────────────────────────────────────────
    if re.search(r"\blast\s+hour\b|\bprevious\s+hour\b|\bthis\s+hour\b", q):
        delta = _dt.timedelta(hours=1)
        return TimeWindow(start=_iso(now - delta), end=_iso(now))

    if re.search(r"\b(yesterday|since\s+yesterday)\b", q):
        y = now - _dt.timedelta(days=1)
        return TimeWindow(start=_iso(_start_of_day(y)), end=_iso(_end_of_day(y)))

    if re.search(r"\btoday\b|\btoday'?s\b|\bearlier\s+today\b", q):
        return TimeWindow(start=_iso(_start_of_day(now)), end=_iso(now))

    if re.search(r"\blast\s+week\b|\bprevious\s+week\b", q):
        return TimeWindow(start=_iso(now - _dt.timedelta(days=7)), end=_iso(now))

    if re.search(r"\bthis\s+week\b", q):
        start = _start_of_day(now - _dt.timedelta(days=now.weekday()))
        return TimeWindow(start=_iso(start), end=_iso(now))

    if re.search(r"\blast\s+month\b|\bprevious\s+month\b", q):
        return TimeWindow(start=_iso(now - _dt.timedelta(days=30)), end=_iso(now))

    if re.search(r"\bthis\s+month\b", q):
        start = _start_of_day(now.replace(day=1))
        return TimeWindow(start=_iso(start), end=_iso(now))

    if re.search(r"\blast\s+year\b|\bprevious\s+year\b", q):
        return TimeWindow(start=_iso(now - _dt.timedelta(days=365)), end=_iso(now))

    if re.search(r"\bthis\s+year\b", q):
        start = _start_of_day(now.replace(month=1, day=1))
        return TimeWindow(start=_iso(start), end=_iso(now))

    # Day-of-week: "last monday", "this tuesday", etc.
    m = re.search(
        r"\b(last|previous|this)\s+"
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        q,
    )
    if m:
        prefix = m.group(1)
        target = _DAY_NAMES[m.group(2)]
        # Compute days back from now.weekday() to the target weekday.
        diff = (now.weekday() - target) % 7
        if prefix in ("last", "previous") and diff == 0:
            diff = 7
        ref = now - _dt.timedelta(days=diff)
        return TimeWindow(
            start=_iso(_start_of_day(ref)),
            end=_iso(_end_of_day(ref)),
        )

    # "this morning|afternoon|evening" / "since this morning|afternoon|midnight|noon"
    if re.search(r"\b(?:since\s+)?(?:this\s+)?morning\b", q):
        start = _start_of_day(now)
        return TimeWindow(start=_iso(start), end=_iso(now))
    if re.search(r"\bsince\s+midnight\b", q):
        start = _start_of_day(now)
        return TimeWindow(start=_iso(start), end=_iso(now))
    if re.search(r"\bsince\s+noon\b", q):
        start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        return TimeWindow(start=_iso(start), end=_iso(now))
    if re.search(r"\b(?:this\s+)?afternoon\b", q):
        start = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if start > now:
            return None
        return TimeWindow(start=_iso(start), end=_iso(now))
    if re.search(r"\b(?:this\s+)?evening\b", q):
        start = now.replace(hour=18, minute=0, second=0, microsecond=0)
        if start > now:
            return None
        return TimeWindow(start=_iso(start), end=_iso(now))
    if re.search(r"\blast\s+night\b", q):
        y = now - _dt.timedelta(days=1)
        start = y.replace(hour=18, minute=0, second=0, microsecond=0)
        end = _end_of_day(y)
        return TimeWindow(start=_iso(start), end=_iso(end))

    # "<N> <unit>s ago" — point reference; treat as 1-unit window ending then
    m = re.search(
        r"\b(\d+)\s+(hour|minute|day|week|month|year)s?\s+ago\b", q
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = _dt.timedelta(seconds=n * _UNIT_TO_SECONDS[unit])
        unit_delta = _dt.timedelta(seconds=_UNIT_TO_SECONDS[unit])
        end = now - delta
        return TimeWindow(start=_iso(end - unit_delta), end=_iso(end))

    # Bare "ago" / "recently" — vague; default to last 24h
    if re.search(r"\b(ago|recent(ly)?|not\s+long\s+ago)\b", q):
        return TimeWindow(start=_iso(now - _dt.timedelta(days=1)), end=_iso(now))

    return None


# ── Main classifier ──────────────────────────────────────────────────────────

def classify(
    query: str,
    now: Optional[_dt.datetime] = None,
) -> TemporalTag:
    """
    Classify a query into VOLATILE / ANCHORED / STATIC.

    Priority order:
        1. VOLATILE  — live-state keywords, urgency, streaming, implicit-now IoT.
        2. ANCHORED  — explicit absolute dates or year anchors
                       (even if query also contains relative phrases like
                       "last week of 2020").
        3. ANCHORED  — relative-time expressions resolved to a concrete
                       window using `now` (e.g. "yesterday" → ISO window).
        4. ANCHORED  — historical-context keywords without explicit dates.
        5. STATIC    — default: metadata / reference queries.

    Input:  query (str), optional now (datetime; defaults to wall clock)
    Output: TemporalTag
    """
    if now is None:
        now = _dt.datetime.now()

    # Rule 1: VOLATILE — live/current keywords take highest priority.
    if _is_volatile(query):
        return TemporalTag(bucket=TemporalBucket.VOLATILE)

    # Rule 2: ANCHORED (explicit dates) — absolute date anchors override
    # relative phrases.  "last week of 2020" has a year → ANCHORED.
    if _has_explicit_date(query):
        iso_dates = _find_iso_dates(query)
        window = _extract_time_window_from_iso(iso_dates)
        return TemporalTag(bucket=TemporalBucket.ANCHORED, time_window=window)

    # Rule 3: RELATIVE → ANCHORED — resolve relative phrases against `now`.
    if _is_relative(query):
        window = _resolve_relative_window(query, now)
        return TemporalTag(bucket=TemporalBucket.ANCHORED, time_window=window)

    # Rule 4: ANCHORED (historical context) — queries about logs, history,
    # time-series, reports, etc. even without explicit dates.
    if _HISTORICAL_CONTEXT_RE.search(query):
        return TemporalTag(bucket=TemporalBucket.ANCHORED)

    # Rule 5: STATIC — metadata, reference, knowledge queries.
    return TemporalTag(bucket=TemporalBucket.STATIC)


# ── Temporal Gate ────────────────────────────────────────────────────────────

def passes_temporal_gate(
    query_tag: TemporalTag,
    cached_bucket: str,
    cached_window_start: Optional[str],
    cached_window_end: Optional[str],
    cached_created_at: float,
    **_kwargs: object,
) -> bool:
    """
    Decide whether a semantic cache hit should be accepted given the
    temporal characteristics of the incoming query and the cached entry.

    VOLATILE queries never reach this gate (they bypass the cache entirely).
    ANCHORED queries pass only when time windows match exactly.
    STATIC queries always pass.

    Parameters
    ----------
    query_tag : TemporalTag
    cached_bucket : str  — "STATIC" or "ANCHORED"
    cached_window_start, cached_window_end : str | None  — ISO strings (ANCHORED only)
    cached_created_at : float  — epoch seconds when SE was stored
    """
    qb = query_tag.bucket

    if qb == TemporalBucket.STATIC:
        return True

    if qb == TemporalBucket.ANCHORED:
        if cached_bucket != TemporalBucket.ANCHORED.value:
            return False
        if query_tag.time_window is None:
            # No parseable window on new query — can't confirm match → reject.
            return False
        if cached_window_start is None or cached_window_end is None:
            return False
        cached_window = TimeWindow(start=cached_window_start, end=cached_window_end)
        return query_tag.time_window.matches(cached_window)

    # VOLATILE (or legacy RELATIVE) should never be cached/hit safely.
    return False
