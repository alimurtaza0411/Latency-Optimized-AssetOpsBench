# Temporal Bucketing Layer for Asteria Cache

## Background & Problem Statement

Asteria currently uses **semantic similarity** (embeddings + reranker judger) and a scalar **staticity score** (1–10) to decide whether a cached answer is reusable. The staticity score controls:
- Whether to cache at all (volatile threshold = 2.0)
- TTL duration: `3600 * (staticity/10) * 24 * 30` seconds

**The gap**: staticity is binary-flavored ("is this stable or not?") and encoded at *insertion time*. It doesn't account for the **temporal context of the query itself**. Two semantically identical queries about the same asset can require completely different answers depending on when the data was requested:

| Query A | Query B | Semantically identical? | Same answer? |
|---------|---------|------------------------|--------------|
| "Chiller 6 history 2020-06-01 00:00–01:00" | "Chiller 6 history 2020-06-01 01:00–02:00" | ≈ Yes (high similarity) | **No** — different time window |
| "Chiller 6 history 2020-06-01 00:00–01:00" | "Chiller 6 data for June 1st midnight to 1am" | ≈ Yes | **Yes** — same window, paraphrase |
| "What sensors does Chiller 6 have?" | Same query, 6 months later | ≈ Yes | **Likely yes** — metadata is static |
| "Current vibration level for Motor_01" | Same query, 5 minutes later | ≈ Yes | **Probably no** — live data changes |

The current Sine+judger pipeline would match Query A→B incorrectly (false positive hit), returning stale/wrong data. The temporal bucketing layer fixes this by adding a **time-awareness gate** before confirming a cache hit.

---

## Proposed Architecture: 3-Bucket Temporal Classification

### Why 3 Buckets?

After analyzing all the query patterns in AssetOpsBench (IoT `assets/sensors/history`, vibration analysis, work orders, FMSR, TSFM), the queries naturally fall into three temporal regimes:

| Bucket | Name | Description | Examples | Cache Strategy |
|--------|------|-------------|----------|----------------|
| **T1** | **Static / Metadata** | Schema, configuration, reference data. Answers rarely change. | "What assets at site MAIN?", "What sensors does Chiller 6 have?", "What vibration analysis capabilities are available?", bearing geometry, ISO 10816 thresholds | **Always trust cache hit** if semantic match passes. Long TTL (days–months). |
| **T2** | **Historical / Bounded-Window** | Time-series queries with explicit date ranges. The answer is *permanently correct* for that window — but a different window needs a different answer. | "Chiller 6 history from 2020-06-01T00:00 to 2020-06-01T01:00", "Fetch vibration data from 2024-01-15 to 2024-01-15T01:00" | **Trust cache hit ONLY if time-window matches.** Effectively infinite TTL within window. If cache hit returns different window → treat as **miss**. |
| **T3** | **Live / Real-Time** | Current state queries, latest readings, "now" data. Answer changes continuously. | "Current vibration level", "Latest sensor reading", "What is the live status of Motor_01?" | **Never trust cache** unless query was within a configurable freshness window (e.g. staleness_threshold = 30s–5min). Very aggressive TTL. |

### Why Not 2 or 4?

- **2 buckets** (static vs. dynamic) lose the critical distinction between T2 (historical = permanently valid for its window) and T3 (live = always stale). Historical data is just as cacheable as metadata *if you match the time window*. Collapsing T2+T3 would force overly conservative TTLs on historical queries.
- **4+ buckets** (e.g. splitting T1 into "immutable facts" vs. "slowly changing config") adds classifier complexity for minimal cache-behavior difference. The cache action for both is "trust the hit, long TTL."

---

## Design Details

### 1. Temporal Classifier (`temporal_classifier.py`)

A lightweight, rule-based classifier that runs **before** the Sine lookup. No ML model needed — temporal signals are syntactic and very reliable:

```
classify(query: str) → TemporalBucket (T1 | T2 | T3)
```

**Classification logic (ordered rules):**

1. **T3 detection** — regex/keyword scan for live-data indicators:
   - Keywords: `current`, `now`, `latest`, `live`, `real-time`, `right now`, `at this moment`, `today`
   - Heuristic: query references no explicit date/time range

2. **T2 detection** — regex for explicit time-window markers:
   - ISO dates: `\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?`
   - Natural dates: `January`, `Jan 1`, `from ... to ...`, `between ... and ...`, `last week`, `yesterday`
   - The key signal: presence of any parseable date-range = T2

3. **T1 default** — neither T3 nor T2 → metadata/knowledge query

**Time-window extraction** (for T2 only):
```
extract_time_window(query: str) → Optional[TimeWindow(start, end)]
```
Uses dateutil-style parsing to pull `(start, end)` from the query text. This canonical window becomes part of the cache key fingerprint.

### 2. `TemporalBucket` enum + `TimeWindow` dataclass

```python
class TemporalBucket(Enum):
    STATIC = "T1"        # metadata, reference data
    HISTORICAL = "T2"    # bounded time-window queries
    REALTIME = "T3"      # live/current data

@dataclass
class TemporalTag:
    bucket: TemporalBucket
    time_window: Optional[TimeWindow]  # only for T2
    classified_at: float               # time.time() for staleness checks
```

### 3. Modified Cache Lookup Flow

Current flow:
```
query → embed → Sine ANN → Judger → HIT/MISS
```

New flow with temporal gate:
```
query → temporal_classify(query) → embed → Sine ANN → Judger → TEMPORAL GATE → HIT/MISS
                                                                      ↓
                                                      Compare query bucket + window
                                                      against cached SE's bucket + window
                                                                      ↓
                                                      T1: always pass
                                                      T2: pass only if windows match
                                                      T3: pass only if cache entry 
                                                          age < freshness_threshold
```

The temporal gate sits **after** the semantic judger (to avoid the cost of temporal classification on cold misses) but **before** confirming the hit.

### 4. `SemanticElement` Extension

Add to the existing `SemanticElement` dataclass:

```python
# Temporal metadata (new fields)
temporal_bucket: str = "T1"           # "T1", "T2", "T3"
time_window_start: Optional[str] = None   # ISO string, T2 only
time_window_end: Optional[str] = None     # ISO string, T2 only
```

These get set at `insert()` time, stored alongside the SE, and checked at `lookup()` time.

### 5. Configuration Extension

New fields in `AsteriaConfig`:

```python
# Temporal bucketing (new)
enable_temporal_bucketing: bool = True
t3_freshness_threshold_s: float = 60.0    # 1 min - max staleness for T3 cache hits
t2_window_match_strict: bool = True       # require exact window match for T2
```

---

## Proposed File Changes

### Asteria core

#### [NEW] [temporal_classifier.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/temporal_classifier.py)
- `TemporalBucket` enum
- `TimeWindow` dataclass
- `TemporalTag` dataclass
- `classify(query: str) → TemporalTag` — rule-based classifier
- `extract_time_window(query: str) → Optional[TimeWindow]` — date parser
- `passes_temporal_gate(query_tag, cached_se) → bool` — the gate logic

#### [MODIFY] [config.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/config.py)
- Add `enable_temporal_bucketing`, `t3_freshness_threshold_s`, `t2_window_match_strict` fields

#### [MODIFY] [semantic_element.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/semantic_element.py)
- Add `temporal_bucket`, `time_window_start`, `time_window_end` fields to the dataclass

#### [MODIFY] [cache.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/cache.py)
- Import and use `TemporalClassifier`
- `lookup()`: after judger confirms a semantic hit, run the temporal gate — reject if bucket/window mismatch
- `insert()`: classify the query, attach `TemporalTag` to SE, adjust TTL based on bucket:
  - T1: keep existing staticity-based TTL (or even increase)
  - T2: set very long TTL (data is permanently valid for its window)
  - T3: set aggressively short TTL (= `t3_freshness_threshold_s`)

#### [MODIFY] [__init__.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/__init__.py)
- Export new temporal classifier symbols

---

### Integration layer

#### [MODIFY] [full_asteria_adapter.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/asteria/integrations/assetops/full_asteria_adapter.py)
- Pass `enable_temporal_bucketing` through `build_asteria_cache_stack()`

---

### Tests

#### [NEW] [test_temporal_classifier.py](file:///c:/Users/sajal/Downloads/Columbia/Academics/Semester%202/COMS6998%20(HPML)/Final%20Project/Ali%20Repo/AssetOpsBench/tests/asteria_unit/test_temporal_classifier.py)
- Classification correctness for T1/T2/T3 queries
- Time-window extraction accuracy
- Temporal gate pass/reject scenarios
- Edge cases (ambiguous queries, no dates, relative dates)

---

## Design Decisions (Resolved)

> [!NOTE]
> **T3 freshness threshold**: Set to `t3_freshness_threshold_s = 60.0` (1 minute). Aggressive enough for industrial IoT where live sensor readings change continuously.

> [!NOTE]
> **T2 window matching**: **Strict mode** — if cached window is `00:00–01:00` and new query asks for `00:00–01:30`, treat as MISS. Safest approach.

> [!NOTE]
> **Relative dates** ("last hour", "yesterday"): Classified as **T3** (real-time). Since their meaning changes with wall-clock time, treating them as live data is the safest approach.

### Why Not Rely on the LLM Judger for Temporal Gating?

The Qwen3-Reranker-0.6B already sees `(new_query, cached_answer)` — couldn't it detect temporal mismatches?

In practice, the judger is **insufficient as a sole temporal defense**:
1. **Semantic prompt, not temporal**: The judger prompt asks "does this cached answer sufficiently answer the new query, even if the wording differs?" — a 0.6B prefill-only model (single forward pass, no CoT) is not reliably doing date arithmetic. Queries like "history 00:00–01:00" vs "history 01:00–02:00" are ~95% textually identical and will likely score above τ_lsm.
2. **Cached answers may lack explicit timestamps**: If the stored response is `"Chiller 6 readings: [data...]"` without echoing the time window, the judger has no temporal signal to compare.
3. **Determinism vs. probability**: Even if the judger catches temporal mismatches 70% of the time, the other 30% silently returns wrong data. A rule-based gate is 100% reliable for syntactic date signals at near-zero cost.

**Conclusion**: The temporal gate and judger are **complementary layers** — the gate provides deterministic temporal correctness, the judger provides semantic relevance scoring. Neither alone is sufficient.

---

## Verification Plan

### Automated Tests
1. **Unit tests** for `temporal_classifier.py`:
   ```
   pytest tests/asteria_unit/test_temporal_classifier.py -v
   ```
   - Verify all three bucket classifications on queries from `cache_stress_test.py` and `vibration_utterance.json`
   - Verify time-window extraction accuracy
   - Verify temporal gate correctly blocks/allows cache hits

2. **Existing tests pass**:
   ```
   pytest tests/asteria_unit/ -v
   ```
   - `test_query_intent_cache.py` — unchanged, should still pass
   - `test_profiled_runner.py` — unchanged, should still pass (temporal bucketing defaults to enabled but doesn't break non-temporal queries)

### Manual Verification
- Run the cache stress test with temporal-aware queries (e.g., same question with two different time windows) and confirm the cache correctly rejects the stale hit
- Verify that static/metadata queries continue to cache normally with high hit rates

---

## Open Questions

All open questions have been resolved per user feedback above. Ready for implementation.
