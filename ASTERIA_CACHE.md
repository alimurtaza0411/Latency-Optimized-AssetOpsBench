# Temporal Asteria Caching System

End-to-end semantic caching layer for AssetOpsBench's plan-execute pipeline.
Built on the Asteria paper (arXiv 2509.17360) with a temporal bucketing layer
on top.

## Setup — first-time reload

Anyone cloning the `TemporalAsteriaCache` branch on a new machine needs:

### Prerequisites

- Python 3.12 (managed by `uv`)
- `uv` package manager (`brew install uv` or [astral.sh/uv](https://docs.astral.sh/uv/))
- Docker Desktop (for CouchDB)
- WatsonX API credentials (via IBM Cloud)
- ~10 GB disk for Qwen3 embedding + reranker models (auto-downloaded on first run)

### One-time bootstrap

```bash
# 1. Clone + branch
git clone https://github.com/alimurtaza0411/AssetOpsBench.git
cd AssetOpsBench
git checkout TemporalAsteriaCache

# 2. Install Python deps (creates .venv automatically)
uv sync

# 3. Activate venv
source .venv/bin/activate

# 4. Configure secrets — copy template and fill in
cp .env.example .env   # if .env.example exists; otherwise create .env manually
```

Required `.env` contents (replace placeholders with real values):

```env
# ── CouchDB (IoT/WO/Vibration MCP servers) ───────────────────────────────────
COUCHDB_URL=http://localhost:5984
IOT_DBNAME=chiller
WO_DBNAME=workorder
VIBRATION_DBNAME=vibration
COUCHDB_USERNAME=admin
COUCHDB_PASSWORD=<your-couchdb-password>

# ── IBM WatsonX (planner/executor LLMs) ──────────────────────────────────────
WATSONX_APIKEY=<your-watsonx-key>
WATSONX_PROJECT_ID=178e05b2-3352-4f21-8388-572e6b13d65d
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# ── LiteLLM (optional alternate provider) ────────────────────────────────────
LITELLM_BASE_URL=
LITELLM_API_KEY=
```

### Bring up CouchDB

```bash
cd src/couchdb
docker compose up -d
cd ../..

# Verify
curl http://localhost:5984/
# → {"couchdb":"Welcome", ...}
```

If CouchDB is empty (fresh machine), seed the asset data:

```bash
PYTHONPATH=src uv run python src/couchdb/init_asset_data.py
```

### First-time model download

The first time `bench_cache.py`, `timer.py`, or `tools/recalibrate.py` runs
with Asteria enabled, it will download:

- `Qwen/Qwen3-Embedding-0.6B` (~1.2 GB)
- `Qwen/Qwen3-Reranker-0.6B` (~1.2 GB)

Subsequent runs reload from local cache (~30 s warm-up).

### Smoke test the install

```bash
# Pure-Python unit tests (no GPU/MCP needed)
PYTHONPATH=src:. uv run pytest tests/asteria_unit/ -q
# → expect 146 passed

# End-to-end single query (loads Qwen models + CouchDB + MCP)
PYTHONPATH=src:. uv run python timer.py --asteria --skip-summary "What assets are at site MAIN?"
```

If both succeed the project is ready to use. Continue to the **Testing** section below.

---

## Why

The plan-execute workflow takes 25–100 s per query (planning LLM call + MCP
tool calls + summary LLM call). For queries that repeat — exact text, a
paraphrase, or a different time window with the same intent — we want to skip
the pipeline and return a cached answer.

A naive hash-keyed KV cache only matches identical strings. A pure semantic
cache hits paraphrases but mishandles time-sensitive queries: "yesterday"
asked today must NOT hit a cached answer to "yesterday" asked last week.

Temporal Asteria solves this with a regex-only temporal classifier in front
of the ANN + judger pipeline.

---

## Architecture

```
                ┌─────────────────────────────────────────────────────┐
                │  user query  +  optional now (simulated wall clock) │
                └─────────────────────────┬───────────────────────────┘
                                          ▼
                          ┌──────────────────────────┐
                          │  Temporal Classifier      │  regex only, no LLM
                          │  (asteria/temporal_       │
                          │   classifier.py)          │
                          └──────────────┬───────────┘
                                         │
                            ┌────────────┼────────────┐
                            ▼            ▼            ▼
                       VOLATILE      ANCHORED       STATIC
                            │           │             │
                            ▼           ▼             ▼
                       BYPASS      Sine ──────────► Sine
                       (no cache)   ANN+Judger      ANN+Judger
                                    (window match    (vanilla
                                     in instruction)  judger prompt)
                                         │
                                         ▼
                                       HIT?
                                ┌───────┴────────┐
                              yes              no
                               │                │
                               ▼                ▼
                          return SE.answer    full pipeline →
                          (skip pipeline)     INSERT (subject to
                                              staticity gate)
```

### Components

| Module | Role |
|---|---|
| `asteria/temporal_classifier.py` | Regex classifier → VOLATILE / ANCHORED / STATIC; resolves relative phrases ("yesterday") to ISO windows using `now` |
| `asteria/embedding_model.py` | Qwen3-Embedding-0.6B → 1024-dim vec |
| `asteria/semantic_judger.py` | Qwen3-Reranker-0.6B; relevance score (Role 1) + staticity score (Role 2) |
| `asteria/sine_index.py` | FAISS IndexFlatIP + judger validation. Two-stage retrieval: ANN coarse → judger fine |
| `asteria/cache.py` | Top-level `AsteriaCache`. LCFU eviction, TTL purge, Markov prefetch, VOLATILE bypass |
| `asteria/semantic_element.py` | One cache entry. Stores embedding, answer, staticity, ttl, temporal_bucket, time_window |
| `asteria/recalibrator.py` | Algorithm 1: τ_lsm offline recalibration (paper-faithful) |

---

## Temporal Buckets

Four buckets exist in the enum; classifier returns three. (RELATIVE is legacy
— relative phrases now resolve to ANCHORED at classification time.)

| Bucket | Examples | Cache behaviour |
|---|---|---|
| **VOLATILE** | "current temperature", "live status", "right now" | Bypass cache entirely. Never insert. |
| **ANCHORED** | "from 2020-06-01 to 2020-06-02"; "yesterday"; "last 3 hours"; "history of …" | Cached. Lookup checks query window matches cached window. TTL = 1 year. |
| **STATIC** | "What assets at MAIN?", "List sensors for Chiller 6" | Cached. No window logic. TTL = `(staticity/10) × 30 days`. |

### Regex priority order (`classify()`)

1. **VOLATILE** — live-state / urgency / streaming / status / implicit-now IoT keywords
2. **ANCHORED (explicit)** — ISO date, year anchor, natural date, slash/dot date, time-of-day, epoch
3. **ANCHORED (relative-resolved)** — "yesterday", "last N hours", "this week", etc. → resolved to concrete window via `now`
4. **ANCHORED (historical context)** — "history", "trend", "logs", "downtime report", etc.
5. **STATIC** — fallthrough

`classify(query, now=datetime)` accepts a simulated wall clock so tests can
inject a deterministic "now" — `query_run_at` from the CSV.

---

## Cache Mechanics

### SemanticElement (one cache entry)

```
query, answer
embedding         (1024-dim numpy vec)
cost, latency_ms  (LCFU inputs)
staticity         (1.0–10.0, judger Role 2)
frequency         (incremented on each HIT)
size_tokens       (LCFU denominator)
created_at, ttl_seconds
temporal_bucket   ("STATIC" or "ANCHORED")
time_window_start, time_window_end   (ANCHORED only)
```

### Where it lives

- `AsteriaCache.ses: Dict[str, SemanticElement]` — Python dict, RAM only
- `SineIndex.index = faiss.IndexFlatIP(1024)` — FAISS in-memory
- **No persistence.** Process exit → cache gone.

### TTL

- ANCHORED → 1 year (the window itself is the validity bound)
- STATIC → `(staticity / 10) × 30 days` (≤ 2.0 staticity = discarded)

### Capacity + Eviction (Algorithm 2)

50-entry cap. Eviction order:
1. Purge expired entries (TTL elapsed)
2. While over capacity, drop entry with lowest LCFU score:

```
value_score = log(freq+1) × log(cost×1000+1) × log(latency+1) × log(staticity+1)
              ─────────────────────────────────────────────────────────────────
                                       size_tokens
```

### Lookup → Sine (Algorithm in §4.2)

```
1. classify(query, now)
2. If VOLATILE → return None (bypass).
3. Embed query.
4. Check Markov prefetch store. Hit → return.
5. FAISS ANN over all SEs → top-K candidates with cos ≥ τ_sim.
6. For each candidate, call judger with TemporalContext.
   - STATIC: vanilla "does cached answer satisfy new query?"
   - ANCHORED: extended prompt embedding both windows + cached timestamp,
     model must say 'no' if windows don't match.
7. Best score ≥ τ_lsm → HIT (return that SE.answer).
8. Else MISS.
```

**Critical:** there is **no pre-filter by bucket** before ANN. Temporal
context flows through to the judger as instruction text, not as a candidate
filter. (Confirmed from paper.)

### Insert

```
1. classify(query, now)
2. VOLATILE → skip.
3. Compute staticity via judger Role 2.
4. staticity ≤ 2.0 → skip.   ← gate that affects ANCHORED too
5. Pick TTL (ANCHORED=1y, STATIC=staticity-based).
6. _evict_if_needed() → purge expired, evict by LCFU if full.
7. Embed answer's query, build SE, add to FAISS, store in dict.
```

### Markov Prefetcher (Algorithm 3)

First-order Markov model over confirmed HITs.
- `observe(query)` — record transition from previous query.
- On each hit, predict next query, prefetch if probability ≥ θ (0.30).
- Prefetched entries land in cache with frequency=0; if not used, get evicted
  next time capacity bites.

---

## Configuration

[asteria/config.py](asteria/config.py)

| Setting | Default | What it does |
|---|---|---|
| `embedding_dim` | 1024 | Qwen3-Embedding-0.6B output |
| `tau_sim` | 0.75 | ANN cosine threshold |
| `ann_top_k` | 5 | Max candidates forwarded to judger |
| `tau_lsm` | 0.80 | Judger HIT threshold |
| `staticity_volatile` | 2.0 | Insert skipped below this |
| `cache_capacity` | 50 | Max SEs in cache |
| `default_ttl` | 3600 s | Fallback TTL |
| `markov_theta` | 0.30 | Prefetch probability threshold |
| `enable_temporal_bucketing` | True | Toggle classifier |

---

## Testing

End-to-end test pipeline produces synthetic paraphrases of the 152 utterances
in `all_utterance.csv`, pre-warms a cache with one set, then measures hit
rate / latency on a different set.

### Step 1 — Generate paraphrase CSVs

```bash
set -a; source .env; set +a

# seed CSV — what the cache pre-warms with
PYTHONPATH=. uv run python generate_scenarios.py \
    --output cache_seed.csv \
    --max-rows 25 \
    --paraphrases-per-row 2 \
    --anchored-shifts-per-row 1 \
    --seed 42

# test CSV — what we measure (wider parent range so half MISS, half HIT)
PYTHONPATH=. uv run python generate_scenarios.py \
    --output cache_test.csv \
    --max-rows 50 \
    --paraphrases-per-row 2 \
    --anchored-shifts-per-row 1 \
    --seed 99
```

What [generate_scenarios.py](generate_scenarios.py) does:

1. Loads N rows from `all_utterance.csv` (5 query types: IoT, Workorder, TSFM, multiagent, FMSA)
2. Per row, calls LLM to generate `--paraphrases-per-row` paraphrases of the original query (`similarity_tier="paraphrase"`)
3. For temporally-eligible rows (have date / "yesterday" / "history" / etc., NOT live-state), additionally generates `--anchored-shifts-per-row` paraphrases that REPLACE the original time reference with a synthetic ISO window (`similarity_tier="shifted_anchored"`)
4. Writes augmented CSV with extra columns:
   - `parent_id` — original utterance row this row was derived from
   - `similarity_tier` — `paraphrase` or `shifted_anchored`
   - `synthetic_window_start` / `synthetic_window_end` — ISO window for shifted_anchored rows; empty otherwise
   - `query_run_at` — simulated wall clock for THIS query, deterministic per `(parent_id, variant_index)`. Drives `classify(text, now=…)` for relative phrases.

**Determinism:** windows and `query_run_at` are seeded by `parent_id` (not by `--seed`), so the SAME parent in seed and test CSVs gets the SAME window and the SAME run-time. Different `--seed` only affects negative subsampling.

### Step 2 — Inspect overlap

```bash
PYTHONPATH=. uv run python - <<'PY'
import csv, json
seed = list(csv.DictReader(open("cache_seed.csv")))
test = list(csv.DictReader(open("cache_test.csv")))
sp = {r["parent_id"] for r in seed}
tp = {r["parent_id"] for r in test}
print(json.dumps({
    "seed_rows": len(seed), "test_rows": len(test),
    "seed_unique_parents": len(sp), "test_unique_parents": len(tp),
    "parents_in_both_HIT_candidates": len(sp & tp),
    "test_parents_NOT_in_seed_MISS_candidates": len(tp - sp),
}, indent=2))
PY
```

### Step 3 — Run the benchmark

Requires CouchDB up (`docker compose up -d` in `src/couchdb/`).

```bash
PYTHONPATH=src:. uv run python bench_cache.py \
    --seed-csv cache_seed.csv \
    --test-csv cache_test.csv \
    --sample-count 10 \
    --skip-summary \
    --max-seed-rows 25 \
    --sample-seed 7
```

What [bench_cache.py](bench_cache.py) does:

1. **Pre-warm pass:** runs every row of `cache_seed.csv` (capped by `--max-seed-rows`) through `ProfiledRunner --asteria`. Each row goes through full plan-execute, the answer is inserted into Asteria.
2. **Sample pass:** picks `--sample-count` rows from `cache_test.csv` via deterministic `--sample-seed`.
3. **Baseline pass:** runs sampled rows through a SECOND `ProfiledRunner` with Asteria disabled. Pure end-to-end latency.
4. **Cached pass:** runs the SAME sampled rows through the warmed runner from step 1. Records latency + hit rate.
5. Prints summary: avg / median / min / max baseline vs cached, hit count, speedup, per-row diff.

### Step 4 — Quick single-query check

To eyeball one query's bucketing without full bench:

```bash
PYTHONPATH=src:. uv run python timer.py --asteria --skip-summary "What happened yesterday with Chiller 6 at MAIN?"
```

Output includes `Temporal: tag=… asteria_bucket=… window=…` showing the
classifier's verdict and resolved window.

### Step 5 — Recalibrate τ_lsm (optional, paper Algorithm 1)

```bash
PYTHONPATH=src:. uv run python tools/recalibrate.py --target-precision 0.95
```

Builds paraphrase + cross-scenario validation pairs from
`cache_stress_test.QUERY_SCENARIOS`, scores each through the real Qwen judger,
returns the smallest τ_lsm that achieves the target precision. Update
[asteria/config.py:tau_lsm](asteria/config.py) manually based on output.

---

## What to Look For in Bench Results

| Signal | Meaning |
|---|---|
| Per-row HIT with ≥ 2× speedup | Cache working. |
| MISS overhead +3–7 s vs baseline | Asteria's lookup cost on miss (embed + judger). Real cost; expected. |
| Cross-parent HIT (test parent_id NOT in seed, still hits) | Working as designed. ANN matches on semantic similarity, not provenance. |
| Cached avg slower than baseline avg | Either too many MISSes + small N (overhead > savings), or LiteLLM rate-limit retry noise. Need ≥ 20 samples for stable numbers. |
| 0 % hit rate when seed parents fully cover test parents | Bug. Check temporal classification, τ_sim, τ_lsm in run logs. |

---

## Scope and Known Limitations

### Recommended scope: knowledge-style queries

This implementation is a faithful reproduction of Asteria's pure-semantic cache plus a temporal layer on top. It works **well** for queries where paraphrase variation dominates and the answer is fully determined by the intent (knowledge queries, reference lookups, classification questions). Examples that hit reliably:

- "List sensors for Chiller 6" / "What sensors does Chiller 6 have"
- FMSA failure-mode lookups
- TSFM model-support boolean questions
- IoT site/asset enumeration ("What assets at MAIN?")

It works **poorly** for parameter-rich data fetches where the same query shape with different parameter values needs different answers. Examples that produce false-positive HITs:

- "Tonnage Chiller 6 June 2020" vs "Tonnage Chiller 9 June 2020" — embeddings ~0.95 cosine, judger can't tell them apart
- "List failure modes for Chiller 6" vs "List sensors for Chiller 6" — verb collision, judger sees same domain
- IoT history queries with different (asset, sensor, time-window) tuples

**Bench evaluation should restrict to knowledge-style queries** (FMSA + TSFM + IoT-knowledge + a subset of multiagent). Parameter-rich data-fetch queries are explicitly out of scope and deferred to future work.

### Known friction points

1. **False-positive hits on parameter-rich queries.** Documented above. Mitigated partially by raising `tau_lsm` from the paper default 0.80 to 0.92, which cuts most low-confidence false positives at the cost of fewer hits. Cannot be fully eliminated within pure semantic caching.

2. **Staticity gate blocks ANCHORED inserts when the answer doesn't anchor itself.** When the answer text doesn't include the resolved window, the judger scores its staticity ~1–2 and the insert is skipped. Current workaround: windowless-ANCHORED queries are demoted to STATIC at lookup and insert time so the staticity-based TTL applies. Long-term fix: prepend the resolved window to stored answer text so the judger sees an anchor.

3. **No persistence.** Cache resets on every process. Production deployment needs `pickle` or `faiss.write_index()` round-trip.

4. **LiteLLM rate-limit noise.** Long bench runs see retries that distort latency averages. `bench_cache.py` exposes `--llm-retries` + `--retry-delay-s` to make retry behaviour explicit and visible in logs.

### Future work

1. **Parameter-aware caching.** Extract structured params from each query (entity ID, sensor name, time window, action verb) via lightweight LLM call or rule-based extractor; cache per `(canonical_intent, param_combo)` bucket. Eliminates the false-positive class entirely for parameterised queries.

2. **Hybrid retrieval.** Param-extraction layer in front of the semantic+temporal cache. Param-exact hit short-circuits to direct lookup; semantic match only fires when params overlap. Combines the precision of hash-keyed caching with the paraphrase-robustness of Asteria.

3. **Bucket-based pre-filter before ANN.** Currently candidates are pre-filtered by window-overlap for ANCHORED queries. Could be extended to entity/sensor pre-filter for STATIC queries to further narrow the candidate pool.

4. **Cache persistence layer.** Pickle-based snapshot of `AsteriaCache.ses` plus `faiss.write_index()` round-trip, with versioning and TTL-aware reload on startup.

5. **Online τ_lsm recalibration.** The current `tools/recalibrate.py` is offline. Algorithm 1 from the paper specifies online sampling via `FetchGT(q)`; implementing this needs an oracle callback so production traffic can drive τ adjustment in real time.

---

## File Map (cheat sheet)

```
asteria/
├── temporal_classifier.py   ← regex bucketing, RELATIVE→ANCHORED resolver
├── embedding_model.py       ← Qwen3 embeddings
├── semantic_judger.py       ← Qwen3 reranker, dual-role
├── sine_index.py            ← FAISS + judger 2-stage
├── cache.py                 ← AsteriaCache, eviction, prefetch, gates
├── semantic_element.py      ← SE dataclass, lcfu_score, ttl
├── recalibrator.py          ← Algorithm 1
├── config.py                ← thresholds + capacity
└── integrations/assetops/full_asteria_adapter.py
                             ← build_asteria_cache_stack() entry point

generate_scenarios.py        ← paraphrase + shifted_anchored CSV generator
bench_cache.py               ← seed + sample + baseline-vs-cached harness
timer.py                     ← single-query CLI + ProfiledRunner
sample_queries.py            ← CSV row helpers (filter, sample, load)
cache_stress_test.py         ← Zipfian workload runner (older, hand-coded scenarios)
tools/recalibrate.py         ← τ_lsm offline calibration

tests/asteria_unit/
├── test_temporal_classifier.py   ← 138 tests (no ML deps)
└── test_profiled_runner.py        ← 8 tests (mocked runner, stub cache)

all_utterance.csv            ← 152 source utterances
cache_seed.csv               ← generated, pre-warm corpus
cache_test.csv               ← generated, measurement corpus
bench_overlap.json           ← run-time stats on parent overlap
```
