# Asteria × AssetOpsBench Integration Report

**Date:** 2026-04-19
**Scope:** Integrate the Asteria semantic cache (arXiv:2509.17360) into AssetOpsBench's plan-execute workflow, add a latency profiler with cache modes, and build HuggingFace-sourced workloads for cache-performance testing.

---

## 1. Motivation

AssetOpsBench evaluates agentic LLM pipelines on industrial asset-operations tasks. Every user question triggers the full **discover → plan → execute → summarise** chain, which hits multiple MCP servers and calls the LLM several times. Re-asked or paraphrased questions ("What assets are at MAIN?" vs "List assets on site MAIN") pay the full cost every time.

Asteria proposes a two-component semantic cache — a Qwen3-Embedding + Qwen3-Reranker pipeline fronted by a SINE index — that can short-circuit a repeated or paraphrased request back to its stored answer. The goal of this integration was to:

1. Make Asteria importable from AssetOpsBench without forcing heavy deps (torch, faiss, sentence-transformers) onto every user.
2. Wire it into the profiler (`timer.py`) so we can measure real latency deltas under cache modes.
3. Decide at which layer semantic caching actually adds value (spoiler: **query-level only**).
4. Ship tests that run in CI without model downloads.
5. Build realistic workloads from the real AssetOpsBench scenarios dataset so cache behaviour can be stress-tested.

---

## 2. Architecture

### 2.1 Where caching lives in the request path

```
┌──────────────────────────────────────────────────────────────────┐
│                        User question                             │
└──────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: Query-level semantic cache  ← THIS is where we cache   │
│  ────────────────────────────────────                            │
│   • Full Asteria: Qwen-Emb → SINE ANN → Qwen-Reranker judger     │
│   • Lightweight fallback: QueryIntentCache (difflib, zero deps)  │
│                                                                  │
│   HIT  → return stored answer, skip everything below             │
│   MISS → proceed, then insert(question, final_answer)            │
└──────────────────────────────────────────────────────────────────┘
               │ miss
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Discovery  → list MCP tools across IoT / FMSR / TSFM / Utils    │
│  Planning   → LLM decomposes question into ordered steps         │
│  Execution  → per step: resolve args (LLM) + MCP tool call       │
│  Summary    → LLM synthesises final answer                       │
└──────────────────────────────────────────────────────────────────┘
```

**Why only at the query level?** An earlier prototype also cached at the MCP tool-call level. We removed it deliberately — tool arguments are already structured JSON (`{"site": "MAIN"}`), so "semantic" matching there degenerates to exact match, which the MCP server can do just as well. Embedding+reranking structured JSON adds latency without finding new hits. The code comment in `timer.py` captures the decision:

> Tool-call-level caching was removed on purpose: tool args are already structured JSON, so a semantic cache there adds no value over exact match and the MCP call itself. Semantic reuse only matters at the question level.

### 2.2 Package layout

```
AssetOpsBench/
├── asteria/                               ← paper implementation
│   ├── cache.py                           AsteriaCache (SINE + judger + LCFU + Markov prefetch)
│   ├── sine_index.py                      SINE ANN index
│   ├── embedding_model.py                 Qwen3-Embedding-0.6B wrapper
│   ├── semantic_judger.py                 Qwen3-Reranker-0.6B wrapper
│   ├── temporal_classifier.py             T1/T2/T3 bucket classifier
│   ├── workload.py                        reference Zipfian/bursty/sequential generators
│   ├── config.py                          DEFAULT_CONFIG (tau_sim=0.75, tau_lsm=0.80, …)
│   └── integrations/
│       └── assetops/                      ← this integration
│           ├── __init__.py                re-exports QueryIntentCache, build_asteria_cache_stack
│           ├── query_intent_cache.py      lightweight dependency-free cache
│           └── full_asteria_adapter.py    build_asteria_cache_stack(...) factory
├── timer.py                               latency profiler with --query-cache / --full-asteria flags
├── notebook/
│   └── asteria_workloads.ipynb            pulls HF scenarios → builds workloads → replays
├── src/scenarios/huggingface/workloads/
│   ├── zipfian.json                       300 requests
│   ├── bursty.json                        300 requests
│   └── sequential.json                    160 requests
└── tests/asteria_unit/
    ├── test_query_intent_cache.py
    ├── test_profiled_runner.py
    └── test_temporal_classifier.py
```

### 2.3 Cache modes exposed to users

| Mode | Flag | Deps | What runs |
|------|------|------|-----------|
| **None (baseline)** | *(default)* | stdlib | Full plan-execute every call |
| **Query-intent (lightweight)** | `--query-cache` | stdlib only | Pre-planner cache using exact + `difflib.SequenceMatcher` similarity |
| **Full Asteria** | `--full-asteria` | torch, faiss-cpu, sentence-transformers, transformers | Pre-planner cache using Qwen-Embedding + SINE ANN + Qwen-Reranker judger, with LCFU eviction and Markov prefetching |

The two cache flags are mutually exclusive (enforced in `_main()`).

### 2.4 `QueryIntentCache` (lightweight)

[asteria/integrations/assetops/query_intent_cache.py](AssetOpsBench/asteria/integrations/assetops/query_intent_cache.py) — 100 lines, no heavy deps. Provides:
- `lookup(query) -> (hit, payload)` — exact match first, then `difflib.SequenceMatcher` ratio against every live key
- `store(query, payload)` — stamps a TTL-expiry timestamp
- `summary()` — hits/misses/hit_rate/entries
- Stats & `last_event` for `timer.py` to surface in run output

Default threshold 0.92, default TTL 1800 s. Intentionally simple so it runs in CI and on laptops without a GPU.

### 2.5 `build_asteria_cache_stack` factory

[asteria/integrations/assetops/full_asteria_adapter.py](AssetOpsBench/asteria/integrations/assetops/full_asteria_adapter.py) — 63 lines. One entry point:

```python
build_asteria_cache_stack(
    capacity=None, tau_sim=None, tau_lsm=None,
    enable_prefetch=True,
    enable_temporal_bucketing=None,
    t3_freshness_threshold_s=None,
)
```

- Catches the ImportError on `torch` / `sentence-transformers` / `faiss` and re-raises with a one-line installation hint, so missing deps never surface as a confusing stack trace.
- Delegates every `None` kwarg to `DEFAULT_CONFIG` from `asteria.config`.
- Constructs `EmbeddingModel()` + `SemanticJudger()` + `AsteriaCache(...)` with all parameters forwarded.

`asteria/__init__.py` guards the heavy imports behind a `try/except ModuleNotFoundError` so `from asteria import QueryIntentCache` always works, even without torch.

### 2.6 `ProfiledRunner` integration ([timer.py](AssetOpsBench/timer.py))

Constructor wires one of three modes based on flags:

```python
if full_asteria:
    self._asteria_cache = build_asteria_cache_stack()
elif query_cache_enabled:
    self._query_cache = QueryIntentCache(threshold, ttl)
```

`run(question)` checks the caches before discovery:

```python
# 0a. Full Asteria
if self._asteria_cache is not None:
    cached_ans, _dbg = self._asteria_cache.lookup(question)
    if cached_ans is not None:
        timing.asteria_query_hit = True
        timing.total_s = time.perf_counter() - run_start
        return timing   # short-circuit

# 0b. Query-intent
if self._query_cache is not None:
    hit, payload = self._query_cache.lookup(question)
    if hit:
        return timing   # short-circuit
```

On a miss, the full pipeline runs, and on success the answer is inserted back into whichever cache is active.

`print_run` / `print_summary` display per-run cache status and an aggregate hit rate across multi-run invocations (`--runs 3`).

---

## 3. Latency profiler (`timer.py`) changes

| Flag | Purpose |
|---|---|
| `--runs N` | Repeat the same question N times; prints mean/stddev per phase |
| `--query-cache` | Enable `QueryIntentCache` |
| `--query-cache-threshold` | Override semantic threshold (default 0.92) |
| `--query-cache-ttl-seconds` | Override TTL (default 1800) |
| `--full-asteria` | Use the paper stack; incompatible with `--query-cache` |
| `--skip-summary` | Skip the final summarisation LLM call (big speed-up for large histories) |
| `--summary-max-chars` | Cap context sent to summariser |
| `--step-response-max-chars` | Cap per-step response in summary prompt |

Recommended baseline vs cache comparison:

```bash
uv run python timer.py --runs 3 --skip-summary "What assets are at MAIN?"
uv run python timer.py --runs 3 --skip-summary --query-cache "What assets are at MAIN?"
uv run python timer.py --runs 3 --skip-summary --full-asteria "What assets are at MAIN?"
```

Run 1 is always a miss for both cache modes; runs 2–N hit the cache and collapse to near-zero wall clock (short-circuit returns before discovery).

---

## 4. Workloads from HuggingFace ([notebook/asteria_workloads.ipynb](AssetOpsBench/notebook/asteria_workloads.ipynb))

### 4.1 Source

`ibm-research/AssetOpsBench` (HF), `train` split — **152 rows**. Columns: `id, type, text, category, deterministic, characteristic_form, group, entity, note`.

### 4.2 Topic classification

The dataset's coarse `category` field only has a few values (Knowledge Query / Actionable / …) and collapsed everything into 6 buckets. We instead use `type:entity` (agent × asset) to get semantically meaningful topics.

**Observed topic distribution (after reclassification):**

| Paraphrases | Topic |
|---:|---|
| 41 | multiagent:chiller |
| 33 | workorder:equipment |
| 18 | fmsa:chiller |
| 15 | tsfm:equipment |
| 14 | workorder:chiller |
| 10 | iot:chiller |
| 8 | tsfm:chiller |
| 6 | iot:ahu |
| 3 | iot:site |
| 2 | fmsa:windturbine |

Topics with fewer than 2 paraphrases are dropped (a semantic cache is meaningless without paraphrases). Result: **10 usable topics**, up from 6 in the first iteration.

### 4.3 Staticity mapping (per agent)

Staticity = 0–10 score for how stable an answer is over time (feeds Asteria's TTL/retention logic).

| Agent | Staticity | Rationale |
|---|---:|---|
| `iot` | 5.5 | Live sensor values change constantly |
| `fmsa` | 8.5 | Failure-mode knowledge is stable |
| `tsfm` | 7.0 | Model metadata stable, forecast outputs vary |
| `workorder` | 6.5 | WO state evolves as orders land |
| `multiagent` | 6.0 | Mixed-agent compositions |

### 4.4 Generators (mirror `asteria/workload.py`)

- **Zipfian** (`n=300, alpha=0.99`) — power-law over topics, topic 0 gets most traffic. Tests steady-state hit rate under realistic head/tail popularity.
- **Bursty** (`n=300`) — two phases; hot topic shifts halfway. Tests cache adaptation to changing popularity.
- **Sequential** (`n_pairs=80`, 160 requests total) — deterministic A→B pairs. Tests Markov prefetcher effectiveness.

All three serialize to `src/scenarios/huggingface/workloads/*.json` for reuse.

### 4.5 Full-Asteria replay tuning

First full-Asteria replay returned **0% hit rate** on all three workloads. Root cause: `AsteriaCache.lookup()` calls `temporal_classify(query)` and bypasses the cache entirely for any query tagged `T3 REALTIME`. AssetOpsBench utterances are packed with date phrases ("last week of April '20", "May and June 2020", "week of 2020-04-27"), so nearly every query got bypassed.

Fix (in notebook cell):

```python
build_asteria_cache_stack(
    capacity=256,                        # > # unique queries so we see steady state
    tau_sim=0.70,                        # looser similarity gate
    tau_lsm=0.75,                        # looser judger confidence
    enable_temporal_bucketing=False,     # disable T3 bypass for replay experiments
    enable_prefetch=True,
)
```

`build_asteria_cache_stack` already accepted these kwargs so no adapter changes were needed.

---

## 5. Testing

### 5.1 Structure

`tests/asteria_unit/` — unit tests runnable without any MCP server, WatsonX, CouchDB, or Qwen model downloads. The directory name was chosen deliberately: earlier candidates (`tests/asteria/`, `tests/asteria_integration/`) collided with either (a) pytest adding `tests/` to `sys.path` and shadowing the real `asteria` package, or (b) the CLAUDE.md-recommended `-k "not integration"` filter silently deselecting every test.

### 5.2 Test inventory

| File | Covers |
|---|---|
| `test_query_intent_cache.py` | Exact & semantic hits, TTL expiry, stats, `last_event`, threshold boundary |
| `test_profiled_runner.py` | Baseline timing, query-cache hit short-circuit, full-asteria short-circuit via a `StubAsteriaCache` (no model load), multi-run aggregation, flag mutex |
| `test_temporal_classifier.py` | Temporal bucket classification for the REALTIME/HISTORICAL/STATIC cases Asteria uses |

Full-Asteria-layer behaviour is tested with a `StubAsteriaCache` that implements the same `lookup/insert` shape, so tests run in <5 s without downloading a single model weight.

### 5.3 Results

```
$ uv run pytest tests/asteria_unit/ -q
........................................................................ [ 50%]
.......................................................................  [100%]
143 passed, 3 warnings in 4.17s
```

**143/143 passing** as of 2026-04-19.

### 5.4 Workload-level empirical results

Inline replay against the 10-topic KB using `QueryIntentCache` (default threshold 0.85, TTL 3600 s):

| Workload | Requests | Hits | Misses | Hit rate | Unique entries |
|---|---:|---:|---:|---:|---:|
| Zipfian (α=0.99) | 300 | 203 | 97 | **0.677** | 97 |
| Bursty (60/40) | 300 | 202 | 98 | **0.673** | 98 |
| Sequential A→B | 160 | 103 | 57 | **0.644** | 57 |

**Reading these numbers.** Hit rate ≈ (1 − unique/total). The lightweight cache lands a hit on every exact-repeated utterance and paraphrase pairs that exceed the difflib ratio threshold. Zipfian and bursty are near-identical because both distributions concentrate ~60% of traffic on 1–2 hot topics, and the cache has capacity for every entry. Sequential is a touch lower because it cycles between only two topics' paraphrase pools, so early paraphrases miss before being warmed up.

**Full-Asteria hit rates** are not yet re-measured with the corrected settings (temporal bucketing disabled + loosened taus). The notebook is wired to produce them but each run loads Qwen-Embedding + Qwen-Reranker × 3 workloads on CPU (~10 min). The first un-tuned run returned 0% due to T3 bypass; code is in place to produce a meaningful number on the next execution.

---

## 6. Summary of changes

| Area | Change |
|---|---|
| **New package** | `asteria/integrations/assetops/` — factory + lightweight cache |
| **New package exports** | `QueryIntentCache`, `build_asteria_cache_stack`, `compose_stored_answer_from_steps` |
| **`asteria/__init__.py`** | Lazy heavy imports so lightweight use works without torch/faiss |
| **`timer.py`** | New flags `--query-cache`, `--query-cache-threshold`, `--query-cache-ttl-seconds`, `--full-asteria`; pre-planner short-circuit for both cache modes; per-run cache status + multi-run aggregate hit rate in output |
| **Removed from timer.py** | Tool-call-level cache (`--cache` / `_cache` / `AsteriaIoTToolLayer`) — deliberate |
| **Notebook** | `notebook/asteria_workloads.ipynb` — pulls HF dataset, reclassifies topics from `type:entity`, builds 3 workloads, replays against both caches, serialises traces to JSON |
| **Workload artifacts** | `src/scenarios/huggingface/workloads/{zipfian,bursty,sequential}.json` |
| **Tests** | `tests/asteria_unit/` — 143 passing in 4.17s, no heavy deps |

---

## 7. Recommended next steps

1. **Finish full-Asteria measurement.** Run the notebook end-to-end (it's ready) to populate the full-Asteria column in the hit-rate table.
2. **Wire `--workload` into `timer.py`.** Let the profiler consume the serialised JSON traces directly instead of taking a single question on the CLI. This gives us per-phase latency distributions under Zipfian/bursty/sequential pressure.
3. **Sweep `tau_sim` / `tau_lsm` / `alpha`.** Map the accuracy-vs-hit-rate Pareto frontier for AssetOpsBench utterances; the paper's defaults were tuned on Wikipedia-style QA and may not be optimal here.
4. **Compare hit-rate curves** for `QueryIntentCache` vs full `AsteriaCache` on the same trace to quantify what the Qwen embedding + reranker stack actually buys on industrial utterances.
