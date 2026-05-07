# AssetOpsBench: Temporal Semantic Caching and Workflow Optimization

This repository extends **AssetOpsBench (AOB)**, an industrial agent benchmark, by optimizing its latency-sensitive Plan-Execute pipelines. We introduce two complementary optimization layers that dramatically reduce the overhead of multi-hop orchestration across specialized Model Context Protocol (MCP) servers (IoT, FMSR, TSFM, Work Order).

## Architecture & Optimizations

### 1. Temporal Semantic Caching
Traditional semantic caching struggles with parameter-rich, time-sensitive industrial queries (e.g., "What was the temperature of Chiller 6 yesterday?"). We implemented a temporal semantic caching layer that routes queries based on their time-sensitivity before retrieving cached answers:

- **VOLATILE**: Live-state queries (e.g., "current status") bypass the cache entirely.
- **RELATIVE & ANCHORED**: Time-bounded queries are intercepted, resolved into concrete absolute time windows, and forced to match the exact temporal context of the cached answer.
- **STATIC**: Reference metadata queries use standard semantic retrieval.

Our implementation uses a dual-stage retrieval pipeline: an Approximate Nearest Neighbor (ANN) search via `Qwen3-Embedded-0.6B`, followed by a rigorous, time-aware semantic judger using `Qwen3-Reranker-0.6B`.

### 2. MCP Workflow Optimizations
Even on cache misses, the standard Plan-Execute pipeline is slow because it discovers tools and executes plan steps sequentially. We optimized the MCP orchestration layer with:

- **Discovery-Phase Caching**: Tool catalogs from the 4 domain servers are cached to a local `.discovery_cache.json` file. This eliminates the need to spawn subprocesses per query just to fetch tool signatures.
- **Parallel Step Execution**: The planner's output is treated as a directed acyclic graph (DAG). Steps are grouped into topological layers and executed concurrently using `asyncio.gather()`.
- **Persistent Server Pool**: An `MCPServerPool` maintains persistent standard I/O connections to the domain servers across the lifetime of a plan, serializing concurrent tool calls without repeatedly paying subprocess spawn costs.


## Repository Structure

```
.
├── ASTERIA_CACHE.md         # Deep-dive documentation on the caching internals
├── bench_cache.py           # Ablation study benchmarking script
├── timer.py                 # Single-query execution and profiling script
├── generate_scenarios.py    # Synthetic dataset generator for cache scenarios
├── requirements.txt         # Dependencies
├── asteria/                 # Core caching engine
│   ├── temporal_classifier.py
│   ├── semantic_judger.py
│   ├── sine_index.py
│   └── cache.py
└── src/
    ├── agent/plan_execute/  # DAG parallel executor and persistent server pool
    ├── couchdb/             # Dockerized CouchDB backend for simulated assets
    └── servers/             # MCP domain servers (IoT, FMSR, TSFM, WO)
```

## Setup & Reproducibility

### Prerequisites
- Python 3.12 (managed by `uv`)
- Docker Desktop (for CouchDB)
- WatsonX API credentials

### Installation
1. Clone the repository and install dependencies:
   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. Configure your Watson API key and other credentials in `.env` before running anything to get the outputs (refer to `.env.public`).
3. Bring up the CouchDB backend (this will automatically load a subset of the asset data):
   ```bash
   cd src/couchdb
   docker compose up -d
   cd ../..
   ```
   > **Note**: The full `main.json` dataset used for our benchmark results is not provided publicly. However, bringing up the Docker container automatically loads a subset of the data. Queries will run absolutely fine, but the scope of data is restricted to a smaller number of sites and chillers. You do not need to run `init_asset_data.py`.

## Running the Code

### End-to-End Single Query
Run a single query to verify the pipeline and load the Qwen models:
```bash
PYTHONPATH=src:. uv run python timer.py --asteria --skip-summary "What happened yesterday with Chiller 6 at MAIN?"
```

### Full Ablation Benchmark
Generate the datasets and run the full three-phase ablation workflow (Phase 1: Warm, Phase 2A: Baseline, Phase 2B: Cached):

```bash
# 1. Generate test data
PYTHONPATH=. uv run python generate_scenarios.py --output cache_seed.csv --max-rows 25 --paraphrases-per-row 2 --anchored-shifts-per-row 1 --seed 42
PYTHONPATH=. uv run python generate_scenarios.py --output cache_test.csv --max-rows 50 --paraphrases-per-row 2 --anchored-shifts-per-row 1 --seed 99

# 2. Run the benchmark
PYTHONPATH=src:. uv run python bench_cache.py \
    --seed-csv cache_seed.csv \
    --test-csv cache_test.csv \
    --sample-count 100 \
    --skip-summary \
    --max-seed-rows 20 \
    --sample-seed 7 \
    --ablation \
    --verbose
```

## Key Results
- **Temporal Semantic Cache**: In a 50-row temporal-cache benchmark, cache hits achieve a median **30.6x** speedup. 
- **MCP Workflow Optimizations**: Across 20 benchmark queries with 3 runs per query, MCP workflow optimizations reduce the median end-to-end latency from 37.30s to 23.02s, yielding a **1.33x** speedup and a **38.3%** latency reduction.
- **Discovery Cost**: Discovery caching reduces tool discovery overhead from 2.34s to 0.0076s (**296× faster**).
- **Overall Speedup**: The combined optimizations reduce median end-to-end latency by **3.48×** (34.10s → 9.80s).
- **Additive Gains**: Because MCP workflow optimizations apply independently of the cache state, the system is faster than the baseline even on cache misses.
