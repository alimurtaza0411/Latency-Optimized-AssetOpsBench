# Asteria Cache — Temporal Paraphrase Latency Suite

A small, end-to-end experiment that asks two questions:

1. **Cache precision** — given 20 known queries (from the AssetOpsBench HF
   dataset) and 2 LLM-generated paraphrases of each (40 paraphrases total),
   how often does the Asteria cache *correctly* route a paraphrase back to
   its parent? When it routes incorrectly, how often does it pick the wrong
   parent (cross-route)?
2. **Latency improvement** — for the same 60 queries, how much does the cache
   actually save in wall time end-to-end vs running the plan-execute agent
   from scratch?

Both questions are sliced by **temporal class** of the parent query —
`static`, `anchored`, `volatile`, `relative` — so the report can show
whether the cache misbehaves on time-sensitive queries (it should refuse to
serve them).

This branch is **harness-only** — `originals.csv` is committed and verified;
`paraphrases.csv` and the result JSONs are produced by the scripts and
gitignored. Run the three scripts in order (see *Running* below).

---

## The 4 temporal classes

| Class | Definition | Example from this set |
|---|---|---|
| `static` | No time component at all. Asks about intrinsic asset properties or system capabilities. | *"What types of time series analysis are supported?"* |
| `anchored` | Specific date / month / year / week. Question fully resolves to a fixed historical window. | *"Get the work order of equipment CWC04013 for year 2017."* |
| `volatile` | Time-sensitive snapshot. Contains markers like *"latest"*, *"now"*, *"current"*, *"today"*. The answer changes as time passes — a cache that serves it stale is wrong. | *"What was the latest supply humidity from CQPA AHU 1 at site MAIN on sept 3 2015?"* |
| `relative` | Date relative to today. Could be normalised to anchored if the system knows the current date. *"last week"*, *"next week"*, *"yesterday"*, etc. | *"forecast Chiller 6's performance for next week based on data from 2020-04-27 at MAIN."* |

The cache should:

- HIT on `static` and `anchored` paraphrases → save latency.
- MISS (or refuse to serve) on `volatile` paraphrases → staticity gate.
- For `relative` it depends on system clock semantics; the suite reports the
  numbers and lets you decide.

---

## The 20 originals

Manually picked from `data/all_utterance.jsonl` (verbatim mirror of the HF
dataset). Distribution:

| Slice | Count |
|---|---:|
| **By temporal class** | |
| static | 7 |
| anchored | 6 |
| volatile | 3 |
| relative | 4 |
| **By MCP type** | |
| IoT | 6 |
| FMSA | 2 |
| TSFM | 1 |
| Workorder | 5 |
| multiagent | 6 |
| **Total** | **20** |

The full list with HF ids and rationale lives in `data/originals.csv`. Every
row references a real HF scenario id; every `text` field is byte-for-byte
identical to the HF source.

---

## Layout

```
asteria_temporal_eval/
├── README.md                          ← this file
├── .gitignore                         ← excludes generated outputs from git
├── data/
│   ├── all_utterance.jsonl            ← raw HF mirror (152 scenarios)
│   ├── originals.csv                  ← 20 hand-curated queries  ✅ committed
│   ├── paraphrases.csv                ← 60 rows after step 1     (gitignored)
│   ├── cache_precision_results.json   ← output of step 2         (gitignored)
│   └── latency_results.json           ← output of step 3         (gitignored)
├── generate_paraphrases.py            ← step 1: 20 → 60 via WatsonX
├── run_cache_precision.py             ← step 2: cache-only TP/FP/FN (fast)
└── run_latency_benchmark.py           ← step 3: agent latency no-cache vs cache
```

---

## Schemas

### `originals.csv` (committed, 20 rows)

| Column | Meaning |
|---|---|
| `query_id` | int, 1..20 |
| `hf_id` | HF scenario id — provenance |
| `mcp_type` | `IoT` / `FMSA` / `TSFM` / `Workorder` / `multiagent` |
| `temporal_class` | `static` / `anchored` / `volatile` / `relative` |
| `temporal_rationale` | plain-English justification for the temporal label |
| `text` | byte-for-byte from the HF dataset |

### `paraphrases.csv` (generated, 60 rows = 20 + 40)

| Column | Meaning |
|---|---|
| `query_id` | int, 1..60. Originals occupy 1..20; paraphrases occupy 21..60. |
| `role` | `original` or `paraphrase` |
| `parent_query_id` | for paraphrases, the `query_id` of the original; for originals, `query_id` itself |
| `parent_hf_id` | HF id of the original (paraphrases inherit this) |
| `mcp_type` | inherited from parent |
| `temporal_class` | inherited from parent |
| `gen_temperature` | `0.0` for originals; `0.3` (close paraphrase) or `0.9` (loose paraphrase) for paraphrases |
| `text` | for originals: HF text; for paraphrases: WatsonX-generated paraphrase |

---

## Running

All commands run from the repo root.

### Prerequisites

- `.env` with `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`.
- For step 3 (latency): the local MCP servers (`couchdb` etc.) up.
- For step 2 (precision, full mode) and step 3: the Asteria stack and its
  heavy deps (`torch`, `sentence_transformers`, `faiss-cpu`, `transformers`).
- A working `litellm` install — `pip install litellm python-dotenv`.

### Step 1 — generate paraphrases (LLM-bound, ~1–3 min)

```bash
python asteria_temporal_eval/generate_paraphrases.py
# optional: tune retry/sleep behaviour
python asteria_temporal_eval/generate_paraphrases.py --max-retries 3 --sleep 0.5
```

Each original is paraphrased once at `T=0.3` and once at `T=0.9` using
`watsonx/meta-llama/llama-3-3-70b-instruct` (the same model `timer.py`
uses, so the paraphrases reflect what the agent would naturally produce).

The prompt explicitly forbids changing entity names / asset IDs / dates /
parameters — exactly the dimensions our `asteria_eval/` hard-negatives
test discrimination on, so we don't want the paraphraser leaking into them.

Output: `data/paraphrases.csv` (60 rows). **Spot-check it** before
proceeding — open it in your editor and verify the paraphrases are sensible.

### Step 2 — measure cache precision (fast, no agent runs)

```bash
# full Asteria stack (recommended for the report)
python asteria_temporal_eval/run_cache_precision.py --mode full --capacity 256

# difflib-based lightweight cache (sanity check, no heavy deps required)
python asteria_temporal_eval/run_cache_precision.py --mode lightweight
```

What it does:

1. Build a fresh cache.
2. Insert each of the 20 originals with a unique sentinel as the cached
   "answer" (`PARENT_QID_<n>`). The sentinel encodes the parent identity.
3. Look up each of the 40 paraphrases.
4. Compare the served sentinel to the expected parent's sentinel.

Confusion matrix (paraphrases only):

| Outcome | Definition |
|---|---|
| `TP` | Paraphrase hit cache and returned its own parent's sentinel — correct route. |
| `FP` | Paraphrase hit cache but returned some other parent's sentinel — cross-routed. |
| `FN` | Paraphrase missed the cache entirely. |
| (TN) | Not applicable here — every paraphrase is *expected* to hit; staticity gate behaviour is tested in `asteria_eval/asteria_eval_volatile.csv`. |

Reported metrics:

| Metric | Formula |
|---|---|
| precision | `TP / (TP + FP)` |
| recall | `TP / (TP + FN)` |
| F1 | `2 P R / (P + R)` |
| cross-route rate | `FP / (TP + FP)` |

Slices: per `temporal_class`, per `gen_temperature`, per `mcp_type`.

Output: console table + `data/cache_precision_results.json` with full
per-paraphrase detail (which sentinel was served, lookup latency, label).

### Step 3 — measure latency (slow, real agent runs)

```bash
# fast benchmark mode (skips final summarization LLM call)
python asteria_temporal_eval/run_latency_benchmark.py --skip-summary

# only the no-cache baseline (useful as a first dry-run)
python asteria_temporal_eval/run_latency_benchmark.py --no-cached-pass --skip-summary
```

What it does:

- **Pass 1** — runs all 60 queries through `ProfiledRunner` with **no
  cache**. Each query goes through discovery + planning + execution +
  (optional) summarization. Records per-phase timings.
- **Pass 2** — runs the same 60 queries through `ProfiledRunner` with
  `--full-asteria`. Originals are first (they MISS, seeding the cache);
  paraphrases come after (they should HIT). Each pass uses a fresh runner
  so they don't warm each other.

Reported metrics:

- Per-query: `total_s`, `discovery_s`, `planning_s`, `execution_s`,
  `summarization_s`, `asteria_query_hit`.
- Aggregates over all 60, originals-only, paraphrases-only — and
  paraphrases sliced by `temporal_class`.
- **Improvement table** — for each slice, mean / median latency
  no-cache vs cached, absolute saving in seconds and percent.
- Cache hit rate on paraphrases (overall + per temporal class).

Output: console summary + `data/latency_results.json` with per-query
timings for both passes.

---

## What "good" looks like

For a correctly-working full Asteria stack:

| Slice | Pass criterion |
|---|---|
| `precision` overall (step 2) | ≥ 0.95 — when the cache says HIT, it must hit the right parent |
| `cross_route_rate` overall (step 2) | ≤ 0.05 |
| `recall` for `static` paraphrases (step 2) | ≥ 0.85 |
| `recall` for `anchored` paraphrases (step 2) | ≥ 0.75 |
| `recall` for `volatile` paraphrases (step 2) | should be **low** (staticity gate; high recall here is bad) |
| paraphrase hit rate (step 3) | matches step 2's recall closely |
| paraphrase mean latency saving (step 3) | ≥ 60% reduction vs no-cache |

If precision is high but recall is low → judger threshold too tight; widen.
If recall is high but precision is low → threshold too loose; tighten.

---

## Provenance

- 152 scenarios mirrored from
  [`ibm-research/AssetOpsBench`](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
  → `data/all_utterance.jsonl`.
- 20 originals hand-picked and labelled — every row in `originals.csv` is
  verifiable against the HF mirror (matching `hf_id` and byte-exact `text`).
- Paraphrases are LLM-generated and stamped with `gen_temperature` so a
  reviewer can spot-check / re-generate any subset.
- `timer.py` (already in the repo) is the underlying latency profiler;
  this suite is a thin wrapper around it.
