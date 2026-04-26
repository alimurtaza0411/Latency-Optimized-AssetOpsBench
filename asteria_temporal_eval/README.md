# Asteria Temporal Paraphrase — Evaluation Scenarios

Scenario data for measuring **Asteria cache precision and latency** against
LLM paraphrases of real AssetOpsBench queries, sliced by **temporal class**.

This branch is **scenario-only** — just the CSVs and the raw HF mirror.
The evaluation harness (paraphrase generator + cache-precision and
latency benchmark scripts) lives in branch
[`asteria-temporal-paraphrase-eval`](../../tree/asteria-temporal-paraphrase-eval).

---

## What's in here

```
asteria_temporal_eval/
├── README.md                         ← this file
└── data/
    ├── all_utterance.jsonl           ← raw HF mirror (152 scenarios)
    ├── originals.csv                 ← 20 hand-curated parent queries
    └── paraphrases.csv               ← 60 rows (20 originals + 40 paraphrases)
```

- `all_utterance.jsonl` is a byte-exact mirror of
  [`ibm-research/AssetOpsBench`](https://huggingface.co/datasets/ibm-research/AssetOpsBench).
  It is the single source of truth — every original we picked is provable
  against this file.
- `originals.csv` is the hand-curated set of 20 parent queries.
- `paraphrases.csv` contains the 20 originals plus 2 paraphrases of each
  (40 paraphrases) generated with `watsonx/meta-llama/llama-3-3-70b-instruct`.

---

## The two questions this dataset is designed to answer

1. **Cache precision** — given an original and 2 paraphrases of it, how
   often does the cache correctly route a paraphrase back to its parent?
   When it routes incorrectly, how often does it pick the wrong parent
   ("cross-route")?
2. **Latency improvement** — for the same 60 queries, how much does the
   cache actually save in wall time end-to-end vs running the agent from
   scratch?

Both questions are sliced by the **temporal class** of the parent query so
the report can show whether the cache misbehaves on time-sensitive queries
(it should refuse to serve them).

---

## The 4 temporal classes

Every original is labelled with one of:

| Class | Definition | Example from this set |
|---|---|---|
| `static` | No time component at all. Asks about intrinsic asset properties or system capabilities. | *"What types of time series analysis are supported?"* |
| `anchored` | Specific date / month / year / week. Question fully resolves to a fixed historical window. | *"Get the work order of equipment CWC04013 for year 2017."* |
| `volatile` | Time-sensitive snapshot. Contains markers like *"latest"*, *"now"*, *"current"*, *"today"*. The answer changes as time passes — a cache that serves it stale is wrong. | *"What was the latest supply humidity from CQPA AHU 1 at site MAIN on sept 3 2015?"* |
| `relative` | Date relative to today. Could be normalised to anchored if the system knows the current date. *"last week"*, *"next week"*, *"yesterday"*, etc. | *"forecast Chiller 6's performance for next week based on data from 2020-04-27 at MAIN."* |

A correctly-working cache should:

- HIT on `static` and `anchored` paraphrases → save latency.
- MISS (or refuse to serve) on `volatile` paraphrases → staticity gate.
- For `relative` it depends on system clock semantics; report the numbers
  and decide.

---

## The 20 originals — distribution

Manually picked from `data/all_utterance.jsonl`. Coverage was deliberate:
every temporal class is represented, every MCP type is represented.

| Slice | Count |
|---|---:|
| **By temporal class** | |
| `static` | 7 |
| `anchored` | 6 |
| `volatile` | 3 |
| `relative` | 4 |
| **By MCP type** | |
| `IoT` | 6 |
| `FMSA` | 2 |
| `TSFM` | 1 |
| `Workorder` | 5 |
| `multiagent` | 6 |
| **Total** | **20** |

The full list with HF ids and the rationale for each temporal label is in
`data/originals.csv`. **Every row references a real HF scenario id; every
`text` field is byte-for-byte identical to the HF source** — there are no
hallucinated queries.

---

## How the paraphrases were generated

For each of the 20 originals, two paraphrases were generated with
`watsonx/meta-llama/llama-3-3-70b-instruct`:

- one at **`T=0.3`** (close paraphrase)
- one at **`T=0.9`** (loose paraphrase)

The prompt explicitly forbade the model from changing:

- Asset / equipment IDs (`Chiller 6`, `Chiller 9`, `CWC04013`, `CWC04009`, …)
- Site names (`MAIN`, `POKMAIN`)
- Sensor / metric names (`Tonnage`, `Condenser Water Flow`, `Power Input`, …)
- Dates and time periods (`2020-04-27`, `mar 13 '20`, `year 2017`, …)
- File names, numeric parameters, model names (`TTM`, `LSTM`, `Chronos`, …)

In 5 of the 20 cases, the model converged on the same surface form at
both temperatures. Those duplicates were re-rolled at `T=1.0` with a
"do not repeat the previous attempt" prompt, so **all 40 paraphrases in
`paraphrases.csv` are unique surface forms**.

For each row, `gen_temperature` records the temperature actually used —
so a reviewer can identify any re-rolled rows by `gen_temperature ∉ {0.3, 0.9}`.

---

## Schemas

### `originals.csv` (20 rows)

| Column | Meaning |
|---|---|
| `query_id` | int, 1..20 |
| `hf_id` | HF scenario id — provenance |
| `mcp_type` | `IoT` / `FMSA` / `TSFM` / `Workorder` / `multiagent` |
| `temporal_class` | `static` / `anchored` / `volatile` / `relative` |
| `temporal_rationale` | plain-English justification for the temporal label |
| `text` | byte-for-byte from the HF dataset |

### `paraphrases.csv` (60 rows = 20 originals + 40 paraphrases)

| Column | Meaning |
|---|---|
| `query_id` | int, 1..60. Originals occupy 1..20; paraphrases occupy 21..60. |
| `role` | `original` or `paraphrase` |
| `parent_query_id` | for paraphrases, the `query_id` of the original; for originals, `query_id` itself |
| `parent_hf_id` | HF id of the original (paraphrases inherit this) |
| `mcp_type` | inherited from parent |
| `temporal_class` | inherited from parent |
| `gen_temperature` | `0.0` for originals; `0.3` / `0.7` / `0.9` / `1.0` for paraphrases |
| `text` | for originals: HF text; for paraphrases: WatsonX-generated paraphrase |

---

## How to use this dataset to evaluate a cache

The intended evaluation protocol — independent of any specific cache
implementation — is:

1. **Warm the cache with the 20 originals.** Insert each original with a
   unique sentinel value (e.g. `"PARENT_QID_<n>"`) as the cached "answer".
   The sentinel encodes which parent the entry belongs to.
2. **Look up each of the 40 paraphrases in the cache.**
3. **Score each paraphrase lookup against the expected parent's sentinel.**

Confusion matrix on the 40 paraphrases:

| Outcome | Definition |
|---|---|
| `TP` | Paraphrase hit cache and returned its own parent's sentinel — correct route. |
| `FP` | Paraphrase hit cache but returned some other parent's sentinel — cross-routed. |
| `FN` | Paraphrase missed the cache entirely. |

Reported metrics:

| Metric | Formula | Reading |
|---|---|---|
| `precision` | `TP / (TP + FP)` | "When the cache serves a HIT, can I trust it?" |
| `recall` | `TP / (TP + FN)` | "How often does the cache succeed when it should?" |
| `cross-route rate` | `FP / (TP + FP)` | "When the cache hits, how often is the route wrong?" |

Slice all three metrics by `temporal_class`, by `gen_temperature`, and by
`mcp_type`.

For latency, run all 60 queries through the agent twice — once with no
cache, once with the cache populated (originals first to seed it,
paraphrases second to measure HIT latency) — and compare per-phase
timings.

---

## What "good" looks like

For a correctly-working cache:

| Slice | Pass criterion |
|---|---|
| `precision` overall | ≥ 0.95 — when the cache says HIT, it must hit the right parent |
| `cross_route_rate` overall | ≤ 0.05 |
| `recall` for `static` paraphrases | ≥ 0.85 |
| `recall` for `anchored` paraphrases | ≥ 0.75 |
| `recall` for `volatile` paraphrases | should be **low** (staticity gate; high recall here is bad) |
| paraphrase mean latency saving | ≥ 60% reduction vs no-cache |

If precision is high but recall is low → judger threshold too tight; widen.
If recall is high but precision is low → threshold too loose; tighten.

---

## Provenance

- 152 scenarios mirrored from
  [`ibm-research/AssetOpsBench`](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
  → `data/all_utterance.jsonl`.
- 20 originals hand-picked and labelled — every row in `originals.csv`
  is verifiable against the HF mirror (matching `hf_id` and byte-exact
  `text`).
- Paraphrases generated by `watsonx/meta-llama/llama-3-3-70b-instruct`,
  with each row stamped with the `gen_temperature` actually used.
- The full evaluation harness — `generate_paraphrases.py`,
  `run_cache_precision.py`, `run_latency_benchmark.py` — is in branch
  [`asteria-temporal-paraphrase-eval`](../../tree/asteria-temporal-paraphrase-eval).
