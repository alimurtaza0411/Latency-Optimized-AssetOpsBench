# Asteria Cache — Precision/Recall Evaluation Suite

A self-contained labeled-pair test set for evaluating any implementation of
the Asteria semantic cache. Branched off `main` so it does not depend on, and
does not include, any in-flight component-A (parallel-execution / discovery
cache) work.

The suite frames cache evaluation as an **information-retrieval** problem,
not hit-rate. Reporting "the cache hits 80% of paraphrases" is meaningless —
every cache fed its own paraphrases will hit. We instead report:

| Metric | Formula | What it answers |
|---|---|---|
| Precision | `TP / (TP + FP)` | "When the cache says HIT, is it right?" |
| Recall | `TP / (TP + FN)` | "When the cache *should* HIT, does it?" |
| F1 | `2·P·R / (P+R)` | Single-number summary |
| False-positive rate | `FP / (FP + TN)` | Per-category, exposes *which* failure mode is responsible |

`TP/FP/FN/TN` are computed against a **labeled `expected` column** in the
CSVs below.

---

## Methodology — manual curation, not heuristics

Every paraphrase pair and every hard-negative pair has been **manually
verified** against the raw text in `data/all_utterance.jsonl` (mirrored from
[`ibm-research/AssetOpsBench`](https://huggingface.co/datasets/ibm-research/AssetOpsBench)
on Hugging Face).

The earlier version of this suite used a heuristic 5-tuple `intent_signature`
to auto-cluster paraphrases. Auditing that output against the raw text
revealed it was over-aggressive — it grouped "What types of TSFM are
supported?" with "Is LSTM supported?" because their slot-tuples were
identical, even though they ask completely different questions. This version
replaces that with hand-curated tables in `build_asteria_eval_pairs.py`
(`TRUE_PARAPHRASES` and `TRUE_HARD_NEGATIVES`); each row carries a `kind`
column explaining *why* the pair was labelled the way it is, so the
labelling is auditable end-to-end.

The dataset has fewer natural paraphrases than the heuristic claimed — that's
just the reality of how AssetOpsBench was authored. A smaller, defensible
positive set is more useful for a report than a larger, noisy one. The
recall numbers you compute from this suite are over scenarios you (or any
reviewer) can verify by reading `data/asteria_eval_positives.csv`.

Easy negatives and volatile-repeat pairs remain auto-generated; the rules
("different MCP type AND no shared asset" / regex-detected time-sensitivity)
are simple enough to be reliable.

---

## Layout

```
asteria_eval/
├── README.md                           ← this file
├── build_asteria_eval_pairs.py         ← manual-curation generator (deterministic)
├── run_asteria_eval.py                 ← cache-agnostic harness
└── data/
    ├── all_utterance.jsonl             ← raw HF dataset (152 scenarios)
    ├── asteria_eval_pairs.csv          ← UNIFIED test set (97 rows). Feed this to the harness.
    ├── asteria_eval_positives.csv      ← 19 rows, expected=HIT  (manually curated)
    ├── asteria_eval_hard_negatives.csv ← 22 rows, expected=MISS (manually curated)
    ├── asteria_eval_easy_negatives.csv ← 50 rows, expected=MISS (auto-generated, sanity floor)
    ├── asteria_eval_volatile.csv       ← 6 rows, expected=MISS  (verbatim repeats of volatile queries)
    └── asteria_eval_summary.json       ← machine-readable manifest with per-kind breakdowns
```

Distribution: **19 HIT / 78 MISS** (≈ 1:4). Class imbalance is acknowledged —
precision and recall are still well-defined, but a "predict everything MISS"
baseline trivially achieves accuracy 0.80, which is why the report should
quote precision/recall/F1, not accuracy.

---

## What each CSV tests

Every row is a `(seed, probe, expected)` triplet. The harness:

1. Builds a **fresh empty cache**.
2. Calls `cache.insert(seed_text, "<answer>")`.
3. Calls `cache.lookup(probe_text)` → `HIT` if a value comes back, `MISS` otherwise.
4. Compares to the `expected` column → buckets as TP / FP / FN / TN.

Pairs are independent (cache is reset per pair), so each row is a clean test.

### `asteria_eval_positives.csv` — 19 rows, expected = **HIT**

> **Question being tested:** *Two questions that mean the same thing are
> asked in succession. Does the cache notice they mean the same thing?*

Every pair is manually verified. The `kind` column tells you why the pair
qualifies as a paraphrase.

| `kind` | Count | What it tests |
|---|---:|---|
| `verbatim`   | 2 | Character-identical text (e.g. `601`/`602`). A cache that misses these is broken outright. |
| `site_added` | 14 | 1xx FMSA series ↔ 6xx multiagent series; only difference is *"at MAIN site"* qualifier added. Same intent, same answer. |
| `reworded`   | 3 | Different surface form, identical intent (e.g. *"What IoT sites are available?"* ↔ *"Can you list the IoT sites?"*). |

**Example row** (`pair_id=18`, `kind=reworded`):

| | Text |
|---|---|
| seed | *Is there any anomaly detected in Chiller 6's Tonnage in the week of 2020-04-27 at the MAIN site?* |
| probe | *Have there been any anomalies in Chiller 6's Tonnage in the week of 2020-04-27 at MAIN?* |

**Measures recall.**
- HIT → **TP**, MISS → **FN**.
- Asteria target: **recall ≥ 0.7** overall; the 2 `verbatim` pairs should be
  trivially 100%; the 14 `site_added` and 3 `reworded` pairs are the real test.
- Low recall ⇒ embedding too tight or judger threshold too high.

### `asteria_eval_hard_negatives.csv` — 22 rows, expected = **MISS**

> **Question being tested:** *Two questions whose texts differ by 1–2 words
> but whose intent (and therefore answer) is genuinely different. Does the
> cache resist the temptation to confuse them?*

Every pair is manually curated and labelled with the **explicit dimension**
along which they differ:

| `kind` | Count | Example |
|---|---:|---|
| `different_metric` | 5 | `502↔505`: forecast Condenser Water Flow vs forecast Tonnage |
| `different_sensor_set` | 3 | `605↔606`: temp sensors only vs temp + power input sensors |
| `different_model` | 3 | `204↔205`: Is TTM supported? vs Is LSTM supported? |
| `different_asset` | 2 | `509↔514`: Chiller **6**'s vs Chiller **9**'s anomaly (one digit) |
| `different_time` | 2 | `407↔408`: 2021 vs May 2020 |
| `different_feature` | 2 | `207↔208`: Anomaly Detection vs Time Series Classification |
| `different_specificity` | 1 | `101↔102`: asset *"Chiller"* vs asset *"Chiller 6"* |
| `different_param` | 1 | `209↔210`: context length 96 vs 1024 |
| `different_operation` | 1 | `402↔403`: preventive vs corrective work order |
| `different_sensor` | 1 | `106↔107`: Supply Temperature vs general temperature |
| `different_site` | 1 | `120↔620`: POKMAIN vs MAIN |

**Example row** (`pair_id=20`, `kind=different_asset`):

| | Text |
|---|---|
| seed | *Can you detect any anomalies in Chiller 6's Condenser Water Flow in the week of 2020-04-27 at MAIN?* |
| probe | *Can you detect any anomalies in Chiller 9's Condenser Water Flow in the week of 2020-04-27 at MAIN?* |

These are surgical traps — only one digit differs but the answers are
completely independent.

**Measures FPR on the hardest negatives.**
- MISS → **TN**, HIT → **FP**.
- Asteria target: **FPR ≤ 0.10** (≤ 2 of 22 incorrectly served).
- High FPR ⇒ judger threshold too loose; the cache is letting surface
  similarity override semantic distinction.

### `asteria_eval_easy_negatives.csv` — 50 rows, expected = **MISS**

> **Question being tested:** *Two questions about completely different
> things. Does the cache leave them alone?*

Auto-generated: different MCP type **and** no overlapping asset. Random
sample with `random.seed(42)` for reproducibility.

**Example row** (`kind=different_topic`):

| | Text |
|---|---|
| seed | *Get failure modes for Chiller 6 and only include in final response those that can be monitored using the available sensors.* |
| probe | *Download the metadata for Chiller 3 at the MAIN facility.* |

**Sanity-floor for FPR.** Should be ~0; anything > 0 is a fundamental
embedding problem. Use this row in the report to *defend* hard-negative
numbers: *"the few false positives we observe come exclusively from the
surgical near-duplicates; on unrelated questions FPR is zero."*

### `asteria_eval_volatile.csv` — 6 rows, expected = **MISS**

> **Question being tested:** *The user asks the exact same time-sensitive
> question twice. Does the cache correctly refuse to serve the stale answer?*

Verbatim repeats of queries flagged volatile by regex match for *latest /
today / now / current / recent / live / right now*.

**Example row** (`pair_id=92`):

| | Text |
|---|---|
| seed | *What was the latest supply humidity from CQPA AHU 1 at site MAIN on sept 3 2015? return in CSV.* |
| probe | (identical to seed) |

**Tests the staticity gate.** A correctly-functioning Asteria must keep this
at FPR=0. The baseline `ExactMatchCache` posts FPR=1.0 here (verbatim text
trivially hits string match) — that's the empirical motivation for the gate.

---

## CSV schema (all five files)

| Column | Meaning |
|---|---|
| `pair_id` | int, 1-based, contiguous within each file |
| `pair_type` | `paraphrase` / `hard_negative` / `easy_negative` / `volatile_repeat` |
| `expected` | Ground truth: `HIT` or `MISS` |
| `kind` | Sub-category — paraphrase reason or confusion type |
| `rationale` | Plain-English justification for this row |
| `seed_id` | int — scenario id from the HF dataset |
| `seed_type` | str — MCP type (IoT / TSFM / Workorder / FMSA / multiagent) |
| `seed_text` | str — exact byte-for-byte copy from `all_utterance.jsonl` |
| `probe_id` | (same fields, for the lookup query) |
| `probe_type` | |
| `probe_text` | |
| `shared_type` | bool — true if seed and probe come from the same MCP type |

---

## Running the harness

The repo is the working directory.

### Fast sanity baseline (no model downloads)

```bash
python asteria_eval/run_asteria_eval.py --mode exact
```

Self-contained string-match dict baseline. Expected output:

```
TP=  2   FP=  6   FN= 17   TN= 72
precision=0.250  recall=0.105  F1=0.148  FPR=0.077
  paraphrase       support= 19  P=1.00  R=0.10  F1=0.19  FPR=0.00
  hard_negative    support= 22  P=0.00  R=0.00  F1=0.00  FPR=0.00
  easy_negative    support= 50  P=0.00  R=0.00  F1=0.00  FPR=0.00
  volatile_repeat  support=  6  P=0.00  R=0.00  F1=0.00  FPR=1.00
```

Reads exactly as expected:

- `paraphrase` recall=0.10 — string match catches **only** the 2 verbatim
  pairs (`601`/`602` and `606`/`616`); the 14 site-qualifier pairs and 3
  reworded pairs all miss.
- `volatile_repeat` FPR=1.0 — verbatim text → trivial string-match hit; no
  staticity gate present.
- Other negatives: FPR=0 — string match doesn't fire on differing surfaces.

If your run reproduces these numbers, the harness is wired correctly and you
have an empirical floor to compare your real cache against.

### Real Asteria run

```bash
python asteria_eval/run_asteria_eval.py --mode full --capacity 256
```

Instantiates `asteria.cache.AsteriaCache` (embeddings + Sine + Qwen
reranker + LCFU + Markov), then resets it once per pair. First run
downloads ~1 GB of model weights. Requires `torch` +
`sentence_transformers`.

### Plugging in a different cache

The harness only needs this two-method contract:

```python
class MyCache:
    def insert(self, query: str, answer: str) -> None: ...
    def lookup(self, query: str) -> tuple[Optional[str], dict]: ...
```

(`lookup` returns `(None, debug)` on miss and `(answer, debug)` on hit;
`debug` may be empty.)

To wire in your cache, add a new factory in `run_asteria_eval.py` next to
`_full_factory` and a new `--mode` value, or import the harness from your own
script and pass your factory to `evaluate(pairs, factory)`.

---

## Outputs

Each run writes `asteria_eval/data/asteria_eval_results.json`:

```json
{
  "mode": "exact" | "full",
  "elapsed_s": 0.0,
  "n_pairs": 97,
  "confusion_matrix": { "TP": …, "FP": …, "FN": …, "TN": … },
  "metrics": { "precision": …, "recall": …, "f1": …, "false_positive_rate": …, "support": 97 },
  "per_pair_type": { "paraphrase": {…}, "hard_negative": {…}, "easy_negative": {…}, "volatile_repeat": {…} },
  "rows": [
    { "pair_id": 1, "pair_type": "paraphrase", "expected": "HIT", "actual": "MISS",
      "label": "FN", "seed_id": 601, "probe_id": 602, "judger_score": null },
    …
  ]
}
```

Use `rows` to inspect *which* queries the cache got wrong:

```bash
jq '.rows[] | select(.label == "FP")' asteria_eval/data/asteria_eval_results.json
jq '.rows[] | select(.label == "FN")' asteria_eval/data/asteria_eval_results.json
```

For `--mode full`, `judger_score` carries the reranker's actual probability,
useful as a citation in the report.

---

## Pass criteria

For a *correctly-working* Asteria cache:

| Pair category | Pass criterion | If it fails, the bug is in… |
|---|---|---|
| `paraphrase` | recall ≥ 0.7 | embedding too tight, or judger threshold too high |
| `hard_negative` | FPR ≤ 0.10 | judger threshold too loose |
| `easy_negative` | FPR ≈ 0 | embedding model fundamentally broken |
| `volatile_repeat` | FPR = 0 | staticity gate not firing on insert |

The `kind` column inside each CSV lets you go one level deeper — e.g. if
`hard_negative` FPR is too high, look at which `kind` is responsible
(`different_asset` vs `different_model` vs `different_metric`) and you know
exactly what dimension the judger is failing to discriminate on.

---

## Regenerating the CSVs

The CSVs in `data/` are checked in and are the canonical reference.
`build_asteria_eval_pairs.py` is deterministic (`random.seed(42)`); re-runs
are bit-identical. You only need to regenerate if you edit the
`TRUE_PARAPHRASES` or `TRUE_HARD_NEGATIVES` tables in the script (e.g. to add
a new manually-verified pair).

```bash
python asteria_eval/build_asteria_eval_pairs.py
```

Source: `data/all_utterance.jsonl` — a verbatim mirror of
`https://huggingface.co/datasets/ibm-research/AssetOpsBench/resolve/main/data/scenarios/all_utterance.jsonl`.

---

## Provenance

- Source dataset: `ibm-research/AssetOpsBench` on Hugging Face — 152 scenarios.
- Pair construction: every paraphrase and every hard-negative pair was
  manually verified against the raw `text` field. Easy negatives and
  volatile repeats are auto-generated under simple, easy-to-audit rules.
- Reproducibility: deterministic with `random.seed(42)`.
