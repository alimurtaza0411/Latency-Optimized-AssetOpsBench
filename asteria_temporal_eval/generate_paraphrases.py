"""Generate 2 paraphrases per original via LiteLLM/WatsonX.

For each of the 20 hand-picked originals in `data/originals.csv` we call the
same model the agent uses (`watsonx/meta-llama/llama-3-3-70b-instruct`) at
two LLM sampling temperatures:

    T = 0.3  → close paraphrase   (small wording changes, structure mostly intact)
    T = 0.9  → loose paraphrase   (different sentence structure, wider word swaps)

The result is `data/paraphrases.csv` — 60 rows: 20 originals + 40 paraphrases.
Every paraphrase row points to its parent via `parent_query_id` so the cache
precision harness can later route HIT-results back to the correct expected
parent.

Constraints baked into the prompt (do NOT modify these without re-validating
your dataset):

  * Asset/equipment IDs unchanged   (e.g., Chiller 6, CWC04013, MAIN, POKMAIN)
  * Sensor/metric names unchanged   (e.g., Tonnage, Condenser Water Flow)
  * Dates and time periods unchanged (e.g., 2020-04-27, May 2020, sept 3 2015)
  * File names unchanged             (e.g., chiller9_test.csv)
  * Numeric parameters unchanged     (e.g., context length 96, 1024)

These are exactly the dimensions our hard_negatives in `asteria_eval/` are
designed to test discrimination on; the paraphrase generator must not leak
into them.

Usage (from the repo root):
    python asteria_temporal_eval/generate_paraphrases.py
    python asteria_temporal_eval/generate_paraphrases.py --max-retries 3 --sleep 0.5

Requires: WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL set in the
environment (or in `.env` at the repo root). Uses LiteLLM under the hood.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


PARAPHRASE_PROMPT = """You will paraphrase a question while keeping its intent perfectly intact.

Rules:
1. Do NOT change any of these — copy them character-for-character if they appear:
     - Asset / equipment names and IDs (Chiller 6, Chiller 9, CWC04013, CWC04009, ...)
     - Site names (MAIN, POKMAIN)
     - Sensor / metric names (Tonnage, Condenser Water Flow, Power Input, Setpoint Temperature, ...)
     - Dates and time periods (2020-04-27, May 2020, mar 13 '20, June 2020, year 2017, ...)
     - File names (chiller9_annotated_small_test.csv, ...)
     - Numeric parameters (context length 96, context length 1024, ...)
     - Specific feature names (Anomaly Detection, Time Series Classification, TTM, LSTM, Chronos, ...)
2. The paraphrase must request the same information that would yield the same answer.
3. Only change wording, sentence structure, voice, or formality.
4. Do NOT add new constraints, qualifiers, or assumptions that weren't in the original.
5. Output ONLY the paraphrased question, no preface, no explanation, no quotes.

Original:
{text}

Paraphrase:"""


def call_llm(text: str, temperature: float, model_id: str, max_retries: int, sleep_s: float) -> str:
    """Call LiteLLM with retry-on-error. Returns stripped paraphrase text."""
    import litellm  # imported lazily so the file can at least be parsed without it

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = litellm.completion(
                model=model_id,
                messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
                temperature=temperature,
                max_tokens=200,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out.startswith('"') and out.endswith('"'):
                out = out[1:-1].strip()
            if not out:
                raise RuntimeError("empty response")
            return out
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < max_retries:
                time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def load_originals() -> list[dict[str, str]]:
    path = DATA_DIR / "originals.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run from a clean checkout of asteria-temporal-paraphrase-eval")
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="watsonx/meta-llama/llama-3-3-70b-instruct",
                    help="LiteLLM model string (default: same model timer.py uses).")
    ap.add_argument("--temperatures", default="0.3,0.9",
                    help="Comma-separated list of LLM sampling temperatures, one paraphrase per temperature.")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="Sleep between calls (in seconds) to avoid rate limits.")
    ap.add_argument("--output", default=str(DATA_DIR / "paraphrases.csv"))
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()  # populate WATSONX_APIKEY etc. from .env

    if not os.environ.get("WATSONX_APIKEY"):
        print("WARNING: WATSONX_APIKEY is not set. LiteLLM will fail to authenticate.", file=sys.stderr)

    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]
    if len(temps) != 2:
        print(f"NOTE: --temperatures gave {len(temps)} values; harness expects 2 paraphrases per original.")

    originals = load_originals()
    print(f"loaded {len(originals)} originals from {DATA_DIR / 'originals.csv'}")
    print(f"model     : {args.model_id}")
    print(f"temps     : {temps}")
    print(f"output    : {args.output}\n")

    rows: list[dict[str, object]] = []

    # First, copy originals as their own rows
    for o in originals:
        rows.append({
            "query_id":           int(o["query_id"]),
            "role":               "original",
            "parent_query_id":    int(o["query_id"]),
            "parent_hf_id":       int(o["hf_id"]),
            "mcp_type":           o["mcp_type"],
            "temporal_class":     o["temporal_class"],
            "gen_temperature":    0.0,
            "text":               o["text"],
        })

    next_qid = len(originals) + 1
    for o in originals:
        for t in temps:
            print(f"  [orig {o['query_id']:>2}, T={t}] {o['text'][:70]}", flush=True)
            try:
                paraphrased = call_llm(o["text"], t, args.model_id, args.max_retries, args.sleep)
            except Exception as exc:  # noqa: BLE001
                print(f"    ! FAILED: {exc}", flush=True)
                paraphrased = ""
            print(f"    → {paraphrased[:70]}", flush=True)
            rows.append({
                "query_id":           next_qid,
                "role":               "paraphrase",
                "parent_query_id":    int(o["query_id"]),
                "parent_hf_id":       int(o["hf_id"]),
                "mcp_type":           o["mcp_type"],
                "temporal_class":     o["temporal_class"],
                "gen_temperature":    t,
                "text":               paraphrased,
            })
            next_qid += 1
            time.sleep(args.sleep)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["query_id", "role", "parent_query_id", "parent_hf_id", "mcp_type",
            "temporal_class", "gen_temperature", "text"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_orig = sum(1 for r in rows if r["role"] == "original")
    n_para = sum(1 for r in rows if r["role"] == "paraphrase")
    n_empty = sum(1 for r in rows if r["role"] == "paraphrase" and not r["text"])
    print(f"\nwrote {out_path}")
    print(f"  originals     : {n_orig}")
    print(f"  paraphrases   : {n_para}  (empty/failed: {n_empty})")
    print(f"  total queries : {len(rows)}")


if __name__ == "__main__":
    main()
