"""
Experiment runner for Asteria benchmarks.

run_experiment():
    Input:  workload (list of (query, answer, staticity) tuples),
            cache (AsteriaCache | ExactMatchCache | None),
            mode ("asteria" | "exact" | "vanilla" | "ann_only")
    Output: dict with keys:
            mode, n, hit_rate, hits, misses, total_cost_$,
            avg_latency_ms, p50_latency_ms, p95_latency_ms,
            latencies (list), hit_mask (list[bool]), answers (list[str])

check_accuracy():
    Input:  workload, served_answers
    Output: float — fraction of correct answers
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .cache import AsteriaCache, ExactMatchCache
from .config import DEFAULT_CONFIG


def simulate_remote_api(
    query: str,
    answer: str,
    latency_ms: float = DEFAULT_CONFIG.remote_latency_ms,
) -> Tuple[str, float, float]:
    """
    Simulates a cross-region API call with ±20% jitter.

    Input:  query, ground-truth answer, base latency
    Output: (answer, actual_latency_ms, cost_$)
    """
    jitter = random.uniform(0.8, 1.2)
    actual_ms = latency_ms * jitter
    time.sleep(actual_ms / 1000 * 0.001)  # tiny sleep for non-zero timing
    return answer, actual_ms, DEFAULT_CONFIG.remote_cost_per_call


def run_experiment(
    workload: List[Tuple[str, str, float]],
    cache,
    mode: str = "asteria",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a full workload through the cache and collect metrics."""
    latencies: List[float] = []
    hits: List[bool] = []
    costs: List[float] = []
    answers: List[str] = []

    for i, (query, true_answer, staticity) in enumerate(
        tqdm(workload, desc=f"{mode}", leave=False)
    ):
        if mode == "vanilla":
            answer, api_ms, cost = simulate_remote_api(query, true_answer)
            latencies.append(api_ms)
            hits.append(False)
            costs.append(cost)
            answers.append(answer)

        elif mode == "exact":
            hit_answer = cache.lookup(query)
            if hit_answer is not None:
                latencies.append(0.5)
                hits.append(True)
                costs.append(0.0)
                answers.append(hit_answer)
            else:
                answer, api_ms, cost = simulate_remote_api(query, true_answer)
                cache.insert(query, answer)
                latencies.append(api_ms)
                hits.append(False)
                costs.append(cost)
                answers.append(answer)

        elif mode in ("asteria", "ann_only"):
            ann_only_flag = mode == "ann_only"
            hit_answer, debug = cache.lookup(query, ann_only=ann_only_flag)
            if hit_answer is not None:
                lat = debug.get("cache_lookup_ms", 5.0)
                latencies.append(lat)
                hits.append(True)
                costs.append(0.0)
                answers.append(hit_answer)
                if verbose:
                    print(f"  HIT  [{i:3d}] {query[:50]}")
            else:
                answer, api_ms, cost = simulate_remote_api(query, true_answer)
                cache.insert(query, answer, latency_ms=api_ms, cost=cost)
                cache_overhead = debug.get("cache_lookup_ms", 0)
                latencies.append(api_ms + cache_overhead)
                hits.append(False)
                costs.append(cost)
                answers.append(answer)
                if verbose:
                    print(f"  MISS [{i:3d}] {query[:50]}")

    hit_array = np.array(hits)
    return {
        "mode": mode,
        "n": len(workload),
        "hit_rate": float(hit_array.mean()),
        "hits": int(hit_array.sum()),
        "misses": int((~hit_array).sum()),
        "total_cost_$": round(sum(costs), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
        "latencies": latencies,
        "hit_mask": hits,
        "answers": answers,
    }


def check_accuracy(
    workload: List[Tuple[str, str, float]],
    served_answers: List[str],
) -> float:
    """
    Fraction of served answers matching ground truth (soft containment match).

    Input:  workload, served_answers
    Output: float [0, 1]
    """
    correct = 0
    for (_, true_answer, _), served in zip(workload, served_answers):
        t = true_answer.lower().strip()
        s = served.lower().strip()
        if t in s or s in t or t[:40] in s:
            correct += 1
    return correct / len(workload)
