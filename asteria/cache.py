"""
Cache implementations  (§4.3, Algorithms 2 & 3)

AsteriaCache — full cache with Sine + LCFU eviction + Markov prefetching
    Input:
        lookup(query: str, ann_only: bool = False)
            → (cached_answer: str | None, debug: dict)
        insert(query: str, answer: str, cost: float, latency_ms: float)
            → None   (SE stored internally; volatile data auto-discarded)
    Output from debug dict:
        hit (bool), source ("prefetch"|"sine"), cache_lookup_ms (float),
        prefetch_triggered (str|None), judger_scores (list)

ExactMatchCache — hash-keyed baseline
    lookup(query: str) → str | None
    insert(query: str, answer: str) → None

LRUSemanticCache / LFUSemanticCache — eviction policy ablations
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_CONFIG, AsteriaConfig
from .embedding_model import EmbeddingModel
from .semantic_element import SemanticElement
from .semantic_judger import SemanticJudger
from .sine_index import SineIndex


# ── Markov Prefetcher (Algorithm 3) ──────────────────────────────────────────

class MarkovPrefetcher:
    """
    First-order Markov model over confirmed cache hits.
    Triggered on every confirmed HIT inside lookup(), not on inserts.

    Input:  observe(query)          — record a hit
            predict(query)          — get likely next queries
    Output: List[(next_query, probability)]
    """

    def __init__(self, theta: float = DEFAULT_CONFIG.markov_theta):
        self.theta = theta
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._last_query: Optional[str] = None

    def observe(self, query: str):
        if self._last_query is not None:
            self.transitions[self._last_query][query] += 1
        self._last_query = query

    def predict(self, query: str) -> List[Tuple[str, float]]:
        counts = self.transitions.get(query, {})
        total = sum(counts.values())
        if total == 0:
            return []
        return sorted(
            [(q, c / total) for q, c in counts.items()],
            key=lambda x: -x[1],
        )


# ── Exact-match baseline ─────────────────────────────────────────────────────

class ExactMatchCache:
    """Hash-keyed KV cache. 0% hit rate on paraphrased queries."""

    def __init__(self, capacity: int = DEFAULT_CONFIG.cache_capacity):
        self.store: Dict[str, str] = {}
        self.capacity = capacity

    def lookup(self, query: str) -> Optional[str]:
        return self.store.get(query.strip().lower())

    def insert(self, query: str, answer: str):
        if len(self.store) >= self.capacity:
            oldest = next(iter(self.store))
            del self.store[oldest]
        self.store[query.strip().lower()] = answer


# ── AsteriaCache ─────────────────────────────────────────────────────────────

class AsteriaCache:
    """
    Full Asteria cache: Sine + model-predicted staticity + LCFU eviction
    + TTL + Markov prefetching.

    Parameters
    ----------
    embedding_model : EmbeddingModel
    judger          : SemanticJudger
    capacity        : max SEs
    tau_sim         : ANN similarity threshold
    tau_lsm         : judger confidence threshold
    enable_prefetch : toggle Markov prefetching
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        judger: SemanticJudger,
        capacity: int = DEFAULT_CONFIG.cache_capacity,
        tau_sim: float = DEFAULT_CONFIG.tau_sim,
        tau_lsm: float = DEFAULT_CONFIG.tau_lsm,
        enable_prefetch: bool = True,
    ):
        self.emb_model = embedding_model
        self.judger = judger
        self.capacity = capacity
        self.enable_prefetch = enable_prefetch

        self.sine = SineIndex(
            embedding_model.dim, judger, tau_sim=tau_sim, tau_lsm=tau_lsm
        )
        self.ses: Dict[str, SemanticElement] = {}   # str(faiss_id) → SE
        self.prefetcher = MarkovPrefetcher()

        # Metrics
        self.total_api_cost: float = 0.0
        self.total_api_calls: int = 0
        self.total_cache_hits: int = 0
        self.latency_log: List[float] = []
        self._prefetch_store: Dict[str, str] = {}

    # ── lookup ────────────────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        ann_only: bool = False,
    ) -> Tuple[Optional[str], dict]:
        """
        Input:  query (str), ann_only (bool — skip judger for ablation)
        Output: (cached_answer | None, debug_dict)
        """
        t0 = time.perf_counter()
        vec = self.emb_model.encode_one(query)

        # Check prefetch store first
        prefetch_hit = self._prefetch_store.pop(query.strip().lower(), None)
        if prefetch_hit is not None:
            cache_ms = (time.perf_counter() - t0) * 1000
            self.total_cache_hits += 1
            self.latency_log.append(cache_ms)
            return prefetch_hit, {
                "hit": True,
                "source": "prefetch",
                "cache_lookup_ms": round(cache_ms, 2),
            }

        se, debug = self.sine.lookup(query, vec, ann_only=ann_only)
        cache_ms = (time.perf_counter() - t0) * 1000
        debug["cache_lookup_ms"] = round(cache_ms, 2)

        if se is not None:
            self.total_cache_hits += 1
            self.latency_log.append(cache_ms)

            # Markov prefetching — triggered on every confirmed HIT
            if self.enable_prefetch:
                self.prefetcher.observe(query)
                for predicted_q, prob in self.prefetcher.predict(query):
                    if prob < self.prefetcher.theta:
                        break
                    pred_vec = self.emb_model.encode_one(predicted_q)
                    pred_se, _ = self.sine.lookup(predicted_q, pred_vec)
                    if pred_se is None:
                        debug["prefetch_triggered"] = predicted_q
                    break

            return se.answer, debug

        return None, debug

    # ── insert ────────────────────────────────────────────────────────────

    def insert(
        self,
        query: str,
        answer: str,
        cost: float = DEFAULT_CONFIG.remote_cost_per_call,
        latency_ms: float = DEFAULT_CONFIG.remote_latency_ms,
    ):
        """
        Insert a new SE after a real API call.
        Staticity predicted by judger. Volatile SEs are auto-discarded.

        Input:  query, answer, cost, latency_ms
        Output: None (SE stored internally or discarded)
        """
        self.total_api_cost += cost
        self.total_api_calls += 1
        self.latency_log.append(latency_ms)

        # Step 1: Predict staticity
        staticity = self.judger.staticity_score(query, answer)

        # Step 2: Discard volatile data
        if staticity <= DEFAULT_CONFIG.staticity_volatile:
            return

        # Step 3: TTL proportional to staticity
        ttl_seconds = 3600.0 * (staticity / 10.0) * 24 * 30

        # Step 4: Evict if needed, then insert
        self._evict_if_needed()

        vec = self.emb_model.encode_one(query)
        se = SemanticElement(
            query=query,
            answer=answer,
            embedding=vec,
            cost=cost,
            latency_ms=latency_ms,
            staticity=staticity,
            ttl_seconds=ttl_seconds,
        )
        fid = self.sine.add(se)
        self.ses[str(fid)] = se

    # ── LCFU eviction (Algorithm 2) ──────────────────────────────────────

    def _evict_if_needed(self):
        # Remove expired
        expired = [k for k, se in self.ses.items() if se.is_expired]
        for k in expired:
            self.sine.remove(self.ses[k])
            del self.ses[k]
        if expired:
            self.sine.rebuild(list(self.ses.values()))

        # Evict by LCFU score
        while len(self.ses) >= self.capacity:
            victim_key = min(self.ses, key=lambda k: self.ses[k].lcfu_score)
            self.sine.remove(self.ses[victim_key])
            del self.ses[victim_key]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats_summary(self) -> dict:
        total = self.total_cache_hits + self.sine.stats["misses"]
        return {
            "cache_hits": self.total_cache_hits,
            "cache_misses": self.sine.stats["misses"],
            "hit_rate_%": round(self.total_cache_hits / max(1, total) * 100, 1),
            "api_calls": self.total_api_calls,
            "api_cost_$": round(self.total_api_cost, 4),
            "ses_in_cache": len(self.ses),
        }


# ── Eviction policy ablations ────────────────────────────────────────────────

class LRUSemanticCache(AsteriaCache):
    """LRU eviction instead of LCFU."""

    def _evict_if_needed(self):
        expired = [k for k, se in self.ses.items() if se.is_expired]
        for k in expired:
            self.sine.remove(self.ses[k])
            del self.ses[k]
        while len(self.ses) >= self.capacity:
            victim = min(self.ses, key=lambda k: self.ses[k].created_at)
            self.sine.remove(self.ses[victim])
            del self.ses[victim]
        if expired:
            self.sine.rebuild(list(self.ses.values()))


class LFUSemanticCache(AsteriaCache):
    """LFU eviction instead of LCFU."""

    def _evict_if_needed(self):
        expired = [k for k, se in self.ses.items() if se.is_expired]
        for k in expired:
            self.sine.remove(self.ses[k])
            del self.ses[k]
        while len(self.ses) >= self.capacity:
            victim = min(self.ses, key=lambda k: self.ses[k].frequency)
            self.sine.remove(self.ses[victim])
            del self.ses[victim]
        if expired:
            self.sine.rebuild(list(self.ses.values()))
