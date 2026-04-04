"""
Semantic Element (SE) — core caching unit (Figure 5 in the paper).

Inputs at construction:
    query      : str          — the agent's tool-call query
    answer     : str          — the retrieved response
    embedding  : np.ndarray   — dense vector from embedding model

Outputs / accessors:
    se.is_expired  → bool     — whether TTL has elapsed
    se.lcfu_score  → float    — LCFU value score for eviction ranking
    se.summary()   → dict     — human-readable snapshot
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULT_CONFIG


@dataclass
class SemanticElement:
    query:       str
    answer:      str
    embedding:   np.ndarray              # shape (embedding_dim,)

    # Performance metadata
    cost:        float = DEFAULT_CONFIG.remote_cost_per_call
    latency_ms:  float = DEFAULT_CONFIG.remote_latency_ms
    staticity:   float = 5.0             # 1–10: how time-stable is this fact?
    frequency:   int   = 0               # confirmed cache hit count
    size_tokens: int   = 0               # token length of answer
    created_at:  float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_CONFIG.default_ttl

    # Internal FAISS tracking
    faiss_id:    int   = -1

    def __post_init__(self):
        if self.size_tokens == 0:
            self.size_tokens = max(1, len(self.answer.split()))

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def lcfu_score(self) -> float:
        """
        LCFU value score — Algorithm 2 in the paper.

        score = log(freq+1) × log(cost×1000+1) × log(latency+1) × log(staticity+1)
                ─────────────────────────────────────────────────────────────────────
                                        size_tokens
        """
        ttl_remaining = self.ttl_seconds - (time.time() - self.created_at)
        if self.size_tokens == 0 or ttl_remaining <= 0:
            return 0.0
        return (
            np.log1p(self.frequency)
            * np.log1p(self.cost * 1000)
            * np.log1p(self.latency_ms)
            * np.log1p(self.staticity)
        ) / self.size_tokens

    def summary(self) -> dict:
        return {
            "query":      self.query[:60],
            "answer":     self.answer[:80],
            "staticity":  self.staticity,
            "frequency":  self.frequency,
            "cost_$":     self.cost,
            "latency_ms": self.latency_ms,
            "lcfu_score": round(self.lcfu_score, 4),
            "expired":    self.is_expired,
        }
