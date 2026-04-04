"""
Asteria configuration — all tuneable parameters.

Paper: Qwen3-Embedding-0.6B (1024-dim) + Qwen3-Reranker-0.6B
"""

from dataclasses import dataclass


@dataclass
class AsteriaConfig:
    # Embedding
    embedding_dim: int = 1024           # Qwen3-Embedding-0.6B output dimension

    # ANN stage  (§4.2)
    tau_sim: float = 0.75               # τ_sim — coarse ANN threshold
    ann_top_k: int = 5                  # max ANN candidates forwarded to judger

    # Judger stage  (§4.2)
    tau_lsm: float = 0.80              # τ_lsm — semantic judger confidence threshold

    # Staticity  (§4.1)
    staticity_volatile: float = 2.0    # SEs with staticity ≤ this are NOT cached

    # Cache  (§4.3)
    cache_capacity: int = 50           # max number of SEs in cache
    default_ttl: float = 3600.0        # default TTL in seconds

    # Prefetcher  (Algorithm 3)
    markov_theta: float = 0.30         # prefetch confidence threshold

    # Remote API simulation
    remote_latency_ms: float = 350.0   # simulated cross-region latency
    remote_cost_per_call: float = 0.005  # $ per API call

    # Reproducibility
    seed: int = 42


# Singleton default config
DEFAULT_CONFIG = AsteriaConfig()
