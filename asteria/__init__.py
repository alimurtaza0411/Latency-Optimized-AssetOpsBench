"""
Asteria — Semantic-Aware Cross-Region Caching for Agentic LLM Tool Access.

Based on arXiv:2509.17360 (Ruan et al., 2025).

Quick start:
    from asteria import EmbeddingModel, SemanticJudger, AsteriaCache

    emb = EmbeddingModel()               # Qwen3-Embedding-0.6B
    judger = SemanticJudger()             # Qwen3-Reranker-0.6B
    cache = AsteriaCache(emb, judger)     # full Asteria cache

    # Lookup (returns (answer|None, debug_dict))
    answer, debug = cache.lookup("Who painted the Mona Lisa?")

    # Insert on miss
    if answer is None:
        cache.insert("Who painted the Mona Lisa?",
                     "Leonardo da Vinci painted the Mona Lisa.",
                     cost=0.005, latency_ms=320)
"""

from .config import AsteriaConfig, DEFAULT_CONFIG
from .integrations.assetops import QueryIntentCache

# Optional heavy imports (sentence-transformers/torch/faiss stack).
# Keep lightweight integrations importable even when those deps are absent.
try:  # pragma: no cover - availability depends on local environment
    from .cache import (
        AsteriaCache,
        ExactMatchCache,
        LFUSemanticCache,
        LRUSemanticCache,
        MarkovPrefetcher,
    )
    from .embedding_model import EmbeddingModel
    from .experiments import check_accuracy, run_experiment, simulate_remote_api
    from .semantic_element import SemanticElement
    from .semantic_judger import SemanticJudger
    from .sine_index import SineIndex
    from .workload import (
        QA_KNOWLEDGE_BASE,
        make_bursty_workload,
        make_sequential_workload,
        make_zipfian_workload,
    )
    _HEAVY_IMPORTS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _HEAVY_IMPORTS_AVAILABLE = False

__all__ = [
    "AsteriaConfig",
    "DEFAULT_CONFIG",
    "QueryIntentCache",
]

if _HEAVY_IMPORTS_AVAILABLE:
    __all__.extend(
        [
            "SemanticElement",
            "EmbeddingModel",
            "SemanticJudger",
            "SineIndex",
            "AsteriaCache",
            "ExactMatchCache",
            "LRUSemanticCache",
            "LFUSemanticCache",
            "MarkovPrefetcher",
            "run_experiment",
            "check_accuracy",
            "simulate_remote_api",
            "make_zipfian_workload",
            "make_bursty_workload",
            "make_sequential_workload",
            "QA_KNOWLEDGE_BASE",
        ]
    )
