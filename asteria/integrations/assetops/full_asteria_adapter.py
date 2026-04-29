"""Query-level Asteria helpers for AssetOpsBench profiling."""

from __future__ import annotations

from typing import Any

from asteria.config import DEFAULT_CONFIG


def build_asteria_cache_stack(
    capacity: int | None = None,
    tau_sim: float | None = None,
    tau_lsm: float | None = None,
    enable_prefetch: bool = True,
    enable_temporal_bucketing: bool | None = None,
) -> Any:
    """Construct :class:`asteria.cache.AsteriaCache` with default paper models.

    Raises ImportError with a short message if optional deps (torch, faiss, etc.) are missing.
    """
    try:
        from asteria.cache import AsteriaCache
        from asteria.embedding_model import EmbeddingModel
        from asteria.semantic_judger import SemanticJudger
    except ImportError as e:
        raise ImportError(
            "Full Asteria requires optional dependencies (e.g. torch, sentence-transformers, "
            "faiss-cpu, transformers). Install them in this environment, then retry."
        ) from e

    cap = capacity if capacity is not None else DEFAULT_CONFIG.cache_capacity
    ts = tau_sim if tau_sim is not None else DEFAULT_CONFIG.tau_sim
    tl = tau_lsm if tau_lsm is not None else DEFAULT_CONFIG.tau_lsm
    etb = enable_temporal_bucketing if enable_temporal_bucketing is not None else DEFAULT_CONFIG.enable_temporal_bucketing
    emb = EmbeddingModel()
    judger = SemanticJudger()
    return AsteriaCache(
        emb,
        judger,
        capacity=cap,
        tau_sim=ts,
        tau_lsm=tl,
        enable_prefetch=enable_prefetch,
        enable_temporal_bucketing=etb,
    )


def compose_stored_answer_from_steps(ordered: list, context: dict) -> str:
    """Flatten step results into one string for query-level Asteria insert."""
    lines: list[str] = []
    for step in ordered:
        r = context.get(step.step_number)
        if r is None:
            continue
        if r.success:
            lines.append(f"Step {r.step_number} — {r.task}: {r.response}")
        else:
            lines.append(f"Step {r.step_number} — ERROR: {r.error}")
    return "\n".join(lines)
