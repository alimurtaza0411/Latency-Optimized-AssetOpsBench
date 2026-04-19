"""
Sine — Semantic Retrieval Index  (§4.2)

Two-stage lookup pipeline:
    Stage 1 (coarse): FAISS IndexFlatIP → candidates with sim ≥ τ_sim
    Stage 2 (fine):   SemanticJudger.score_batch() → confirmed hit if ≥ τ_lsm

Input:
    lookup(query: str, query_vec: np.ndarray, ann_only: bool)
Output:
    (matched_se: SemanticElement | None, debug: dict)

    debug keys: ann_candidates, judger_scores, hit, cache_lookup_ms
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import DEFAULT_CONFIG
from .semantic_element import SemanticElement
from .semantic_judger import SemanticJudger, TemporalContext


class SineIndex:

    def __init__(
        self,
        dim: int,
        judger: SemanticJudger,
        tau_sim: float = DEFAULT_CONFIG.tau_sim,
        tau_lsm: float = DEFAULT_CONFIG.tau_lsm,
        top_k: int = DEFAULT_CONFIG.ann_top_k,
    ):
        self.dim = dim
        self.judger = judger
        self.tau_sim = tau_sim
        self.tau_lsm = tau_lsm
        self.top_k = top_k

        self.index = faiss.IndexFlatIP(dim)
        self._id_to_se: Dict[int, SemanticElement] = {}
        self._next_id = 0

        self.stats = {
            "ann_candidates_total": 0,
            "judger_calls": 0,
            "hits": 0,
            "misses": 0,
            "ann_only_hits": 0,
        }

    # ── Index management ─────────────────────────────────────────────────

    def add(self, se: SemanticElement) -> int:
        vec = se.embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)
        fid = self._next_id
        se.faiss_id = fid
        self._id_to_se[fid] = se
        self._next_id += 1
        return fid

    def remove(self, se: SemanticElement):
        if se.faiss_id in self._id_to_se:
            del self._id_to_se[se.faiss_id]

    def rebuild(self, ses: List[SemanticElement]):
        self.index = faiss.IndexFlatIP(self.dim)
        self._id_to_se = {}
        self._next_id = 0
        for se in ses:
            self.add(se)

    # ── Two-stage lookup ─────────────────────────────────────────────────

    def lookup(
        self,
        query: str,
        query_vec: np.ndarray,
        ann_only: bool = False,
        query_temporal_tag: object | None = None,
    ) -> Tuple[Optional[SemanticElement], dict]:
        """
        Returns (matched_se, debug_info).  matched_se is None on a miss.

        Parameters
        ----------
        query_temporal_tag : TemporalTag | None
            If provided, temporal context is built per candidate and
            forwarded to the judger so it can make a unified
            semantic + temporal decision.
        """
        debug = {"ann_candidates": 0, "judger_scores": [], "hit": False}

        if self.index.ntotal == 0:
            self.stats["misses"] += 1
            return None, debug

        # Stage 1: ANN coarse filter
        k = min(self.top_k, self.index.ntotal)
        vec = query_vec.reshape(1, -1).astype(np.float32)
        sims, ids = self.index.search(vec, k)
        sims, ids = sims[0], ids[0]

        candidates = []
        for sim, fid in zip(sims, ids):
            if fid == -1:
                continue
            se = self._id_to_se.get(int(fid))
            if se is None or se.is_expired:
                continue
            if sim >= self.tau_sim:
                candidates.append((sim, se))

        self.stats["ann_candidates_total"] += len(candidates)
        debug["ann_candidates"] = len(candidates)

        if ann_only:
            if candidates:
                best_se = max(candidates, key=lambda x: x[0])[1]
                self.stats["ann_only_hits"] += 1
                debug["hit"] = True
                return best_se, debug
            self.stats["misses"] += 1
            return None, debug

        if not candidates:
            self.stats["misses"] += 1
            return None, debug

        # Stage 2: Semantic Judger — batch score all candidates
        #          with optional temporal context
        pairs = [(query, se.answer) for _, se in candidates]

        temporal_ctxs: list[TemporalContext | None] | None = None
        if query_temporal_tag is not None:
            temporal_ctxs = []
            qb = query_temporal_tag.bucket.value
            qws = None
            qwe = None
            if query_temporal_tag.time_window is not None:
                qws = query_temporal_tag.time_window.start
                qwe = query_temporal_tag.time_window.end
            for _, se in candidates:
                temporal_ctxs.append(
                    TemporalContext(
                        query_bucket=qb,
                        query_window_start=qws,
                        query_window_end=qwe,
                        cached_bucket=se.temporal_bucket,
                        cached_window_start=se.time_window_start,
                        cached_window_end=se.time_window_end,
                        cached_created_at=se.created_at,
                    )
                )

        scores = self.judger.score_batch(pairs, temporal_ctxs=temporal_ctxs)
        self.stats["judger_calls"] += len(pairs)
        debug["judger_scores"] = [round(s, 3) for s in scores]

        best_score, best_se = -1.0, None
        for (sim, se), score in zip(candidates, scores):
            if score >= self.tau_lsm and score > best_score:
                best_score = score
                best_se = se

        if best_se is not None:
            best_se.frequency += 1
            self.stats["hits"] += 1
            debug["hit"] = True
            return best_se, debug

        self.stats["misses"] += 1
        return None, debug
