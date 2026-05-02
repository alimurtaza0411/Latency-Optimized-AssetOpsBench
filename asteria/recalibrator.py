"""
Algorithm 1 — Periodic τ_lsm threshold recalibration  (§4.2)

Offline calibration of the SemanticJudger threshold τ_lsm using a
labelled validation set of (query, candidate_answer, is_correct) triples.

Inputs (per call to record()):
    query             : str          — incoming query
    candidate_answer  : str          — cached answer being judged
    judger_score      : float [0,1]  — output of SemanticJudger.score()
    is_correct        : bool         — ground-truth label for this pair

Outputs:
    find_threshold(target_precision)  → float | None
        — Smallest τ such that observed precision over kept entries
          (score ≥ τ) is ≥ target_precision.  Returns None if no τ
          achieves target.

    precision_curve()  → list[(threshold, precision)]
        — Full sweep, descending by judger score.  Useful for plotting.

The recalibrator does NOT mutate any cache.  Callers consume the
suggested τ_lsm and update AsteriaConfig / SineIndex.tau_lsm
out of band.

This implementation diverges from the paper in one detail: the paper
samples online ground-truth via FetchGT(q).  Here we expect callers to
supply labels offline (from a paraphrase validation set or expert
labelling pass).  Both flows feed the same scoring math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CalibrationEntry:
    query:            str
    candidate_answer: str
    judger_score:     float
    is_correct:       bool


class Recalibrator:
    def __init__(self, target_precision: float = 0.95):
        self.target_precision = target_precision
        self.log: List[CalibrationEntry] = []

    # ── recording ─────────────────────────────────────────────────────────

    def record(
        self,
        query: str,
        candidate_answer: str,
        judger_score: float,
        is_correct: bool,
    ) -> None:
        self.log.append(
            CalibrationEntry(
                query=query,
                candidate_answer=candidate_answer,
                judger_score=judger_score,
                is_correct=is_correct,
            )
        )

    # ── Algorithm 1: threshold search ────────────────────────────────────

    def find_threshold(
        self,
        target_precision: Optional[float] = None,
    ) -> Optional[float]:
        """
        Smallest τ such that precision over entries with score ≥ τ
        is still ≥ target_precision.

        Sweep observed scores in descending order.  At each rank i,
        kept = entries[:i+1] = entries with score ≥ entries[i].score.
        Precision over kept = (# correct in kept) / |kept|.

        Returns the smallest τ at which precision is still ≥ target.
        If precision never reaches target, returns None.
        """
        target = target_precision if target_precision is not None else self.target_precision
        if not self.log:
            return None

        sorted_log = sorted(self.log, key=lambda e: -e.judger_score)
        best: Optional[float] = None
        tp = 0
        for i, entry in enumerate(sorted_log, start=1):
            if entry.is_correct:
                tp += 1
            precision = tp / i
            if precision >= target:
                best = entry.judger_score
        return best

    def precision_curve(self) -> List[Tuple[float, float, int, int]]:
        """
        Full descending sweep — returns [(τ, precision, kept, tp)].

        kept   = number of entries with score ≥ τ
        tp     = number of correct entries among kept
        """
        if not self.log:
            return []
        sorted_log = sorted(self.log, key=lambda e: -e.judger_score)
        out: List[Tuple[float, float, int, int]] = []
        tp = 0
        for i, entry in enumerate(sorted_log, start=1):
            if entry.is_correct:
                tp += 1
            precision = tp / i
            out.append((entry.judger_score, precision, i, tp))
        return out

    def summary(self, target_precision: Optional[float] = None) -> dict:
        target = target_precision if target_precision is not None else self.target_precision
        tau = self.find_threshold(target)
        n_pos = sum(1 for e in self.log if e.is_correct)
        n_neg = sum(1 for e in self.log if not e.is_correct)
        return {
            "target_precision": target,
            "suggested_tau_lsm": tau,
            "n_entries": len(self.log),
            "n_positives": n_pos,
            "n_negatives": n_neg,
        }
