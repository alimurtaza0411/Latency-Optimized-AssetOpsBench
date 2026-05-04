"""
Semantic Judger — Qwen/Qwen3-Reranker-0.6B  (§4.1, §4.2)

Three roles:
    Role 1 — Relevance scoring (query time):
        Input:  score(new_query, cached_answer, temporal_ctx?)
        Output: float [0,1] — P(cached answer sufficiently answers new query)
        Cache hit confirmed only if output ≥ τ_lsm.
        When temporal context is provided, the instruction is enriched with
        bucket type, time windows, and cached-answer age so the model makes
        a unified semantic + temporal judgement.

    Role 2 — Staticity scoring (insertion time):
        Input:  staticity_score(query: str, answer: str)
        Output: float [1,10] — how time-invariant the answer is
        SEs with score ≤ STATICITY_VOLATILE are NOT inserted.

Both use prefill-only inference (single forward pass, no generation).
The empty <think></think> block suppresses chain-of-thought.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Temporal context passed from cache → Sine → Judger ───────────────────────

@dataclass
class TemporalContext:
    """Temporal metadata provided per (query, candidate) pair.

    Created by AsteriaCache.lookup() from the query's TemporalTag and
    the candidate SemanticElement's stored temporal fields.
    """
    query_bucket: str                        # "STATIC", "RELATIVE", or "ANCHORED"  (VOLATILE bypasses cache)
    query_window_start: Optional[str] = None # ISO str, ANCHORED only
    query_window_end: Optional[str] = None   # ISO str, ANCHORED only
    cached_bucket: str = "STATIC"
    cached_window_start: Optional[str] = None
    cached_window_end: Optional[str] = None
    cached_created_at: Optional[float] = None  # epoch seconds


class SemanticJudger:

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            from .config import DEFAULT_CONFIG
            model_name = DEFAULT_CONFIG.judger_model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16
        )
        self.model.eval()

        # Pick the best available device (CUDA > MPS > CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

        # Official model card: lowercase yes/no tokens
        self.token_yes = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
        self.token_no = self.tokenizer.encode("no", add_special_tokens=False)[-1]

        # Empty <think> block forces next-token = yes/no
        self._suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        print(f"Loaded judger: {model_name}  device={self.device}")
        print(f"  yes token id : {self.token_yes}")
        print(f"  no  token id : {self.token_no}")

    # ── Internal ──────────────────────────────────────────────────────────

    def _build_prompt(self, instruction: str, query: str, document: str) -> str:
        return (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the "
            "Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".\n'
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
            + self._suffix
        )

    def _yes_prob(self, prompt: str) -> float:
        """Single forward pass → P(yes) via softmax over (no, yes) logits."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_logits = outputs.logits[:, -1, :]
        yes_logit = last_logits[:, self.token_yes]
        no_logit = last_logits[:, self.token_no]
        pair_probs = F.softmax(torch.stack([no_logit, yes_logit], dim=1), dim=-1)
        return float(pair_probs[0, 1].item())

    # ── Temporal instruction builder ───────────────────────────────────────

    @staticmethod
    def _build_temporal_instruction(ctx: Optional[TemporalContext] = None) -> str:
        """Build a relevance-scoring instruction, enriched with temporal
        context when available.

        STATIC:
            Neutral — returns the vanilla base prompt.

        ANCHORED (Time-Bounded):
            Explicitly state both time windows and the cached-answer
            timestamp so the model can compare them.  The model must
            say 'no' when the windows don't align.
        """
        base = (
            "Given a cached IoT agent answer, does it sufficiently answer "
            "the new query, even if the wording differs?"
        )
        if ctx is None:
            return base

        # ── STATIC ──────────────────────────────────────────────────
        # Neutral: use the vanilla prompt.
        if ctx.query_bucket == "STATIC":
            return base

        # ── ANCHORED (Time-Bounded) ─────────────────────────────────
        if ctx.query_bucket == "ANCHORED":
            parts = [base]
            parts.append(
                " This is a time-bounded historical data query. "
                "The cached answer is ONLY valid if the time window "
                "it covers matches the time window requested by the new query."
            )

            if ctx.query_window_start and ctx.query_window_end:
                parts.append(
                    f" Requested time window: {ctx.query_window_start} "
                    f"to {ctx.query_window_end}."
                )
            elif ctx.query_window_start:
                parts.append(
                    f" Requested start time: {ctx.query_window_start}."
                )

            if ctx.cached_window_start and ctx.cached_window_end:
                parts.append(
                    f" Cached answer time window: {ctx.cached_window_start} "
                    f"to {ctx.cached_window_end}."
                )
            elif ctx.cached_window_start:
                parts.append(
                    f" Cached answer start time: {ctx.cached_window_start}."
                )

            if ctx.cached_created_at is not None:
                ts = datetime.datetime.fromtimestamp(
                    ctx.cached_created_at, tz=datetime.timezone.utc
                ).isoformat()
                parts.append(f" Cached answer was stored at {ts}.")

            parts.append(
                " Answer 'yes' ONLY if the cached answer covers the EXACT "
                "same time window as the new query AND the semantic meaning "
                "matches. If the time windows differ, answer 'no'."
            )
            return "".join(parts)

        # Fallback (should not reach here in normal flow).
        return base

    # ── Role 1: Relevance scoring (query time) ───────────────────────────

    def score(
        self,
        new_query: str,
        cached_answer: str,
        temporal_ctx: Optional[TemporalContext] = None,
    ) -> float:
        """
        P(yes) that cached_answer sufficiently answers new_query.
        Called during Sine Stage 2 for every ANN candidate.

        Input:  new_query (str), cached_answer (str), temporal_ctx (optional)
        Output: float [0,1]
        """
        instruction = self._build_temporal_instruction(temporal_ctx)
        prompt = self._build_prompt(instruction, new_query, cached_answer)
        return self._yes_prob(prompt)

    def score_batch(
        self,
        pairs: List[Tuple[str, str]],
        temporal_ctxs: Optional[List[Optional[TemporalContext]]] = None,
    ) -> List[float]:
        """
        Batch scoring — more efficient when |candidates| > 1.

        Input:  list of (query, cached_answer) tuples,
                optional list of TemporalContext per pair
        Output: list of float [0,1] scores
        """
        if not pairs:
            return []

        if temporal_ctxs is None:
            temporal_ctxs = [None] * len(pairs)

        prompts = []
        for (q, a), ctx in zip(pairs, temporal_ctxs):
            instruction = self._build_temporal_instruction(ctx)
            prompts.append(self._build_prompt(instruction, q, a))

        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt",
            truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_logits = outputs.logits[:, -1, :]
        yes_logits = last_logits[:, self.token_yes]
        no_logits = last_logits[:, self.token_no]
        pair_probs = F.softmax(
            torch.stack([no_logits, yes_logits], dim=1), dim=-1
        )
        return pair_probs[:, 1].tolist()

    # ── Role 2: Staticity scoring (insertion time) ───────────────────────

    def staticity_score(self, query: str, answer: str) -> float:
        """
        Estimate time-stability of the answer. P(yes) scaled to [1,10].

        Input:  query (str), answer (str)
        Output: float [1.0, 10.0]
        """
        instruction = (
            "Is this answer a stable, time-invariant fact that will remain "
            "correct for months or years? "
            "Answer yes for permanent or slowly-changing facts "
            "(e.g. asset configurations, failure mode mappings, site metadata). "
            "Answer no for answers that change frequently "
            "(e.g. live sensor readings, current stock prices, today's weather)."
        )
        prompt = self._build_prompt(instruction, query, answer)
        prob_stable = self._yes_prob(prompt)
        return round(1.0 + prob_stable * 9.0, 2)
