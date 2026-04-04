"""
Semantic Judger — Qwen/Qwen3-Reranker-0.6B  (§4.1, §4.2)

Two roles:
    Role 1 — Relevance scoring (query time):
        Input:  score(new_query: str, cached_answer: str)
        Output: float [0,1] — P(cached answer sufficiently answers new query)
        Cache hit confirmed only if output ≥ τ_lsm.

    Role 2 — Staticity scoring (insertion time):
        Input:  staticity_score(query: str, answer: str)
        Output: float [1,10] — how time-invariant the answer is
        SEs with score ≤ STATICITY_VOLATILE are NOT inserted.

Both use prefill-only inference (single forward pass, no generation).
The empty <think></think> block suppresses chain-of-thought.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class SemanticJudger:

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16
        )
        self.model.eval()

        # Official model card: lowercase yes/no tokens
        self.token_yes = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
        self.token_no = self.tokenizer.encode("no", add_special_tokens=False)[-1]

        # Empty <think> block forces next-token = yes/no
        self._suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        print(f"Loaded judger: {model_name}")
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
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_logits = outputs.logits[:, -1, :]
        yes_logit = last_logits[:, self.token_yes]
        no_logit = last_logits[:, self.token_no]
        pair_probs = F.softmax(torch.stack([no_logit, yes_logit], dim=1), dim=-1)
        return float(pair_probs[0, 1].item())

    # ── Role 1: Relevance scoring (query time) ───────────────────────────

    def score(self, new_query: str, cached_answer: str) -> float:
        """
        P(yes) that cached_answer sufficiently answers new_query.
        Called during Sine Stage 2 for every ANN candidate.

        Input:  new_query (str), cached_answer (str)
        Output: float [0,1]
        """
        instruction = (
            "Given a cached IoT agent answer, does it sufficiently answer "
            "the new query, even if the wording differs?"
        )
        prompt = self._build_prompt(instruction, new_query, cached_answer)
        return self._yes_prob(prompt)

    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Batch scoring — more efficient when |candidates| > 1.

        Input:  list of (query, cached_answer) tuples
        Output: list of float [0,1] scores
        """
        if not pairs:
            return []
        instruction = (
            "Given a cached IoT agent answer, does it sufficiently answer "
            "the new query, even if the wording differs?"
        )
        prompts = [self._build_prompt(instruction, q, a) for q, a in pairs]
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt",
            truncation=True, max_length=512
        )
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
