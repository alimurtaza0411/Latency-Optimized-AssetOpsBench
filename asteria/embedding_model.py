"""
Embedding Model — converts queries to dense vectors for ANN search.

Uses Qwen/Qwen3-Embedding-0.6B (1024-dim, last-token pooling).

Input:
    encode(texts: List[str])  → np.ndarray shape (N, 1024) float32
    encode_one(text: str)     → np.ndarray shape (1024,) float32

Output:
    L2-normalised float32 vectors. Inner product = cosine similarity.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps Qwen3-Embedding-0.6B via the SentenceTransformer interface.

    SentenceTransformer automatically applies last-token pooling (correct
    for decoder-based Qwen3 — NOT CLS pooling).
    Returns L2-normalised float32 vectors of dim=1024.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)
        self.dim = 1024
        print(f"Loaded embedding model: {model_name}  dim={self.dim}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts → (N, 1024) float32 L2-normalised."""
        vecs = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=16,
        )
        return vecs.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text → (1024,) float32 L2-normalised."""
        return self.encode([text])[0]
