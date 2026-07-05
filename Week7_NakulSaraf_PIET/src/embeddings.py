from __future__ import annotations
from typing import List
import numpy as np


class SentenceTransformerEmbedder:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer 
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


class HashingEmbedder:


    def __init__(self, dimension: int = 384):
        self.model_name = "hashing-bag-of-words (offline fallback, no real semantics)"
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype="float32")
        for word in text.lower().split():
            idx = hash(word) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts])


def get_embedder(offline_mode: bool = False, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if offline_mode:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name=model_name)

