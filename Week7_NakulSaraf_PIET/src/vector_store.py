from __future__ import annotations
import json
import os
from typing import List, Dict, Tuple

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict] = [] 

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        assert vectors.shape[0] == len(metadata), "vectors/metadata length mismatch"
        assert vectors.shape[1] == self.dimension, (
            f"Embedding dimension mismatch: index expects {self.dimension}, "
            f"got {vectors.shape[1]}"
        )
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, folder_path: str) -> None:
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, folder_path: str) -> "VectorStore":
        index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        store = cls(dimension=index.d)
        store.index = index
        store.metadata = metadata
        return store
