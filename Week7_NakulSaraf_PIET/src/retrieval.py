from __future__ import annotations
from typing import List, Dict, Literal
import numpy as np
from rank_bm25 import BM25Okapi

from src.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder, chunks: List[Dict]):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query])[0]

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        query_vector = self.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        return [{**meta, "score": score} for meta, score in results]

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top_indices]

    def _merge_hybrid(
        self, vector_results: List[Dict], keyword_results: List[Dict], top_k: int
    ) -> List[Dict]:
        rrf_scores: Dict[str, float] = {}
        combined_lookup: Dict[str, Dict] = {}

        for rank_list in (vector_results, keyword_results):
            for rank, item in enumerate(rank_list):
                key = f"{item['source']}::{item['chunk_id']}"
                combined_lookup[key] = item
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (60 + rank)

        ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [combined_lookup[k] for k in ranked_keys[:top_k]]

    def _rerank_by_keyword_overlap(self, query: str, candidates: List[Dict]) -> List[Dict]:
        query_words = set(query.lower().split())

        def overlap_score(chunk_text: str) -> int:
            return len(query_words & set(chunk_text.lower().split()))

        return sorted(candidates, key=lambda c: overlap_score(c["text"]), reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        mode: Literal["vector", "hybrid"] = "vector",
        rerank: bool = False,
    ) -> List[Dict]:
        pool_size = top_k * 3 if rerank else top_k

        if mode == "vector":
            results = self._vector_search(query, top_k=pool_size)
        elif mode == "hybrid":
            vector_results = self._vector_search(query, top_k=pool_size)
            keyword_results = self._bm25_search(query, top_k=pool_size)
            results = self._merge_hybrid(vector_results, keyword_results, top_k=pool_size)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        if rerank:
            results = self._rerank_by_keyword_overlap(query, results)

        return results[:top_k]
