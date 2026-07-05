from __future__ import annotations
from typing import List, Dict, Literal

from src.document_ingestion import load_documents_from_folder, load_huggingface_dataset
from src.text_chunking import chunk_documents
from src.embeddings import get_embedder
from src.vector_store import VectorStore
from src.retrieval import Retriever
from src.generation import get_generator, build_prompt


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        offline_mode: bool = False,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generation_model_name: str = "google/flan-t5-base",
        generation_backend: str = "local",
        cohere_api_key: str | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = get_embedder(offline_mode=offline_mode, model_name=embedding_model_name)
        self.generator = get_generator(
            offline_mode=offline_mode,
            model_name=generation_model_name,
            backend=generation_backend,
            cohere_api_key=cohere_api_key,
        )
        self.vector_store: VectorStore | None = None
        self.retriever: Retriever | None = None
        self.chunks: List[Dict] = []

    def ingest_folder(self, folder_path: str) -> None:
        documents = load_documents_from_folder(folder_path)
        self._build_index(documents)

    def ingest_huggingface_dataset(self, dataset_name: str, text_column: str, max_docs: int = 200) -> None:
        documents = load_huggingface_dataset(dataset_name, text_column=text_column, max_docs=max_docs)
        self._build_index(documents)

    def _build_index(self, documents: List[Dict[str, str]]) -> None:
        print(f"[1/4] Loaded {len(documents)} document(s)")

        self.chunks = chunk_documents(documents, self.chunk_size, self.chunk_overlap)
        print(f"[2/4] Split into {len(self.chunks)} chunks")

        texts = [c["text"] for c in self.chunks]
        vectors = self.embedder.encode(texts)
        print(f"[3/4] Created {vectors.shape[0]} embeddings (dim={vectors.shape[1]})")

        self.vector_store = VectorStore(dimension=vectors.shape[1])
        self.vector_store.add(vectors, self.chunks)
        self.retriever = Retriever(self.vector_store, self.embedder, self.chunks)
        print("[4/4] Vector store ready")

    _SUMMARY_TRIGGERS = (
        "summarize", "summarise", "summary", "tl;dr", "tldr",
        "overview", "what is this document about", "what's this about",
    )

    def _is_summary_request(self, question: str) -> bool:
        q = question.lower()
        return any(trigger in q for trigger in self._SUMMARY_TRIGGERS)

    def _sample_chunks_for_summary(self, top_k: int) -> List[Dict]:
        n = len(self.chunks)
        if n <= top_k:
            return list(self.chunks)
        step = n / top_k
        indices = [int(i * step) for i in range(top_k)]
        return [self.chunks[i] for i in indices]

    def ask(
        self,
        question: str,
        top_k: int = 4,
        mode: Literal["vector", "hybrid"] = "vector",
        rerank: bool = False,
        return_context: bool = False,
    ):
        if self.retriever is None:
            raise RuntimeError("Call ingest_folder() or ingest_huggingface_dataset() first.")

        is_summary = self._is_summary_request(question)
        if is_summary:
            retrieved_chunks = self._sample_chunks_for_summary(top_k=max(top_k, 6))
        else:
            retrieved_chunks = self.retriever.retrieve(question, top_k=top_k, mode=mode, rerank=rerank)

        answer = self.generator.generate(question, retrieved_chunks, is_summary=is_summary)

        if return_context:
            return {
                "answer": answer,
                "context": retrieved_chunks,
                "prompt": build_prompt(question, retrieved_chunks),
            }
        return answer

    def save_index(self, folder_path: str) -> None:
        if self.vector_store is None:
            raise RuntimeError("No index built yet.")
        self.vector_store.save(folder_path)

    def load_index(self, folder_path: str) -> None:
        self.vector_store = VectorStore.load(folder_path)
        self.chunks = self.vector_store.metadata
        self.retriever = Retriever(self.vector_store, self.embedder, self.chunks)