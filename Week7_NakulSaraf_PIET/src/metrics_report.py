from __future__ import annotations
import time
from typing import Dict

from src.rag_pipeline import RAGPipeline


def collect_system_metrics(pipeline: RAGPipeline) -> Dict:
    if pipeline.vector_store is None:
        raise RuntimeError("Pipeline has no index built yet — call ingest_folder() first.")

    chunk_lengths = [len(c["text"]) for c in pipeline.chunks]
    embedder = pipeline.embedder
    generator = pipeline.generator

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunking_profile": {
            "chunk_size_chars": pipeline.chunk_size,
            "chunk_overlap_chars": pipeline.chunk_overlap,
            "total_chunks": len(pipeline.chunks),
            "avg_chunk_length_chars": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0,
            "min_chunk_length_chars": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length_chars": max(chunk_lengths) if chunk_lengths else 0,
        },
        "embedding_model": {
            "class": type(embedder).__name__,
            "model_name": getattr(embedder, "model_name", "unknown"),
            "embedding_dimension": embedder.dimension,
        },
        "vector_store": {
            "backend": "FAISS",
            "index_type": type(pipeline.vector_store.index).__name__,
            "similarity_metric": "cosine (via normalized inner product)",
            "vectors_stored": pipeline.vector_store.index.ntotal,
        },
        "generation_model": {
            "class": type(generator).__name__,
            "model_name": getattr(generator, "model_name", "unknown"),
            "max_input_tokens": getattr(generator, "max_input_tokens", "N/A"),
        },
        "document_sources": sorted(set(c["source"] for c in pipeline.chunks)),
    }
    return metrics


def write_metrics_report(pipeline: RAGPipeline, output_path: str = "logs/system_metrics_report.md") -> None:
    import os
    metrics = collect_system_metrics(pipeline)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = ["# RAG System Metrics Report", "", f"Generated: {metrics['generated_at']}", ""]

    lines.append("## Chunking Profile")
    for k, v in metrics["chunking_profile"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Embedding Model")
    for k, v in metrics["embedding_model"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Vector Store")
    for k, v in metrics["vector_store"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Generation Model")
    for k, v in metrics["generation_model"].items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Indexed Document Sources")
    for src in metrics["document_sources"]:
        lines.append(f"- {src}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"System metrics report written to {output_path}")
