from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
from src.rag_pipeline import RAGPipeline


@dataclass
class TestQuestion:
    question: str
    expected_source: Optional[str] = None  



def generate_test_questions(pipeline):
    grouped = defaultdict(list)

    for chunk in pipeline.chunks:
        grouped[chunk["source"]].append(chunk)

    test_questions = []

    for source, chunks in grouped.items():

        preview = chunks[0]["text"][:300].lower()
        questions = [
            f"What is the document '{source}' about?",
            f"Summarize '{source}'.",
            f"What are the main topics in '{source}'?",
            f"What important information does '{source}' contain?",
            f"Explain the key points of '{source}'."
        ]

        for q in questions:
            test_questions.append(
                TestQuestion(
                    question=q,
                    expected_source=source
                )
            )

    return test_questions

@dataclass
class ValidationResult:
    question: str
    answer: str
    retrieved: List[Dict]
    expected_source: Optional[str]
    found_expected: Optional[bool]
    rank_of_expected: Optional[int]
    latency_seconds: float


def run_validation(
    pipeline: RAGPipeline,
    test_questions: List[TestQuestion],
    top_k: int = 4,
    mode: str = "vector",
    rerank: bool = False,
) -> List[ValidationResult]:
    results: List[ValidationResult] = []

    for tq in test_questions:
        start = time.time()
        output = pipeline.ask(tq.question, top_k=top_k, mode=mode, rerank=rerank, return_context=True)
        elapsed = time.time() - start

        found_expected = None
        rank_of_expected = None
        if tq.expected_source:
            for i, chunk in enumerate(output["context"]):
                if tq.expected_source.lower() in chunk["source"].lower():
                    found_expected = True
                    rank_of_expected = i + 1  
                    break
            if found_expected is None:
                found_expected = False

        results.append(
            ValidationResult(
                question=tq.question,
                answer=output["answer"],
                retrieved=output["context"],
                expected_source=tq.expected_source,
                found_expected=found_expected,
                rank_of_expected=rank_of_expected,
                latency_seconds=round(elapsed, 3),
            )
        )
    return results


def compute_retrieval_metrics(results: List[ValidationResult]) -> Dict:
    labeled = [r for r in results if r.expected_source is not None]
    if not labeled:
        return {"labeled_questions": 0, "note": "No expected_source labels provided; no accuracy metrics computed."}

    hits = sum(1 for r in labeled if r.found_expected)
    reciprocal_ranks = [1.0 / r.rank_of_expected for r in labeled if r.found_expected]
    mrr = sum(reciprocal_ranks) / len(labeled) if labeled else 0.0

    return {
        "labeled_questions": len(labeled),
        "hit_rate": round(hits / len(labeled), 3),
        "hits": hits,
        "mean_reciprocal_rank": round(mrr, 3),
    }


def write_validation_log(
    results: List[ValidationResult],
    metrics: Dict,
    output_path: str = "logs/validation_log.md",
) -> None:
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = ["# RAG Pipeline Validation Log", ""]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Retrieval Accuracy Summary")
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Per-Question Results")

    for i, r in enumerate(results, 1):
        lines.append(f"\n### Q{i}: {r.question}")
        lines.append(f"- Latency: {r.latency_seconds}s")
        if r.expected_source:
            lines.append(f"- Expected source: `{r.expected_source}` | Found: {r.found_expected} | Rank: {r.rank_of_expected}")
        lines.append(f"- **Answer**: {r.answer}")
        lines.append("- Retrieved chunks:")
        for c in r.retrieved:
            preview = c["text"][:120].replace("\n", " ")
            lines.append(f"  - `{c['source']}` chunk {c['chunk_id']} (score={c.get('score', 0):.3f}): {preview}...")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = output_path.rsplit(".", 1)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "results": [
                    {
                        "question": r.question,
                        "answer": r.answer,
                        "expected_source": r.expected_source,
                        "found_expected": r.found_expected,
                        "rank_of_expected": r.rank_of_expected,
                        "latency_seconds": r.latency_seconds,
                        "retrieved": r.retrieved,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )

    print(f"Validation log written to {output_path} and {json_path}")
