
import argparse

from src.rag_pipeline import RAGPipeline
from src.evaluation import (
    generate_test_questions,
    run_validation,
    compute_retrieval_metrics,
    write_validation_log,
)
from src.metrics_report import write_metrics_report

def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline validation + metrics reporting")
    parser.add_argument("--docs", type=str, default="data")
    parser.add_argument("--mode", type=str, default="vector", choices=["vector", "hybrid"])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="No model downloads; tests plumbing only")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    pipeline = RAGPipeline(offline_mode=args.offline)
    pipeline.ingest_folder(args.docs)

    test_questions = generate_test_questions(pipeline)

    print(f"\nRunning {len(test_questions)} validation question(s)...\n")

    results = run_validation(
        pipeline,
        test_questions,
        top_k=args.top_k,
        mode=args.mode,
        rerank=args.rerank
        )

    metrics = compute_retrieval_metrics(results)
    print("Retrieval accuracy metrics:", metrics)
    write_validation_log(results, metrics, output_path=f"{args.log_dir}/validation_log.md")
    write_metrics_report(pipeline, output_path=f"{args.log_dir}/system_metrics_report.md")


if __name__ == "__main__":
    main()
