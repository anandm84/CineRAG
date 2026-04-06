"""RAGAS evaluation runner for CineRAG.

Runs the CineRAG chain on all test questions and evaluates using RAGAS metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are the retrieved chunks relevant?
- Context Recall: Were all necessary chunks retrieved?

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --limit 10       # Only run first 10 questions
    python -m evaluation.run_eval --query-type trend  # Only run trend questions
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from generation.chains import CineRAGChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def load_test_set(
    query_type: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load the curated test set, optionally filtering by query type."""
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    if query_type:
        test_set = [t for t in test_set if t["query_type"] == query_type]

    if limit:
        test_set = test_set[:limit]

    return test_set


def run_chain_on_test_set(test_set: list[dict]) -> list[dict]:
    """Run CineRAGChain on all test questions and collect results.

    Returns a list of result dicts with both the chain output and the test metadata.
    """
    chain = CineRAGChain()

    # Health check
    if not chain.llm.health_check():
        logger.error("Ollama is not available. Start it with: ollama serve")
        logger.error("Then pull the model: ollama pull " + config.LLM_MODEL)
        sys.exit(1)

    results = []
    total = len(test_set)

    for i, test in enumerate(test_set, 1):
        question = test["question"]
        logger.info(f"\n[{i}/{total}] {question}")

        start = time.time()
        try:
            output = chain.run(question)
            elapsed = time.time() - start

            result = {
                "question": question,
                "query_type": test["query_type"],
                "expected_query_type": test["query_type"],
                "detected_query_type": output.get("query_type", ""),
                "ground_truth": test["ground_truth_answer"],
                "answer": output.get("answer", ""),
                "contexts": [s.get("snippet", "") for s in output.get("sources", [])],
                "sources": output.get("sources", []),
                "filters": output.get("filters", {}),
                "relevant_movies": test.get("relevant_movies", []),
                "expected_source_types": test.get("expected_source_types", []),
                "elapsed_seconds": round(elapsed, 1),
                "error": None,
            }

            logger.info(f"  Type: {result['detected_query_type']} | Time: {elapsed:.1f}s")
            logger.info(f"  Answer preview: {result['answer'][:100]}...")

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"  ERROR: {e}")
            result = {
                "question": question,
                "query_type": test["query_type"],
                "expected_query_type": test["query_type"],
                "detected_query_type": "",
                "ground_truth": test["ground_truth_answer"],
                "answer": "",
                "contexts": [],
                "sources": [],
                "filters": {},
                "relevant_movies": test.get("relevant_movies", []),
                "expected_source_types": test.get("expected_source_types", []),
                "elapsed_seconds": round(elapsed, 1),
                "error": str(e),
            }

        results.append(result)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute evaluation metrics from the results.

    Attempts RAGAS evaluation if the library is available and properly configured.
    Falls back to simpler heuristic metrics otherwise.
    """
    metrics = _compute_heuristic_metrics(results)

    # Try RAGAS evaluation
    try:
        ragas_metrics = _compute_ragas_metrics(results)
        metrics["ragas"] = ragas_metrics
    except Exception as e:
        logger.warning(f"RAGAS evaluation skipped: {e}")
        metrics["ragas"] = None
        metrics["ragas_skip_reason"] = str(e)

    return metrics


def _compute_heuristic_metrics(results: list[dict]) -> dict:
    """Compute simple heuristic metrics that don't require RAGAS."""
    total = len(results)
    if total == 0:
        return {}

    errors = sum(1 for r in results if r.get("error"))
    successful = total - errors

    # Query type classification accuracy
    type_correct = sum(
        1 for r in results
        if r["detected_query_type"] == r["expected_query_type"]
    )

    # Source retrieval: did we retrieve from expected source types?
    source_hits = 0
    for r in results:
        expected = set(r.get("expected_source_types", []))
        actual = set(s.get("source_type", "") for s in r.get("sources", []))
        if expected & actual:
            source_hits += 1

    # Relevant movie hits: did retrieved sources include expected movies?
    movie_hits = 0
    for r in results:
        expected = set(m.lower() for m in r.get("relevant_movies", []))
        actual = set(s.get("movie", "").lower() for s in r.get("sources", []))
        if expected & actual:
            movie_hits += 1

    # Average response time
    times = [r["elapsed_seconds"] for r in results if not r.get("error")]
    avg_time = sum(times) / len(times) if times else 0

    # Per-type breakdown
    by_type = {}
    for qtype in ("general", "trend", "pattern", "hit_flop"):
        type_results = [r for r in results if r["query_type"] == qtype]
        if type_results:
            type_errors = sum(1 for r in type_results if r.get("error"))
            type_times = [r["elapsed_seconds"] for r in type_results if not r.get("error")]
            by_type[qtype] = {
                "count": len(type_results),
                "errors": type_errors,
                "avg_time": round(sum(type_times) / len(type_times), 1) if type_times else 0,
            }

    return {
        "total_questions": total,
        "successful": successful,
        "errors": errors,
        "query_type_accuracy": round(type_correct / total, 3) if total else 0,
        "source_type_hit_rate": round(source_hits / total, 3) if total else 0,
        "movie_hit_rate": round(movie_hits / total, 3) if total else 0,
        "avg_response_time_s": round(avg_time, 1),
        "by_type": by_type,
    }


def _compute_ragas_metrics(results: list[dict]) -> dict:
    """Run RAGAS evaluation on the results.

    RAGAS requires an LLM for evaluation. We configure it to use Ollama.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    # Filter to successful results only
    valid = [r for r in results if not r.get("error") and r.get("answer")]
    if not valid:
        raise ValueError("No valid results to evaluate")

    # Build the RAGAS dataset
    data = {
        "question": [r["question"] for r in valid],
        "answer": [r["answer"] for r in valid],
        "contexts": [r["contexts"] for r in valid],
        "ground_truth": [r["ground_truth"] for r in valid],
    }
    dataset = Dataset.from_dict(data)

    # Run RAGAS evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    return {k: round(v, 4) for k, v in result.items()}


def run_evaluation(
    query_type: str | None = None,
    limit: int | None = None,
):
    """Main evaluation entry point."""
    logger.info("=" * 60)
    logger.info("CineRAG Evaluation Runner")
    logger.info("=" * 60)

    # Load test set
    test_set = load_test_set(query_type=query_type, limit=limit)
    logger.info(f"Loaded {len(test_set)} test questions")
    if query_type:
        logger.info(f"Filtered to query type: {query_type}")

    # Run chain
    start = time.time()
    results = run_chain_on_test_set(test_set)
    total_time = time.time() - start

    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = compute_metrics(results)
    metrics["total_eval_time_s"] = round(total_time, 1)

    # Save results
    output = {
        "metrics": metrics,
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {RESULTS_PATH}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total questions:       {metrics.get('total_questions', 0)}")
    logger.info(f"  Successful:            {metrics.get('successful', 0)}")
    logger.info(f"  Errors:                {metrics.get('errors', 0)}")
    logger.info(f"  Query type accuracy:   {metrics.get('query_type_accuracy', 0):.1%}")
    logger.info(f"  Source type hit rate:   {metrics.get('source_type_hit_rate', 0):.1%}")
    logger.info(f"  Movie hit rate:         {metrics.get('movie_hit_rate', 0):.1%}")
    logger.info(f"  Avg response time:     {metrics.get('avg_response_time_s', 0):.1f}s")
    logger.info(f"  Total eval time:       {total_time:.0f}s ({total_time / 60:.1f} min)")

    if metrics.get("ragas"):
        logger.info("")
        logger.info("  RAGAS Scores:")
        for k, v in metrics["ragas"].items():
            logger.info(f"    {k}: {v:.4f}")
    elif metrics.get("ragas_skip_reason"):
        logger.info(f"\n  RAGAS skipped: {metrics['ragas_skip_reason']}")

    if metrics.get("by_type"):
        logger.info("")
        logger.info("  Per-type breakdown:")
        for qtype, stats in metrics["by_type"].items():
            logger.info(f"    {qtype}: {stats['count']} questions, "
                        f"{stats['errors']} errors, {stats['avg_time']:.1f}s avg")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="CineRAG Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_eval                       # Run full evaluation
  python -m evaluation.run_eval --limit 5             # Quick test with 5 questions
  python -m evaluation.run_eval --query-type trend    # Only evaluate trend queries
        """,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only evaluate the first N questions",
    )
    parser.add_argument(
        "--query-type",
        choices=["general", "trend", "pattern", "hit_flop"],
        help="Only evaluate a specific query type",
    )

    args = parser.parse_args()
    run_evaluation(query_type=args.query_type, limit=args.limit)


if __name__ == "__main__":
    main()
