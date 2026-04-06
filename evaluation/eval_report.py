"""Evaluation report generator — produces a markdown summary from eval results.

Usage:
    python -m evaluation.eval_report
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
RESULTS_PATH = EVAL_DIR / "results.json"
REPORT_PATH = EVAL_DIR / "EVAL_REPORT.md"


def load_results() -> dict:
    """Load evaluation results from JSON."""
    if not RESULTS_PATH.exists():
        logger.error(f"Results file not found: {RESULTS_PATH}")
        logger.error("Run the evaluation first: python -m evaluation.run_eval")
        sys.exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(data: dict) -> str:
    """Generate a markdown evaluation report."""
    metrics = data.get("metrics", {})
    results = data.get("results", [])

    sections = []

    # Header
    sections.append("# CineRAG Evaluation Report\n")

    # Overall summary
    sections.append("## Overall Summary\n")
    sections.append("| Metric | Value |")
    sections.append("|---|---|")
    sections.append(f"| Total Questions | {metrics.get('total_questions', 0)} |")
    sections.append(f"| Successful | {metrics.get('successful', 0)} |")
    sections.append(f"| Errors | {metrics.get('errors', 0)} |")
    sections.append(f"| Query Type Accuracy | {metrics.get('query_type_accuracy', 0):.1%} |")
    sections.append(f"| Source Type Hit Rate | {metrics.get('source_type_hit_rate', 0):.1%} |")
    sections.append(f"| Movie Hit Rate | {metrics.get('movie_hit_rate', 0):.1%} |")
    sections.append(f"| Avg Response Time | {metrics.get('avg_response_time_s', 0):.1f}s |")
    sections.append(f"| Total Eval Time | {metrics.get('total_eval_time_s', 0):.0f}s |")
    sections.append("")

    # RAGAS scores
    ragas = metrics.get("ragas")
    if ragas:
        sections.append("## RAGAS Scores\n")
        sections.append("| Metric | Score |")
        sections.append("|---|---|")
        for k, v in ragas.items():
            sections.append(f"| {k} | {v:.4f} |")
        sections.append("")
    else:
        skip_reason = metrics.get("ragas_skip_reason", "Not available")
        sections.append(f"## RAGAS Scores\n\n*Skipped: {skip_reason}*\n")

    # Per-type breakdown
    by_type = metrics.get("by_type", {})
    if by_type:
        sections.append("## Per-Query-Type Breakdown\n")
        sections.append("| Query Type | Count | Errors | Avg Time |")
        sections.append("|---|---|---|---|")
        for qtype in ("general", "trend", "pattern", "hit_flop"):
            stats = by_type.get(qtype, {})
            if stats:
                sections.append(
                    f"| {qtype} | {stats['count']} | {stats['errors']} | {stats['avg_time']:.1f}s |"
                )
        sections.append("")

    # Query routing accuracy detail
    sections.append("## Query Routing Accuracy\n")
    correct = 0
    misrouted = []
    for r in results:
        if r["detected_query_type"] == r["expected_query_type"]:
            correct += 1
        else:
            misrouted.append(r)

    sections.append(f"Correctly routed: {correct}/{len(results)}\n")
    if misrouted:
        sections.append("**Misrouted queries:**\n")
        for r in misrouted:
            sections.append(
                f"- \"{r['question'][:80]}...\" — expected `{r['expected_query_type']}`, "
                f"got `{r['detected_query_type']}`"
            )
        sections.append("")

    # Worst performing queries (by error or missing sources)
    sections.append("## Potential Issues\n")

    # Errors
    error_results = [r for r in results if r.get("error")]
    if error_results:
        sections.append("### Failed Queries\n")
        for r in error_results:
            sections.append(f"- **\"{r['question'][:80]}\"**")
            sections.append(f"  - Error: `{r['error']}`")
        sections.append("")

    # No sources retrieved
    no_sources = [r for r in results if not r.get("error") and not r.get("sources")]
    if no_sources:
        sections.append("### No Sources Retrieved\n")
        for r in no_sources:
            sections.append(f"- \"{r['question'][:80]}\"")
        sections.append("")

    # Source type misses
    source_misses = []
    for r in results:
        if r.get("error"):
            continue
        expected = set(r.get("expected_source_types", []))
        actual = set(s.get("source_type", "") for s in r.get("sources", []))
        if expected and not (expected & actual):
            source_misses.append(r)

    if source_misses:
        sections.append("### Source Type Mismatches\n")
        sections.append("Queries where retrieved sources didn't match expected types:\n")
        for r in source_misses:
            expected = r.get("expected_source_types", [])
            actual = list(set(s.get("source_type", "") for s in r.get("sources", [])))
            sections.append(
                f"- \"{r['question'][:60]}...\" — expected {expected}, got {actual}"
            )
        sections.append("")

    # Retrieval quality: movie hit analysis
    sections.append("## Retrieval Quality\n")
    movie_hits = 0
    movie_misses = []
    for r in results:
        if r.get("error"):
            continue
        expected = set(m.lower() for m in r.get("relevant_movies", []))
        actual = set(s.get("movie", "").lower() for s in r.get("sources", []))
        if expected & actual:
            movie_hits += 1
        elif expected:
            movie_misses.append({
                "question": r["question"][:60],
                "expected": list(r.get("relevant_movies", []))[:3],
                "actual": [s.get("movie", "") for s in r.get("sources", [])][:3],
            })

    valid = [r for r in results if not r.get("error")]
    sections.append(
        f"Retrieved at least one expected movie: {movie_hits}/{len(valid)} "
        f"({movie_hits / len(valid):.1%})\n" if valid else "No valid results.\n"
    )

    if movie_misses[:5]:
        sections.append("**Top missed retrievals:**\n")
        for m in movie_misses[:5]:
            sections.append(
                f"- \"{m['question']}...\" — expected {m['expected']}, got {m['actual']}"
            )
        sections.append("")

    # Footer
    sections.append("---\n")
    sections.append("*Generated by `python -m evaluation.eval_report`*\n")

    return "\n".join(sections)


def main():
    logger.info("Generating evaluation report...")
    data = load_results()
    report = generate_report(data)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved to {REPORT_PATH}")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
