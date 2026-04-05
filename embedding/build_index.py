"""Index builder — embeds all chunks and populates ChromaDB."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from chunking.script_chunker import chunk_all_scripts
from chunking.review_chunker import chunk_all_reviews
from chunking.chunk_models import DocumentChunk
from embedding.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Rough estimate: ~50ms per chunk for BGE-M3 on CPU
ESTIMATED_MS_PER_CHUNK = 50


def _load_movie_metadata() -> dict[int, dict]:
    """Load all processed movie metadata into a lookup dict keyed by tmdb_id."""
    metadata_dir = config.PROCESSED_DIR / "metadata"
    lookup = {}

    if not metadata_dir.exists():
        return lookup

    for filepath in metadata_dir.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tmdb_id = meta.get("tmdb_id")
            if tmdb_id:
                lookup[tmdb_id] = meta
        except Exception:
            continue

    return lookup


def build_index(
    rebuild: bool = False,
    source_type: str | None = None,
):
    """Build the ChromaDB index from processed chunks.

    Args:
        rebuild: If True, delete the existing collection and rebuild from scratch.
        source_type: If set, only index chunks of this type ('script' or 'review').
    """
    logger.info("=" * 60)
    logger.info("CineRAG Index Builder")
    logger.info("=" * 60)

    store = VectorStore()

    if rebuild:
        logger.info("Rebuilding index — deleting existing collection...")
        store.delete_collection()

    # Load movie metadata for enriching chunks
    logger.info("Loading movie metadata...")
    movie_metadata = _load_movie_metadata()
    logger.info(f"Loaded metadata for {len(movie_metadata)} movies")

    all_chunks: list[DocumentChunk] = []

    # Collect script chunks
    if source_type is None or source_type == "script":
        logger.info("")
        logger.info("-" * 40)
        logger.info("Chunking scripts...")
        logger.info("-" * 40)
        script_chunks = chunk_all_scripts(movie_metadata)
        all_chunks.extend(script_chunks)
        logger.info(f"Script chunks: {len(script_chunks)}")

    # Collect review chunks
    if source_type is None or source_type == "review":
        logger.info("")
        logger.info("-" * 40)
        logger.info("Chunking reviews...")
        logger.info("-" * 40)
        review_chunks = chunk_all_reviews(movie_metadata)
        all_chunks.extend(review_chunks)
        logger.info(f"Review chunks: {len(review_chunks)}")

    if not all_chunks:
        logger.warning("No chunks to index. Run the ingestion pipeline first.")
        return

    # Estimate time
    estimated_seconds = (len(all_chunks) * ESTIMATED_MS_PER_CHUNK) / 1000
    estimated_minutes = estimated_seconds / 60
    logger.info("")
    logger.info(f"Total chunks to index: {len(all_chunks)}")
    logger.info(f"Estimated embedding time: ~{estimated_minutes:.1f} minutes (CPU)")

    # Count by type and unique movies
    script_count = sum(1 for c in all_chunks if c.source_type == "script")
    review_count = sum(1 for c in all_chunks if c.source_type == "review")
    unique_movies = len(set(c.movie_title for c in all_chunks))

    logger.info(f"  Scripts: {script_count} chunks")
    logger.info(f"  Reviews: {review_count} chunks")
    logger.info(f"  Movies:  {unique_movies}")

    # Embed and index
    logger.info("")
    logger.info("-" * 40)
    logger.info("Embedding and indexing...")
    logger.info("-" * 40)

    start_time = time.time()
    indexed = store.add_documents(all_chunks)
    elapsed = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Indexed {indexed} chunks ({script_count} script, {review_count} review)")
    logger.info(f"Covering {unique_movies} movies")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    stats = store.get_stats()
    logger.info(f"Collection total: {stats['total_chunks']} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="CineRAG Index Builder — embed chunks and populate ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m embedding.build_index                     # Build full index
  python -m embedding.build_index --rebuild           # Rebuild from scratch
  python -m embedding.build_index --source-type review  # Only index reviews
        """,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    parser.add_argument(
        "--source-type",
        choices=["script", "review"],
        help="Only index a specific source type",
    )

    args = parser.parse_args()
    build_index(rebuild=args.rebuild, source_type=args.source_type)


if __name__ == "__main__":
    main()
