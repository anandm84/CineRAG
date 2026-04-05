"""Chunker for movie reviews — typically one review per chunk."""

import json
import logging
import re
from pathlib import Path

import config
from chunking.chunk_models import DocumentChunk

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _split_review(text: str, max_tokens: int) -> list[str]:
    """Split a long review at paragraph boundaries."""
    max_chars = max_tokens * CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) <= 1:
        # No paragraph breaks — split at sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        paragraphs = sentences

    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_reviews(review_data: dict, movie_meta: dict) -> list[DocumentChunk]:
    """Chunk reviews for a single movie into DocumentChunks.

    Args:
        review_data: Parsed review dict from the scraper. Expected keys:
            - movie_title, imdb_id, reviews (list of review dicts).
        movie_meta: Movie metadata dict with keys:
            tmdb_id, title, year, language, genres, director.

    Returns:
        List of DocumentChunk objects.
    """
    tmdb_id = movie_meta.get("tmdb_id", 0)
    title = movie_meta.get("title", review_data.get("movie_title", "Unknown"))
    year = movie_meta.get("year", 0)
    language = movie_meta.get("language", "en")
    genres = movie_meta.get("genres", [])
    director = movie_meta.get("director", "Unknown")
    max_tokens = config.REVIEW_CHUNK_MAX_TOKENS

    reviews = review_data.get("reviews", [])
    chunks = []
    chunk_idx = 0

    for review in reviews:
        review_text = review.get("review_text", "").strip()
        if not review_text:
            continue

        rating = review.get("rating")
        review_date = review.get("date", "")
        review_title = review.get("title", "")

        # Prepend review title if available
        full_text = f"{review_title}\n\n{review_text}" if review_title else review_text

        meta = {
            "rating": rating,
            "review_date": review_date,
            "review_title": review_title,
            "imdb_id": review_data.get("imdb_id", ""),
        }

        if _estimate_tokens(full_text) <= max_tokens:
            chunks.append(DocumentChunk(
                chunk_id=f"{tmdb_id}_review_{chunk_idx}",
                movie_title=title,
                movie_year=year,
                language=language,
                genres=genres,
                director=director,
                source_type="review",
                content=full_text,
                metadata=meta,
            ))
            chunk_idx += 1
        else:
            # Split long reviews
            sub_texts = _split_review(full_text, max_tokens)
            for j, sub in enumerate(sub_texts):
                chunks.append(DocumentChunk(
                    chunk_id=f"{tmdb_id}_review_{chunk_idx}",
                    movie_title=title,
                    movie_year=year,
                    language=language,
                    genres=genres,
                    director=director,
                    source_type="review",
                    content=sub,
                    metadata={**meta, "sub_chunk": j},
                ))
                chunk_idx += 1

    if chunks:
        logger.info(f"  Chunked reviews for '{title}': {len(chunks)} chunks from {len(reviews)} reviews")
    return chunks


def chunk_all_reviews(movie_metadata: dict[int, dict] | None = None) -> list[DocumentChunk]:
    """Chunk all processed reviews found in data/processed/reviews/.

    Args:
        movie_metadata: Optional dict mapping tmdb_id -> metadata dict.
            If not provided, uses minimal metadata from the review files.

    Returns:
        List of all DocumentChunk objects across all review files.
    """
    processed_dir = config.PROCESSED_DIR / "reviews"

    if not processed_dir.exists():
        logger.info("No processed reviews directory found")
        return []

    review_files = list(processed_dir.glob("*.json"))
    if not review_files:
        logger.info("No processed review files found")
        return []

    logger.info(f"Chunking {len(review_files)} review files")
    all_chunks = []

    for filepath in review_files:
        with open(filepath, "r", encoding="utf-8") as f:
            review_data = json.load(f)

        title = review_data.get("movie_title", filepath.stem)

        # Build movie_meta from the metadata lookup or minimal info
        movie_meta = {"title": title, "language": "en", "genres": [], "director": "Unknown"}
        if movie_metadata:
            for meta in movie_metadata.values():
                if meta.get("title", "").lower() == title.lower():
                    movie_meta = meta
                    break

        chunks = chunk_reviews(review_data, movie_meta)
        all_chunks.extend(chunks)

    logger.info(f"Total review chunks: {len(all_chunks)}")
    return all_chunks
