"""Chunker for movie scripts — splits by scene boundaries."""

import json
import logging
import re
from pathlib import Path

import config
from chunking.chunk_models import DocumentChunk

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    return len(text) // CHARS_PER_TOKEN


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int = 50) -> list[str]:
    """Split text that exceeds max_tokens at paragraph boundaries.

    Falls back to hard character splits if no paragraph breaks exist.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    max_chars = max_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # Try splitting at paragraph boundaries (double newlines)
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return _merge_paragraphs(paragraphs, max_chars, overlap_chars)

    # Fallback: split at single newlines
    lines = text.split("\n")
    if len(lines) > 1:
        return _merge_paragraphs(lines, max_chars, overlap_chars)

    # Last resort: hard character split with overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a space
        if end < len(text):
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunks.append(text[start:end].strip())
        start = end - overlap_chars
    return [c for c in chunks if c]


def _merge_paragraphs(paragraphs: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """Merge paragraphs into chunks that fit within max_chars."""
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
            # If a single paragraph exceeds max, include it as-is
            # (it'll be a slightly oversized chunk, which is acceptable)
            current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def chunk_script(script_data: dict, movie_meta: dict) -> list[DocumentChunk]:
    """Chunk a single parsed script into DocumentChunks.

    Args:
        script_data: Parsed script dict from the scraper. Expected keys:
            - movie_title, scenes (list of {scene_number, heading, content}),
              and optionally script_text (full text fallback).
        movie_meta: Movie metadata dict with keys:
            tmdb_id, title, year, language, genres, director.

    Returns:
        List of DocumentChunk objects.
    """
    tmdb_id = movie_meta.get("tmdb_id", 0)
    title = movie_meta.get("title", script_data.get("movie_title", "Unknown"))
    year = movie_meta.get("year", 0)
    language = movie_meta.get("language", "en")
    genres = movie_meta.get("genres", [])
    director = movie_meta.get("director", "Unknown")
    max_tokens = config.SCRIPT_CHUNK_MAX_TOKENS

    scenes = script_data.get("scenes", [])
    chunks = []
    chunk_idx = 0

    # If no scenes were parsed, fall back to full script text splitting
    if not scenes:
        full_text = script_data.get("script_text", "")
        if not full_text:
            logger.warning(f"No scenes or script_text for: {title}")
            return []

        text_pieces = _split_long_text(full_text, max_tokens)
        for piece in text_pieces:
            chunks.append(DocumentChunk(
                chunk_id=f"{tmdb_id}_script_{chunk_idx}",
                movie_title=title,
                movie_year=year,
                language=language,
                genres=genres,
                director=director,
                source_type="script",
                content=piece,
                metadata={"scene_heading": "UNSTRUCTURED", "scene_number": 0},
            ))
            chunk_idx += 1
        return chunks

    # Process scene by scene
    for scene in scenes:
        heading = scene.get("heading", "")
        content = scene.get("content", "")
        scene_number = scene.get("scene_number", 0)

        if not content.strip():
            continue

        # Prepend the scene heading to content for context
        full_scene = f"{heading}\n\n{content}" if heading else content

        if _estimate_tokens(full_scene) <= max_tokens:
            # Scene fits in one chunk
            chunks.append(DocumentChunk(
                chunk_id=f"{tmdb_id}_script_{chunk_idx}",
                movie_title=title,
                movie_year=year,
                language=language,
                genres=genres,
                director=director,
                source_type="script",
                content=full_scene,
                metadata={"scene_heading": heading, "scene_number": scene_number},
            ))
            chunk_idx += 1
        else:
            # Scene too long — split at paragraph boundaries
            sub_chunks = _split_long_text(content, max_tokens)
            for j, sub in enumerate(sub_chunks):
                # Prepend heading to the first sub-chunk for context
                text = f"{heading}\n\n{sub}" if j == 0 else sub
                chunks.append(DocumentChunk(
                    chunk_id=f"{tmdb_id}_script_{chunk_idx}",
                    movie_title=title,
                    movie_year=year,
                    language=language,
                    genres=genres,
                    director=director,
                    source_type="script",
                    content=text,
                    metadata={
                        "scene_heading": heading,
                        "scene_number": scene_number,
                        "sub_chunk": j,
                    },
                ))
                chunk_idx += 1

    logger.info(f"  Chunked script for '{title}': {len(chunks)} chunks from {len(scenes)} scenes")
    return chunks


def chunk_all_scripts(movie_metadata: dict[int, dict] | None = None) -> list[DocumentChunk]:
    """Chunk all processed scripts found in data/processed/scripts/.

    Args:
        movie_metadata: Optional dict mapping tmdb_id -> metadata dict.
            If not provided, uses minimal metadata from the script files.

    Returns:
        List of all DocumentChunk objects across all scripts.
    """
    processed_dir = config.PROCESSED_DIR / "scripts"
    raw_dir = config.RAW_DIR / "scripts"

    if not processed_dir.exists():
        logger.info("No processed scripts directory found")
        return []

    script_files = list(processed_dir.glob("*.json"))
    if not script_files:
        logger.info("No processed script files found")
        return []

    logger.info(f"Chunking {len(script_files)} script files")
    all_chunks = []

    for filepath in script_files:
        with open(filepath, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        title = script_data.get("movie_title", filepath.stem)

        # The processed JSON may not have script_text (it was stripped to save space).
        # Load it from the raw text file if available.
        if "script_text" not in script_data:
            raw_path = raw_dir / f"{filepath.stem}.txt"
            if raw_path.exists():
                with open(raw_path, "r", encoding="utf-8") as f:
                    script_data["script_text"] = f.read()

        # Build movie_meta from the metadata lookup or from minimal info
        movie_meta = {"title": title, "language": "en", "genres": [], "director": "Unknown"}
        if movie_metadata:
            # Try to find by title match
            for meta in movie_metadata.values():
                if meta.get("title", "").lower() == title.lower():
                    movie_meta = meta
                    break

        chunks = chunk_script(script_data, movie_meta)
        all_chunks.extend(chunks)

    logger.info(f"Total script chunks: {len(all_chunks)}")
    return all_chunks
