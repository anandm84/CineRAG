"""Pydantic models for document chunks used across the chunking pipeline."""

from pydantic import BaseModel


class DocumentChunk(BaseModel):
    """A single chunk of text with movie metadata attached.

    This is the universal unit that flows from chunking -> embedding -> retrieval.
    """

    chunk_id: str               # Unique ID: "{tmdb_id}_{source_type}_{index}"
    movie_title: str
    movie_year: int
    language: str               # 'te', 'hi', 'en'
    genres: list[str]
    director: str
    source_type: str            # 'script' or 'review'
    content: str                # The actual text chunk
    metadata: dict              # Additional source-specific metadata
    # For scripts: scene_heading, scene_number
    # For reviews: rating, review_date, review_title
