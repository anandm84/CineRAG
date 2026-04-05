"""Core retriever — semantic search over ChromaDB with metadata filtering."""

import logging

import config
from chunking.chunk_models import DocumentChunk
from embedding.vector_store import VectorStore

logger = logging.getLogger(__name__)


class CineRetriever:
    """Retrieves relevant document chunks from ChromaDB.

    Wraps VectorStore.query() and converts raw results back into DocumentChunk
    objects with similarity scores attached.
    """

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[DocumentChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query text.
            top_k: Number of results to return (default from config).
            filters: ChromaDB `where` clause for metadata filtering.

        Returns:
            List of DocumentChunk objects sorted by relevance (best first).
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        logger.info(f"Retrieving top {top_k} chunks for: {query[:80]}...")
        if filters:
            logger.info(f"  Filters: {filters}")

        results = self.vector_store.query(
            query_text=query,
            top_k=top_k,
            filters=filters,
        )

        chunks = []
        for result in results:
            chunk = _result_to_chunk(result)
            if chunk:
                chunks.append(chunk)

        logger.info(f"  Retrieved {len(chunks)} chunks")
        return chunks


def _result_to_chunk(result: dict) -> DocumentChunk | None:
    """Convert a raw ChromaDB result dict back into a DocumentChunk."""
    try:
        meta = result.get("metadata", {})

        # Reconstruct genres from comma-separated string
        genres_str = meta.get("genres", "")
        genres = [g.strip() for g in genres_str.split(",") if g.strip()] if genres_str else []

        # Reconstruct source-specific metadata
        source_type = meta.get("source_type", "")
        extra_meta: dict = {"distance": result.get("distance", 0)}

        if source_type == "script":
            extra_meta["scene_heading"] = meta.get("scene_heading", "")
            extra_meta["scene_number"] = meta.get("scene_number", 0)
        elif source_type == "review":
            if "rating" in meta:
                extra_meta["rating"] = meta["rating"]
            extra_meta["review_date"] = meta.get("review_date", "")

        return DocumentChunk(
            chunk_id=result.get("chunk_id", ""),
            movie_title=meta.get("movie_title", ""),
            movie_year=meta.get("movie_year", 0),
            language=meta.get("language", ""),
            genres=genres,
            director=meta.get("director", ""),
            source_type=source_type,
            content=result.get("content", ""),
            metadata=extra_meta,
        )
    except Exception as e:
        logger.warning(f"Failed to convert result to chunk: {e}")
        return None
