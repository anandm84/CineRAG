"""Optional cross-encoder reranker to improve retrieval precision.

Takes the top-K results from the initial retrieval and reranks them using a
cross-encoder model that scores (query, document) pairs directly. This is more
accurate than bi-encoder similarity but slower since it can't be precomputed.

Togglable via config.USE_RERANKER.
"""

import logging

from sentence_transformers import CrossEncoder

import config
from chunking.chunk_models import DocumentChunk

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker for improving retrieval precision."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
    ) -> list[DocumentChunk]:
        """Rerank chunks by cross-encoder relevance score.

        Args:
            query: The original query text.
            chunks: List of chunks from initial retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of DocumentChunks (best first), truncated to top_k.
        """
        top_k = top_k or config.TOP_K_RERANK

        if not chunks:
            return []

        if len(chunks) <= top_k:
            return chunks

        # Score all (query, document) pairs
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        reranked = [chunk for chunk, score in scored[:top_k]]

        logger.info(
            f"Reranked {len(chunks)} -> {len(reranked)} chunks "
            f"(top score: {scored[0][1]:.3f}, cutoff: {scored[top_k - 1][1]:.3f})"
        )

        return reranked


# Module-level singleton — lazy loaded
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Get or create the singleton reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def maybe_rerank(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int | None = None,
) -> list[DocumentChunk]:
    """Rerank if USE_RERANKER is enabled, otherwise pass through.

    This is the main entry point for the reranker. Call it unconditionally —
    it respects the config toggle.
    """
    if not config.USE_RERANKER:
        # Just truncate to top_k without reranking
        top_k = top_k or config.TOP_K_RERANK
        return chunks[:top_k]

    reranker = get_reranker()
    return reranker.rerank(query, chunks, top_k)
