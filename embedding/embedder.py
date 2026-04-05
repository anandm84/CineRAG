"""Embedding wrapper for BGE-M3 (multilingual) with lightweight fallback."""

import logging
import math

from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers model for encoding text into embeddings.

    Uses BGE-M3 by default (multilingual, ~2.3GB).
    Falls back to all-MiniLM-L6-v2 (~80MB, English-only) if configured.
    """

    def __init__(self, model_name: str | None = None):
        if model_name:
            self.model_name = model_name
        elif config.USE_LIGHTWEIGHT_EMBEDDINGS:
            self.model_name = config.LIGHTWEIGHT_EMBEDDING_MODEL
            logger.info("Using lightweight embedding model (English-only)")
        else:
            self.model_name = config.EMBEDDING_MODEL

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, texts: list[str], batch_size: int = 32,
              show_progress: bool = True) -> list[list[float]]:
        """Embed a list of texts in batches.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to encode at once.
            show_progress: Log progress every batch.

        Returns:
            List of embedding vectors (list of floats).
        """
        total = len(texts)
        if total == 0:
            return []

        total_batches = math.ceil(total / batch_size)
        all_embeddings = []

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch, normalize_embeddings=True, show_progress_bar=False
            )
            all_embeddings.extend(embeddings.tolist())

            if show_progress:
                done = min(i + batch_size, total)
                batch_num = (i // batch_size) + 1
                logger.info(f"Embedded {done}/{total} texts (batch {batch_num}/{total_batches})")

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.model.encode(query, normalize_embeddings=True).tolist()
