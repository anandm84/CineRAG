"""ChromaDB vector store wrapper for CineRAG."""

import logging
from pathlib import Path

import chromadb

import config
from chunking.chunk_models import DocumentChunk
from embedding.embedder import Embedder

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "cinerag"


class VectorStore:
    """Persistent ChromaDB-backed vector store with metadata filtering."""

    def __init__(self, persist_dir: Path | None = None, embedder: Embedder | None = None):
        persist_dir = persist_dir or config.CHROMA_DIR
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedder = embedder
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_embedder(self) -> Embedder:
        """Lazy-load the embedder on first use."""
        if self.embedder is None:
            self.embedder = Embedder()
        return self.embedder

    def get_collection(self, name: str = DEFAULT_COLLECTION) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add_documents(self, chunks: list[DocumentChunk],
                      collection_name: str = DEFAULT_COLLECTION,
                      batch_size: int = 100) -> int:
        """Embed and upsert DocumentChunks into ChromaDB.

        Args:
            chunks: List of DocumentChunk objects to index.
            collection_name: Name of the ChromaDB collection.
            batch_size: Number of chunks to upsert at a time.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        collection = self.get_collection(collection_name)
        embedder = self._get_embedder()

        # Extract texts and embed all at once
        texts = [c.content for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = embedder.embed(texts)

        # Prepare ChromaDB-compatible metadata (flat key-value, no nested dicts/lists)
        ids = []
        metadatas = []
        documents = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            metadatas.append(_flatten_metadata(chunk))

        # Upsert in batches
        indexed = 0
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                documents=documents[i:end],
            )
            indexed = end
            logger.info(f"Upserted {indexed}/{total} chunks into ChromaDB")

        return indexed

    def query(self, query_text: str, top_k: int | None = None,
              filters: dict | None = None,
              collection_name: str = DEFAULT_COLLECTION) -> list[dict]:
        """Semantic search with optional metadata filters.

        Args:
            query_text: The natural language query.
            top_k: Number of results to return.
            filters: ChromaDB `where` clause dict for metadata filtering.
                Examples:
                    {"language": "te"}
                    {"source_type": "review"}
                    {"movie_year": {"$gte": 2000, "$lte": 2015}}
            collection_name: Name of the ChromaDB collection.

        Returns:
            List of result dicts with keys: chunk_id, content, metadata, distance.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        collection = self.get_collection(collection_name)
        embedder = self._get_embedder()

        query_embedding = embedder.embed_query(query_text)

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_params["where"] = filters

        results = collection.query(**query_params)

        # Unpack ChromaDB results into a cleaner format
        output = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                output.append({
                    "chunk_id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        return output

    def delete_collection(self, name: str = DEFAULT_COLLECTION):
        """Delete a collection entirely (for rebuilds)."""
        try:
            self.client.delete_collection(name)
            self._collections.pop(name, None)
            logger.info(f"Deleted collection: {name}")
        except ValueError:
            logger.info(f"Collection '{name}' does not exist, nothing to delete")

    def get_stats(self, collection_name: str = DEFAULT_COLLECTION) -> dict:
        """Get collection statistics."""
        collection = self.get_collection(collection_name)
        count = collection.count()
        return {"collection": collection_name, "total_chunks": count}


def _flatten_metadata(chunk: DocumentChunk) -> dict:
    """Convert a DocumentChunk into flat metadata for ChromaDB.

    ChromaDB metadata values must be str, int, float, or bool — no lists or nested dicts.
    """
    meta = {
        "movie_title": chunk.movie_title,
        "movie_year": chunk.movie_year,
        "language": chunk.language,
        "genres": ", ".join(chunk.genres) if chunk.genres else "",
        "director": chunk.director,
        "source_type": chunk.source_type,
    }

    # Attach source-specific metadata
    extra = chunk.metadata or {}
    if chunk.source_type == "script":
        meta["scene_heading"] = extra.get("scene_heading", "")
        meta["scene_number"] = extra.get("scene_number", 0)
    elif chunk.source_type == "review":
        rating = extra.get("rating")
        if rating is not None:
            meta["rating"] = float(rating)
        meta["review_date"] = extra.get("review_date", "")

    return meta
