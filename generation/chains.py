"""RAG chains — the full query pipeline from question to answer with sources."""

import logging

from chunking.chunk_models import DocumentChunk
from generation.llm_client import OllamaClient
from generation.prompts import SYSTEM_MESSAGE, get_template
from generation import query_router
from ingestion.metadata_store import MetadataStore
from retrieval.hybrid_search import HybridSearcher, extract_filters, build_chroma_filters
from retrieval.reranker import maybe_rerank

logger = logging.getLogger(__name__)


class CineRAGChain:
    """Full RAG pipeline: route -> retrieve -> rerank -> generate."""

    def __init__(
        self,
        searcher: HybridSearcher | None = None,
        llm: OllamaClient | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.searcher = searcher or HybridSearcher()
        self.llm = llm or OllamaClient()
        self.metadata_store = metadata_store

    def run(self, query: str, explicit_filters: dict | None = None) -> dict:
        """Execute the full RAG pipeline.

        Args:
            query: Natural language question.
            explicit_filters: Optional filters from frontend (override extraction).

        Returns:
            Dict with keys: answer, query_type, filters, sources.
        """
        # 1. Classify the query
        query_type = query_router.classify(query)
        logger.info(f"Query type: {query_type}")

        # 2. Route to the appropriate strategy
        if query_type == "hit_flop":
            return self._run_hit_flop(query, explicit_filters)

        if query_type == "trend":
            return self._run_trend(query, explicit_filters)

        # pattern and general use the same retrieval flow
        return self._run_standard(query, query_type, explicit_filters)

    def _run_standard(self, query: str, query_type: str,
                      explicit_filters: dict | None = None) -> dict:
        """Standard retrieval flow for general Q&A and pattern queries."""
        # Retrieve
        chunks, extracted = self.searcher.search(
            query, explicit_filters=explicit_filters
        )

        # Rerank
        chunks = maybe_rerank(query, chunks)

        # Format context
        context = _format_context(chunks)

        # Generate
        template = get_template(query_type)
        prompt = template.format(context=context, question=query)
        answer = self.llm.generate(prompt, system=SYSTEM_MESSAGE)

        return _build_response(answer, query_type, extracted, chunks)

    def _run_trend(self, query: str,
                   explicit_filters: dict | None = None) -> dict:
        """Trend analysis — retrieves and sorts context chronologically."""
        chunks, extracted = self.searcher.search(
            query, explicit_filters=explicit_filters
        )

        # Rerank
        chunks = maybe_rerank(query, chunks)

        # Sort chronologically for trend analysis
        chunks.sort(key=lambda c: c.movie_year)

        context = _format_context(chunks)
        template = get_template("trend")
        prompt = template.format(context=context, question=query)
        answer = self.llm.generate(prompt, system=SYSTEM_MESSAGE)

        return _build_response(answer, "trend", extracted, chunks)

    def _run_hit_flop(self, query: str,
                      explicit_filters: dict | None = None) -> dict:
        """Hit/flop comparison — separate retrieval for hits and flops."""
        base_extracted = extract_filters(query)
        if explicit_filters:
            base_extracted.update(explicit_filters)

        # Build base filters without hit/flop
        hit_extracted = {k: v for k, v in base_extracted.items() if k != "is_hit"}
        flop_extracted = {k: v for k, v in base_extracted.items() if k != "is_hit"}

        # Try to use SQLite to get hit/flop movie titles for filtering
        hit_titles, flop_titles = self._get_hit_flop_titles(base_extracted)

        hit_chunks = self._retrieve_for_group(query, hit_extracted, hit_titles, is_hit=True)
        flop_chunks = self._retrieve_for_group(query, flop_extracted, flop_titles, is_hit=False)

        # Rerank each group separately
        hit_chunks = maybe_rerank(query, hit_chunks)
        flop_chunks = maybe_rerank(query, flop_chunks)

        hit_context = _format_context(hit_chunks)
        flop_context = _format_context(flop_chunks)

        template = get_template("hit_flop")
        prompt = template.format(
            hit_context=hit_context or "(No hit film data available)",
            flop_context=flop_context or "(No flop film data available)",
            question=query,
        )
        answer = self.llm.generate(prompt, system=SYSTEM_MESSAGE)

        all_chunks = hit_chunks + flop_chunks
        return _build_response(answer, "hit_flop", base_extracted, all_chunks)

    def _get_hit_flop_titles(self, filters: dict) -> tuple[list[str], list[str]]:
        """Query SQLite for hit and flop movie titles matching the filters."""
        if not self.metadata_store:
            try:
                self.metadata_store = MetadataStore()
            except Exception:
                return [], []

        base_kwargs = {}
        if "language" in filters:
            base_kwargs["language"] = filters["language"]
        if "year_min" in filters:
            base_kwargs["year_min"] = filters["year_min"]
        if "year_max" in filters:
            base_kwargs["year_max"] = filters["year_max"]
        if "genre" in filters:
            base_kwargs["genre"] = filters["genre"]

        hits = self.metadata_store.get_movies_by_filter(is_hit=True, **base_kwargs)
        flops = self.metadata_store.get_movies_by_filter(is_hit=False, **base_kwargs)

        hit_titles = [m["title"] for m in hits]
        flop_titles = [m["title"] for m in flops]

        logger.info(f"Hit/flop split: {len(hit_titles)} hits, {len(flop_titles)} flops")
        return hit_titles, flop_titles

    def _retrieve_for_group(
        self,
        query: str,
        base_filters: dict,
        titles: list[str],
        is_hit: bool,
    ) -> list[DocumentChunk]:
        """Retrieve chunks for a group of movies (hits or flops).

        If we have specific titles from SQLite, filter by them.
        Otherwise fall back to general retrieval.
        """
        if titles:
            # Retrieve for each title and merge (ChromaDB doesn't support $in on strings easily)
            all_chunks = []
            for title in titles[:10]:  # Limit to avoid too many queries
                title_filter = {**build_chroma_filters(base_filters) or {}}
                # Use a simple title filter
                chroma_where = {"movie_title": title}
                try:
                    chunks = self.searcher.retriever.retrieve(
                        query=query, top_k=5, filters=chroma_where
                    )
                    all_chunks.extend(chunks)
                except Exception:
                    continue
            return all_chunks

        # Fallback: general retrieval
        chroma_filters = build_chroma_filters(base_filters)
        try:
            return self.searcher.retriever.retrieve(
                query=query, top_k=10, filters=chroma_filters
            )
        except Exception:
            return []


def _format_context(chunks: list[DocumentChunk]) -> str:
    """Format chunks into a readable context string for the LLM prompt."""
    if not chunks:
        return "(No relevant context found)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk.source_type.capitalize()
        header = f"[{i}] {chunk.movie_title} ({chunk.movie_year}) — {source_label}"

        # Add extra context from metadata
        extras = []
        if chunk.source_type == "script":
            heading = chunk.metadata.get("scene_heading", "")
            if heading:
                extras.append(f"Scene: {heading}")
        elif chunk.source_type == "review":
            rating = chunk.metadata.get("rating")
            if rating:
                extras.append(f"Rating: {rating}/10")

        if extras:
            header += f" ({', '.join(extras)})"

        parts.append(f"{header}\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


def _build_response(
    answer: str,
    query_type: str,
    filters: dict,
    chunks: list[DocumentChunk],
) -> dict:
    """Build the standard response dict."""
    return {
        "answer": answer,
        "query_type": query_type,
        "filters": filters,
        "sources": [
            {
                "movie": c.movie_title,
                "year": c.movie_year,
                "source_type": c.source_type,
                "language": c.language,
                "snippet": c.content[:200],
            }
            for c in chunks[:5]
        ],
    }
