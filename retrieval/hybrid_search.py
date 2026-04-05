"""Hybrid search — combines semantic search with metadata-based pre-filtering.

Extracts structured filters from natural language queries using keyword matching
and regex patterns. No LLM is used for filter extraction — it's fast and deterministic.
"""

import logging
import re

from chunking.chunk_models import DocumentChunk
from retrieval.retriever import CineRetriever

logger = logging.getLogger(__name__)

# Language mappings
LANGUAGE_MAP = {
    "telugu": "te", "tollywood": "te",
    "hindi": "hi", "bollywood": "hi",
    "english": "en", "hollywood": "en",
}

# Genre keywords (lowercased)
GENRE_KEYWORDS = {
    "action", "comedy", "drama", "thriller", "horror", "romance", "romantic",
    "sci-fi", "science fiction", "fantasy", "animation", "crime", "mystery",
    "adventure", "family", "war", "history", "historical", "musical", "western",
    "documentary", "biography", "sport", "sports",
}

# Decade patterns
DECADE_RANGES = {
    "70s": (1970, 1979), "1970s": (1970, 1979),
    "80s": (1980, 1989), "1980s": (1980, 1989),
    "90s": (1990, 1999), "1990s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
}


def extract_filters(query: str) -> dict:
    """Extract structured metadata filters from a natural language query.

    Returns a dict with optional keys:
        - language: str ("te", "hi", "en")
        - year_min: int
        - year_max: int
        - genre: str
        - director: str
        - source_type: str ("script" or "review")
        - is_hit: bool
    """
    query_lower = query.lower()
    filters: dict = {}

    # Extract language
    for keyword, lang_code in LANGUAGE_MAP.items():
        if keyword in query_lower:
            filters["language"] = lang_code
            break

    # Extract year range: "from 2005 to 2010", "between 2005 and 2010"
    year_range = re.search(
        r"(?:from|between)\s+(\d{4})\s+(?:to|and)\s+(\d{4})", query_lower
    )
    if year_range:
        filters["year_min"] = int(year_range.group(1))
        filters["year_max"] = int(year_range.group(2))

    # Extract single year: "in 2015", "of 2015"
    if "year_min" not in filters:
        single_year = re.search(r"(?:in|of|from|year)\s+(\d{4})", query_lower)
        if single_year:
            year = int(single_year.group(1))
            if 1950 <= year <= 2030:
                filters["year_min"] = year
                filters["year_max"] = year

    # Extract decade: "in the 2010s", "90s"
    if "year_min" not in filters:
        for decade_key, (y_min, y_max) in DECADE_RANGES.items():
            if decade_key in query_lower:
                filters["year_min"] = y_min
                filters["year_max"] = y_max
                break

    # Extract genre
    for genre in GENRE_KEYWORDS:
        if genre in query_lower:
            # Normalize some genre names
            normalized = genre
            if genre == "romantic":
                normalized = "Romance"
            elif genre in ("sci-fi", "science fiction"):
                normalized = "Science Fiction"
            elif genre in ("sport", "sports"):
                normalized = "Sport"
            elif genre in ("history", "historical"):
                normalized = "History"
            else:
                normalized = genre.title()
            filters["genre"] = normalized
            break

    # Extract director: "directed by X", "X's films", "director X"
    director_match = re.search(
        r"(?:directed by|director)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        query,
    )
    if director_match:
        filters["director"] = director_match.group(1)
    else:
        # "Rajamouli's films", "Nolan's movies"
        possessive = re.search(
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'s\s+(?:films?|movies?|work|style|cinema)",
            query,
        )
        if possessive:
            filters["director"] = possessive.group(1)

    # Extract source type
    if "script" in query_lower or "scene" in query_lower or "dialogue" in query_lower:
        filters["source_type"] = "script"
    elif "review" in query_lower or "audience" in query_lower or "critic" in query_lower:
        filters["source_type"] = "review"

    # Extract hit/flop
    hit_keywords = {"hit", "successful", "blockbuster", "superhit", "box office success"}
    flop_keywords = {"flop", "failure", "disaster", "bombed", "box office failure"}

    if any(kw in query_lower for kw in hit_keywords):
        filters["is_hit"] = True
    if any(kw in query_lower for kw in flop_keywords):
        filters["is_hit"] = False

    return filters


def build_chroma_filters(extracted: dict) -> dict | None:
    """Convert extracted filters into a ChromaDB `where` clause.

    ChromaDB uses `$and` to combine multiple conditions, and operators like
    `$gte`, `$lte` for range queries.
    """
    conditions = []

    if "language" in extracted:
        conditions.append({"language": extracted["language"]})

    if "year_min" in extracted and "year_max" in extracted:
        conditions.append({"movie_year": {"$gte": extracted["year_min"]}})
        conditions.append({"movie_year": {"$lte": extracted["year_max"]}})
    elif "year_min" in extracted:
        conditions.append({"movie_year": {"$gte": extracted["year_min"]}})
    elif "year_max" in extracted:
        conditions.append({"movie_year": {"$lte": extracted["year_max"]}})

    if "genre" in extracted:
        conditions.append({"genres": {"$contains": extracted["genre"]}})

    if "director" in extracted:
        conditions.append({"director": {"$contains": extracted["director"]}})

    if "source_type" in extracted:
        conditions.append({"source_type": extracted["source_type"]})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class HybridSearcher:
    """Combines semantic search with metadata pre-filtering.

    Usage:
        searcher = HybridSearcher()
        results = searcher.search("Telugu action movies from 2010 to 2020")
    """

    def __init__(self, retriever: CineRetriever | None = None):
        self.retriever = retriever or CineRetriever()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        explicit_filters: dict | None = None,
    ) -> tuple[list[DocumentChunk], dict]:
        """Run hybrid search: extract filters from query, then semantic search.

        Args:
            query: Natural language query.
            top_k: Number of results.
            explicit_filters: Optional manually-specified filters that override
                extraction. Useful when the frontend passes filter checkboxes.

        Returns:
            Tuple of (results, extracted_filters).
        """
        # Extract filters from the query text
        extracted = extract_filters(query)
        logger.info(f"Extracted filters: {extracted}")

        # Merge with explicit filters (explicit takes precedence)
        if explicit_filters:
            extracted.update(explicit_filters)

        # Build ChromaDB where clause
        chroma_filters = build_chroma_filters(extracted)

        # Run retrieval with filters
        try:
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=chroma_filters,
            )
        except Exception as e:
            # If filtered query fails (e.g., no results match filters),
            # fall back to unfiltered semantic search
            logger.warning(f"Filtered search failed ({e}), falling back to unfiltered")
            results = self.retriever.retrieve(query=query, top_k=top_k)

        # If filtered search returned nothing, try without filters
        if not results and chroma_filters:
            logger.info("No results with filters, retrying without filters")
            results = self.retriever.retrieve(query=query, top_k=top_k)
            extracted["fallback_unfiltered"] = True

        return results, extracted
