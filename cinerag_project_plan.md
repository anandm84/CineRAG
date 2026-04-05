# CineRAG: Film Analysis Engine Powered by RAG

## Project Overview

CineRAG is a Retrieval-Augmented Generation (RAG) system that ingests movie scripts, reviews, and metadata across Telugu, Bollywood, and Hollywood cinema. It supports analytical queries like trend analysis, structural pattern detection, success/failure analysis, and semantic search across scripts and reviews.

The goal is to build a production-quality RAG project that demonstrates: multilingual retrieval, metadata-filtered hybrid search, cross-document analysis, multiple data modalities, and a real evaluation framework.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Primary language |
| Generation LLM | Llama 3.1 8B (via Ollama) | Free, runs locally, strong general capability |
| Embedding Model | BGE-M3 (via sentence-transformers) | Multilingual (Telugu, Hindi, English in one space), MIT license |
| Vector Store | ChromaDB | Local, free, supports metadata filtering |
| Metadata Store | SQLite | Lightweight, no setup, great for structured metadata |
| Framework | LangChain | For retrieval chains, multi-step reasoning, Ollama integration |
| Frontend | Flask + Jinja2 | Already familiar, lightweight |
| Evaluation | RAGAS | Industry-standard RAG evaluation (faithfulness, relevance, etc.) |
| Data Viz | Plotly (optional) | For trend visualizations in the frontend |
| Testing | pytest | Unit and integration tests |

### External APIs and Data Sources

- **TMDb API** (https://developer.themoviedb.org/): Free API key, provides movie metadata (cast, crew, genre, budget, revenue, ratings, release dates). Rate limit: 40 requests/10 seconds.
- **IMDb Reviews**: Scrape user reviews using `cinemagoer` (formerly IMDbPY) Python package. No API key needed.
- **IMSDb** (https://imsdb.com/): Hollywood movie scripts in HTML format. Scrape with BeautifulSoup.

---

## Project Structure

```
cinerag/
├── README.md
├── requirements.txt
├── .env                          # API keys (TMDB_API_KEY, etc.)
├── config.py                     # Central config (paths, model names, chunk sizes)
│
├── data/
│   ├── raw/
│   │   ├── scripts/              # Raw movie scripts (text files)
│   │   ├── reviews/              # Raw review JSON files
│   │   └── metadata/             # Raw TMDb JSON responses
│   ├── processed/
│   │   ├── scripts/              # Parsed script JSON (scenes extracted)
│   │   ├── reviews/              # Cleaned review JSON
│   │   └── metadata/             # Normalized metadata JSON
│   ├── chroma/                   # ChromaDB persistent vector store
│   ├── cinerag.db                # SQLite database for structured metadata
│   └── seed_movies.json          # Curated list of 150 movies
│
├── ingestion/
│   ├── __init__.py
│   ├── tmdb_client.py            # TMDb API wrapper
│   ├── script_scraper.py         # IMSDb scraper
│   ├── review_scraper.py         # IMDb review scraper via cinemagoer
│   ├── metadata_store.py         # SQLite schema and CRUD operations
│   └── run_ingestion.py          # Master ingestion orchestrator
│
├── chunking/
│   ├── __init__.py
│   ├── chunk_models.py           # Pydantic DocumentChunk model
│   ├── script_chunker.py         # Chunk scripts by scene/act boundaries
│   └── review_chunker.py         # Chunk reviews (1 review = 1 chunk typically)
│
├── embedding/
│   ├── __init__.py
│   ├── embedder.py               # BGE-M3 embedding wrapper (with lightweight fallback)
│   ├── vector_store.py           # ChromaDB operations (create, upsert, query)
│   └── build_index.py            # Script to embed all chunks and populate ChromaDB
│
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py              # Core retrieval logic with metadata filtering
│   ├── reranker.py               # Optional reranking step (cross-encoder)
│   └── hybrid_search.py          # Combine semantic search + metadata filters
│
├── generation/
│   ├── __init__.py
│   ├── llm_client.py             # Ollama client wrapper for Llama 3.1
│   ├── prompts.py                # Prompt templates for each query type
│   ├── chains.py                 # RAG chains for multi-step reasoning
│   └── query_router.py           # Route queries to appropriate chain
│
├── evaluation/
│   ├── __init__.py
│   ├── test_set.json             # 50 curated Q&A pairs with ground truth
│   ├── run_eval.py               # RAGAS evaluation runner
│   └── eval_report.py            # Generate evaluation summary/report
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # Flask app entry point
│   ├── routes.py                 # API routes and page routes
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html            # Main search/query interface
│   │   └── results.html          # Results display with sources
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_chains.py
```

---

## Implementation Plan

### Phase 1: Data Collection and Ingestion Pipeline [DONE]

#### Task 1.1: Project Setup [DONE]
- Initialized project structure, `requirements.txt`, `config.py`, `.env.example`, `.gitignore`

#### Task 1.2: TMDb Metadata Client [DONE]
- File: `ingestion/tmdb_client.py`
- `TMDbClient` class with rate-limited API calls, search + ID-based lookups
- Fetches: title, year, genres, director, top 5 cast, overview, vote_average, budget, revenue, runtime
- Saves raw JSON to `data/raw/metadata/`, normalized JSON to `data/processed/metadata/`
- Curated seed list of 150 movies (50 Telugu, 50 Bollywood, 50 Hollywood) in `data/seed_movies.json`
  - Deliberate mix of hits and flops for each language
  - All entries include pre-populated `tmdb_id` for reliable lookups

#### Task 1.3: Script Scraper (Hollywood only) [DONE]
- File: `ingestion/script_scraper.py`
- `ScriptScraper` class that scrapes movie scripts from IMSDb
- Parses scene boundaries via regex (INT./EXT. headings)
- Saves raw text to `data/raw/scripts/`, parsed JSON to `data/processed/scripts/`
- Rate-limited with 2s delay between requests

#### Task 1.4: Review Scraper [DONE]
- File: `ingestion/review_scraper.py`
- `ReviewScraper` class using cinemagoer for IMDb user reviews
- Fetches up to 25 reviews per movie (text, rating, date, helpful votes)
- Robust error handling — gracefully skips movies that fail

#### Task 1.5: SQLite Metadata Store [DONE]
- File: `ingestion/metadata_store.py`
- `MetadataStore` class with full CRUD, filtered queries, hit/flop classification
- Auto-classifies movies: hit (revenue > 2x budget OR vote_avg > 7.0), flop (revenue < budget OR vote_avg < 5.0)
- Data availability flags: `has_script`, `has_reviews`

#### Task 1.6: Master Ingestion Orchestrator [DONE]
- File: `ingestion/run_ingestion.py`
- CLI with `--skip-existing`, `--movies-only`, `--scripts-only`, `--reviews-only`
- Runs full pipeline: TMDb metadata -> SQLite -> scripts -> reviews -> summary

---

### Phase 2: Chunking Pipeline [DONE]

#### Task 2.1: Chunk Data Models [DONE]
- File: `chunking/chunk_models.py`
- `DocumentChunk` Pydantic model — universal unit from chunking to embedding to retrieval
- Fields: `chunk_id`, `movie_title`, `movie_year`, `language`, `genres`, `director`, `source_type`, `content`, `metadata`

#### Task 2.2: Script Chunker [DONE]
- File: `chunking/script_chunker.py`
- Scene-based splitting: each scene = 1 chunk (with heading prepended to content)
- Long scenes (>512 tokens) split at paragraph boundaries
- Unstructured scripts fall back to recursive character splitting with overlap
- Token estimation: `len(text) // 4`

#### Task 2.3: Review Chunker [DONE]
- File: `chunking/review_chunker.py`
- 1 review = 1 chunk (most are under 512 tokens)
- Long reviews split at paragraph then sentence boundaries
- Review title prepended to body for richer embeddings

---

### Phase 3: Embedding and Vector Store [DONE]

#### Task 3.1: Embedding Wrapper [DONE]
- File: `embedding/embedder.py`
- `Embedder` class wrapping sentence-transformers
- Default: BGE-M3 (~2.3GB, multilingual Telugu/Hindi/English)
- Fallback: `all-MiniLM-L6-v2` (~80MB, English-only) via `USE_LIGHTWEIGHT_EMBEDDINGS=true`
- Batched encoding (batch_size=32) with progress logging

#### Task 3.2: ChromaDB Vector Store [DONE]
- File: `embedding/vector_store.py`
- `VectorStore` class wrapping ChromaDB PersistentClient
- Flat metadata for each chunk: movie_title, movie_year, language, genres (comma-separated), director, source_type, plus source-specific fields (scene_heading, rating, etc.)
- `query()` supports ChromaDB `where` filters: language, year range, source_type, genre
- `add_documents()` handles embedding + upsert in batches
- Cosine similarity via HNSW index

#### Task 3.3: Index Builder [DONE]
- File: `embedding/build_index.py`
- CLI script: `python -m embedding.build_index`
- Loads processed chunks from Phase 2, embeds, upserts into ChromaDB
- `--rebuild` flag to clear and rebuild the entire index
- `--source-type script|review` to index selectively
- Prints time estimate before starting and summary after completion

---

### Phase 4: Retrieval Pipeline [DONE]

#### Task 4.1: Core Retriever [DONE]
- File: `retrieval/retriever.py`
- `CineRetriever` class wrapping VectorStore.query()
- Converts raw ChromaDB results back into `DocumentChunk` objects (reconstructs genres, source-specific metadata)
- Passes through ChromaDB `where` filters for language, year range, genre, director, source_type

#### Task 4.2: Hybrid Search [DONE]
- File: `retrieval/hybrid_search.py`
- `extract_filters()` — deterministic keyword/regex-based filter extraction from natural language:
  - Language: Telugu/Bollywood/Hollywood -> te/hi/en
  - Year ranges: "from 2005 to 2010", "in 2015", decade mentions ("2010s", "90s")
  - Genre: 25+ genre keywords normalized to TMDb genre names
  - Director: "directed by X", "X's films"
  - Source type: script/review inferred from keywords like "scene", "dialogue", "audience"
  - Hit/flop: "blockbuster", "disaster", etc.
- `build_chroma_filters()` — converts extracted filters to ChromaDB `$and` / `$gte` / `$lte` / `$contains` syntax
- `HybridSearcher` — runs filtered search with automatic fallback to unfiltered if no results match
- Supports explicit frontend filters that override extraction

#### Task 4.3: Reranker [DONE]
- File: `retrieval/reranker.py`
- `Reranker` class using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks top 20 -> top 5 by cross-encoder relevance score
- Togglable via `USE_RERANKER=true` in config
- `maybe_rerank()` convenience function — respects config toggle, pass-through when disabled
- Lazy singleton pattern to avoid loading the model until needed

---

### Phase 5: Generation Pipeline [DONE]

#### Task 5.1: Ollama LLM Client [DONE]
- File: `generation/llm_client.py`
- `OllamaClient` with `generate()` and `health_check()`
- Timeout handling (120s default), connection error messages with fix instructions
- Logs generation timing from Ollama's response metadata

#### Task 5.2: Prompt Templates [DONE]
- File: `generation/prompts.py`
- Four templates: General Q&A, Trend Analysis, Pattern Detection, Hit/Flop Comparison
- Shared system message establishing the film analysis expert persona
- All templates instruct the LLM to cite movies, years, and source types

#### Task 5.3: Query Router [DONE]
- File: `generation/query_router.py`
- Keyword-based scoring classification (<1ms per query, no LLM)
- Year range regex boosts trend score; possessive patterns boost pattern score
- Returns: 'trend', 'pattern', 'hit_flop', or 'general'

#### Task 5.4: RAG Chains [DONE]
- File: `generation/chains.py`
- `CineRAGChain` with three retrieval strategies:
  - Standard: general Q&A and pattern queries
  - Trend: retrieves + sorts context chronologically
  - Hit/Flop: queries SQLite for hit/flop titles, separate retrieval per group
- `_format_context()` builds numbered, labeled context blocks with source metadata
- Response includes: answer, query_type, extracted filters, top 5 source snippets

---

### Phase 6: Evaluation Framework

#### Task 6.1: Curated Test Set
- File: `evaluation/test_set.json`
- 50 Q&A pairs: 10 trend, 10 pattern, 10 hit/flop, 20 general
- Each entry: question, query_type, ground_truth_answer, relevant_movies, expected_source_types

#### Task 6.2: RAGAS Evaluation Runner
- File: `evaluation/run_eval.py`
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall

#### Task 6.3: Evaluation Report
- File: `evaluation/eval_report.py`
- Markdown report with per-query-type breakdown and worst-performing queries

---

### Phase 7: Frontend

#### Task 7.1: Flask Application
- Routes: `GET /`, `POST /query`, `GET /movies`, `GET /stats`

#### Task 7.2: Search Interface
- Dark-themed, minimal UI with search bar, filter sidebar, result cards
- AJAX-based query submission

#### Task 7.3: Stats Dashboard (Optional)
- Movie counts by language, chunk distribution, top reviewed movies, RAGAS scores

---

### Phase 8: Documentation and Polish

#### Task 8.1: README.md [DONE]
#### Task 8.2: Architecture Diagram

---

## Key Design Decisions and Rationale

1. **Why ChromaDB over FAISS?** ChromaDB supports metadata filtering natively, which is essential for our hybrid search (filter by year, language, genre before semantic search). FAISS would require a separate metadata store and manual filtering logic.

2. **Why keyword-based query routing instead of LLM-based?** Speed and cost. Routing happens on every query and should be <10ms. Using the LLM for routing would add 5-10 seconds of latency. Keyword-based classification is "good enough" for four well-defined categories.

3. **Why BGE-M3 for embeddings?** It's the best multilingual embedding model that handles Telugu, Hindi, and English in a single embedding space. This means a Telugu query can find semantically similar English content and vice versa.

4. **Why separate SQLite from ChromaDB?** ChromaDB metadata is great for filtering during retrieval, but SQLite gives us structured queries for the hit/flop analysis (joins, aggregations) and the stats dashboard. They serve different purposes.

5. **Why Ollama instead of direct Hugging Face transformers?** Ollama handles quantization, model management, and memory optimization out of the box. Running a raw Hugging Face model requires manual GGUF/GPTQ setup. Ollama is one command: `ollama pull llama3.1:8b`.

---

## Testing Strategy

- **Unit tests**: Test each component in isolation (script chunker, review chunker, filter extraction, prompt formatting)
- **Integration tests**: Test the full retrieval pipeline (query -> filter extraction -> ChromaDB search -> response)
- **Evaluation tests**: RAGAS metrics on the 50-question test set
- Run tests with: `pytest tests/ -v`

---

## Environment Setup Prerequisites

Before running any code:

1. **Install Ollama**: https://ollama.com/download
2. **Pull the model**: `ollama pull llama3.1:8b`
3. **Verify Ollama is running**: `curl http://localhost:11434/api/tags` should return a JSON response
4. **Get TMDb API key**: Sign up at https://developer.themoviedb.org/ (free)
5. **Python 3.11+** installed
6. **At least 16GB RAM recommended** (8B model + BGE-M3 embeddings)
7. **~10GB disk space** for models and data
