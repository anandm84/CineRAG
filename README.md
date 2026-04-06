# CineRAG

A Retrieval-Augmented Generation (RAG) engine for film analysis across Telugu, Bollywood, and Hollywood cinema. Ingest scripts, reviews, and metadata — then query with natural language for trend analysis, pattern detection, hit/flop comparisons, and semantic search.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       DATA SOURCES                           │
│  TMDb API (metadata)  ·  IMSDb (scripts)  ·  IMDb (reviews) │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐      ┌─────────────────────┐
│   Ingestion Pipeline     │─────▶│  SQLite (metadata)   │
│  tmdb_client, scrapers   │      │  + Raw/Processed     │
└──────────────┬───────────┘      │    JSON files        │
               │                  └─────────────────────┘
               ▼
┌──────────────────────────┐
│   Chunking Pipeline      │
│  script_chunker (scenes) │
│  review_chunker (per-rev)│
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│   Embedding (BGE-M3)     │
│  Multilingual: te/hi/en  │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────┐
│   ChromaDB Vector Store  │
│  Metadata-filtered HNSW  │
└──────────────┬───────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│              Query Pipeline                       │
│                                                   │
│  User Query ──▶ Query Router ──▶ Filter          │
│                 (keyword)        Extraction       │
│                                    │              │
│                              Hybrid Retrieval     │
│                              (semantic + filters) │
│                                    │              │
│                              ┌─────┴──────┐      │
│                              │  Reranker   │      │
│                              │ (optional)  │      │
│                              └─────┬──────┘      │
│                                    ▼              │
│                              LLM Generation       │
│                             (Llama 3.1 8B)        │
│                                    │              │
│                                    ▼              │
│                          Response + Sources        │
└──────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────┐
│              Flask Frontend                       │
│  Search UI · Filter sidebar · Source cards        │
│  Stats dashboard · Dark theme                     │
└──────────────────────────────────────────────────┘
```

## Features

- **Multilingual retrieval** — BGE-M3 embeds Telugu, Hindi, and English into a shared vector space
- **Metadata-filtered hybrid search** — filter by language, year range, genre, director before semantic search
- **4 query types** — Trend Analysis, Pattern Detection, Hit/Flop Comparison, General Q&A
- **Keyword-based query routing** — deterministic classification (<1ms), no LLM overhead
- **150 curated movies** — 50 Telugu, 50 Bollywood, 50 Hollywood (deliberate mix of hits and flops)
- **Scene-level script chunking** — preserves screenplay structure (INT./EXT. headings)
- **Cross-encoder reranking** — optional precision boost via `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **RAGAS evaluation framework** — 50 curated Q&A pairs, faithfulness/relevancy/precision/recall metrics
- **Dark-themed web UI** — search bar, filter sidebar, source cards, stats dashboard
- **Fully local** — no paid APIs for inference (Ollama + open-source models)

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed
- TMDb API key ([free signup](https://developer.themoviedb.org/))
- 16GB RAM recommended (8B LLM + BGE-M3 embeddings)
- ~10GB disk space for models and data

### Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd cinerag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

### Pull the LLM

```bash
ollama pull llama3.1:8b
# Verify Ollama is running:
curl http://localhost:11434/api/tags
```

### Run the Pipeline

```bash
# Step 1: Ingest data (metadata, scripts, reviews)
python -m ingestion.run_ingestion

# Step 2: Build the vector index
python -m embedding.build_index

# Step 3: Start the app
python -m app.main
# Opens at http://localhost:5000
```

### Partial / Incremental Runs

```bash
# Only fetch TMDb metadata
python -m ingestion.run_ingestion --movies-only

# Only scrape scripts
python -m ingestion.run_ingestion --scripts-only

# Skip already-fetched data
python -m ingestion.run_ingestion --skip-existing

# Rebuild the vector index from scratch
python -m embedding.build_index --rebuild

# Only index reviews
python -m embedding.build_index --source-type review
```

### Evaluation

```bash
# Run the full evaluation (50 questions, requires Ollama running)
python -m evaluation.run_eval

# Quick test with 5 questions
python -m evaluation.run_eval --limit 5

# Only evaluate trend queries
python -m evaluation.run_eval --query-type trend

# Generate the markdown report
python -m evaluation.eval_report
```

### Lightweight Mode

If your machine can't handle BGE-M3 (~2.3GB), use the lighter English-only model:

```bash
# In .env:
USE_LIGHTWEIGHT_EMBEDDINGS=true
```

This switches to `all-MiniLM-L6-v2` (~80MB) — faster but loses multilingual capability.

### Optional: Enable Reranking

```bash
# In .env:
USE_RERANKER=true
```

Loads `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank top 20 results down to top 5. Improves precision but adds latency.

## Example Queries

| Query | Type | What It Does |
|---|---|---|
| "How did action choreography in Telugu cinema evolve from 2000 to 2020?" | Trend | Filters by language + year range, sorts context chronologically |
| "What storytelling patterns does SS Rajamouli use across his films?" | Pattern | Extracts director name, retrieves across multiple films |
| "What differentiates hit Telugu comedies from flops in the 2010s?" | Hit/Flop | Queries SQLite for hit/flop splits, separate retrieval per group |
| "Find intense confrontation scenes similar to The Dark Knight interrogation" | General | Pure semantic search across scripts and reviews |
| "Why did Pushpa succeed while Saaho underperformed?" | Hit/Flop | Comparative analysis with per-movie retrieval |
| "What do reviews say about Oppenheimer's cinematography?" | General | Filtered to reviews, single-movie deep dive |

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Primary language |
| Generation LLM | Llama 3.1 8B via Ollama | Free, local, strong general capability |
| Embeddings | BGE-M3 (sentence-transformers) | Multilingual (te/hi/en in one vector space) |
| Vector Store | ChromaDB | Local, free, native metadata filtering |
| Metadata Store | SQLite | Structured queries for hit/flop analysis and stats |
| Frontend | Flask + vanilla JS | Lightweight, no build step |
| Evaluation | RAGAS + heuristic metrics | Industry-standard RAG evaluation |
| Testing | pytest | Unit and integration tests |

## Data Sources

| Source | What | Coverage |
|---|---|---|
| TMDb API | Metadata (cast, crew, genres, ratings, budget, revenue) | All 150 movies |
| IMSDb | Movie scripts (full text with scene structure) | Hollywood subset |
| IMDb (cinemagoer) | User reviews (up to 25 per movie) | All 150 movies |

## Project Structure

```
cinerag/
├── config.py                     # Central configuration (paths, models, thresholds)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
│
├── data/
│   ├── seed_movies.json          # 150 curated movies (50 te, 50 hi, 50 en)
│   ├── raw/                      # Raw scraped data (scripts, reviews, metadata)
│   ├── processed/                # Normalized JSON (scripts, reviews, metadata)
│   ├── chroma/                   # ChromaDB persistent vector store
│   └── cinerag.db                # SQLite metadata database
│
├── ingestion/                    # Phase 1: Data collection
│   ├── tmdb_client.py            # TMDb API wrapper (rate-limited, ID + search)
│   ├── script_scraper.py         # IMSDb scraper (scene boundary parsing)
│   ├── review_scraper.py         # IMDb reviews via cinemagoer
│   ├── metadata_store.py         # SQLite CRUD + hit/flop classification
│   └── run_ingestion.py          # CLI orchestrator (--movies-only, --skip-existing, etc.)
│
├── chunking/                     # Phase 2: Text chunking
│   ├── chunk_models.py           # DocumentChunk Pydantic model
│   ├── script_chunker.py         # Scene-based splitting (paragraph fallback)
│   └── review_chunker.py         # Per-review chunking (sentence-split overflow)
│
├── embedding/                    # Phase 3: Embedding + indexing
│   ├── embedder.py               # BGE-M3 wrapper (batch, progress, lightweight fallback)
│   ├── vector_store.py           # ChromaDB wrapper (upsert, query, metadata filters)
│   └── build_index.py            # CLI index builder (--rebuild, --source-type)
│
├── retrieval/                    # Phase 4: Search pipeline
│   ├── retriever.py              # CineRetriever (ChromaDB -> DocumentChunk)
│   ├── hybrid_search.py          # Filter extraction + filtered semantic search
│   └── reranker.py               # Optional cross-encoder reranking
│
├── generation/                   # Phase 5: LLM generation
│   ├── llm_client.py             # Ollama API wrapper (health check, timeout)
│   ├── prompts.py                # 4 prompt templates (general, trend, pattern, hit/flop)
│   ├── query_router.py           # Keyword-based query classification
│   └── chains.py                 # CineRAGChain (full pipeline orchestrator)
│
├── evaluation/                   # Phase 6: Evaluation
│   ├── test_set.json             # 50 curated Q&A pairs with ground truth
│   ├── run_eval.py               # Evaluation runner (RAGAS + heuristic metrics)
│   └── eval_report.py            # Markdown report generator
│
├── app/                          # Phase 7: Frontend
│   ├── main.py                   # Flask app entry point
│   ├── routes.py                 # Routes: /, /query, /movies, /stats, /api/stats
│   ├── templates/
│   │   ├── base.html             # Layout with navbar
│   │   ├── index.html            # Search interface
│   │   └── stats.html            # Stats dashboard
│   └── static/
│       ├── css/style.css         # Dark theme
│       └── js/app.js             # AJAX search + stats rendering
│
└── tests/                        # Test suite
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_chains.py
```

## Key Design Decisions

1. **ChromaDB over FAISS** — Native metadata filtering is essential for hybrid search. FAISS would need a separate metadata store and manual pre-filtering.

2. **Keyword-based query routing** — Classification in <1ms vs 5-10s with an LLM. Four well-defined categories don't need neural classification.

3. **BGE-M3 for embeddings** — Best multilingual model for Telugu + Hindi + English in a single vector space. A Telugu query retrieves semantically similar English content.

4. **Separate SQLite and ChromaDB** — SQLite handles structured queries (hit/flop joins, aggregations, stats). ChromaDB handles vector similarity with metadata filters. Different tools for different jobs.

5. **Ollama over raw HuggingFace** — One command (`ollama pull`) handles quantization, memory management, and serving. No manual GGUF/GPTQ configuration.

6. **Two-tier evaluation** — Heuristic metrics (type accuracy, source hit rate, movie hit rate) always work locally. RAGAS metrics are optional and require additional LLM configuration.
