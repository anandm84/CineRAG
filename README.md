# CineRAG

A Retrieval-Augmented Generation (RAG) engine for film analysis across Telugu, Bollywood, and Hollywood cinema. Ingest scripts, reviews, and metadata — then query with natural language for trend analysis, pattern detection, hit/flop comparisons, and semantic search.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
│  TMDb API (metadata)  ·  IMSDb (scripts)  ·  IMDb (reviews)│
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────┐     ┌────────────────────┐
│   Ingestion Pipeline     │────▶│  SQLite (metadata)  │
│  tmdb_client, scrapers   │     │  + Raw/Processed    │
└──────────────┬───────────┘     │    JSON files       │
               │                 └────────────────────┘
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
┌──────────────────────────────────────────────┐
│              Query Pipeline                   │
│  User Query ──▶ Query Router ──▶ Hybrid      │
│                 (keyword)        Retrieval    │
│                                    │         │
│                              ┌─────┴──────┐  │
│                              │  Reranker   │  │
│                              │ (optional)  │  │
│                              └─────┬──────┘  │
│                                    ▼         │
│                              LLM Generation  │
│                             (Llama 3.1 8B)   │
│                                    │         │
│                                    ▼         │
│                          Response + Sources   │
└──────────────────────────────────────────────┘
```

## Features

- **Multilingual retrieval** — BGE-M3 embeds Telugu, Hindi, and English into a shared vector space
- **Metadata-filtered hybrid search** — filter by language, year range, genre, director before semantic search
- **4 query types** — Trend Analysis, Pattern Detection, Hit/Flop Comparison, General Q&A
- **150 curated movies** — 50 Telugu, 50 Bollywood, 50 Hollywood (mix of hits and flops)
- **Scene-level script chunking** — preserves screenplay structure (INT./EXT. headings)
- **RAGAS evaluation framework** — faithfulness, relevancy, precision, recall metrics
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

### Lightweight Mode

If your machine can't handle BGE-M3 (~2.3GB), use the lighter English-only model:

```bash
# In .env:
USE_LIGHTWEIGHT_EMBEDDINGS=true
```

This switches to `all-MiniLM-L6-v2` (~80MB) — faster but loses multilingual capability.

## Example Queries

| Query | Type | What It Does |
|---|---|---|
| "How did action choreography in Telugu cinema evolve from 2000 to 2020?" | Trend | Filters by year range, sorts chronologically |
| "What storytelling patterns does SS Rajamouli use across his films?" | Pattern | Filters by director, retrieves across multiple films |
| "What differentiates hit Telugu comedies from flops in the 2010s?" | Hit/Flop | Separate retrieval for hits vs flops |
| "Find intense confrontation scenes similar to The Dark Knight interrogation" | General | Pure semantic search across scripts |

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11+ |
| Generation LLM | Llama 3.1 8B via Ollama |
| Embeddings | BGE-M3 (sentence-transformers) |
| Vector Store | ChromaDB |
| Metadata Store | SQLite |
| Framework | LangChain |
| Frontend | Flask + Jinja2 |
| Evaluation | RAGAS |
| Testing | pytest |

## Data Sources

| Source | What | Coverage |
|---|---|---|
| TMDb API | Metadata (cast, crew, genres, ratings, budget, revenue) | All 150 movies |
| IMSDb | Movie scripts (full text with scene structure) | Hollywood subset |
| IMDb (cinemagoer) | User reviews (up to 25 per movie) | All 150 movies |

## Project Structure

```
cinerag/
├── config.py              # Central configuration
├── data/                  # Raw + processed data, SQLite, ChromaDB
├── ingestion/             # Data collection pipeline
├── chunking/              # Text chunking (scripts, reviews)
├── embedding/             # BGE-M3 embeddings + ChromaDB
├── retrieval/             # Hybrid search + reranking
├── generation/            # LLM client, prompts, RAG chains
├── evaluation/            # RAGAS test set + evaluation runner
├── app/                   # Flask frontend
└── tests/                 # pytest test suite
```
