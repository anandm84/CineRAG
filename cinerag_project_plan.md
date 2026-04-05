# CineRAG: Film Analysis Engine Powered by RAG

## Project Overview

CineRAG is a Retrieval-Augmented Generation (RAG) system that ingests movie scripts, subtitles, reviews, and metadata across Telugu, Bollywood, and Hollywood cinema. It supports analytical queries like trend analysis, structural pattern detection, success/failure analysis, and semantic dialogue search.

The goal is to build a production-quality RAG project that demonstrates: multilingual retrieval, metadata-filtered hybrid search, cross-document analysis, multiple data modalities, and a real evaluation framework.

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| Language | Python 3.11+ | Primary language |
| Generation LLM | Llama 3.3 8B (Q4_K_M via Ollama) | Free, runs locally, strong general capability |
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
- **OpenSubtitles**: Subtitle files (.srt) for dialogue data across languages. Use the `opensubtitles-com` Python package or download SRT files directly.
- **IMDb Reviews**: Scrape user reviews using `cinemagoer` (formerly IMDbPY) Python package. No API key needed.
- **IMSDb** (https://imsdb.com/): Hollywood movie scripts in HTML format. Scrape with BeautifulSoup.
- **Letterboxd**: Optional additional review source. Scrape with requests + BeautifulSoup.

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
│   │   ├── subtitles/            # Raw .srt files organized by language
│   │   ├── scripts/              # Raw movie scripts (HTML or text)
│   │   ├── reviews/              # Raw review JSON files
│   │   └── metadata/             # Raw TMDb JSON responses
│   ├── processed/
│   │   ├── subtitles/            # Parsed subtitle JSON
│   │   ├── scripts/              # Parsed script JSON
│   │   ├── reviews/              # Cleaned review JSON
│   │   └── metadata/             # Normalized metadata JSON
│   └── cinerag.db                # SQLite database for structured metadata
│
├── ingestion/
│   ├── __init__.py
│   ├── tmdb_client.py            # TMDb API wrapper
│   ├── subtitle_parser.py        # SRT file parser
│   ├── script_scraper.py         # IMSDb scraper
│   ├── review_scraper.py         # IMDb/Letterboxd review scraper
│   ├── metadata_store.py         # SQLite schema and CRUD operations
│   └── run_ingestion.py          # Master ingestion orchestrator
│
├── chunking/
│   ├── __init__.py
│   ├── subtitle_chunker.py       # Chunk subtitles by dialogue blocks
│   ├── script_chunker.py         # Chunk scripts by scene/act boundaries
│   ├── review_chunker.py         # Chunk reviews (1 review = 1 chunk typically)
│   └── chunk_models.py           # Pydantic models for chunk schema
│
├── embedding/
│   ├── __init__.py
│   ├── embedder.py               # BGE-M3 embedding wrapper
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
│   ├── llm_client.py             # Ollama client wrapper for Llama 3.3
│   ├── prompts.py                # Prompt templates for each query type
│   ├── chains.py                 # LangChain chains for multi-step reasoning
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
    ├── test_subtitle_parser.py
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_chains.py
```

---

## Implementation Plan

### Phase 1: Data Collection and Ingestion Pipeline

#### Task 1.1: Project Setup
- Initialize the project structure as shown above
- Create `requirements.txt` with dependencies:
  ```
  langchain>=0.3.0
  langchain-community>=0.3.0
  langchain-ollama>=0.2.0
  chromadb>=0.5.0
  sentence-transformers>=3.0.0
  flask>=3.0
  pydantic>=2.0
  requests>=2.31
  beautifulsoup4>=4.12
  cinemagoer>=2023.5.1
  pysrt>=1.1.2
  ragas>=0.2.0
  plotly>=5.18
  python-dotenv>=1.0
  pytest>=8.0
  ```
- Create `config.py` with:
  ```python
  from pathlib import Path
  from dotenv import load_dotenv
  import os

  load_dotenv()

  # Paths
  BASE_DIR = Path(__file__).parent
  DATA_DIR = BASE_DIR / "data"
  RAW_DIR = DATA_DIR / "raw"
  PROCESSED_DIR = DATA_DIR / "processed"
  DB_PATH = DATA_DIR / "cinerag.db"
  CHROMA_DIR = DATA_DIR / "chroma"

  # API Keys
  TMDB_API_KEY = os.getenv("TMDB_API_KEY")

  # Model Config
  EMBEDDING_MODEL = "BAAI/bge-m3"
  LLM_MODEL = "llama3.1:8b"  # Ollama model name
  OLLAMA_BASE_URL = "http://localhost:11434"

  # Chunking Config
  SUBTITLE_CHUNK_WINDOW = 30       # seconds of dialogue per chunk
  SCRIPT_CHUNK_MAX_TOKENS = 512
  REVIEW_CHUNK_MAX_TOKENS = 512

  # Retrieval Config
  TOP_K_RETRIEVAL = 20
  TOP_K_RERANK = 5
  ```
- Create `.env.example` with placeholder for `TMDB_API_KEY`

#### Task 1.2: TMDb Metadata Client
- File: `ingestion/tmdb_client.py`
- Build a client class `TMDbClient` that:
  - Accepts a list of movie titles (or TMDb IDs) and fetches metadata
  - Retrieves: title, original_title, release_date, genres, overview, vote_average, vote_count, budget, revenue, runtime, original_language, production_countries, credits (director, top 5 cast)
  - Handles rate limiting (40 req/10s) with a simple sleep-based throttle
  - Saves raw JSON responses to `data/raw/metadata/`
  - Normalizes and saves processed metadata to `data/processed/metadata/`
- Include a curated seed list of ~100-150 movies to start:
  - ~50 Telugu films (mix of hits and flops across 2000-2024, covering masala, comedy, drama): e.g., "Pokiri", "Magadheera", "Baahubali", "Ala Vaikunthapurramuloo", "Arya", "Bommarillu", "Khaleja", "Eega", "Jersey", "Pushpa", "RRR", "Athadu", etc.
  - ~50 Bollywood films: e.g., "Lagaan", "Dil Chahta Hai", "Gangs of Wasseypur", "3 Idiots", "Dangal", "Andhadhun", "Tumbbad", "Jawan", "Animal", etc.
  - ~50 Hollywood films: e.g., "The Dark Knight", "Inception", "Parasite", "Everything Everywhere All at Once", "Oppenheimer", "Pulp Fiction", "The Godfather", "Mad Max Fury Road", etc.
- Store the seed list in a `data/seed_movies.json` file with fields: `title`, `year`, `language`, `tmdb_id` (if known)

#### Task 1.3: Subtitle Parser
- File: `ingestion/subtitle_parser.py`
- Use `pysrt` to parse .srt files
- Output format per subtitle file (save as JSON):
  ```json
  {
    "movie_title": "Pokiri",
    "language": "te",
    "source_file": "pokiri_2006.srt",
    "dialogues": [
      {
        "index": 1,
        "start_time": "00:01:23,456",
        "end_time": "00:01:26,789",
        "text": "Dialogue line here",
        "start_seconds": 83.456,
        "end_seconds": 86.789
      }
    ]
  }
  ```
- Handle encoding issues (Telugu subtitles may be UTF-8 or other encodings)
- Strip HTML tags and formatting artifacts from subtitle text
- NOTE: For the initial build, if .srt files are hard to source programmatically, create a manual download workflow documented in README. User downloads SRT files from OpenSubtitles.org and places them in `data/raw/subtitles/{language}/`. The parser processes whatever is there.

#### Task 1.4: Script Scraper (Hollywood only)
- File: `ingestion/script_scraper.py`
- Scrape movie scripts from IMSDb (imsdb.com)
- Parse HTML to extract clean script text
- Save raw HTML and processed text
- Only scrape scripts for movies in the seed list that are available on IMSDb
- Respect rate limits and add delays between requests
- Output format (save as JSON):
  ```json
  {
    "movie_title": "The Dark Knight",
    "source": "imsdb",
    "script_text": "Full script text here...",
    "scenes": [
      {
        "scene_number": 1,
        "heading": "INT. BANK - DAY",
        "content": "Scene content..."
      }
    ]
  }
  ```
- Use regex patterns to detect scene boundaries:
  - Scene headings typically match: `^(INT\.|EXT\.|INT/EXT\.).*`
  - Also look for `FADE IN:`, `CUT TO:`, `DISSOLVE TO:` as scene transitions

#### Task 1.5: Review Scraper
- File: `ingestion/review_scraper.py`
- Use `cinemagoer` package to fetch IMDb user reviews
- For each movie in the seed list:
  - Fetch up to 25 user reviews
  - Extract: review text, rating (if available), review date, helpful votes
- Save as JSON per movie:
  ```json
  {
    "movie_title": "Pokiri",
    "imdb_id": "tt0492578",
    "reviews": [
      {
        "review_text": "...",
        "rating": 9,
        "date": "2006-05-15",
        "helpful_votes": 42,
        "title": "Review title"
      }
    ]
  }
  ```
- Add retry logic and error handling for failed requests
- If cinemagoer doesn't work well for a particular movie, skip and log it

#### Task 1.6: SQLite Metadata Store
- File: `ingestion/metadata_store.py`
- Create SQLite schema:
  ```sql
  CREATE TABLE movies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      tmdb_id INTEGER UNIQUE,
      imdb_id TEXT,
      title TEXT NOT NULL,
      original_title TEXT,
      year INTEGER,
      language TEXT,           -- 'te', 'hi', 'en'
      genres TEXT,             -- JSON array stored as text
      director TEXT,
      top_cast TEXT,           -- JSON array stored as text
      overview TEXT,
      vote_average REAL,
      vote_count INTEGER,
      budget INTEGER,
      revenue INTEGER,
      runtime INTEGER,
      is_hit INTEGER,          -- 1 = hit, 0 = flop, NULL = unknown (based on revenue > 2x budget or vote_average > 7)
      has_subtitles INTEGER DEFAULT 0,
      has_script INTEGER DEFAULT 0,
      has_reviews INTEGER DEFAULT 0
  );
  ```
- Provide functions: `insert_movie()`, `get_movie_by_title()`, `get_movies_by_filter()`, `update_data_availability()`
- Populate from processed TMDb metadata

#### Task 1.7: Master Ingestion Orchestrator
- File: `ingestion/run_ingestion.py`
- CLI script that runs the full ingestion pipeline:
  1. Load seed movie list
  2. Fetch TMDb metadata for all movies
  3. Populate SQLite database
  4. Parse any subtitle files found in `data/raw/subtitles/`
  5. Scrape available scripts from IMSDb
  6. Scrape IMDb reviews
  7. Update data availability flags in SQLite
  8. Print summary: "Ingested X movies, Y subtitles, Z scripts, W reviews"
- Support `--skip-existing` flag to avoid re-fetching data
- Support `--movies-only`, `--subtitles-only`, `--reviews-only` flags for partial runs

---

### Phase 2: Chunking Pipeline

#### Task 2.1: Chunk Data Models
- File: `chunking/chunk_models.py`
- Define Pydantic models:
  ```python
  from pydantic import BaseModel
  from typing import Optional

  class DocumentChunk(BaseModel):
      chunk_id: str                    # Unique ID: "{movie_id}_{source}_{index}"
      movie_title: str
      movie_year: int
      language: str                    # 'te', 'hi', 'en'
      genres: list[str]
      director: str
      source_type: str                 # 'subtitle', 'script', 'review'
      content: str                     # The actual text chunk
      metadata: dict                   # Additional source-specific metadata
      # For subtitles: start_time, end_time
      # For scripts: scene_heading, act_number
      # For reviews: rating, review_date
  ```

#### Task 2.2: Subtitle Chunker
- File: `chunking/subtitle_chunker.py`
- Chunking strategy: Group consecutive dialogue lines into chunks based on a time window (default 30 seconds)
- Each chunk should contain:
  - Combined dialogue text from the time window
  - Start and end timestamps of the window
  - Movie metadata attached
- Handle edge cases:
  - Very long monologues (split at sentence boundaries)
  - Action descriptions in subtitles (keep them, they provide context)
  - Song lyrics in subtitles (tag them as "song" in metadata if detectable)
- Output: list of `DocumentChunk` objects

#### Task 2.3: Script Chunker
- File: `chunking/script_chunker.py`
- Chunking strategy: Split by scene boundaries (detected via scene headings)
- If a scene is too long (>512 tokens), split at paragraph boundaries within the scene
- Preserve scene headings as metadata (`scene_heading` field)
- If scene boundaries can't be detected (unformatted script), fall back to recursive character text splitting with 512 token chunks and 50 token overlap
- Output: list of `DocumentChunk` objects

#### Task 2.4: Review Chunker
- File: `chunking/review_chunker.py`
- Chunking strategy: Each review is typically 1 chunk (most reviews are under 512 tokens)
- If a review exceeds 512 tokens, split at paragraph boundaries
- Attach rating and review date as metadata
- Output: list of `DocumentChunk` objects

---

### Phase 3: Embedding and Vector Store

#### Task 3.1: Embedding Wrapper
- File: `embedding/embedder.py`
- Load BGE-M3 model using `sentence-transformers`:
  ```python
  from sentence_transformers import SentenceTransformer

  class Embedder:
      def __init__(self, model_name: str = "BAAI/bge-m3"):
          self.model = SentenceTransformer(model_name)

      def embed(self, texts: list[str]) -> list[list[float]]:
          return self.model.encode(texts, normalize_embeddings=True).tolist()

      def embed_query(self, query: str) -> list[float]:
          return self.model.encode(query, normalize_embeddings=True).tolist()
  ```
- Add batching support for large document sets (batch size 32)
- Log progress during embedding (e.g., "Embedded 500/2000 chunks")
- NOTE: BGE-M3 is ~2.3GB. First run will download the model. This is expected.
- FALLBACK: If the machine cannot handle BGE-M3 (needs ~4GB RAM for the model), fall back to `all-MiniLM-L6-v2` (~80MB) which is much lighter but English-only. Add a config flag `USE_LIGHTWEIGHT_EMBEDDINGS=true` in config.py to toggle this.

#### Task 3.2: ChromaDB Vector Store
- File: `embedding/vector_store.py`
- Wrapper class `VectorStore`:
  - `create_collection(name)`: Create or get a ChromaDB collection
  - `add_documents(chunks: list[DocumentChunk])`: Embed and upsert chunks with metadata
  - `query(query_text, top_k, filters)`: Semantic search with optional metadata filters
  - Metadata fields stored in ChromaDB for each chunk:
    - `movie_title`, `movie_year`, `language`, `genres` (as comma-separated string), `director`, `source_type`, `rating` (for reviews)
  - Support filter syntax for ChromaDB's `where` clause:
    - Filter by language: `{"language": "te"}`
    - Filter by year range: `{"movie_year": {"$gte": 2000, "$lte": 2015}}`
    - Filter by source type: `{"source_type": "review"}`
    - Filter by genre: Use `$contains` on genres string
- Use persistent ChromaDB storage at `data/chroma/`

#### Task 3.3: Index Builder
- File: `embedding/build_index.py`
- Script that:
  1. Loads all processed chunks from Phase 2
  2. Embeds them in batches using the Embedder
  3. Upserts into ChromaDB with metadata
  4. Prints summary: "Indexed X chunks (Y subtitle, Z script, W review) for N movies"
- Support `--rebuild` flag to clear and rebuild the entire index
- Support `--source-type subtitle|script|review` to index only one type
- Estimate and print total embedding time before starting

---

### Phase 4: Retrieval Pipeline

#### Task 4.1: Core Retriever
- File: `retrieval/retriever.py`
- Class `CineRetriever`:
  - `retrieve(query, top_k=20, filters=None) -> list[DocumentChunk]`
  - Embeds the query using BGE-M3
  - Queries ChromaDB with optional metadata filters
  - Returns ranked list of chunks with similarity scores
  - Supports filtering by: language, year range, genre, director, source_type

#### Task 4.2: Hybrid Search
- File: `retrieval/hybrid_search.py`
- Combine semantic search with metadata-based pre-filtering
- Query parsing logic to extract structured filters from natural language:
  - "Telugu movies from 2005 to 2010" -> extract `language="te"`, `year_range=(2005, 2010)`
  - "comedy films directed by Trivikram" -> extract `genre="comedy"`, `director="Trivikram Srinivas"`
  - Use simple keyword matching and regex patterns for filter extraction. Do NOT use the LLM for this; keep it fast and deterministic.
  - Filters that can be extracted: language (Telugu/Bollywood/Hollywood -> te/hi/en), year/decade mentions, genre keywords, director names, "hit"/"flop" keywords
- If filter extraction finds structured filters, apply them as ChromaDB `where` clauses before semantic search
- If no filters are detected, fall back to pure semantic search

#### Task 4.3: Reranker (Optional Enhancement)
- File: `retrieval/reranker.py`
- Use a cross-encoder model for reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Takes the top 20 results from retrieval, reranks to top 5
- This is optional -- build it but make it togglable via config (`USE_RERANKER=true/false`)
- Reranking improves precision significantly but adds latency

---

### Phase 5: Generation Pipeline

#### Task 5.1: Ollama LLM Client
- File: `generation/llm_client.py`
- Wrapper for Ollama API:
  ```python
  import requests

  class OllamaClient:
      def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
          self.model = model
          self.base_url = base_url

      def generate(self, prompt: str, system: str = None, temperature: float = 0.3) -> str:
          payload = {
              "model": self.model,
              "prompt": prompt,
              "system": system or "",
              "stream": False,
              "options": {"temperature": temperature}
          }
          response = requests.post(f"{self.base_url}/api/generate", json=payload)
          return response.json()["response"]
  ```
- Add timeout handling (local models can be slow)
- Add a health check method to verify Ollama is running before queries

#### Task 5.2: Prompt Templates
- File: `generation/prompts.py`
- Define prompt templates for each query type:

  **General Q&A prompt:**
  ```
  You are a film analysis expert with deep knowledge of Telugu, Bollywood, and Hollywood cinema.
  Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so.
  Always cite which movie and source (review/script/subtitle) you're drawing from.

  Context:
  {context}

  Question: {question}

  Answer:
  ```

  **Trend Analysis prompt:**
  ```
  You are a film historian analyzing trends across cinema.
  Based on the following excerpts from movies spanning different years, identify and describe the trend or evolution the user is asking about.
  Organize your analysis chronologically. Cite specific movies and years as evidence.

  Context (sorted by year):
  {context}

  Question: {question}

  Analysis:
  ```

  **Comparison/Pattern prompt:**
  ```
  You are a film critic analyzing patterns across a director's or genre's body of work.
  Based on the following excerpts, identify recurring patterns, similarities, and differences.
  Be specific -- reference actual dialogue, scenes, or review sentiments.

  Context:
  {context}

  Question: {question}

  Pattern Analysis:
  ```

  **Hit vs Flop Analysis prompt:**
  ```
  You are a film industry analyst.
  Based on the following reviews and metadata for films categorized as hits and flops, analyze what differentiates them.
  Consider: storytelling elements, audience reception themes, critical praise/criticism patterns.

  Hit Films Context:
  {hit_context}

  Flop Films Context:
  {flop_context}

  Question: {question}

  Analysis:
  ```

#### Task 5.3: Query Router
- File: `generation/query_router.py`
- Classify incoming queries into one of four types and route to the appropriate retrieval + prompt strategy:
  1. **Trend Analysis**: Detected by keywords like "evolve", "change over time", "trend", "how did X change", "from YEAR to YEAR", "over the years", "decade"
  2. **Pattern Detection**: Detected by keywords like "pattern", "reuse", "similar structure", "recurring", "signature style", "always does"
  3. **Hit/Flop Analysis**: Detected by keywords like "hit", "flop", "succeed", "fail", "work vs", "what makes", "box office"
  4. **Semantic Search / General Q&A**: Default fallback for everything else, including "find similar scenes", "dialogues like", etc.
- Use keyword-based classification (simple and fast). Do NOT use the LLM for routing.
- Each query type triggers a different retrieval strategy:
  - Trend: Filter by year range, sort chunks chronologically, use trend prompt
  - Pattern: Filter by director, retrieve across multiple films, use pattern prompt
  - Hit/Flop: Separate retrieval for hit and flop films (using `is_hit` flag from SQLite), use comparison prompt
  - General: Standard retrieval with hybrid search, use general Q&A prompt

#### Task 5.4: RAG Chains
- File: `generation/chains.py`
- Implement the full RAG pipeline using LangChain:
  ```python
  class CineRAGChain:
      def __init__(self, retriever, llm_client, query_router):
          self.retriever = retriever
          self.llm_client = llm_client
          self.query_router = query_router

      def run(self, query: str) -> dict:
          # 1. Route the query
          query_type = self.query_router.classify(query)

          # 2. Extract filters from query
          filters = self.query_router.extract_filters(query)

          # 3. Retrieve relevant chunks
          chunks = self.retriever.retrieve(query, filters=filters)

          # 4. Select prompt template
          prompt = self.query_router.get_prompt(query_type)

          # 5. Format context from chunks
          context = self._format_context(chunks, query_type)

          # 6. Generate response
          response = self.llm_client.generate(
              prompt.format(context=context, question=query)
          )

          # 7. Return response with sources
          return {
              "answer": response,
              "query_type": query_type,
              "sources": [
                  {
                      "movie": c.movie_title,
                      "year": c.movie_year,
                      "source_type": c.source_type,
                      "snippet": c.content[:200]
                  }
                  for c in chunks[:5]
              ]
          }
  ```

---

### Phase 6: Evaluation Framework

#### Task 6.1: Curated Test Set
- File: `evaluation/test_set.json`
- Create 50 question-answer pairs across the four query types:
  - 10 trend analysis questions (e.g., "How did action choreography in Telugu cinema evolve from 2000 to 2020?")
  - 10 pattern detection questions (e.g., "What storytelling patterns does SS Rajamouli use across his films?")
  - 10 hit/flop analysis questions (e.g., "What differentiates hit Telugu comedies from flops in the 2010s?")
  - 20 general Q&A / semantic search questions (e.g., "Find me intense confrontation dialogues similar to the courtroom scene in Vakeel Saab")
- Each entry should have:
  ```json
  {
    "question": "...",
    "query_type": "trend|pattern|hit_flop|general",
    "ground_truth_answer": "...",
    "relevant_movies": ["Pokiri", "Magadheera"],
    "expected_source_types": ["review", "subtitle"]
  }
  ```
- Ground truth answers can be brief (2-3 sentences) -- they're for evaluation metrics, not perfection

#### Task 6.2: RAGAS Evaluation Runner
- File: `evaluation/run_eval.py`
- Run the CineRAG chain on all 50 test questions
- Evaluate using RAGAS metrics:
  - **Faithfulness**: Is the answer grounded in the retrieved context?
  - **Answer Relevancy**: Is the answer relevant to the question?
  - **Context Precision**: Are the retrieved chunks relevant?
  - **Context Recall**: Were all necessary chunks retrieved?
- Save results to `evaluation/results.json`
- Print summary table with per-query-type scores

#### Task 6.3: Evaluation Report
- File: `evaluation/eval_report.py`
- Generate a markdown report summarizing:
  - Overall RAGAS scores
  - Per-query-type breakdown
  - Worst-performing queries (to identify improvement areas)
  - Retrieval quality analysis (are the right chunks being retrieved?)
- Save to `evaluation/EVAL_REPORT.md`

---

### Phase 7: Frontend

#### Task 7.1: Flask Application
- File: `app/main.py`
- Create Flask app with routes:
  - `GET /` -- Main search interface
  - `POST /query` -- Submit a query, returns JSON response
  - `GET /movies` -- List all indexed movies with metadata
  - `GET /stats` -- Show index statistics (chunk counts, movie counts by language)

#### Task 7.2: Search Interface
- File: `app/templates/index.html`
- Clean, minimal search interface:
  - Large search bar at top
  - Example queries as clickable suggestions below the search bar
  - Filter sidebar with checkboxes for: Language (Telugu, Hindi, English), Source Type (Scripts, Reviews, Subtitles), Year range slider
  - Results displayed as cards showing:
    - Answer text with highlighted source citations
    - Source cards below showing movie title, year, source type, and relevant snippet
    - Query type badge (Trend, Pattern, Hit/Flop, Search)
- Use a simple, dark-themed design. No frameworks needed -- vanilla HTML/CSS/JS.
- AJAX for query submission (no page reload)

#### Task 7.3: Stats Dashboard (Optional)
- A simple `/stats` page showing:
  - Total movies indexed by language (bar chart)
  - Chunk distribution by source type (pie chart)
  - Top 10 most-reviewed movies
  - RAGAS evaluation scores
- Use Plotly.js for charts

---

### Phase 8: Documentation and Polish

#### Task 8.1: README.md
- Project title and one-line description
- Architecture diagram (text-based, using boxes and arrows)
- Features list
- Setup instructions:
  1. Prerequisites (Python 3.11+, Ollama installed)
  2. Clone repo
  3. Install dependencies
  4. Get TMDb API key
  5. Download subtitle files (manual step, with instructions)
  6. Run ingestion: `python -m ingestion.run_ingestion`
  7. Build index: `python -m embedding.build_index`
  8. Start Ollama: `ollama run llama3.1:8b`
  9. Start app: `python -m app.main`
- Example queries with screenshots
- Evaluation results summary
- Tech stack section
- Future improvements section

#### Task 8.2: Architecture Diagram
- Create a text-based or mermaid diagram showing:
  ```
  [Data Sources] -> [Ingestion Pipeline] -> [SQLite + Raw Files]
                                                    |
                                            [Chunking Pipeline]
                                                    |
                                            [Embedding (BGE-M3)]
                                                    |
                                            [ChromaDB Vector Store]
                                                    |
  [User Query] -> [Query Router] -> [Hybrid Retrieval] -> [Reranker]
                                                              |
                                                    [LLM Generation (Llama 3.3)]
                                                              |
                                                    [Response + Sources]
  ```

---

## Key Design Decisions and Rationale

1. **Why ChromaDB over FAISS?** ChromaDB supports metadata filtering natively, which is essential for our hybrid search (filter by year, language, genre before semantic search). FAISS would require a separate metadata store and manual filtering logic.

2. **Why keyword-based query routing instead of LLM-based?** Speed and cost. Routing happens on every query and should be <10ms. Using the LLM for routing would add 5-10 seconds of latency. Keyword-based classification is "good enough" for four well-defined categories.

3. **Why BGE-M3 for embeddings?** It's the best multilingual embedding model that handles Telugu, Hindi, and English in a single embedding space. This means a Telugu query can find semantically similar English content and vice versa.

4. **Why separate SQLite from ChromaDB?** ChromaDB metadata is great for filtering during retrieval, but SQLite gives us structured queries for the hit/flop analysis (joins, aggregations) and the stats dashboard. They serve different purposes.

5. **Why Ollama instead of direct Hugging Face transformers?** Ollama handles quantization, model management, and memory optimization out of the box. Running a raw Hugging Face model requires manual GGUF/GPTQ setup. Ollama is one command: `ollama pull llama3.1:8b`.

---

## Testing Strategy

- **Unit tests**: Test each component in isolation (subtitle parser, chunker, filter extraction, prompt formatting)
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
