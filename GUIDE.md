# CineRAG: A Complete Beginner's Guide

This guide explains every part of the CineRAG project as if you're new to software engineering and AI. We'll walk through what the project does, why each piece exists, and how they all connect together.

---

## Table of Contents

1. [What Is This Project?](#what-is-this-project)
2. [What Is RAG?](#what-is-rag)
3. [The Big Picture](#the-big-picture)
4. [Project Files Overview](#project-files-overview)
5. [Phase 1: Getting the Data (Ingestion)](#phase-1-getting-the-data-ingestion)
6. [Phase 2: Cutting Text Into Pieces (Chunking)](#phase-2-cutting-text-into-pieces-chunking)
7. [Phase 3: Converting Text to Numbers (Embedding)](#phase-3-converting-text-to-numbers-embedding)
8. [Phase 4: Finding Relevant Pieces (Retrieval)](#phase-4-finding-relevant-pieces-retrieval)
9. [Phase 5: Generating Answers (Generation)](#phase-5-generating-answers-generation)
10. [Phase 6: Measuring Quality (Evaluation)](#phase-6-measuring-quality-evaluation)
11. [Phase 7: The Web Interface (Frontend)](#phase-7-the-web-interface-frontend)
12. [Key Concepts Explained](#key-concepts-explained)
13. [How a Query Flows Through the System](#how-a-query-flows-through-the-system)
14. [Why We Chose Each Tool](#why-we-chose-each-tool)

---

## What Is This Project?

CineRAG is a **film analysis engine**. You can ask it questions about movies — Telugu, Bollywood, and Hollywood — and it gives you intelligent answers based on real data (scripts, reviews, and metadata).

For example, you could ask:
- "How did action movies in Telugu cinema evolve from 2000 to 2020?"
- "What storytelling patterns does SS Rajamouli use across his films?"
- "Why did Pushpa succeed while Saaho underperformed?"

The system doesn't just make up answers. It **finds relevant information** from its database of movie scripts and reviews, and then uses an AI language model to **write an answer based on that evidence**. This is what makes it a RAG system.

---

## What Is RAG?

**RAG stands for Retrieval-Augmented Generation.** Let's break that down:

- **Generation**: An AI language model (like ChatGPT or Llama) generates a text answer.
- **Retrieval**: Before generating, the system retrieves relevant documents from a database.
- **Augmented**: The retrieved documents are added to the prompt, so the AI's answer is grounded in real data.

### Why not just ask the AI directly?

If you ask a language model "What do reviews say about Rangasthalam?", it might:
- Give a generic answer based on its training data
- Make up facts ("hallucinate")
- Not know about lesser-known films

With RAG, we first **find actual reviews of Rangasthalam** from our database, then give those reviews to the AI and say "answer based on THESE reviews." This makes the answer factual and specific.

### The RAG formula

```
User Question
      |
      v
[RETRIEVE relevant documents from our database]
      |
      v
[COMBINE the question + retrieved documents into a prompt]
      |
      v
[GENERATE an answer using the AI model]
      |
      v
Answer + Sources
```

---

## The Big Picture

Here's how data flows through the entire system:

```
STEP 1: COLLECT DATA
   TMDb API ──> movie metadata (titles, cast, genres, ratings)
   IMSDb ──> movie scripts (The Dark Knight, Inception, etc.)
   IMDb ──> user reviews (what audiences thought)
         |
         v
STEP 2: STORE RAW DATA
   SQLite database (structured metadata)
   JSON files (scripts, reviews)
         |
         v
STEP 3: CHUNK THE TEXT
   Break scripts into scenes
   Keep each review as one piece
         |
         v
STEP 4: CONVERT TO NUMBERS (EMBED)
   Each text chunk becomes a list of numbers (a "vector")
   Store vectors in ChromaDB
         |
         v
STEP 5: USER ASKS A QUESTION
   "What patterns does Nolan use?"
         |
         v
STEP 6: FIND RELEVANT CHUNKS
   Search ChromaDB for chunks similar to the question
   Filter by language, year, genre, etc.
         |
         v
STEP 7: GENERATE ANSWER
   Send the question + relevant chunks to Llama 3.1
   AI writes an answer citing specific movies
         |
         v
STEP 8: SHOW TO USER
   Display answer + source cards in the web UI
```

---

## Project Files Overview

Here's every file in the project and what it does:

### Root files

| File | What it does |
|---|---|
| `config.py` | **The control center.** Every setting lives here — file paths, model names, chunk sizes, API keys. When you want to change how the system behaves, you change this file. |
| `requirements.txt` | Lists every Python library the project needs. When you run `pip install -r requirements.txt`, Python downloads all of them. |
| `.env` | Stores secret values (like your TMDb API key). This file is never uploaded to GitHub — that's why it's in `.gitignore`. |
| `.env.example` | A template showing what `.env` should look like, without the actual secret values. |
| `.gitignore` | Tells Git which files to NOT track — secrets (`.env`), large data files, temporary Python files, etc. |
| `data/seed_movies.json` | A hand-picked list of 150 movies: 50 Telugu, 50 Bollywood, 50 Hollywood. This is the starting point for all data collection. Each entry has the movie title, year, language, and TMDb ID. |

---

## Phase 1: Getting the Data (Ingestion)

**Goal:** Collect movie information from the internet and store it locally.

Think of this like a researcher going to a library, photocopying relevant books, and filing them in folders.

### `ingestion/tmdb_client.py` — The Metadata Collector

**What it does:** Talks to the TMDb (The Movie Database) API to get structured information about each movie — title, year, genres, director, cast, ratings, budget, revenue.

**Key concepts:**
- **API (Application Programming Interface):** A way for programs to request data from a web service. Think of it like a waiter at a restaurant — you give your order (request), and the waiter brings you food (data). TMDb's API lets us say "give me info about movie #155" and it returns The Dark Knight's data.
- **Rate limiting:** TMDb only allows 40 requests every 10 seconds. If we go faster, they'll block us. So the code has a `_throttle()` method that automatically slows down when we're going too fast.
- **API key:** A password that identifies you to the service. You get one free from TMDb's website.

**Why TMDb?** It's the best free movie database API. IMDb doesn't have a free public API. TMDb has everything we need: metadata, credits, and financial data (budget/revenue) which we use to classify hits vs flops.

### `ingestion/script_scraper.py` — The Script Collector

**What it does:** Visits IMSDb.com (Internet Movie Script Database) and downloads movie scripts for Hollywood films.

**Key concepts:**
- **Web scraping:** When a website doesn't have an API, you can read the webpage itself and extract data. This is called scraping. We download the HTML page, then use BeautifulSoup (a Python library) to find the script text within it.
- **Scene parsing:** Movie scripts have a standard format. Scene headings look like `INT. BANK - DAY` (interior, bank, daytime) or `EXT. PARK - NIGHT` (exterior, park, nighttime). We use **regex** (regular expressions — text pattern matching) to find these headings and split the script into individual scenes.
- **Politeness delay:** We wait 2 seconds between requests so we don't overwhelm the website's server. This is good internet etiquette.

**Why only Hollywood scripts?** IMSDb only has English-language scripts. Telugu and Bollywood scripts aren't available in a scrape-friendly format online. That's why reviews become the primary data source for non-English films.

### `ingestion/review_scraper.py` — The Review Collector

**What it does:** Uses the `cinemagoer` Python library to fetch user reviews from IMDb for every movie in our seed list.

**Key concepts:**
- **cinemagoer:** A Python library that can access IMDb data without an official API. It works but can be unreliable, which is why we have extensive error handling (try/except blocks) and gracefully skip movies that fail.
- **Graceful failure:** When one movie fails to fetch, we don't crash the entire pipeline. We log the error and move on. This is important because real-world data collection always has failures.

**Why IMDb reviews?** They're the most widely available source of audience opinions across all three languages. IMDb has reviews for Telugu, Bollywood, and Hollywood films.

### `ingestion/metadata_store.py` — The Database

**What it does:** Creates and manages a SQLite database that stores structured movie information.

**Key concepts:**
- **SQLite:** A database that lives in a single file (`data/cinerag.db`). Unlike MySQL or PostgreSQL, it doesn't need a separate server — it's just a file on your disk. Perfect for projects like this.
- **Schema:** The structure of the database — what columns exist, what type of data each holds. Our `movies` table has columns like `title` (text), `year` (number), `vote_average` (decimal), etc.
- **Hit/flop classification:** The code automatically classifies movies as hits or flops using two rules:
  - If revenue > 2x budget, it's a **hit**
  - If revenue < budget, it's a **flop**
  - If we don't have financial data, we fall back to ratings: above 7.0 with 500+ votes = hit, below 5.0 = flop
  - Otherwise, it's "unknown"

**Why both SQLite AND ChromaDB (introduced later)?** They serve different purposes:
- SQLite is great for **structured queries**: "give me all Telugu comedies from 2010-2020 that were flops" — this is SQL's strength.
- ChromaDB is great for **semantic search**: "find text chunks similar to this question" — this requires vector math, not SQL.

### `ingestion/run_ingestion.py` — The Orchestrator

**What it does:** Runs the entire data collection pipeline in order:
1. Fetch TMDb metadata for all 150 movies
2. Store metadata in SQLite
3. Scrape scripts from IMSDb
4. Fetch reviews from IMDb
5. Print a summary

**Key concept — CLI flags:** You can control what runs:
- `--movies-only`: Only fetch metadata (fast, good for testing)
- `--scripts-only`: Only scrape scripts
- `--reviews-only`: Only fetch reviews
- `--skip-existing`: Don't re-fetch data you already have

This saves time during development. You don't want to re-scrape 150 movies every time you test a change.

---

## Phase 2: Cutting Text Into Pieces (Chunking)

**Goal:** Break large texts (full scripts, long reviews) into smaller pieces that are manageable for the AI.

### Why do we need chunking?

Imagine you have the entire script of The Dark Knight (about 30,000 words). If someone asks "What happens in the interrogation scene?", you don't want to send the ENTIRE script to the AI. You want to send just the relevant scene.

But how does the system know which scene is relevant? It needs to compare the question against individual pieces. That's why we cut the script into scenes first — so each scene can be independently searched.

**Also:** AI models have a limited input size (called a "context window"). You can't fit 50 full movie scripts into a single prompt. Chunking ensures each piece is small enough to be useful.

### `chunking/chunk_models.py` — The Blueprint

**What it does:** Defines the `DocumentChunk` data model using **Pydantic**.

```
A DocumentChunk contains:
- chunk_id:     unique identifier (e.g., "155_script_42")
- movie_title:  "The Dark Knight"
- movie_year:   2008
- language:     "en"
- genres:       ["Action", "Crime", "Drama"]
- director:     "Christopher Nolan"
- source_type:  "script" or "review"
- content:      the actual text
- metadata:     extra info (scene heading, review rating, etc.)
```

**Key concept — Pydantic:** A Python library for defining data structures with automatic validation. If you try to create a `DocumentChunk` with `movie_year="not a number"`, Pydantic will raise an error. This catches bugs early.

**Why attach all this metadata to every chunk?** When we search later, we need to know which movie each chunk came from, what language it's in, who directed it, etc. This metadata travels with the chunk through the entire pipeline and eventually gets stored in ChromaDB for filtering.

### `chunking/script_chunker.py` — Script Cutter

**What it does:** Splits movie scripts into chunks, one per scene.

**Strategy:**
1. Find scene headings (like `INT. BANK - DAY`) using regex
2. Each scene becomes one chunk
3. If a scene is too long (>512 tokens), split it further at paragraph breaks
4. If no scene headings are found (unformatted script), fall back to splitting every ~512 tokens with overlap

**Key concept — Token estimation:** AI models think in "tokens" (roughly word-pieces). We estimate tokens as `len(text) / 4` — meaning 4 characters per token on average. This isn't exact, but it's fast and close enough for chunking decisions.

**Key concept — Overlap:** When you hard-split text, you might cut a sentence in half. Overlap means the end of chunk N appears again at the start of chunk N+1, so no information is lost at the boundary.

### `chunking/review_chunker.py` — Review Cutter

**What it does:** Turns each review into a chunk. Most reviews are short enough to be one chunk. Long ones get split at paragraph boundaries.

**Strategy:**
1. One review = one chunk (most are under 512 tokens)
2. Review title is prepended to the body ("A masterpiece\n\nThis film...")
3. If over 512 tokens, split at paragraph breaks, then sentence breaks

**Why prepend the title?** Review titles carry strong sentiment. "Absolute disaster" or "Best film of the decade" are powerful signals. Including them helps the embedding model understand the chunk's meaning.

---

## Phase 3: Converting Text to Numbers (Embedding)

**Goal:** Turn text chunks into numerical representations (vectors) so we can mathematically compare them.

### Why convert text to numbers?

Computers can't understand text the way humans do. But they're excellent at comparing numbers. An **embedding** converts text into a list of numbers (called a **vector**) where:

- Similar texts get similar vectors
- Different texts get different vectors

Example (simplified — real vectors have 1024 numbers):
```
"intense fight scene"   -> [0.8, 0.1, 0.9, 0.2, ...]
"brutal action sequence" -> [0.7, 0.1, 0.8, 0.3, ...]  <- similar!
"romantic dinner scene"  -> [0.1, 0.9, 0.2, 0.7, ...]  <- very different
```

When someone searches "find me intense action scenes", we convert their query to a vector and find which stored vectors are closest. This is **semantic search** — searching by meaning, not by exact word matching.

### `embedding/embedder.py` — The Text-to-Numbers Converter

**What it does:** Loads the BGE-M3 model and converts text into 1024-dimensional vectors.

**Key concepts:**
- **BGE-M3:** An embedding model made by BAAI (Beijing Academy of AI). The "M3" stands for Multi-lingual, Multi-granularity, Multi-functionality. It understands Telugu, Hindi, and English in the same vector space.
- **Why multilingual matters:** If someone asks a question in English about a Telugu movie, the system can still find relevant Telugu content because BGE-M3 maps both languages into the same numerical space.
- **Batching:** Instead of embedding one text at a time, we process 32 at once. This is much faster because GPUs (and even CPUs) are optimized for bulk operations.
- **Normalization:** We normalize the vectors (make them all length 1). This means we can use **cosine similarity** to compare them, which measures the angle between vectors — a reliable measure of text similarity.
- **Lightweight fallback:** BGE-M3 is 2.3GB and needs ~4GB RAM. If your machine can't handle it, setting `USE_LIGHTWEIGHT_EMBEDDINGS=true` switches to `all-MiniLM-L6-v2` (80MB, English-only). You lose multilingual support but gain speed.

### `embedding/vector_store.py` — The Vector Database

**What it does:** Stores vectors in ChromaDB and lets you search them.

**Key concepts:**
- **ChromaDB:** A vector database designed for AI applications. It stores vectors alongside metadata and lets you search by similarity. Think of it as a library where books are organized by meaning, not alphabetically.
- **Persistent storage:** ChromaDB saves to disk at `data/chroma/`. When you restart the app, your vectors are still there.
- **Metadata filtering:** When you search, you can say "find similar text, but ONLY among Telugu movies from 2010-2020." ChromaDB handles this by filtering metadata before (or during) the vector search.
- **HNSW index:** The algorithm ChromaDB uses to find similar vectors quickly. Without it, you'd have to compare your query against every single vector (slow). HNSW builds a graph structure that lets you find approximate nearest neighbors in milliseconds.
- **Flat metadata:** ChromaDB only supports simple key-value metadata (strings, numbers, booleans). It can't store lists or nested objects. So we convert `genres: ["Action", "Drama"]` to `genres: "Action, Drama"` (a comma-separated string).

### `embedding/build_index.py` — The Index Builder

**What it does:** Ties chunking and embedding together:
1. Loads all processed scripts and reviews
2. Runs them through the chunkers to create `DocumentChunk` objects
3. Embeds all chunks using BGE-M3
4. Stores everything in ChromaDB

**CLI flags:**
- `--rebuild`: Delete the existing index and start fresh
- `--source-type review`: Only index reviews (useful for testing)

---

## Phase 4: Finding Relevant Pieces (Retrieval)

**Goal:** When a user asks a question, find the most relevant chunks from our database.

### `retrieval/retriever.py` — The Core Searcher

**What it does:** Takes a question, converts it to a vector, and finds the closest chunks in ChromaDB.

It's a thin wrapper around VectorStore that converts raw ChromaDB results back into `DocumentChunk` objects. This keeps the rest of the code clean — everything works with `DocumentChunk`, not raw database dictionaries.

### `retrieval/hybrid_search.py` — The Smart Searcher

**What it does:** Extracts structured filters from natural language, then combines them with semantic search.

**This is one of the most interesting files in the project.** Here's what it does:

```
User query: "Telugu action movies from 2005 to 2010"

Step 1 - Extract filters:
  "Telugu"     -> language: "te"
  "action"     -> genre: "Action"
  "2005 to 2010" -> year_min: 2005, year_max: 2010

Step 2 - Build ChromaDB filter:
  {"$and": [
    {"language": "te"},
    {"genres": {"$contains": "Action"}},
    {"movie_year": {"$gte": 2005}},
    {"movie_year": {"$lte": 2010}}
  ]}

Step 3 - Search ChromaDB with BOTH the semantic vector AND the metadata filter
```

**Key concept — Hybrid search:** Pure semantic search might return English action movies when you asked for Telugu ones (because the meaning is similar). By extracting metadata filters, we narrow down to exactly the right subset first, THEN find the most semantically relevant chunks within that subset.

**Key concept — Deterministic filter extraction:** We use keyword matching and regex patterns, NOT the AI model. Why? Because:
1. It's instant (<1ms) vs 5-10 seconds for an AI call
2. It's predictable — the same query always produces the same filters
3. It doesn't need the LLM to be running

**Automatic fallback:** If filtered search returns zero results (maybe the filters were too restrictive), the system automatically retries without filters. The user always gets some results.

### `retrieval/reranker.py` — The Precision Booster (Optional)

**What it does:** Takes the top 20 results from the initial search and re-scores them with a more accurate (but slower) model, keeping only the top 5.

**Key concept — Bi-encoder vs Cross-encoder:**
- The initial search uses a **bi-encoder** (BGE-M3). It embeds the query and documents separately, then compares. This is fast because document embeddings are pre-computed.
- The reranker uses a **cross-encoder**. It looks at the query and each document TOGETHER, which is much more accurate but can't be pre-computed — it has to run for every query.

Think of it like this:
- Bi-encoder: Looking at two photos side by side and saying "they seem similar" (fast, approximate)
- Cross-encoder: Carefully comparing every detail of both photos (slow, precise)

**Why optional?** Reranking adds 1-2 seconds of latency per query. For many use cases, the initial retrieval is good enough. Toggle it with `USE_RERANKER=true` in `.env`.

---

## Phase 5: Generating Answers (Generation)

**Goal:** Take the retrieved chunks and use an AI language model to generate a human-readable answer.

### `generation/llm_client.py` — The AI Model Interface

**What it does:** Sends prompts to Ollama (which runs Llama 3.1 locally on your machine) and gets back generated text.

**Key concepts:**
- **Ollama:** A tool that runs AI language models locally on your computer. Instead of paying OpenAI for API calls, you download a model once and run it for free. The tradeoff is speed (local is slower than cloud GPUs) and hardware requirements (~8GB RAM for the 8B model).
- **Llama 3.1 8B:** An open-source language model by Meta. "8B" means 8 billion parameters (the numbers that define the model's knowledge). It's a good balance of quality and speed for local use.
- **Health check:** Before sending a query, we verify Ollama is running and has the model loaded. This prevents confusing error messages.
- **Temperature:** Controls randomness. Low temperature (0.3, our default) makes the model more focused and factual. High temperature (0.9+) makes it more creative but less reliable. For factual film analysis, we want low temperature.

### `generation/prompts.py` — The Instruction Templates

**What it does:** Defines four prompt templates — one for each query type.

**Key concept — Prompt engineering:** The quality of an AI's answer depends heavily on how you ask the question. A prompt template is a pre-written instruction format that consistently produces good results.

Our four templates:

1. **General Q&A:** "Answer based ONLY on the provided context. Cite which movie and source type you're drawing from."
2. **Trend Analysis:** "Organize your analysis chronologically. Cite specific movies and years as evidence."
3. **Pattern Detection:** "Identify recurring patterns. Be specific — reference actual dialogue, scenes, or review sentiments."
4. **Hit/Flop Analysis:** Given separate hit and flop contexts, "analyze what differentiates them."

**Why "based ONLY on the provided context"?** This instruction prevents hallucination. Without it, the model might make up facts from its training data. By constraining it to our retrieved chunks, answers stay grounded in evidence.

**System message:** Every prompt also includes a system message: "You are a film analysis expert with deep knowledge of Telugu, Bollywood, and Hollywood cinema." This sets the model's persona and expertise level.

### `generation/query_router.py` — The Traffic Director

**What it does:** Looks at the user's question and decides which of the four query types it is.

**How it works:** Simple keyword scoring. Each category has a list of trigger words:
- **Trend:** "evolve", "change over time", "trend", "decade", "from YEAR to YEAR"
- **Pattern:** "pattern", "recurring", "signature style", "across his films"
- **Hit/Flop:** "hit", "flop", "succeed", "fail", "box office", "blockbuster"
- **General:** Everything else (the default)

The category with the most keyword matches wins. Year ranges (like "from 2000 to 2020") give a bonus to the trend category.

**Why not use the AI for routing?** Speed. Routing takes <1ms with keyword matching vs 5-10 seconds with an AI call. Since routing happens on every single query, this adds up. And for four well-defined categories, keywords work well enough.

### `generation/chains.py` — The Full Pipeline

**What it does:** This is the **heart of the system**. It connects everything:

```
User Question
    |
    v
Query Router: classify as trend/pattern/hit_flop/general
    |
    v
Hybrid Search: extract filters + semantic search
    |
    v
Reranker: (optional) refine top results
    |
    v
Format Context: turn chunks into numbered, labeled text blocks
    |
    v
LLM: generate answer using the appropriate prompt template
    |
    v
Response: {answer, query_type, filters, sources}
```

**Three retrieval strategies:**

1. **Standard** (general + pattern): Retrieve chunks, rerank, generate.
2. **Trend**: Same as standard, but sorts chunks chronologically before sending to the LLM, so the model can analyze change over time.
3. **Hit/Flop**: Queries SQLite for which movies are hits vs flops, retrieves chunks separately for each group, then sends both sets to the LLM for comparison.

**Context formatting:** Each chunk becomes a labeled block:
```
[1] The Dark Knight (2008) — Script (Scene: INT. INTERROGATION ROOM - NIGHT)
The Joker sits across from Batman...

---

[2] Pokiri (2006) — Review (Rating: 9/10)
One of the best action films in Telugu cinema...
```

This formatting helps the LLM understand where each piece of information comes from and cite it properly.

---

## Phase 6: Measuring Quality (Evaluation)

**Goal:** Objectively measure how good the system's answers are.

### Why evaluate?

Without measurement, you're guessing. "The answers seem okay" isn't engineering. Evaluation tells you:
- Are the retrieved chunks actually relevant?
- Is the AI's answer faithful to the evidence?
- Which query types work best? Which need improvement?

### `evaluation/test_set.json` — The Exam Paper

**What it does:** 50 hand-written question-answer pairs that serve as a "final exam" for the system.

Each entry has:
- **question**: The test question
- **query_type**: Which category it should be classified as
- **ground_truth_answer**: The expected answer (2-3 sentences)
- **relevant_movies**: Which movies should appear in the results
- **expected_source_types**: Should the answer come from scripts, reviews, or both?

**Why hand-written?** Automated test generation is unreliable. A human (who knows cinema) writes questions that test specific capabilities — multi-movie trends, director patterns, hit/flop comparisons, and factual retrieval.

### `evaluation/run_eval.py` — The Exam Grader

**What it does:** Runs all 50 questions through the full pipeline and measures performance.

**Two tiers of metrics:**

**Tier 1: Heuristic metrics (always work, no extra dependencies):**
- **Query type accuracy:** Did the router classify "How did Telugu action evolve?" as "trend"? Measures routing correctness.
- **Source type hit rate:** If we expected reviews, did we actually retrieve reviews?
- **Movie hit rate:** If the question is about Pokiri and Magadheera, did those movies appear in the sources?
- **Average response time:** How fast is the system?

**Tier 2: RAGAS metrics (optional, needs additional LLM configuration):**
- **Faithfulness:** Does the answer only contain claims supported by the retrieved context? (i.e., no hallucination)
- **Answer Relevancy:** Is the answer actually about what was asked?
- **Context Precision:** Are the retrieved chunks relevant to the question?
- **Context Recall:** Did we retrieve ALL the chunks needed to answer properly?

**RAGAS (Retrieval Augmented Generation Assessment)** is an industry-standard framework for evaluating RAG systems. It's the equivalent of standardized testing for AI systems.

### `evaluation/eval_report.py` — The Report Card

**What it does:** Takes the raw evaluation results and generates a readable markdown report (`EVAL_REPORT.md`) with:
- Overall scores table
- Per-query-type breakdown
- Misrouted queries (where the router got it wrong)
- Failed queries (errors)
- Source type mismatches
- Retrieval quality analysis

---

## Phase 7: The Web Interface (Frontend)

**Goal:** Give users a visual way to interact with the system.

### `app/main.py` — The Server

**What it does:** Creates and starts a Flask web server on port 5000.

**Key concept — Flask:** A lightweight Python web framework. It handles incoming HTTP requests (like when you visit a URL or click a button) and returns responses (HTML pages or JSON data). It's the waiter between the user's browser and our Python code.

### `app/routes.py` — The URL Handler

**What it does:** Defines what happens when you visit each URL:

| URL | Method | What it does |
|---|---|---|
| `/` | GET | Shows the search page |
| `/query` | POST | Receives a question, runs the full RAG pipeline, returns the answer as JSON |
| `/movies` | GET | Returns a JSON list of all movies in the database |
| `/stats` | GET | Shows the stats dashboard page |
| `/api/stats` | GET | Returns database and vector store statistics as JSON |

**Key concept — Lazy initialization:** The `CineRAGChain` (which loads the embedding model, connects to ChromaDB, etc.) isn't created when the server starts. It's created on the first query. This makes startup fast and avoids loading heavy models if you're just checking the stats page.

### `app/templates/` — The HTML Pages

- **`base.html`:** The layout shared by all pages — navbar with "Search" and "Stats" links.
- **`index.html`:** The search page — input box, filter dropdowns, example query buttons, results area.
- **`stats.html`:** The stats dashboard — populated dynamically by JavaScript.

**Key concept — Jinja2 templating:** Flask uses Jinja2 to inject Python data into HTML. `{% extends "base.html" %}` means "use base.html as the skeleton." `{{ url_for('static', filename='css/style.css') }}` generates the correct URL for the CSS file.

### `app/static/css/style.css` — The Visual Design

A dark-themed design with purple accents. Key elements:
- CSS custom properties (`--bg`, `--accent`, etc.) for consistent colors
- Responsive layout that works on mobile
- Color-coded badges for query types (blue=general, green=trend, purple=pattern, yellow=hit_flop)

### `app/static/js/app.js` — The Interactive Behavior

**What it does:** Handles everything that happens without a page reload:
- When you click "Search" or press Enter, sends the question to `/query` via AJAX (background HTTP request)
- Shows a loading spinner while waiting
- Renders the answer and source cards when the response arrives
- Populates the stats dashboard by fetching `/api/stats`

**Key concept — AJAX:** Instead of reloading the entire page for every query, JavaScript sends the request in the background and updates just the results area. This makes the experience much smoother.

---

## Key Concepts Explained

### Vectors and Vector Databases

A **vector** is just a list of numbers: `[0.12, -0.45, 0.78, ...]`. In our case, each vector has 1024 numbers.

The magic is that the embedding model assigns these numbers so that **similar texts are close together** in this 1024-dimensional space. "Brilliant action sequences" and "amazing fight choreography" get similar vectors even though they share no words.

A **vector database** (ChromaDB) is optimized for finding the closest vectors to a query vector. Regular databases (SQL) can't do this efficiently.

### Cosine Similarity

The way we measure "closeness" between two vectors. It calculates the angle between them:
- **1.0** = identical direction (very similar text)
- **0.0** = perpendicular (unrelated)
- **-1.0** = opposite direction (opposite meaning)

ChromaDB returns results sorted by cosine similarity (though it reports "distance" = 1 - similarity).

### Tokens

AI models don't read words — they read **tokens**. A token is roughly 3/4 of a word:
- "The Dark Knight" = 3 tokens
- "cinematography" = 3-4 tokens (it's a long word, so it gets split)

Our rough estimate: **1 token ≈ 4 characters**. This is used for chunking decisions (e.g., "is this scene over 512 tokens?").

### SQLite vs ChromaDB — Why Two Databases?

| Need | SQLite | ChromaDB |
|---|---|---|
| "List all Telugu comedies" | Great (SQL WHERE clause) | Not designed for this |
| "Which movies are hits vs flops?" | Great (is_hit column) | Not designed for this |
| "Find text similar to this question" | Impossible | Exactly what it's for |
| "Find similar text but only Telugu films from 2010s" | Can pre-filter | Can filter + search |

They complement each other. SQLite handles structured data; ChromaDB handles semantic search.

---

## How a Query Flows Through the System

Let's trace a real query: **"How did action choreography in Telugu cinema evolve from 2000 to 2020?"**

### Step 1: Frontend sends the query
Browser sends POST to `/query` with `{"question": "How did action choreography..."}`.

### Step 2: Query Router classifies it
Keywords found: "evolve" (trend), "from 2000 to 2020" (year range = trend bonus).
Result: **trend** query type.

### Step 3: Filter Extraction
- "Telugu" -> `language: "te"`
- "from 2000 to 2020" -> `year_min: 2000, year_max: 2020`

ChromaDB filter: `{"$and": [{"language": "te"}, {"movie_year": {"$gte": 2000}}, {"movie_year": {"$lte": 2020}}]}`

### Step 4: Embedding + Search
The question is converted to a 1024-dimensional vector. ChromaDB finds the 20 closest chunks that match the Telugu + 2000-2020 filters.

### Step 5: Reranking (if enabled)
Cross-encoder re-scores the 20 chunks, keeps the top 5.

### Step 6: Chronological Sorting
Because it's a trend query, chunks are sorted by `movie_year`:
1. Simhadri (2003) — Review
2. Pokiri (2006) — Review
3. Magadheera (2009) — Review
4. Baahubali (2015) — Review
5. RRR (2022) — Review

### Step 7: Context Formatting
```
[1] Simhadri (2003) — Review (Rating: 8/10)
NTR's raw physicality defined early 2000s Telugu action...

---

[2] Pokiri (2006) — Review (Rating: 9/10)
Mahesh Babu's stylish gunplay and rooftop chases set a new standard...

... (and so on)
```

### Step 8: LLM Generation
The trend analysis prompt template is filled with the context and question, then sent to Llama 3.1. The model writes an answer citing specific movies and years.

### Step 9: Response
```json
{
  "answer": "Telugu action choreography underwent a significant evolution...",
  "query_type": "trend",
  "filters": {"language": "te", "year_min": 2000, "year_max": 2020},
  "sources": [
    {"movie": "Simhadri", "year": 2003, "source_type": "review", ...},
    {"movie": "Pokiri", "year": 2006, "source_type": "review", ...},
    ...
  ]
}
```

### Step 10: Frontend renders
JavaScript displays the answer, a green "TREND" badge, and 5 source cards.

---

## Why We Chose Each Tool

| Tool | What it does | Why we chose it over alternatives |
|---|---|---|
| **Python** | Programming language | The dominant language for AI/ML. Every library we need has Python support. |
| **Ollama** | Runs LLMs locally | One command to download and run models. No GPU configuration, quantization, or VRAM management needed. Alternative (raw HuggingFace) requires manual GGUF setup. |
| **Llama 3.1 8B** | Generates answers | Open source, free, runs on 16GB RAM machines. GPT-4 is better but costs money. Llama 3.1 8B is the sweet spot of quality and accessibility. |
| **BGE-M3** | Converts text to vectors | Only model that handles Telugu + Hindi + English in one vector space. Alternative (all-MiniLM) is English-only. |
| **ChromaDB** | Stores and searches vectors | Native metadata filtering (crucial for hybrid search). Simple Python API. Persistent storage. Alternative (FAISS) has no metadata filtering — you'd have to build it yourself. |
| **SQLite** | Stores structured movie data | Zero setup — it's just a file. No database server to install. Perfect for this scale (150 movies, not millions). |
| **Flask** | Web server | Already familiar, minimal, no magic. Alternative (FastAPI) is better for APIs but Flask's template rendering is simpler for full web apps. |
| **Pydantic** | Data validation | Catches type errors early. If `movie_year` is accidentally a string, Pydantic tells you immediately instead of crashing later. |
| **RAGAS** | Evaluation metrics | Industry standard for RAG evaluation. Gives you the exact metrics hiring managers and teams look for. |
| **sentence-transformers** | Loads embedding models | The standard library for loading and running embedding models in Python. Clean API, great documentation. |
| **BeautifulSoup** | Parses HTML | The go-to Python library for web scraping. Handles messy HTML gracefully. |
| **cinemagoer** | Fetches IMDb data | The only actively maintained Python library for accessing IMDb data without an official API. |

---

*This guide is part of the CineRAG project. For setup instructions, see [README.md](README.md). For the detailed implementation plan, see [cinerag_project_plan.md](cinerag_project_plan.md).*
