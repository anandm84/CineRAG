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
LLM_MODEL = "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"

# Set to True to use lightweight English-only embeddings (~80MB vs ~2.3GB)
USE_LIGHTWEIGHT_EMBEDDINGS = os.getenv("USE_LIGHTWEIGHT_EMBEDDINGS", "false").lower() == "true"
LIGHTWEIGHT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Config
SCRIPT_CHUNK_MAX_TOKENS = 512
REVIEW_CHUNK_MAX_TOKENS = 512

# Retrieval Config
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
