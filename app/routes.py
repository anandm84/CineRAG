"""Flask routes for CineRAG."""

import json
import logging
import traceback

from flask import Flask, render_template, request, jsonify

from embedding.vector_store import VectorStore
from generation.chains import CineRAGChain
from ingestion.metadata_store import MetadataStore

logger = logging.getLogger(__name__)

# Lazy-initialized singletons
_chain: CineRAGChain | None = None
_metadata_store: MetadataStore | None = None


def _get_chain() -> CineRAGChain:
    global _chain
    if _chain is None:
        _chain = CineRAGChain()
    return _chain


def _get_store() -> MetadataStore:
    global _metadata_store
    if _metadata_store is None:
        _metadata_store = MetadataStore()
    return _metadata_store


def register_routes(app: Flask):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/query", methods=["POST"])
    def query():
        """Submit a query and get a RAG-generated answer with sources."""
        data = request.get_json()
        if not data or not data.get("question"):
            return jsonify({"error": "Missing 'question' field"}), 400

        question = data["question"].strip()
        if not question:
            return jsonify({"error": "Empty question"}), 400

        # Build explicit filters from frontend checkboxes
        explicit_filters = {}
        if data.get("language"):
            explicit_filters["language"] = data["language"]
        if data.get("source_type"):
            explicit_filters["source_type"] = data["source_type"]
        if data.get("year_min"):
            explicit_filters["year_min"] = int(data["year_min"])
        if data.get("year_max"):
            explicit_filters["year_max"] = int(data["year_max"])

        try:
            chain = _get_chain()
            result = chain.run(question, explicit_filters=explicit_filters or None)
            return jsonify(result)
        except ConnectionError as e:
            return jsonify({"error": f"LLM unavailable: {e}"}), 503
        except Exception as e:
            logger.error(f"Query failed: {traceback.format_exc()}")
            return jsonify({"error": f"Internal error: {str(e)}"}), 500

    @app.route("/movies")
    def movies():
        """List all indexed movies with metadata."""
        store = _get_store()
        language = request.args.get("language")
        is_hit_param = request.args.get("is_hit")

        is_hit = None
        if is_hit_param == "1":
            is_hit = True
        elif is_hit_param == "0":
            is_hit = False

        movies_list = store.get_movies_by_filter(language=language, is_hit=is_hit)

        # Parse JSON fields for display
        for m in movies_list:
            try:
                m["genres"] = json.loads(m.get("genres", "[]"))
            except (json.JSONDecodeError, TypeError):
                m["genres"] = []
            try:
                m["top_cast"] = json.loads(m.get("top_cast", "[]"))
            except (json.JSONDecodeError, TypeError):
                m["top_cast"] = []

        return jsonify({"movies": movies_list, "count": len(movies_list)})

    @app.route("/stats")
    def stats_page():
        """Stats dashboard — serves HTML, JS fetches data from /api/stats."""
        return render_template("stats.html")

    @app.route("/api/stats")
    def stats_api():
        """Index and database statistics (JSON API)."""
        result = {}

        # Database stats
        try:
            store = _get_store()
            result["database"] = store.get_stats()
        except Exception as e:
            result["database"] = {"error": str(e)}

        # Vector store stats
        try:
            vs = VectorStore()
            result["vector_store"] = vs.get_stats()
        except Exception as e:
            result["vector_store"] = {"error": str(e)}

        # Eval results if available
        try:
            from pathlib import Path
            eval_path = Path(__file__).parent.parent / "evaluation" / "results.json"
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                result["evaluation"] = eval_data.get("metrics", {})
        except Exception:
            pass

        return jsonify(result)
