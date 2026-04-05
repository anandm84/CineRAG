"""SQLite metadata store for structured movie information."""

import json
import logging
import sqlite3
from pathlib import Path

import config

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id INTEGER UNIQUE,
    imdb_id TEXT,
    title TEXT NOT NULL,
    original_title TEXT,
    year INTEGER,
    language TEXT,
    genres TEXT,
    director TEXT,
    top_cast TEXT,
    overview TEXT,
    vote_average REAL,
    vote_count INTEGER,
    budget INTEGER,
    revenue INTEGER,
    runtime INTEGER,
    is_hit INTEGER,
    has_script INTEGER DEFAULT 0,
    has_reviews INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_movies_language ON movies(language);
CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);
CREATE INDEX IF NOT EXISTS idx_movies_director ON movies(director);
CREATE INDEX IF NOT EXISTS idx_movies_is_hit ON movies(is_hit);
"""


class MetadataStore:
    """SQLite-backed store for movie metadata."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
        logger.info(f"Database initialized at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def insert_movie(self, metadata: dict) -> int | None:
        """Insert or update a movie record from normalized TMDb metadata.

        Returns the row ID, or None if the insert failed.
        """
        is_hit = self._determine_hit_status(metadata)

        values = {
            "tmdb_id": metadata.get("tmdb_id"),
            "imdb_id": metadata.get("imdb_id"),
            "title": metadata.get("title", ""),
            "original_title": metadata.get("original_title", ""),
            "year": metadata.get("year"),
            "language": metadata.get("language", "en"),
            "genres": json.dumps(metadata.get("genres", [])),
            "director": metadata.get("director", "Unknown"),
            "top_cast": json.dumps(metadata.get("top_cast", [])),
            "overview": metadata.get("overview", ""),
            "vote_average": metadata.get("vote_average", 0),
            "vote_count": metadata.get("vote_count", 0),
            "budget": metadata.get("budget", 0),
            "revenue": metadata.get("revenue", 0),
            "runtime": metadata.get("runtime"),
            "is_hit": is_hit,
        }

        sql = """
            INSERT INTO movies (
                tmdb_id, imdb_id, title, original_title, year, language,
                genres, director, top_cast, overview, vote_average, vote_count,
                budget, revenue, runtime, is_hit
            ) VALUES (
                :tmdb_id, :imdb_id, :title, :original_title, :year, :language,
                :genres, :director, :top_cast, :overview, :vote_average, :vote_count,
                :budget, :revenue, :runtime, :is_hit
            )
            ON CONFLICT(tmdb_id) DO UPDATE SET
                imdb_id=excluded.imdb_id,
                title=excluded.title,
                original_title=excluded.original_title,
                year=excluded.year,
                language=excluded.language,
                genres=excluded.genres,
                director=excluded.director,
                top_cast=excluded.top_cast,
                overview=excluded.overview,
                vote_average=excluded.vote_average,
                vote_count=excluded.vote_count,
                budget=excluded.budget,
                revenue=excluded.revenue,
                runtime=excluded.runtime,
                is_hit=excluded.is_hit
        """

        try:
            with self._connect() as conn:
                cursor = conn.execute(sql, values)
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Failed to insert movie {metadata.get('title')}: {e}")
            return None

    def _determine_hit_status(self, metadata: dict) -> int | None:
        """Determine if a movie is a hit or flop.

        Criteria: revenue > 2x budget OR vote_average > 7.0
        Returns: 1 (hit), 0 (flop), None (unknown/insufficient data)
        """
        budget = metadata.get("budget", 0)
        revenue = metadata.get("revenue", 0)
        vote_avg = metadata.get("vote_average", 0)

        if budget and revenue:
            if revenue > 2 * budget:
                return 1
            elif revenue < budget:
                return 0

        if vote_avg >= 7.0 and metadata.get("vote_count", 0) > 500:
            return 1
        elif vote_avg < 5.0 and metadata.get("vote_count", 0) > 100:
            return 0

        return None

    def get_movie_by_title(self, title: str) -> dict | None:
        """Look up a movie by title (case-insensitive)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM movies WHERE LOWER(title) = LOWER(?)", (title,)
            ).fetchone()
            return dict(row) if row else None

    def get_movie_by_tmdb_id(self, tmdb_id: int) -> dict | None:
        """Look up a movie by TMDb ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM movies WHERE tmdb_id = ?", (tmdb_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_movies_by_filter(
        self,
        language: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        genre: str | None = None,
        director: str | None = None,
        is_hit: bool | None = None,
    ) -> list[dict]:
        """Query movies with optional filters."""
        conditions = []
        params = []

        if language:
            conditions.append("language = ?")
            params.append(language)
        if year_min:
            conditions.append("year >= ?")
            params.append(year_min)
        if year_max:
            conditions.append("year <= ?")
            params.append(year_max)
        if genre:
            conditions.append("genres LIKE ?")
            params.append(f"%{genre}%")
        if director:
            conditions.append("LOWER(director) LIKE LOWER(?)")
            params.append(f"%{director}%")
        if is_hit is not None:
            conditions.append("is_hit = ?")
            params.append(1 if is_hit else 0)

        sql = "SELECT * FROM movies"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY year DESC, vote_average DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def update_data_availability(self, tmdb_id: int,
                                  has_script: bool | None = None,
                                  has_reviews: bool | None = None):
        """Update the data availability flags for a movie."""
        updates = []
        params = []

        if has_script is not None:
            updates.append("has_script = ?")
            params.append(1 if has_script else 0)
        if has_reviews is not None:
            updates.append("has_reviews = ?")
            params.append(1 if has_reviews else 0)

        if not updates:
            return

        params.append(tmdb_id)
        sql = f"UPDATE movies SET {', '.join(updates)} WHERE tmdb_id = ?"

        with self._connect() as conn:
            conn.execute(sql, params)

    def get_stats(self) -> dict:
        """Get summary statistics about the database."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
            by_lang = conn.execute(
                "SELECT language, COUNT(*) as cnt FROM movies GROUP BY language ORDER BY cnt DESC"
            ).fetchall()
            hits = conn.execute("SELECT COUNT(*) FROM movies WHERE is_hit = 1").fetchone()[0]
            flops = conn.execute("SELECT COUNT(*) FROM movies WHERE is_hit = 0").fetchone()[0]
            with_scripts = conn.execute("SELECT COUNT(*) FROM movies WHERE has_script = 1").fetchone()[0]
            with_reviews = conn.execute("SELECT COUNT(*) FROM movies WHERE has_reviews = 1").fetchone()[0]

        return {
            "total_movies": total,
            "by_language": {row[0]: row[1] for row in by_lang},
            "hits": hits,
            "flops": flops,
            "with_scripts": with_scripts,
            "with_reviews": with_reviews,
        }

    def populate_from_metadata(self, metadata_list: list[dict]) -> int:
        """Bulk insert movies from a list of normalized metadata dicts.

        Returns the number of successfully inserted movies.
        """
        count = 0
        for meta in metadata_list:
            result = self.insert_movie(meta)
            if result is not None:
                count += 1
        logger.info(f"Populated {count}/{len(metadata_list)} movies into database")
        return count
