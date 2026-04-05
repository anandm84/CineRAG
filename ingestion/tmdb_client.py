"""TMDb API client for fetching movie metadata."""

import json
import time
import logging
from pathlib import Path

import requests

import config

logger = logging.getLogger(__name__)


class TMDbClient:
    """Wrapper around the TMDb API for fetching movie metadata."""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or config.TMDB_API_KEY
        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY not set. Add it to .env or pass it directly."
            )
        self._request_times: list[float] = []

    def _throttle(self):
        """Simple rate limiter: max 40 requests per 10 seconds."""
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 10]
        if len(self._request_times) >= 38:
            sleep_time = 10 - (now - self._request_times[0]) + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit throttle: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
        self._request_times.append(time.time())

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a GET request to the TMDb API."""
        self._throttle()
        params = params or {}
        params["api_key"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def search_movie(self, title: str, year: int | None = None) -> dict | None:
        """Search for a movie by title and optional year. Returns first result."""
        params = {"query": title}
        if year:
            params["year"] = year

        data = self._get("search/movie", params)
        results = data.get("results", [])
        if not results:
            logger.warning(f"No TMDb results for: {title} ({year})")
            return None
        return results[0]

    def get_movie_details(self, tmdb_id: int) -> dict:
        """Fetch full movie details including credits."""
        details = self._get(f"movie/{tmdb_id}", {"append_to_response": "credits"})
        return details

    def fetch_movie(self, title: str, year: int | None = None,
                    tmdb_id: int | None = None) -> dict | None:
        """Fetch full metadata for a movie. Uses tmdb_id if available, else searches."""
        try:
            if tmdb_id:
                details = self.get_movie_details(tmdb_id)
            else:
                search_result = self.search_movie(title, year)
                if not search_result:
                    return None
                details = self.get_movie_details(search_result["id"])

            return self._normalize(details)

        except requests.HTTPError as e:
            logger.error(f"HTTP error fetching {title}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {title}: {e}")
            return None

    def _normalize(self, details: dict) -> dict:
        """Normalize raw TMDb response into our standard format."""
        credits = details.get("credits", {})

        # Extract director
        directors = [
            c["name"] for c in credits.get("crew", [])
            if c.get("job") == "Director"
        ]
        director = directors[0] if directors else "Unknown"

        # Extract top 5 cast
        cast_list = credits.get("cast", [])[:5]
        top_cast = [c["name"] for c in cast_list]

        # Extract genres
        genres = [g["name"] for g in details.get("genres", [])]

        # Extract production countries
        countries = [c["iso_3166_1"] for c in details.get("production_countries", [])]

        return {
            "tmdb_id": details["id"],
            "imdb_id": details.get("imdb_id"),
            "title": details.get("title", ""),
            "original_title": details.get("original_title", ""),
            "year": int(details["release_date"][:4]) if details.get("release_date") else None,
            "release_date": details.get("release_date"),
            "original_language": details.get("original_language"),
            "genres": genres,
            "director": director,
            "top_cast": top_cast,
            "overview": details.get("overview", ""),
            "vote_average": details.get("vote_average", 0),
            "vote_count": details.get("vote_count", 0),
            "budget": details.get("budget", 0),
            "revenue": details.get("revenue", 0),
            "runtime": details.get("runtime"),
            "production_countries": countries,
        }

    def fetch_all(self, seed_list: list[dict], skip_existing: bool = False) -> list[dict]:
        """Fetch metadata for all movies in the seed list.

        Args:
            seed_list: List of dicts with keys: title, year, language, tmdb_id (optional).
            skip_existing: If True, skip movies whose raw JSON already exists.

        Returns:
            List of normalized metadata dicts.
        """
        raw_dir = config.RAW_DIR / "metadata"
        processed_dir = config.PROCESSED_DIR / "metadata"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(seed_list)

        for i, movie in enumerate(seed_list, 1):
            title = movie["title"]
            year = movie.get("year")
            language = movie.get("language", "en")
            tmdb_id = movie.get("tmdb_id")

            safe_name = _safe_filename(title, year)
            raw_path = raw_dir / f"{safe_name}.json"
            processed_path = processed_dir / f"{safe_name}.json"

            if skip_existing and processed_path.exists():
                logger.info(f"[{i}/{total}] Skipping (exists): {title}")
                with open(processed_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue

            logger.info(f"[{i}/{total}] Fetching: {title} ({year})")
            metadata = self.fetch_movie(title, year, tmdb_id)

            if metadata is None:
                logger.warning(f"[{i}/{total}] Failed: {title}")
                continue

            # Attach the language from our seed list (TMDb's original_language
            # may differ from what we consider the primary language)
            metadata["language"] = language

            # Save raw response (re-fetch the raw details for archival)
            try:
                if tmdb_id:
                    raw_data = self._get(f"movie/{tmdb_id}", {"append_to_response": "credits"})
                else:
                    raw_data = metadata  # fallback
                with open(raw_path, "w", encoding="utf-8") as f:
                    json.dump(raw_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not save raw data for {title}: {e}")

            # Save processed metadata
            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            results.append(metadata)

        logger.info(f"Fetched metadata for {len(results)}/{total} movies")
        return results


def _safe_filename(title: str, year: int | None = None) -> str:
    """Create a filesystem-safe filename from a movie title and year."""
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe = safe.strip().replace(" ", "_").lower()
    if year:
        safe = f"{safe}_{year}"
    return safe


def load_seed_list(path: Path | None = None) -> list[dict]:
    """Load the seed movie list from JSON."""
    path = path or config.DATA_DIR / "seed_movies.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
