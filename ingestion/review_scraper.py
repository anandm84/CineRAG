"""Scraper for IMDb user reviews using the cinemagoer package."""

import json
import logging
import time
from pathlib import Path

import config

logger = logging.getLogger(__name__)

MAX_REVIEWS_PER_MOVIE = 25
REQUEST_TIMEOUT = 30  # cinemagoer can be slow


class ReviewScraper:
    """Fetches IMDb user reviews using cinemagoer."""

    def __init__(self):
        try:
            from imdb import Cinemagoer
            self.ia = Cinemagoer()
        except ImportError:
            raise ImportError(
                "cinemagoer is required. Install with: pip install cinemagoer"
            )

    def search_movie(self, title: str, year: int | None = None) -> dict | None:
        """Search IMDb for a movie and return its IMDb object."""
        try:
            results = self.ia.search_movie(title)
            if not results:
                return None

            # Try to match by year if provided
            if year:
                for movie in results[:10]:
                    movie_year = movie.get("year")
                    if movie_year and abs(movie_year - year) <= 1:
                        return movie
            # Fall back to first result
            return results[0]
        except Exception as e:
            logger.error(f"Search failed for {title}: {e}")
            return None

    def fetch_reviews(self, title: str, year: int | None = None,
                      imdb_id: str | None = None) -> dict | None:
        """Fetch user reviews for a movie.

        Args:
            title: Movie title.
            year: Release year (helps disambiguate search results).
            imdb_id: Direct IMDb ID if known.

        Returns:
            Dict with movie_title, imdb_id, and reviews list.
            None if fetching fails.
        """
        try:
            if imdb_id:
                # Strip leading 'tt' if present
                clean_id = imdb_id.lstrip("t")
                movie = self.ia.get_movie(clean_id)
            else:
                movie = self.search_movie(title, year)
                if movie is None:
                    logger.warning(f"Movie not found on IMDb: {title}")
                    return None

            movie_id = movie.movieID
            logger.info(f"  -> IMDb ID: tt{movie_id}")

            # Fetch reviews
            self.ia.update(movie, info=["reviews"])
            raw_reviews = movie.get("reviews", [])

            if not raw_reviews:
                logger.info(f"  -> No reviews found for: {title}")
                return {
                    "movie_title": title,
                    "imdb_id": f"tt{movie_id}",
                    "review_count": 0,
                    "reviews": [],
                }

            reviews = []
            for review in raw_reviews[:MAX_REVIEWS_PER_MOVIE]:
                reviews.append({
                    "review_text": review.get("content", ""),
                    "rating": review.get("rating"),
                    "date": review.get("date", ""),
                    "helpful_votes": review.get("helpful", 0),
                    "title": review.get("title", ""),
                    "author": review.get("author", ""),
                })

            return {
                "movie_title": title,
                "imdb_id": f"tt{movie_id}",
                "review_count": len(reviews),
                "reviews": reviews,
            }

        except Exception as e:
            logger.error(f"Error fetching reviews for {title}: {e}")
            return None


def scrape_all_reviews(seed_list: list[dict], skip_existing: bool = False) -> list[dict]:
    """Scrape reviews for all movies in the seed list.

    Args:
        seed_list: Full seed movie list.
        skip_existing: If True, skip movies whose review JSON already exists.

    Returns:
        List of review dicts.
    """
    processed_dir = config.PROCESSED_DIR / "reviews"
    raw_dir = config.RAW_DIR / "reviews"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    scraper = ReviewScraper()
    results = []
    total = len(seed_list)

    for i, movie in enumerate(seed_list, 1):
        title = movie["title"]
        year = movie.get("year")
        safe_name = _safe_filename(title, year)
        out_path = processed_dir / f"{safe_name}.json"

        if skip_existing and out_path.exists():
            logger.info(f"[{i}/{total}] Skipping (exists): {title}")
            with open(out_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        logger.info(f"[{i}/{total}] Fetching reviews: {title} ({year})")
        review_data = scraper.fetch_reviews(title, year)

        if review_data is None:
            logger.warning(f"[{i}/{total}] Failed: {title}")
            continue

        # Save to processed directory
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(review_data, f, indent=2, ensure_ascii=False)

        results.append(review_data)
        logger.info(f"  -> {review_data['review_count']} reviews saved")

        # Brief pause between movies to be polite
        time.sleep(1)

    total_reviews = sum(r.get("review_count", 0) for r in results)
    logger.info(f"Scraped reviews for {len(results)}/{total} movies ({total_reviews} total reviews)")
    return results


def _safe_filename(title: str, year: int | None = None) -> str:
    """Create a filesystem-safe filename."""
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe = safe.strip().replace(" ", "_").lower()
    if year:
        safe = f"{safe}_{year}"
    return safe
