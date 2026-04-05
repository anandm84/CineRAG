"""Master ingestion orchestrator for the CineRAG pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path so config can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from ingestion.tmdb_client import TMDbClient, load_seed_list
from ingestion.script_scraper import scrape_all_scripts
from ingestion.review_scraper import scrape_all_reviews
from ingestion.metadata_store import MetadataStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_ingestion(
    skip_existing: bool = False,
    movies_only: bool = False,
    scripts_only: bool = False,
    reviews_only: bool = False,
):
    """Run the full ingestion pipeline.

    Args:
        skip_existing: Skip movies/data that already have output files.
        movies_only: Only fetch TMDb metadata.
        scripts_only: Only scrape scripts.
        reviews_only: Only scrape reviews.
    """
    run_all = not any([movies_only, scripts_only, reviews_only])

    logger.info("=" * 60)
    logger.info("CineRAG Ingestion Pipeline")
    logger.info("=" * 60)

    # Load seed list
    seed_list = load_seed_list()
    logger.info(f"Loaded {len(seed_list)} movies from seed list")

    metadata_list = []
    script_count = 0
    review_count = 0

    # Step 1: Fetch TMDb metadata
    if run_all or movies_only:
        logger.info("")
        logger.info("-" * 40)
        logger.info("Step 1: Fetching TMDb metadata")
        logger.info("-" * 40)

        if not config.TMDB_API_KEY:
            logger.error(
                "TMDB_API_KEY not set! Add it to .env file. "
                "Get a free key at https://developer.themoviedb.org/"
            )
            if not run_all:
                return
        else:
            client = TMDbClient()
            metadata_list = client.fetch_all(seed_list, skip_existing=skip_existing)
            logger.info(f"Fetched metadata for {len(metadata_list)} movies")

    # Step 2: Populate SQLite database
    if (run_all or movies_only) and metadata_list:
        logger.info("")
        logger.info("-" * 40)
        logger.info("Step 2: Populating SQLite database")
        logger.info("-" * 40)

        store = MetadataStore()
        inserted = store.populate_from_metadata(metadata_list)
        logger.info(f"Inserted {inserted} movies into database")

    # Step 3: Scrape scripts from IMSDb
    if run_all or scripts_only:
        logger.info("")
        logger.info("-" * 40)
        logger.info("Step 3: Scraping scripts from IMSDb")
        logger.info("-" * 40)

        scripts = scrape_all_scripts(seed_list, skip_existing=skip_existing)
        script_count = len(scripts)
        logger.info(f"Scraped {script_count} scripts")

        # Update data availability in SQLite
        if scripts:
            store = MetadataStore()
            for script in scripts:
                movie = store.get_movie_by_title(script.get("movie_title", ""))
                if movie:
                    store.update_data_availability(
                        movie["tmdb_id"], has_script=True
                    )

    # Step 4: Scrape IMDb reviews
    if run_all or reviews_only:
        logger.info("")
        logger.info("-" * 40)
        logger.info("Step 4: Scraping IMDb reviews")
        logger.info("-" * 40)

        reviews = scrape_all_reviews(seed_list, skip_existing=skip_existing)
        review_count = len(reviews)
        total_reviews = sum(r.get("review_count", 0) for r in reviews)
        logger.info(f"Scraped reviews for {review_count} movies ({total_reviews} total reviews)")

        # Update data availability in SQLite
        if reviews:
            store = MetadataStore()
            for review in reviews:
                movie = store.get_movie_by_title(review.get("movie_title", ""))
                if movie and review.get("review_count", 0) > 0:
                    store.update_data_availability(
                        movie["tmdb_id"], has_reviews=True
                    )

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)

    if run_all or movies_only:
        logger.info(f"  Movies metadata:  {len(metadata_list)}")
    if run_all or scripts_only:
        logger.info(f"  Scripts scraped:  {script_count}")
    if run_all or reviews_only:
        logger.info(f"  Reviews scraped:  {review_count}")

    # Print database stats if available
    try:
        store = MetadataStore()
        stats = store.get_stats()
        logger.info("")
        logger.info("Database Statistics:")
        logger.info(f"  Total movies:     {stats['total_movies']}")
        for lang, count in stats["by_language"].items():
            logger.info(f"    {lang}: {count}")
        logger.info(f"  Classified hits:  {stats['hits']}")
        logger.info(f"  Classified flops: {stats['flops']}")
        logger.info(f"  With scripts:     {stats['with_scripts']}")
        logger.info(f"  With reviews:     {stats['with_reviews']}")
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("Ingestion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="CineRAG Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ingestion.run_ingestion                    # Run full pipeline
  python -m ingestion.run_ingestion --skip-existing    # Skip already-fetched data
  python -m ingestion.run_ingestion --movies-only      # Only fetch TMDb metadata
  python -m ingestion.run_ingestion --reviews-only     # Only scrape reviews
        """,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip movies/data that already have output files",
    )
    parser.add_argument(
        "--movies-only",
        action="store_true",
        help="Only fetch TMDb metadata and populate database",
    )
    parser.add_argument(
        "--scripts-only",
        action="store_true",
        help="Only scrape scripts from IMSDb",
    )
    parser.add_argument(
        "--reviews-only",
        action="store_true",
        help="Only scrape IMDb reviews",
    )

    args = parser.parse_args()
    run_ingestion(
        skip_existing=args.skip_existing,
        movies_only=args.movies_only,
        scripts_only=args.scripts_only,
        reviews_only=args.reviews_only,
    )


if __name__ == "__main__":
    main()
