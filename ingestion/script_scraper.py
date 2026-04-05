"""Scraper for movie scripts from IMSDb (imsdb.com)."""

import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

IMSDB_BASE = "https://imsdb.com"
REQUEST_DELAY = 2.0  # seconds between requests to be polite


class ScriptScraper:
    """Scrapes movie scripts from IMSDb."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CineRAG Research Project (educational use)"
        })

    def _get_page(self, url: str) -> BeautifulSoup | None:
        """Fetch a page and return parsed BeautifulSoup."""
        time.sleep(REQUEST_DELAY)
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def get_script_url(self, title: str) -> str | None:
        """Search IMSDb for a script and return the script page URL."""
        # IMSDb script URLs follow a pattern: /scripts/Title.html
        # Try constructing the URL directly first
        formatted = title.replace(" ", "-")
        script_url = f"{IMSDB_BASE}/scripts/{formatted}.html"

        try:
            time.sleep(REQUEST_DELAY)
            response = self.session.get(script_url, timeout=15)
            if response.status_code == 200 and len(response.text) > 1000:
                return script_url
        except requests.RequestException:
            pass

        # Fall back to search
        search_url = f"{IMSDB_BASE}/search"
        soup = self._get_page(f"{search_url}?query={title.replace(' ', '+')}")
        if soup is None:
            return None

        # Find script links in search results
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if "/scripts/" in href and href.endswith(".html"):
                return f"{IMSDB_BASE}{href}" if href.startswith("/") else href

        return None

    def scrape_script(self, url: str) -> str | None:
        """Scrape the script text from an IMSDb script page."""
        soup = self._get_page(url)
        if soup is None:
            return None

        # IMSDb stores scripts in a <pre> tag or a <td class="scrtext"> tag
        script_tag = soup.find("td", class_="scrtext")
        if script_tag:
            return script_tag.get_text(separator="\n").strip()

        script_tag = soup.find("pre")
        if script_tag:
            return script_tag.get_text(separator="\n").strip()

        return None

    def parse_scenes(self, script_text: str) -> list[dict]:
        """Parse script text into scene-level chunks.

        Detects scene boundaries using standard screenplay formatting:
        - INT./EXT. scene headings
        - FADE IN, CUT TO, DISSOLVE TO transitions
        """
        # Pattern for scene headings
        scene_pattern = re.compile(
            r"^((?:INT\.|EXT\.|INT/EXT\.|INT\./EXT\.).*?)$",
            re.MULTILINE | re.IGNORECASE,
        )

        scenes = []
        splits = scene_pattern.split(script_text)

        if len(splits) <= 1:
            # No scene headings detected — return the whole script as one scene
            return [{"scene_number": 1, "heading": "FULL SCRIPT", "content": script_text.strip()}]

        # splits alternates between: text-before, heading, text-after, heading, text-after, ...
        scene_num = 0

        # Handle any text before the first scene heading
        preamble = splits[0].strip()
        if preamble:
            scene_num += 1
            scenes.append({
                "scene_number": scene_num,
                "heading": "PREAMBLE",
                "content": preamble,
            })

        # Process heading + content pairs
        for i in range(1, len(splits), 2):
            heading = splits[i].strip()
            content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            scene_num += 1
            scenes.append({
                "scene_number": scene_num,
                "heading": heading,
                "content": content,
            })

        return scenes

    def fetch_script(self, title: str) -> dict | None:
        """Fetch and parse a script for the given movie title.

        Returns:
            Dict with movie_title, source, script_text, and scenes.
            None if the script isn't available.
        """
        logger.info(f"Searching IMSDb for: {title}")
        url = self.get_script_url(title)
        if url is None:
            logger.info(f"  -> Not found on IMSDb: {title}")
            return None

        logger.info(f"  -> Found: {url}")
        script_text = self.scrape_script(url)
        if not script_text or len(script_text) < 500:
            logger.warning(f"  -> Script too short or empty for: {title}")
            return None

        scenes = self.parse_scenes(script_text)

        return {
            "movie_title": title,
            "source": "imsdb",
            "url": url,
            "script_length": len(script_text),
            "scene_count": len(scenes),
            "script_text": script_text,
            "scenes": scenes,
        }


def scrape_all_scripts(seed_list: list[dict], skip_existing: bool = False) -> list[dict]:
    """Scrape scripts for all Hollywood movies in the seed list.

    Only attempts Hollywood (English) movies since IMSDb primarily has English scripts.

    Args:
        seed_list: Full seed movie list.
        skip_existing: If True, skip movies whose script JSON already exists.

    Returns:
        List of parsed script dicts.
    """
    raw_dir = config.RAW_DIR / "scripts"
    processed_dir = config.PROCESSED_DIR / "scripts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Filter to English-language movies only
    hollywood = [m for m in seed_list if m.get("language") == "en"]
    if not hollywood:
        logger.info("No Hollywood movies in seed list to scrape scripts for")
        return []

    scraper = ScriptScraper()
    results = []
    total = len(hollywood)

    for i, movie in enumerate(hollywood, 1):
        title = movie["title"]
        year = movie.get("year")
        safe_name = f"{title.lower().replace(' ', '_').replace(':', '')}_{year}"
        safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)
        out_path = processed_dir / f"{safe_name}.json"

        if skip_existing and out_path.exists():
            logger.info(f"[{i}/{total}] Skipping (exists): {title}")
            with open(out_path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        logger.info(f"[{i}/{total}] Scraping: {title}")
        script_data = scraper.fetch_script(title)

        if script_data is None:
            continue

        # Save raw script text
        raw_path = raw_dir / f"{safe_name}.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(script_data["script_text"])

        # Save processed JSON (without full script text to save space)
        processed = {k: v for k, v in script_data.items() if k != "script_text"}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)

        results.append(script_data)
        logger.info(f"  -> {script_data['scene_count']} scenes extracted")

    logger.info(f"Scraped {len(results)}/{total} scripts from IMSDb")
    return results
