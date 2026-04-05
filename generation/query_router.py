"""Query router — classifies queries into types using keyword matching.

No LLM is used for routing. This keeps latency under 1ms per classification.
"""

import re

from generation.prompts import get_template

# Keywords that signal each query type (checked against lowercased query)
TREND_KEYWORDS = {
    "evolve", "evolved", "evolution", "change over time", "changed over",
    "trend", "trends", "how did", "how has", "over the years",
    "over the decades", "decade", "progression", "shift", "shifted",
    "transformation", "from the", "history of",
}

PATTERN_KEYWORDS = {
    "pattern", "patterns", "recurring", "recurrence", "reuse", "reuses",
    "similar structure", "signature style", "signature", "always does",
    "trademark", "trademarks", "hallmark", "hallmarks", "consistent",
    "consistently", "across his films", "across her films", "across their",
    "body of work", "filmography", "common theme", "common themes",
}

HIT_FLOP_KEYWORDS = {
    "hit", "hits", "flop", "flops", "succeed", "success", "successful",
    "fail", "failure", "failed", "work vs", "what makes", "box office",
    "blockbuster", "disaster", "bombed", "commercial", "differentiates",
    "differentiate", "why did", "why do some",
}

# Year range pattern suggests trend analysis
YEAR_RANGE_PATTERN = re.compile(
    r"(?:from|between)\s+\d{4}\s+(?:to|and)\s+\d{4}", re.IGNORECASE
)


def classify(query: str) -> str:
    """Classify a query into one of four types.

    Returns one of: 'trend', 'pattern', 'hit_flop', 'general'.
    """
    q = query.lower()

    # Score each category by counting keyword hits
    trend_score = sum(1 for kw in TREND_KEYWORDS if kw in q)
    pattern_score = sum(1 for kw in PATTERN_KEYWORDS if kw in q)
    hit_flop_score = sum(1 for kw in HIT_FLOP_KEYWORDS if kw in q)

    # Year range in query is a strong signal for trend analysis
    if YEAR_RANGE_PATTERN.search(q):
        trend_score += 2

    # Pick the category with the highest score (if any matched)
    scores = {
        "trend": trend_score,
        "pattern": pattern_score,
        "hit_flop": hit_flop_score,
    }

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "general"


def get_prompt(query_type: str) -> str:
    """Get the prompt template for the classified query type."""
    return get_template(query_type)
