"""
Semantic search service for the Ask Filmory assistant.

Supports:
1. Catalog Lookup mode (two-stage exact title and token matching).
2. Reference Title resolution and similarity search.
3. Personalized Recommendation filtering with strict constraints.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from sqlalchemy import cast, String, and_, or_
from sqlalchemy.orm import Session

from app.models.db_models import Movie
from app.schemas.assistant import MovieIntent

logger = logging.getLogger("filmory.search")

CONVERSATIONAL_STOP_PHRASES = [
    r"\b(?:all\s+the\s+movies?|all\s+the\s+films?|all\s+movies?|all\s+films?|movies?|films?)\b",
    r"\b(?:show\s+me|give\s+me|recommend\s+me|recommend|suggest|list\s+all|list|browse|find)\b",
    r"\b(?:please|can\s+you|could\s+you|i\s+want|i\s+would\s+like)\b",
]

GENERIC_STOP_WORDS = {
    "a", "an", "the", "all", "any", "movie", "movies", "film", "films",
    "show", "shows", "me", "watch", "good", "best", "popular", "something",
    "like", "recommend", "of", "in", "for", "with", "without", "from",
}


def clean_query_filler(query: str) -> str:
    """
    Strip conversational filler while preserving core title keywords.
    E.g. 'avengers all movie' -> 'avengers'
    """
    cleaned = query.strip()
    for pattern in CONVERSATIONAL_STOP_PHRASES:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned


def find_catalog_matches(
    query_str: str,
    db: Session,
    exclude_ids: set[int] | None = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    limit: int = 50,
) -> list[Movie]:
    """
    Two-stage title and franchise catalog search:
    Stage 1: Direct exact or normalized title match (preserves short titles like 'It', 'All About Eve').
    Stage 2: Conversational filler removal and word-boundary token matching (e.g. 'avengers all movie' -> 'avengers').
    """
    exclude_ids = exclude_ids or set()
    raw_trimmed = query_str.strip()
    if not raw_trimmed:
        return []

    # Extract year if present in raw string (e.g. "The Avengers 1998")
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", raw_trimmed)
    explicit_year = int(year_match.group(1)) if year_match else None
    effective_min_year = min_year or explicit_year
    effective_max_year = max_year or explicit_year

    raw_without_year = re.sub(r"\b(19\d\d|20\d\d)\b", "", raw_trimmed).strip() if year_match else raw_trimmed

    def _run_title_query(term: str) -> list[Movie]:
        if not term:
            return []
        core_term = re.sub(r"^(?:the|a|an)\s+", "", term, flags=re.IGNORECASE).strip()
        search_kw = core_term if core_term else term
        base_q = db.query(Movie)
        if exclude_ids:
            base_q = base_q.filter(~Movie.movie_id.in_(exclude_ids))
        if effective_min_year is not None:
            base_q = base_q.filter(Movie.year >= effective_min_year)
        if effective_max_year is not None:
            base_q = base_q.filter(Movie.year <= effective_max_year)

        candidates = (
            base_q.filter(
                or_(
                    Movie.title.ilike(f"%{term}%"),
                    Movie.title.ilike(f"%{search_kw}%"),
                )
            )
            .limit(limit * 2)
            .all()
        )

        # Word boundary check so "avengers" matches "The Avengers" but NOT "Scavenger Hunt"
        filtered = []
        regex_pattern = rf"\b{re.escape(search_kw)}\b"
        for m in candidates:
            if re.search(regex_pattern, m.title, re.IGNORECASE):
                filtered.append(m)
        return filtered

    # Stage 1: Try direct match on raw title
    matches = _run_title_query(raw_without_year)
    if matches:
        def _rank_match(m: Movie) -> tuple[int, int]:
            t_low = m.title.lower()
            q_low = raw_without_year.lower()
            exact_match = 0 if q_low in t_low else 1
            yr = m.year or 0
            return (exact_match, -yr)

        matches.sort(key=_rank_match)
        return matches[:limit]

    # Stage 2: Strip conversational filler (e.g. "avengers all movie" -> "avengers")
    cleaned = clean_query_filler(raw_without_year)
    if not cleaned or cleaned.lower() in GENERIC_STOP_WORDS:
        return []

    matches = _run_title_query(cleaned)
    if matches:
        def _rank_cleaned(m: Movie) -> tuple[int, int]:
            t_low = m.title.lower()
            c_low = cleaned.lower()
            exact_match = 0 if c_low in t_low else 1
            yr = m.year or 0
            return (exact_match, -yr)

        matches.sort(key=_rank_cleaned)
        return matches[:limit]

    return []


def search_movies_by_intent(
    intent: MovieIntent,
    db: Session,
    exclude_ids: set[int] | None = None,
    limit: int = 50,
) -> tuple[list[Movie], list[int]]:
    """
    Retrieve candidate movies based on intent.
    Routes based on intent.request_mode:
    - 'catalog_lookup': Uses two-stage title matching; returns verified title matches only.
    - 'similar_movies' / 'personalized_recommendation': Applies hard constraints and keyword filters.
    """
    exclude_ids = exclude_ids or set()
    resolved_reference_ids: list[int] = []

    # Resolve any reference titles first
    if intent.reference_titles:
        for title in intent.reference_titles[:3]:
            ref_m = (
                db.query(Movie)
                .filter(Movie.title.ilike(f"%{title.strip()}%"))
                .order_by(Movie.rating_count.desc())
                .first()
            )
            if ref_m:
                resolved_reference_ids.append(ref_m.movie_id)

    # 1. CATALOG LOOKUP MODE
    if intent.request_mode == "catalog_lookup":
        search_term = ""
        if intent.reference_titles:
            search_term = intent.reference_titles[0]
        elif intent.semantic_query:
            search_term = intent.semantic_query

        if not search_term:
            return [], resolved_reference_ids

        catalog_movies = find_catalog_matches(
            query_str=search_term,
            db=db,
            exclude_ids=exclude_ids,
            min_year=intent.min_year,
            max_year=intent.max_year,
            limit=limit,
        )
        return catalog_movies, resolved_reference_ids

    # 2. PERSONALIZED RECOMMENDATION OR SIMILAR MOVIES MODE
    filters = []

    # Genre inclusion
    if intent.preferred_genres:
        genre_conditions = [
            cast(Movie.genres, String).ilike(f"%{g}%") for g in intent.preferred_genres
        ]
        filters.append(or_(*genre_conditions))

    # Genre exclusion
    for g in intent.excluded_genres:
        filters.append(~cast(Movie.genres, String).ilike(f"%{g}%"))

    # Year range
    if intent.min_year is not None:
        filters.append(Movie.year >= intent.min_year)
    if intent.max_year is not None:
        filters.append(Movie.year <= intent.max_year)

    # Runtime
    if intent.max_runtime_minutes is not None:
        filters.append(Movie.runtime > 0)
        filters.append(Movie.runtime <= intent.max_runtime_minutes)

    # Rating
    if intent.min_rating is not None:
        filters.append(Movie.rating >= intent.min_rating)

    # Exclusions
    if exclude_ids:
        filters.append(~Movie.movie_id.in_(exclude_ids))

    # Title / Synopsis keywords for discovery (skip generic stop words!)
    if intent.semantic_query:
        cleaned_query = clean_query_filler(intent.semantic_query)
        words = [
            w.lower()
            for w in cleaned_query.split()
            if len(w) >= 3 and w.lower() not in GENERIC_STOP_WORDS
        ]
        if words:
            kw_conditions = []
            for kw in words[:3]:
                kw_conditions.append(
                    or_(
                        Movie.title.ilike(f"%{kw}%"),
                        Movie.description.ilike(f"%{kw}%"),
                    )
                )
            filters.append(or_(*kw_conditions))

    query = db.query(Movie)
    if filters:
        query = query.filter(and_(*filters))

    query = query.filter(Movie.description != "")
    query = query.order_by(Movie.rating_count.desc())

    movies = query.limit(limit).all()

    logger.info(
        "Search found %d movies (mode=%s, genres=%s, excluded=%s, year=%s-%s, runtime<=%s)",
        len(movies),
        intent.request_mode,
        intent.preferred_genres,
        intent.excluded_genres,
        intent.min_year,
        intent.max_year,
        intent.max_runtime_minutes,
    )

    return movies, resolved_reference_ids
