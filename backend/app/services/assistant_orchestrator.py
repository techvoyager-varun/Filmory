"""
Assistant Orchestrator — the main workflow engine for Ask Filmory.

Coordinates:
1. Session management (create / load / update)
2. LLM intent extraction
3. Intent merging with session state (follow-up support)
4. Candidate retrieval (metadata search + personalized pipeline)
5. DAMR scoring
6. Evidence construction
7. Grounded explanation generation
8. Response validation + persistence
"""
from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Optional

import torch
from sqlalchemy.orm import Session

from app.config import settings
from app.ml.model_service import model_service
from app.ml.recommender import (
    get_personalized_recommendations,
    get_similar_movies,
    movie_model_to_schema,
    rerank_candidate_pool_with_damr,
)
from app.ml.tmdb import ensure_movie_posters
from app.models.db_models import (
    AssistantFeedback,
    AssistantMessageRecord,
    AssistantSession,
    Interaction,
    Movie,
    User,
    WatchHistory,
)
from app.schemas.assistant import (
    ChatResponse,
    ExplanationItem,
    FeedbackRequest,
    IntentDelta,
    MessageSchema,
    MovieEvidence,
    MovieIntent,
    SessionDetail,
    SessionSummary,
    StructuredExplanation,
)
from app.schemas.schemas import ScoredMovieSchema
from app.services.llm_client import extract_intent, fallback_extract_intent, generate_explanation
from app.services.semantic_search import search_movies_by_intent

logger = logging.getLogger("filmory.orchestrator")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def _get_or_create_session(
    user: User, session_id: Optional[str], db: Session
) -> AssistantSession:
    """Load an existing session (with ownership check) or create a new one."""
    if session_id:
        sess = (
            db.query(AssistantSession)
            .filter(
                AssistantSession.id == session_id,
                AssistantSession.user_id == user.id,
            )
            .first()
        )
        if sess:
            return sess
        logger.warning("Session %s not found for user %s, creating new", session_id, user.id)

    sess = AssistantSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        active_intent={},
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    db.add(sess)
    db.flush()
    return sess


def _build_conversation_history(session: AssistantSession) -> list[dict]:
    """Build a list of {role, content} dicts from session messages for LLM context."""
    history = []
    for msg in session.messages[-settings.ASSISTANT_MAX_HISTORY :]:
        history.append({"role": msg.role, "content": msg.content})
    return history


# ---------------------------------------------------------------------------
# Intent merging — explicit delta updates & ordinal exclusions
# ---------------------------------------------------------------------------
def _matches_hard_constraints(movie: Movie, intent: MovieIntent) -> bool:
    """Validate that a candidate movie strictly satisfies all hard constraints."""
    movie_genres = [g.lower() for g in (movie.genres if isinstance(movie.genres, list) else [])]

    # Excluded genres
    for ex in intent.excluded_genres:
        if ex.lower() in movie_genres:
            return False

    # Preferred genres
    if intent.preferred_genres:
        if not any(pref.lower() in movie_genres for pref in intent.preferred_genres):
            return False

    # Year range
    if intent.min_year is not None and (movie.year is None or movie.year < intent.min_year):
        return False
    if intent.max_year is not None and (movie.year is None or movie.year > intent.max_year):
        return False

    # Strict runtime policy: when max_runtime_minutes is set, unknown/null runtime is strictly excluded!
    if intent.max_runtime_minutes is not None:
        if movie.runtime is None or movie.runtime <= 0 or movie.runtime > intent.max_runtime_minutes:
            return False

    # Min rating
    if intent.min_rating is not None and (movie.rating is None or movie.rating < intent.min_rating):
        return False

    return True


def _merge_intents(
    previous: dict,
    delta: IntentDelta,
    previous_movie_ids: list[int] | None = None,
) -> MovieIntent:
    """
    Explicitly merge an extracted IntentDelta with the session's accumulated intent:
      1. If clear_fields contains 'all', reset all constraints.
      2. Handle explicit clear_fields (e.g. 'max_runtime_minutes', 'excluded_genres', 'preferred_genres:Comedy').
      3. Handle explicit set_fields (e.g. {'max_runtime_minutes': 90}).
      4. Handle ordinal exclusions ("exclude the second movie" -> target_ordinal_exclusion=2 maps to previous_movie_ids[1]).
      5. Handle explicit add_excluded_movie_ids.
      6. Accumulate or update non-empty delta fields.
    """
    merged = dict(previous)

    # 1. Clear "all"
    if "all" in delta.clear_fields or "all_restrictions" in delta.clear_fields:
        merged = {
            "semantic_query": "",
            "preferred_genres": [],
            "excluded_genres": [],
            "min_year": None,
            "max_year": None,
            "max_runtime_minutes": None,
            "min_rating": None,
            "reference_titles": [],
            "excluded_movie_ids": merged.get("excluded_movie_ids", []),
            "mood": "",
            "needs_clarification": False,
            "clarification_question": "",
        }
    else:
        for field in delta.clear_fields:
            if field == "max_runtime_minutes":
                merged["max_runtime_minutes"] = None
            elif field in ("min_year", "max_year"):
                merged[field] = None
            elif field == "min_rating":
                merged["min_rating"] = None
            elif field == "excluded_genres":
                merged["excluded_genres"] = []
            elif field == "preferred_genres":
                merged["preferred_genres"] = []
            elif field == "mood":
                merged["mood"] = ""
            elif field.startswith("excluded_genres:"):
                genre_to_remove = field.split(":", 1)[1]
                merged["excluded_genres"] = [
                    g for g in merged.get("excluded_genres", []) if g.lower() != genre_to_remove.lower()
                ]
            elif field.startswith("preferred_genres:"):
                genre_to_remove = field.split(":", 1)[1]
                merged["preferred_genres"] = [
                    g for g in merged.get("preferred_genres", []) if g.lower() != genre_to_remove.lower()
                ]

    # 2. Apply set_fields
    for k, v in delta.set_fields.items():
        merged[k] = v

    # 3. Handle target_ordinal_exclusion from previous assistant message
    if delta.target_ordinal_exclusion and previous_movie_ids:
        idx = delta.target_ordinal_exclusion - 1
        if 0 <= idx < len(previous_movie_ids):
            target_id = previous_movie_ids[idx]
            prev_ex = set(merged.get("excluded_movie_ids", []))
            prev_ex.add(target_id)
            merged["excluded_movie_ids"] = list(prev_ex)
            logger.info("[AUDIT:EXCLUSION] Resolved ordinal %d to movie_id %d", delta.target_ordinal_exclusion, target_id)

    # 4. Handle add_excluded_movie_ids
    if delta.add_excluded_movie_ids:
        prev_ex = set(merged.get("excluded_movie_ids", []))
        prev_ex.update(delta.add_excluded_movie_ids)
        merged["excluded_movie_ids"] = list(prev_ex)

    # 5. Non-empty delta values override / accumulate
    if delta.semantic_query:
        merged["semantic_query"] = delta.semantic_query
    if delta.preferred_genres:
        merged["preferred_genres"] = delta.preferred_genres
    if delta.excluded_genres:
        prev_ex_genres = set(merged.get("excluded_genres", []))
        prev_ex_genres.update(delta.excluded_genres)
        merged["excluded_genres"] = list(prev_ex_genres)
    if delta.min_year is not None:
        merged["min_year"] = delta.min_year
    if delta.max_year is not None:
        merged["max_year"] = delta.max_year
    if delta.max_runtime_minutes is not None:
        merged["max_runtime_minutes"] = delta.max_runtime_minutes
    if delta.min_rating is not None:
        merged["min_rating"] = delta.min_rating
    if delta.reference_titles:
        merged["reference_titles"] = delta.reference_titles
    if delta.mood:
        merged["mood"] = delta.mood
    if delta.request_mode:
        merged["request_mode"] = delta.request_mode

    return MovieIntent.model_validate(merged)


# ---------------------------------------------------------------------------
# Evidence builder
# ---------------------------------------------------------------------------
def _build_evidence(
    movie: Movie,
    intent: MovieIntent,
    score: float,
    user: User,
    db: Session,
    variant: str = "damr",
) -> MovieEvidence:
    """Construct a verifiable evidence package for one recommended movie."""
    matched = []
    evidence_keys = [
        f"movie:{movie.movie_id}:genres",
        f"movie:{movie.movie_id}:personalization",
        f"movie:{movie.movie_id}:damr",
    ]
    if movie.year:
        evidence_keys.append(f"movie:{movie.movie_id}:year")
    if movie.runtime:
        evidence_keys.append(f"movie:{movie.movie_id}:runtime")
    if movie.rating:
        evidence_keys.append(f"movie:{movie.movie_id}:rating")

    movie_genres = movie.genres if isinstance(movie.genres, list) else []
    for g in intent.preferred_genres:
        if any(g.lower() == mg.lower() for mg in movie_genres):
            matched.append(f"{g} genre")
            evidence_keys.append(f"movie:{movie.movie_id}:genre_{g.lower()}")

    if intent.max_runtime_minutes and movie.runtime and movie.runtime <= intent.max_runtime_minutes:
        matched.append(f"Under {intent.max_runtime_minutes} minutes ({movie.runtime} min)")
        evidence_keys.append(f"movie:{movie.movie_id}:runtime_constraint")

    if intent.min_year and movie.year and movie.year >= intent.min_year:
        matched.append(f"Released {movie.year} (after {intent.min_year})")
        evidence_keys.append(f"movie:{movie.movie_id}:min_year_constraint")
    if intent.max_year and movie.year and movie.year <= intent.max_year:
        matched.append(f"Released {movie.year} (before {intent.max_year})")
        evidence_keys.append(f"movie:{movie.movie_id}:max_year_constraint")

    if intent.min_rating and movie.rating and movie.rating >= intent.min_rating:
        matched.append(f"Rating {movie.rating:.1f}/10")
        evidence_keys.append(f"movie:{movie.movie_id}:min_rating_constraint")

    # Personalization signals
    signals = []
    recent_interactions = (
        db.query(Interaction)
        .filter(Interaction.user_id == user.id)
        .order_by(Interaction.timestamp.desc())
        .limit(20)
        .all()
    )
    recent_movie_ids = [inter.movie_id for inter in recent_interactions]
    if recent_movie_ids:
        recent_movies = db.query(Movie.genres).filter(Movie.movie_id.in_(recent_movie_ids)).all()
        recent_genres = set()
        for (g_list,) in recent_movies:
            if isinstance(g_list, list):
                recent_genres.update(g_list)
        for g in movie_genres:
            if g in recent_genres:
                signals.append(f"{g} in recent activity")
                evidence_keys.append(f"movie:{movie.movie_id}:recent_genre_{g.lower()}")
                break

    description_snippet = (movie.description or "")[:200]

    return MovieEvidence(
        movie_id=movie.movie_id,
        title=movie.title,
        genres=movie_genres,
        year=movie.year,
        runtime=movie.runtime,
        rating=float(movie.rating) if movie.rating is not None else None,
        description_snippet=description_snippet,
        matched_constraints=matched,
        personalization_signals=signals,
        score=round(score, 4),
        damr_variant=variant,
        evidence_keys=list(set(evidence_keys)),
    )


# ---------------------------------------------------------------------------
# Core message handler — main orchestrator pipeline
# ---------------------------------------------------------------------------
def handle_message(
    user: User,
    message: str,
    session_id: Optional[str],
    db: Session,
) -> ChatResponse:
    """Execute the full conversational recommendation workflow with strict DAMR scoring and audit logging."""
    # 1. Session setup
    session = _get_or_create_session(user, session_id, db)
    conversation_history = _build_conversation_history(session)

    # Previous displayed movie IDs for ordinal resolution ("exclude the second movie")
    previous_displayed_movie_ids: list[int] = []
    if session.messages:
        for prev in reversed(session.messages):
            if prev.role == "assistant" and prev.movie_ids:
                previous_displayed_movie_ids = list(prev.movie_ids)
                break

    # Persist user message
    user_msg = AssistantMessageRecord(
        session_id=session.id,
        role="user",
        content=message,
        created_at=datetime.datetime.utcnow(),
    )
    db.add(user_msg)
    db.flush()

    # 2. Extract intent delta
    ai_interpretation_available = bool(settings.GEMINI_API_KEY)
    try:
        delta = extract_intent(message, conversation_history)
        logger.info("[AUDIT:INTENT] Extracted intent delta: %s", delta.model_dump())
    except Exception as e:
        logger.error("Intent extraction failed: %s", e, exc_info=True)
        ai_interpretation_available = False
        delta = fallback_extract_intent(message)

    # 3. Handle clarification requests
    if delta.needs_clarification and delta.clarification_question:
        assistant_msg = AssistantMessageRecord(
            session_id=session.id,
            role="assistant",
            content=delta.clarification_question,
            intent_snapshot=delta.model_dump(),
            created_at=datetime.datetime.utcnow(),
        )
        db.add(assistant_msg)
        db.commit()

        fallback_intent = MovieIntent(
            semantic_query=message,
            request_mode=delta.request_mode or "catalog_lookup",
            needs_clarification=True,
            clarification_question=delta.clarification_question,
        )
        return ChatResponse(
            session_id=session.id,
            message=delta.clarification_question,
            movies=[],
            intent=fallback_intent,
            evidence=[],
            clarification=delta.clarification_question,
            request_mode=delta.request_mode or "catalog_lookup",
            notice=None,
        )

    # 4. Merge intent with session state using explicit delta
    merged_intent = _merge_intents(session.active_intent or {}, delta, previous_displayed_movie_ids)
    session.active_intent = merged_intent.model_dump()
    session.updated_at = datetime.datetime.utcnow()

    # 5. Get user's watched movie IDs for exclusion
    watched_ids = set()
    watch_records = db.query(WatchHistory.movie_id).filter(WatchHistory.user_id == user.id).all()
    watched_ids.update(r[0] for r in watch_records)
    watched_ids.update(merged_intent.excluded_movie_ids)

    # Also exclude movies already returned in this session
    for prev_msg in session.messages:
        if prev_msg.movie_ids:
            watched_ids.update(prev_msg.movie_ids)

    # Base notice based on AI interpretation status
    notice: Optional[str] = None
    if not ai_interpretation_available:
        notice = "AI interpretation is unavailable. Showing catalog search results."

    # 6. Retrieve candidates via metadata/keyword search enforcing hard constraints
    search_movies, reference_ids = search_movies_by_intent(
        merged_intent, db, exclude_ids=watched_ids, limit=settings.RERANK_POOL_SIZE
    )
    metadata_count = len(search_movies)
    similar_count = 0

    # 7. If we have reference movies in similar_movies mode, also add similar movies
    if merged_intent.request_mode == "similar_movies" and reference_ids:
        for ref_id in reference_ids[:2]:
            try:
                similar = get_similar_movies(ref_id, db, top_k=10)
                search_movie_ids = {m.movie_id for m in search_movies}
                for s_movie in similar:
                    if s_movie.movieId not in search_movie_ids and s_movie.movieId not in watched_ids:
                        db_movie = db.query(Movie).filter(Movie.movie_id == s_movie.movieId).first()
                        if db_movie and _matches_hard_constraints(db_movie, merged_intent):
                            search_movies.append(db_movie)
                            similar_count += 1
            except Exception as e:
                logger.warning("Similar movies lookup failed for ref %d: %s", ref_id, e)

    logger.info("[AUDIT:RETRIEVAL] Mode=%s, candidates: metadata=%d, similar=%d", merged_intent.request_mode, metadata_count, similar_count)
    logger.info("[AUDIT:CONSTRAINTS] Candidates remaining after constraints & exclusions: %d", len(search_movies))

    # 8. Score candidate pool according to request_mode
    has_active_constraints = (
        bool(merged_intent.preferred_genres)
        or bool(merged_intent.excluded_genres)
        or merged_intent.min_year is not None
        or merged_intent.max_year is not None
        or merged_intent.max_runtime_minutes is not None
        or merged_intent.min_rating is not None
        or bool(merged_intent.reference_titles)
        or bool(merged_intent.semantic_query)
    )

    scored_movies: list[ScoredMovieSchema] = []
    damr_audit: dict[str, Any] = {}

    # MODE 1: CATALOG LOOKUP — Return verified matching titles directly!
    # Do NOT allow DAMR to override title matches with unrelated personalized recommendations.
    if merged_intent.request_mode == "catalog_lookup":
        if search_movies:
            scored_movies = []
            for i, m in enumerate(search_movies[:10]):
                score = max(0.99 - (i * 0.02), 0.5)
                scored_movies.append(
                    ScoredMovieSchema(
                        movieId=m.movie_id,
                        title=m.title,
                        genres=m.genres or [],
                        year=m.year,
                        runtime=m.runtime,
                        rating=float(m.rating) if m.rating is not None else None,
                        posterUrl=m.poster_url,
                        score=round(score, 4),
                        explanation=f"Catalog match: '{m.title}'",
                    )
                )
            damr_audit = {
                "mode": "catalog_lookup",
                "matches": len(scored_movies),
                "total_candidates": len(search_movies),
                "final_movie_ids": [m.movieId for m in scored_movies],
            }
            coverage_notice = "Showing matching movies available in Filmory’s catalog. This may not include every franchise installment."
            notice = f"{notice} {coverage_notice}" if notice else coverage_notice
        else:
            # Truthful no-results — do not invent or substitute random popular movies!
            scored_movies = []
            damr_audit = {
                "mode": "catalog_lookup",
                "matches": 0,
                "total_candidates": 0,
                "final_movie_ids": [],
            }

    # MODE 2: SIMILAR MOVIES & PERSONALIZED DISCOVERY — Re-rank candidate pool with DAMR!
    elif search_movies:
        scored_movies, damr_audit = rerank_candidate_pool_with_damr(
            candidate_movies=search_movies,
            user=user,
            db=db,
            top_k=10,
            variant=settings.RERANK_VARIANT,
        )
    elif not has_active_constraints:
        # User has no specific restrictions; fall back to personalized DAMR recommendations
        try:
            rec_type, damr_recs = get_personalized_recommendations(
                user=user, db=db, candidate_k=100, top_k=10, variant=settings.RERANK_VARIANT
            )
            scored_movies = damr_recs
            damr_audit = {
                "total_candidates": len(damr_recs),
                "mapped_candidates": len(damr_recs),
                "unmapped_candidates": 0,
                "variant": settings.RERANK_VARIANT,
                "expert_weights": getattr(damr_recs[0], "expertWeights", {}) if damr_recs else {},
                "user_state": getattr(damr_recs[0], "userState", {}) if damr_recs else {},
                "final_movie_ids": [m.movieId for m in damr_recs],
            }
        except Exception as e:
            logger.warning("DAMR fallback failed: %s", e)
    else:
        # Active constraints were present but produced 0 matches: NEVER silently relax them!
        logger.info("[AUDIT:CONSTRAINTS] Zero candidates matched constraints. Returning honest empty state.")
        damr_audit = {
            "total_candidates": 0,
            "mapped_candidates": 0,
            "unmapped_candidates": 0,
            "variant": settings.RERANK_VARIANT,
            "expert_weights": {},
            "user_state": {},
            "final_movie_ids": [],
        }

    logger.info(
        "[AUDIT:EMBEDDINGS] Model mapping: %d mapped, %d unmapped",
        damr_audit.get("mapped_candidates", 0),
        damr_audit.get("unmapped_candidates", 0),
    )
    logger.info(
        "[AUDIT:DAMR] Mode: %s, variant: %s, weights: %s",
        merged_intent.request_mode,
        damr_audit.get("variant"),
        damr_audit.get("expert_weights"),
    )
    logger.info("[AUDIT:RESULTS] Final ranked movie IDs: %s", damr_audit.get("final_movie_ids"))

    # 9. Build evidence for each movie
    evidence_list = []
    for sm in scored_movies:
        db_movie = db.query(Movie).filter(Movie.movie_id == sm.movieId).first()
        if db_movie:
            ev = _build_evidence(
                db_movie,
                merged_intent,
                sm.score or 0.0,
                user,
                db,
                variant=damr_audit.get("variant", "catalog_match" if merged_intent.request_mode == "catalog_lookup" else "damr"),
            )
            evidence_list.append(ev)

    # 10. Generate grounded explanation
    context_summary = ""
    if conversation_history:
        context_summary = "; ".join(
            f"User: {m['content']}" if m["role"] == "user" else f"Assistant: {m['content'][:80]}"
            for m in conversation_history[-4:]
        )

    if not scored_movies:
        if merged_intent.request_mode == "catalog_lookup":
            search_target = merged_intent.reference_titles[0] if merged_intent.reference_titles else (merged_intent.semantic_query or "your search")
            explanation = f"No movies matching '{search_target}' were found in Filmory's catalog. Please check the spelling or search for another title."
        else:
            active_filters_desc = []
            if merged_intent.preferred_genres:
                active_filters_desc.append(f"genres: {', '.join(merged_intent.preferred_genres)}")
            if merged_intent.excluded_genres:
                active_filters_desc.append(f"excluding: {', '.join(merged_intent.excluded_genres)}")
            if merged_intent.max_runtime_minutes:
                active_filters_desc.append(f"under {merged_intent.max_runtime_minutes} min")
            filter_str = f" ({'; '.join(active_filters_desc)})" if active_filters_desc else ""
            explanation = f"No movies found in the catalog matching all of your active constraints{filter_str}. Try broadening your search or relaxing the runtime/genre filters."
        structured_exp = StructuredExplanation(summary=explanation, items=[])
    elif merged_intent.request_mode == "catalog_lookup":
        titles = [m.title for m in scored_movies]
        explanation = f"Found {len(scored_movies)} matching title{'s' if len(scored_movies) > 1 else ''} in Filmory's catalog: {', '.join(titles[:3])}."
        structured_exp = StructuredExplanation(
            summary=explanation,
            items=[
                ExplanationItem(
                    movie_id=m.movieId,
                    explanation=f"Verified catalog match for '{m.title}'",
                    evidence_ids=[f"movie:{m.movieId}:genres"],
                )
                for m in scored_movies
            ],
        )
    else:
        try:
            explanation, structured_exp = generate_explanation(evidence_list, message, context_summary)
        except Exception as e:
            logger.error("Explanation generation failed: %s", e)
            titles = [m.title for m in scored_movies[:3]]
            explanation = f"Here are picks matching your criteria: {', '.join(titles)}."
            structured_exp = StructuredExplanation(summary=explanation, items=[])

    logger.info("[AUDIT:EXPLANATION] Validated explanation items: %d", len(structured_exp.items))

    # 11. Persist assistant response
    returned_movie_ids = [m.movieId for m in scored_movies]
    assistant_msg = AssistantMessageRecord(
        session_id=session.id,
        role="assistant",
        content=explanation,
        intent_snapshot=merged_intent.model_dump(),
        movie_ids=returned_movie_ids,
        created_at=datetime.datetime.utcnow(),
    )
    db.add(assistant_msg)
    db.commit()

    return ChatResponse(
        session_id=session.id,
        message=explanation,
        movies=scored_movies,
        intent=merged_intent,
        evidence=evidence_list,
        structured_explanation=structured_exp,
        clarification=None,
        notice=notice,
        request_mode=merged_intent.request_mode,
    )


# ---------------------------------------------------------------------------
# Session queries
# ---------------------------------------------------------------------------
def get_user_sessions(user: User, db: Session) -> list[SessionSummary]:
    """List all sessions for a user, most recent first."""
    sessions = (
        db.query(AssistantSession)
        .filter(AssistantSession.user_id == user.id)
        .order_by(AssistantSession.updated_at.desc())
        .limit(50)
        .all()
    )
    result = []
    for s in sessions:
        first_user_msg = ""
        for msg in s.messages:
            if msg.role == "user":
                first_user_msg = msg.content[:80]
                break

        result.append(
            SessionSummary(
                session_id=s.id,
                preview=first_user_msg or "New conversation",
                message_count=len(s.messages),
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
            )
        )
    return result


def get_session_detail(user: User, session_id: str, db: Session) -> Optional[SessionDetail]:
    """Load a session with full message history (ownership verified)."""
    sess = (
        db.query(AssistantSession)
        .filter(
            AssistantSession.id == session_id,
            AssistantSession.user_id == user.id,
        )
        .first()
    )
    if not sess:
        return None

    messages = []
    for msg in sess.messages:
        scored = []
        if msg.movie_ids:
            for mid in msg.movie_ids:
                m = db.query(Movie).filter(Movie.movie_id == mid).first()
                if m:
                    scored.append(
                        ScoredMovieSchema(
                            movieId=m.movie_id,
                            title=m.title,
                            genres=m.genres if isinstance(m.genres, list) else [],
                            year=m.year or 0,
                            rating=float(m.rating or 0.0),
                            runtime=m.runtime or 0,
                            posterUrl=m.poster_url or "",
                            backdropUrl=m.backdrop_url or "",
                            description=m.description or "",
                        )
                    )

        messages.append(
            MessageSchema(
                role=msg.role,
                content=msg.content,
                movies=scored,
                intent=MovieIntent.model_validate(msg.intent_snapshot) if msg.intent_snapshot else None,
                evidence=[],
                timestamp=msg.created_at.isoformat(),
            )
        )

    return SessionDetail(
        session_id=sess.id,
        messages=messages,
        active_intent=MovieIntent.model_validate(sess.active_intent) if sess.active_intent else MovieIntent(),
        created_at=sess.created_at.isoformat(),
        updated_at=sess.updated_at.isoformat(),
    )


def delete_session(user: User, session_id: str, db: Session) -> bool:
    """Delete a session (ownership verified). Returns True if deleted."""
    sess = (
        db.query(AssistantSession)
        .filter(
            AssistantSession.id == session_id,
            AssistantSession.user_id == user.id,
        )
        .first()
    )
    if not sess:
        return False
    db.delete(sess)
    db.commit()
    return True


def submit_feedback(user: User, feedback: FeedbackRequest, db: Session) -> bool:
    """Store user feedback on a recommendation or explanation."""
    # Verify session ownership
    sess = (
        db.query(AssistantSession)
        .filter(
            AssistantSession.id == feedback.session_id,
            AssistantSession.user_id == user.id,
        )
        .first()
    )
    if not sess:
        return False

    fb = AssistantFeedback(
        session_id=feedback.session_id,
        message_id=feedback.message_id,
        movie_id=feedback.movie_id,
        feedback_type=feedback.feedback_type,
        comment=feedback.comment,
        created_at=datetime.datetime.utcnow(),
    )
    db.add(fb)
    db.commit()
    return True


def clear_session_filter(
    user: User,
    session_id: str,
    filter_key: str,
    db: Session,
) -> Optional[MovieIntent]:
    """Explicitly clear a single filter (or 'all') from an active session's intent."""
    sess = (
        db.query(AssistantSession)
        .filter(
            AssistantSession.id == session_id,
            AssistantSession.user_id == user.id,
        )
        .first()
    )
    if not sess:
        return None

    delta = IntentDelta(clear_fields=[filter_key])
    updated_intent = _merge_intents(sess.active_intent or {}, delta)
    sess.active_intent = updated_intent.model_dump()
    sess.updated_at = datetime.datetime.utcnow()
    db.commit()
    logger.info("[AUDIT:FILTER_CLEARED] Cleared filter '%s' on session %s", filter_key, session_id)
    return updated_intent

