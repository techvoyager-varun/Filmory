"""
Pydantic schemas for the Ask Filmory conversational assistant.
"""
from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.schemas.schemas import ScoredMovieSchema


# ---------------------------------------------------------------------------
# Intent — structured output from LLM intent extraction
# ---------------------------------------------------------------------------
class MovieIntent(BaseModel):
    """Validated structured preferences extracted from user's natural language."""

    request_mode: Literal["catalog_lookup", "personalized_recommendation", "similar_movies"] = Field(
        default="personalized_recommendation",
        description="Routing mode: 'catalog_lookup' for franchise/title lookup, 'similar_movies' for recommendations like a movie, or 'personalized_recommendation' for general discovery.",
    )
    semantic_query: str = Field(
        default="",
        description="Free-text description of what the user wants, e.g. 'mind-bending sci-fi'",
    )
    preferred_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user wants, e.g. ['Sci-Fi', 'Comedy']",
    )
    excluded_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user explicitly excluded, e.g. ['Horror']",
    )
    min_year: Optional[int] = Field(default=None, description="Earliest release year")
    max_year: Optional[int] = Field(default=None, description="Latest release year")
    max_runtime_minutes: Optional[int] = Field(default=None, description="Max runtime in minutes")
    min_rating: Optional[float] = Field(default=None, ge=0, le=10, description="Minimum rating (0-10)")
    reference_titles: list[str] = Field(
        default_factory=list,
        description="Movies the user referenced, e.g. ['Inception']",
    )
    excluded_movie_ids: list[int] = Field(
        default_factory=list,
        description="Movie IDs to exclude (e.g. already watched)",
    )
    mood: str = Field(default="", description="Desired mood: 'lighter', 'intense', 'comforting', etc.")
    needs_clarification: bool = Field(
        default=False,
        description="True if the request is too vague to produce useful results",
    )
    clarification_question: str = Field(
        default="",
        description="Question to ask the user if needs_clarification is True",
    )


# ---------------------------------------------------------------------------
# Intent Delta — for explicit follow-up updates (set, clear, ordinal exclusions)
# ---------------------------------------------------------------------------
class IntentDelta(BaseModel):
    """
    Structured delta extracted from user request for explicit updates.
    Allows distinguishing between omitting a field, clearing it, or setting it.
    """

    set_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit fields to set/update, e.g. {'max_runtime_minutes': 100}",
    )
    clear_fields: list[str] = Field(
        default_factory=list,
        description="Fields to remove/clear, e.g. ['excluded_genres', 'max_runtime_minutes']",
    )
    add_excluded_movie_ids: list[int] = Field(
        default_factory=list,
        description="Explicit movie IDs to exclude",
    )
    target_ordinal_exclusion: Optional[int] = Field(
        default=None,
        description="1-indexed position from previous response to exclude (e.g. 1 for 'first', 2 for 'second')",
    )
    request_mode: Optional[Literal["catalog_lookup", "personalized_recommendation", "similar_movies"]] = None
    semantic_query: str = ""
    preferred_genres: list[str] = Field(default_factory=list)
    excluded_genres: list[str] = Field(default_factory=list)
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    max_runtime_minutes: Optional[int] = None
    min_rating: Optional[float] = None
    reference_titles: list[str] = Field(default_factory=list)
    mood: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""


# ---------------------------------------------------------------------------
# Evidence — grounded explanation data for each recommended movie
# ---------------------------------------------------------------------------
class MovieEvidence(BaseModel):
    """Evidence package for one movie. The LLM generates explanations
    using ONLY these facts — no invented directors, themes, or claims."""

    movie_id: int
    title: str
    genres: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    runtime: Optional[int] = None
    rating: Optional[float] = None
    description_snippet: str = Field(
        default="", description="First 200 chars of synopsis for context"
    )
    matched_constraints: list[str] = Field(
        default_factory=list,
        description="Which user constraints this movie satisfies, e.g. ['Sci-Fi genre', 'Under 120 min']",
    )
    personalization_signals: list[str] = Field(
        default_factory=list,
        description="Why personalization favours this, e.g. ['Sci-Fi in recent watches']",
    )
    score: float = 0.0
    damr_variant: str = "damr"
    evidence_keys: list[str] = Field(
        default_factory=list,
        description="Valid verifiable evidence keys, e.g. ['movie:1:genres', 'movie:1:runtime']",
    )


# ---------------------------------------------------------------------------
# Auditable Structured Explanation
# ---------------------------------------------------------------------------
class ExplanationItem(BaseModel):
    """Auditable per-item explanation linked to verified evidence keys."""

    movie_id: int
    explanation: str
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence keys used, e.g. ['movie:123:genres', 'movie:123:runtime']",
    )


class StructuredExplanation(BaseModel):
    """Structured, verifiable response produced by LLM."""

    summary: str = Field(description="High-level conversational overview")
    items: list[ExplanationItem] = Field(
        default_factory=list, description="Per-movie explanations"
    )


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """User sends a message to the assistant."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID. None → create a new session.",
    )


class ChatResponse(BaseModel):
    """Complete assistant response with movies + evidence."""

    session_id: str
    message: str = Field(description="Natural-language assistant response")
    movies: list[ScoredMovieSchema] = Field(
        default_factory=list, description="Movie cards to render"
    )
    intent: MovieIntent = Field(description="Extracted / accumulated intent")
    evidence: list[MovieEvidence] = Field(
        default_factory=list, description="Evidence for transparency panel"
    )
    structured_explanation: Optional[StructuredExplanation] = Field(
        default=None, description="Auditable explanation items"
    )
    clarification: Optional[str] = Field(
        default=None,
        description="Follow-up question if the request was ambiguous",
    )
    notice: Optional[str] = Field(
        default=None,
        description="Informational banner or disclaimer (e.g. catalog completeness or AI fallback)",
    )
    request_mode: Optional[str] = Field(
        default="personalized_recommendation",
        description="Executed request mode: catalog_lookup, similar_movies, or personalized_recommendation",
    )


class SessionSummary(BaseModel):
    """Lightweight session info for the session list."""

    session_id: str
    preview: str = Field(description="First user message as preview")
    message_count: int
    created_at: str
    updated_at: str


class SessionDetail(BaseModel):
    """Full session with message history."""

    session_id: str
    messages: list[MessageSchema]
    active_intent: MovieIntent
    created_at: str
    updated_at: str


class MessageSchema(BaseModel):
    """A single message in the conversation."""

    role: str  # "user" | "assistant"
    content: str
    movies: list[ScoredMovieSchema] = Field(default_factory=list)
    intent: Optional[MovieIntent] = None
    evidence: list[MovieEvidence] = Field(default_factory=list)
    structured_explanation: Optional[StructuredExplanation] = None
    notice: Optional[str] = None
    request_mode: Optional[str] = None
    timestamp: str


class FeedbackRequest(BaseModel):
    """User feedback on a recommendation or explanation."""

    session_id: str
    message_id: Optional[int] = None
    movie_id: Optional[int] = None
    feedback_type: str = Field(
        ...,
        pattern="^(helpful|not_helpful|wrong_movie|bad_explanation|already_watched)$",
    )
    comment: Optional[str] = Field(default=None, max_length=1000)


# Fix forward reference for SessionDetail
SessionDetail.model_rebuild()
