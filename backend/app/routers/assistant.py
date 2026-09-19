"""
API router for the Ask Filmory conversational assistant.

All endpoints require JWT authentication — the user is derived from the
token, never from the request body.
"""
import threading
import time
from collections import defaultdict, deque
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user
from app.database import get_db
from app.models.db_models import User
from app.schemas.assistant import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    MovieIntent,
    SessionDetail,
    SessionSummary,
)
from app.services.assistant_orchestrator import (
    clear_session_filter,
    delete_session,
    get_session_detail,
    get_user_sessions,
    handle_message,
    submit_feedback,
)

router = APIRouter(prefix="/api/assistant", tags=["Assistant"])

# ---------------------------------------------------------------------------
# Simple in-memory per-user rate limiter (sliding window, 1 minute).
# Enforces settings.ASSISTANT_RATE_LIMIT on the /chat endpoint to prevent
# abuse of the LLM API (cost/quota protection).
# ---------------------------------------------------------------------------
_rate_lock = threading.Lock()
_chat_timestamps: dict[int, deque] = defaultdict(deque)


def _enforce_chat_rate_limit(user_id: int) -> None:
    limit = settings.ASSISTANT_RATE_LIMIT
    if limit <= 0:
        return
    now = time.monotonic()
    window = 60.0
    with _rate_lock:
        timestamps = _chat_timestamps[user_id]
        while timestamps and now - timestamps[0] > window:
            timestamps.popleft()
        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait a moment before sending another message.",
            )
        timestamps.append(now)


@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_chat_rate_limit(current_user.id)
    """
    Main chat endpoint. Send a natural-language movie request and receive
    personalized recommendations with evidence-grounded explanations.

    Pass session_id to continue a conversation (follow-up requests).
    Omit session_id to start a new conversation.
    """
    try:
        return handle_message(
            user=current_user,
            message=payload.message,
            session_id=payload.session_id,
            db=db,
        )
    except RuntimeError as e:
        # Catches missing API key
        if "GEMINI_API_KEY" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI assistant is not configured. Please set GEMINI_API_KEY.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request.",
        )
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger("filmory.assistant").error("Chat endpoint error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request.",
        )


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversation sessions for the authenticated user."""
    return get_user_sessions(current_user, db)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Load a session with full message history. Ownership is verified."""
    detail = get_session_detail(current_user, session_id, db)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return detail


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a conversation session. Ownership is verified."""
    if not delete_session(current_user, session_id, db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def post_feedback(
    payload: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit feedback on a recommendation or explanation."""
    if not submit_feedback(current_user, payload, db):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return {"status": "ok"}


@router.delete("/sessions/{session_id}/filters/{filter_key}", response_model=MovieIntent)
def remove_session_filter(
    session_id: str,
    filter_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Explicitly clear a filter chip from an active session (e.g. 'max_runtime_minutes' or 'all')."""
    updated = clear_session_filter(current_user, session_id, filter_key, db)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )
    return updated

