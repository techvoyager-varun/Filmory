from typing import List, Optional
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import User
from app.schemas.schemas import (
    ScoredMovieSchema,
    RecommendationsResponse,
    ColdStartRequest,
)
from app.core.deps import get_optional_current_user
from app.ml.model_service import model_service
from app.ml.recommender import (
    get_personalized_recommendations,
    get_similar_movies,
    get_cold_start_recommendations,
    get_popular_movies,
    get_trending_movies,
    get_movies_by_genre,
    get_taste_state,
)

router = APIRouter(prefix="/api", tags=["Recommendations"])

def _resolve_target_user(user_id: str, current_user: Optional[User], db: Session) -> Optional[User]:
    target_user = current_user
    if user_id != "me" and user_id != "guest" and not target_user:
        try:
            target_user = db.query(User).filter(User.id == int(user_id)).first()
        except ValueError:
            target_user = None
    return target_user

@router.get("/recommendations/{user_id}", response_model=List[ScoredMovieSchema])
def get_recommendations_for_user(
    user_id: str,
    candidate_k: int = Query(100, ge=10, le=500),
    top_k: int = Query(10, ge=1, le=50),
    variant: str = Query("damr", pattern="^(damr|mmr|static)$", description="Stage-4 re-ranker variant"),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    target_user = _resolve_target_user(user_id, current_user, db)

    rec_type, recommendations = get_personalized_recommendations(
        user=target_user,
        db=db,
        candidate_k=candidate_k,
        top_k=top_k,
        variant=variant,
    )
    return recommendations

@router.get("/metrics", tags=["Model"])
def get_model_metrics():
    """
    Offline evaluation results (HR@10, NDCG@10, MRR, AUC, ILD, coverage ...)
    produced by `backend/scripts/evaluate.py` under the leave-one-out protocol.
    """
    path = model_service.ml_dir + "/metrics.json"
    if not path:
        raise HTTPException(status_code=404, detail="Metrics file not configured")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Run backend/scripts/evaluate.py first to generate ml/metrics.json",
        )

@router.get("/taste-state", tags=["Model"])
def get_user_taste_state(
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Live DAMR taste-state for the logged-in user: drift / focus / maturity /
    freshness scalars, the adaptive expert weights, and the long/short genre
    profiles + momentum vector (for the profile-page radar chart).
    """
    if current_user is None:
        raise HTTPException(status_code=401, detail="Login required for taste state")
    return get_taste_state(current_user, db)

@router.get("/similar/{movie_id}", response_model=List[ScoredMovieSchema])
def get_similar(
    movie_id: int,
    top_k: int = Query(14, ge=1, le=30),
    db: Session = Depends(get_db),
):
    return get_similar_movies(movie_id=movie_id, db=db, top_k=top_k)

@router.get("/trending", response_model=List[ScoredMovieSchema])
def get_trending(
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    return get_trending_movies(db=db, limit=limit)

@router.get("/popular", response_model=List[ScoredMovieSchema])
def get_popular(
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    return get_popular_movies(db=db, limit=limit)

@router.get("/movies-by-genre", response_model=List[ScoredMovieSchema])
def get_genre_rail(
    genre: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    return get_movies_by_genre(genre=genre, db=db, limit=limit)

@router.post("/cold-start", response_model=List[ScoredMovieSchema])
def cold_start_recommendations(
    payload: ColdStartRequest,
    top_k: int = Query(10, ge=1, le=30),
    db: Session = Depends(get_db),
):
    return get_cold_start_recommendations(
        favorite_genres=payload.favoriteGenres,
        favorite_movie_ids=payload.favoriteMovieIds,
        db=db,
        top_k=top_k,
    )
