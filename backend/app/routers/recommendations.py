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
    """Return the requester for self/guest aliases or a matching numeric ID.

    Unauthenticated, malformed, and mismatched numeric IDs resolve to ``None``.
    """
    # Never allow an authenticated user to fetch another user's personalized
    # recommendations by guessing their numeric id (IDOR protection).
    if user_id in ("me", "guest"):
        return current_user
    # Only resolve a numeric user_id when it belongs to the requester.
    if current_user is not None:
        try:
            if int(user_id) == current_user.id:
                return current_user
        except ValueError:
            pass
    return None

@router.get("/recommendations/{user_id}", response_model=List[ScoredMovieSchema])
def get_recommendations_for_user(
    user_id: str,
    candidate_k: Optional[int] = Query(None, ge=10, le=1000, description="Omit to use the validated default for this user path"),
    top_k: int = Query(10, ge=1, le=50),
    variant: str = Query("auto", pattern="^(auto|normalized_fixed|damr|mmr|static)$", description="auto uses the validation-selected ranking policy"),
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
    model_service.load_all()
    path = model_service.ml_dir + "/metrics.json"
    if not path:
        raise HTTPException(status_code=404, detail="Metrics file not configured")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "sampled_test" in data:
            from datetime import datetime, timezone
            from pathlib import Path
            names = {"baseline": "NCF Baseline", "hybrid": "NCF Hybrid", "transformer": "Transformer"}
            return {
                "protocol": data["protocol"], "k": 10, "seed": 42, "num_negatives": 99,
                "num_users_evaluated": next(iter(data["sampled_test"].values()))["users"],
                "generated_at": datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat(),
                "dataset": {"num_users_total": model_service.config["num_users"], "num_items": model_service.config["num_items"], "num_genres": model_service.config["num_genres"]},
                "results": {names[name]: {"HR@10": r["hr10"], "NDCG@10": r["ndcg10"], "MRR": r["mrr"]} for name, r in data["sampled_test"].items()},
                "full_catalogue": data["full_catalogue_test"], "limitations": data["limitations"],
                "validation_selected_retriever": data["preferred_retriever_from_validation"],
            }
        data["limitations"] = ["Historical diagnostic metrics: checkpoints and evaluation inference conventions were not independently verified. Use fresh training results for current models."]
        return data
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
