from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import User
from app.schemas.schemas import (
    ScoredMovieSchema,
    RecommendationsResponse,
    ColdStartRequest,
)
from app.core.deps import get_optional_current_user
from app.ml.recommender import (
    get_personalized_recommendations,
    get_similar_movies,
    get_cold_start_recommendations,
    get_popular_movies,
    get_trending_movies,
    get_movies_by_genre,
)

router = APIRouter(prefix="/api", tags=["Recommendations"])

@router.get("/recommendations/{user_id}", response_model=List[ScoredMovieSchema])
def get_recommendations_for_user(
    user_id: str,
    candidate_k: int = Query(100, ge=10, le=500),
    top_k: int = Query(10, ge=1, le=50),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    target_user = current_user
    if user_id != "me" and user_id != "guest" and not target_user:
        try:
            target_user = db.query(User).filter(User.id == int(user_id)).first()
        except ValueError:
            target_user = None

    rec_type, recommendations = get_personalized_recommendations(
        user=target_user,
        db=db,
        candidate_k=candidate_k,
        top_k=top_k,
    )
    return recommendations

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
