from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, or_, cast, String
from app.database import get_db
from app.models.db_models import Movie
from app.schemas.schemas import MovieSchema, MoviePageSchema
from app.ml.recommender import movie_model_to_schema
from app.ml.tmdb import ensure_movie_posters

router = APIRouter(prefix="/api", tags=["Movies"])

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"
]

@router.get("/genres", response_model=List[str])
def get_genres():
    return GENRE_LIST

@router.get("/movies", response_model=MoviePageSchema)
def get_movies(
    offset: int = Query(0, ge=0),
    limit: int = Query(48, ge=1, le=5000),
    genre: str = Query("All"),
    sort: str = Query("popular"),
    query: str = Query(""),
    db: Session = Depends(get_db),
):
    q = db.query(Movie)

    if genre and genre != "All":
        if genre not in GENRE_LIST:
            raise HTTPException(status_code=400, detail="Unknown movie genre")
        q = q.filter(cast(Movie.genres, String).contains('"' + genre + '"', autoescape=True))

    if query.strip():
        q = q.filter(func.lower(Movie.title).contains(query.strip().lower(), autoescape=True))

    total = q.count()

    # Sorting
    if sort == "rating":
        q = q.order_by(desc(Movie.rating).nullslast(), desc(Movie.rating_count).nullslast())
    elif sort == "year":
        q = q.order_by(desc(Movie.year).nullslast(), desc(Movie.rating_count).nullslast())
    elif sort == "title":
        q = q.order_by(asc(Movie.title))
    else:  # default 'popular'
        q = q.order_by(desc(Movie.rating_count).nullslast(), desc(Movie.rating).nullslast())

    movies = q.order_by(Movie.movie_id).offset(offset).limit(limit).all()
    ensure_movie_posters(movies, db)
    return MoviePageSchema(
        movies=[movie_model_to_schema(m) for m in movies],
        total=total,
    )

@router.get("/movies/{movie_id}", response_model=MovieSchema)
def get_movie_by_id(movie_id: int, db: Session = Depends(get_db)):
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} not found in catalog",
        )
    ensure_movie_posters([movie], db)
    return movie_model_to_schema(movie)

@router.get("/search", response_model=List[MovieSchema])
def search_movies(
    query: str = Query("", min_length=1),
    limit: int = Query(60, ge=1, le=100),
    db: Session = Depends(get_db),
):
    q_str = query.strip().lower()
    if not q_str:
        return []

    results = (
        db.query(Movie)
        .filter(
            or_(
                func.lower(Movie.title).contains(q_str, autoescape=True),
                func.lower(cast(Movie.genres, String)).contains(q_str, autoescape=True),
            )
        )
        .order_by(
            desc(func.lower(Movie.title) == q_str),
            desc(func.lower(Movie.title).startswith(q_str, autoescape=True)),
            desc(Movie.rating_count).nullslast(),
            Movie.movie_id,
        )
        .limit(limit)
        .all()
    )
    ensure_movie_posters(results, db)
    return [movie_model_to_schema(m) for m in results]
