import datetime
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.db_models import (
    User,
    Movie,
    Interaction,
    WatchHistory,
    MyList,
    Like,
    UserGenrePreference,
)
from app.schemas.schemas import (
    InteractionCreate,
    InteractionResponse,
    HistoryEntrySchema,
    MovieSchema,
)
from app.core.deps import get_current_user
from app.ml.recommender import movie_model_to_schema

router = APIRouter(prefix="/api", tags=["Interactions & User Data"])

@router.post("/interactions", response_model=InteractionResponse)
def record_user_interaction(
    payload: InteractionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    movie = db.query(Movie).filter(Movie.movie_id == payload.movieId).first()
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {payload.movieId} not found",
        )

    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    interaction = Interaction(
        user_id=current_user.id,
        movie_id=payload.movieId,
        type=payload.type,
        timestamp=now,
    )
    db.add(interaction)

    # 1. If play interaction, update watch history & short-term genre preference signal
    if payload.type == "play":
        history_entry = (
            db.query(WatchHistory)
            .filter(WatchHistory.user_id == current_user.id, WatchHistory.movie_id == payload.movieId)
            .first()
        )
        if history_entry:
            history_entry.watched_at = now
            history_entry.progress = min(95, 20 + ((payload.movieId % 8) * 10))
        else:
            db.add(
                WatchHistory(
                    user_id=current_user.id,
                    movie_id=payload.movieId,
                    watched_at=now,
                    progress=min(95, 20 + ((payload.movieId % 8) * 10)),
                )
            )

        # Update dynamic short-term genre preferences (+1.0 for each genre)
        if isinstance(movie.genres, list):
            for genre in movie.genres:
                gp = (
                    db.query(UserGenrePreference)
                    .filter(UserGenrePreference.user_id == current_user.id, UserGenrePreference.genre == genre)
                    .first()
                )
                if gp:
                    gp.score += 1.0
                else:
                    db.add(UserGenrePreference(user_id=current_user.id, genre=genre, score=1.0))

    elif payload.type == "like":
        existing_like = (
            db.query(Like)
            .filter(Like.user_id == current_user.id, Like.movie_id == payload.movieId)
            .first()
        )
        if not existing_like:
            db.add(Like(user_id=current_user.id, movie_id=payload.movieId))
            # Boost genres on like (+1.5)
            if isinstance(movie.genres, list):
                for genre in movie.genres:
                    gp = (
                        db.query(UserGenrePreference)
                        .filter(UserGenrePreference.user_id == current_user.id, UserGenrePreference.genre == genre)
                        .first()
                    )
                    if gp:
                        gp.score += 1.5
                    else:
                        db.add(UserGenrePreference(user_id=current_user.id, genre=genre, score=1.5))

    elif payload.type == "unlike":
        db.query(Like).filter(Like.user_id == current_user.id, Like.movie_id == payload.movieId).delete()

    elif payload.type == "list_add":
        existing_entry = (
            db.query(MyList)
            .filter(MyList.user_id == current_user.id, MyList.movie_id == payload.movieId)
            .first()
        )
        if not existing_entry:
            db.add(MyList(user_id=current_user.id, movie_id=payload.movieId))

    elif payload.type == "list_remove":
        db.query(MyList).filter(MyList.user_id == current_user.id, MyList.movie_id == payload.movieId).delete()

    db.commit()

    return InteractionResponse(
        userId=str(current_user.id),
        movieId=payload.movieId,
        type=payload.type,
        timestamp=now.isoformat(),
    )


@router.get("/users/me/history", response_model=List[HistoryEntrySchema])
def get_my_watch_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entries = (
        db.query(WatchHistory)
        .filter(WatchHistory.user_id == current_user.id)
        .order_by(desc(WatchHistory.watched_at))
        .limit(50)
        .all()
    )
    
    results = []
    for entry in entries:
        movie = entry.movie
        if movie:
            results.append(
                HistoryEntrySchema(
                    movie=movie_model_to_schema(movie),
                    watchedAt=entry.watched_at.isoformat(),
                    progress=entry.progress,
                )
            )
    return results


@router.get("/users/me/my-list", response_model=List[MovieSchema])
def get_my_list(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entries = (
        db.query(MyList)
        .filter(MyList.user_id == current_user.id)
        .order_by(desc(MyList.created_at))
        .all()
    )
    return [movie_model_to_schema(entry.movie) for entry in entries if entry.movie]


@router.post("/users/me/my-list/{movie_id}", response_model=List[MovieSchema])
def add_to_my_list(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} not found",
        )

    existing = db.query(MyList).filter(MyList.user_id == current_user.id, MyList.movie_id == movie_id).first()
    if not existing:
        db.add(MyList(user_id=current_user.id, movie_id=movie_id))
        db.add(Interaction(user_id=current_user.id, movie_id=movie_id, type="list_add"))
        db.commit()

    return get_my_list(current_user=current_user, db=db)


@router.delete("/users/me/my-list/{movie_id}", response_model=List[MovieSchema])
def remove_from_my_list(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(MyList).filter(MyList.user_id == current_user.id, MyList.movie_id == movie_id).delete()
    db.add(Interaction(user_id=current_user.id, movie_id=movie_id, type="list_remove"))
    db.commit()
    return get_my_list(current_user=current_user, db=db)


@router.get("/users/me/likes", response_model=List[int])
def get_my_likes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    likes = db.query(Like.movie_id).filter(Like.user_id == current_user.id).all()
    return [l[0] for l in likes]


@router.post("/users/me/likes/{movie_id}", response_model=List[int])
def toggle_my_like(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
    if not movie:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movie with ID {movie_id} not found",
        )

    existing = db.query(Like).filter(Like.user_id == current_user.id, Like.movie_id == movie_id).first()
    if existing:
        db.delete(existing)
        db.add(Interaction(user_id=current_user.id, movie_id=movie_id, type="unlike"))
    else:
        db.add(Like(user_id=current_user.id, movie_id=movie_id))
        db.add(Interaction(user_id=current_user.id, movie_id=movie_id, type="like"))
        # Boost genre preference
        if isinstance(movie.genres, list):
            for genre in movie.genres:
                gp = (
                    db.query(UserGenrePreference)
                    .filter(UserGenrePreference.user_id == current_user.id, UserGenrePreference.genre == genre)
                    .first()
                )
                if gp:
                    gp.score += 1.5
                else:
                    db.add(UserGenrePreference(user_id=current_user.id, genre=genre, score=1.5))

    db.commit()
    return get_my_likes(current_user=current_user, db=db)


@router.delete("/users/me/likes/{movie_id}", response_model=List[int])
def unlike_movie(
    movie_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(Like).filter(Like.user_id == current_user.id, Like.movie_id == movie_id).delete()
    db.add(Interaction(user_id=current_user.id, movie_id=movie_id, type="unlike"))
    db.commit()
    return get_my_likes(current_user=current_user, db=db)
