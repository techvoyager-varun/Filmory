"""Rebuild live genre signals from current preferences, plays and current likes."""
from collections import Counter
from sqlalchemy.orm import Session
from app.models.db_models import Movie, Interaction, Like, UserPreference, UserGenrePreference
from app.services.recommendation_feedback import hidden_movie_ids

def refresh_genre_signals(user_id: int, db: Session) -> None:
    db.flush()
    scores = Counter()
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if pref:
        scores.update({g:2.0 for g in (pref.favorite_genres or [])})
    excluded = hidden_movie_ids(user_id, db)
    played_ids = db.query(Interaction.movie_id).filter(Interaction.user_id == user_id, Interaction.type == 'play').distinct()
    for genres, in db.query(Movie.genres).filter(Movie.movie_id.in_(played_ids), ~Movie.movie_id.in_(excluded)):
        if isinstance(genres, list): scores.update(genres)
    for genres, in db.query(Movie.genres).join(Like, Like.movie_id == Movie.movie_id).filter(Like.user_id == user_id, ~Movie.movie_id.in_(excluded)):
        if isinstance(genres, list):
            for genre in genres: scores[genre] += 1.5
    db.query(UserGenrePreference).filter(UserGenrePreference.user_id == user_id).delete(synchronize_session="fetch")
    db.add_all([UserGenrePreference(user_id=user_id, genre=g, score=float(value)) for g,value in scores.items()])
