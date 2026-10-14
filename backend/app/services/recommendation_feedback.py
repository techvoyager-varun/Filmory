"""Local, reversible movie feedback and current positive history.

Feedback is stored in the existing interaction log, so no migration or external
service is required. Timestamp plus ID makes the last decision deterministic.
"""
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.db_models import Interaction, Like


def hidden_movie_ids(user_id: int, db: Session) -> set[int]:
    decisions = {}
    rows = (db.query(Interaction.movie_id, Interaction.type)
            .filter(Interaction.user_id == user_id,
                    Interaction.type.in_(["not_interested", "show_again"]))
            .order_by(Interaction.timestamp.desc(), Interaction.id.desc()))
    for movie_id, kind in rows:
        decisions.setdefault(int(movie_id), kind)
    return {movie_id for movie_id, kind in decisions.items() if kind == "not_interested"}


def positive_history(user_id: int, db: Session) -> list[tuple]:
    """Unique movies, oldest to newest, using each movie's latest positive event.

Removed likes and hidden movies do not remain positive evidence. A play is
activity, not a rating or proof of enjoyment; explicit likes remain separate.
"""
    current_like = (db.query(Like.id)
                    .filter(Like.user_id == user_id,
                            Like.movie_id == Interaction.movie_id).exists())
    rows = (db.query(Interaction.movie_id, Interaction.timestamp)
            .filter(Interaction.user_id == user_id,
                    or_(Interaction.type == "play",
                        (Interaction.type == "like") & current_like))
            .order_by(Interaction.timestamp.desc(), Interaction.id.desc()))
    excluded = hidden_movie_ids(user_id, db)
    seen = set()
    history = []
    for movie_id, timestamp in rows:
        if movie_id not in excluded and movie_id not in seen:
            seen.add(movie_id)
            history.append((int(movie_id), timestamp))
    return list(reversed(history))
