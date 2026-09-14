import datetime
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    Text,
    Boolean,
    JSON,
)
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    model_user_id = Column(Integer, nullable=True, index=True) # Explicit mapping to MovieLens model user index
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    watch_history = relationship("WatchHistory", back_populates="user", cascade="all, delete-orphan")
    my_list = relationship("MyList", back_populates="user", cascade="all, delete-orphan")
    likes = relationship("Like", back_populates="user", cascade="all, delete-orphan")
    genre_preferences = relationship("UserGenrePreference", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", uselist=False, cascade="all, delete-orphan")


class Movie(Base):
    __tablename__ = "movies"

    movie_id = Column(BigInteger, primary_key=True, index=True) # Canonical MovieLens movieId
    title = Column(String(512), nullable=False, index=True)
    genres = Column(JSON, default=list, nullable=False) # List of genre strings
    year = Column(Integer, nullable=True, index=True)
    rating = Column(Float, default=0.0, index=True)
    rating_count = Column(Integer, default=0, index=True)
    runtime = Column(Integer, default=0)
    description = Column(Text, default="")
    poster_url = Column(Text, default="")
    backdrop_url = Column(Text, default="")

    interactions = relationship("Interaction", back_populates="movie")
    watch_history = relationship("WatchHistory", back_populates="movie")
    my_list_entries = relationship("MyList", back_populates="movie")
    likes = relationship("Like", back_populates="movie")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False, index=True)
    type = Column(String(50), nullable=False) # play, like, unlike, list_add, list_remove
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="interactions")
    movie = relationship("Movie", back_populates="interactions")

    __table_args__ = (
        Index("ix_interactions_user_movie_time", "user_id", "movie_id", "timestamp"),
    )


class WatchHistory(Base):
    __tablename__ = "watch_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False, index=True)
    watched_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False, index=True)
    progress = Column(Integer, default=100) # 0 - 100 percentage

    user = relationship("User", back_populates="watch_history")
    movie = relationship("Movie", back_populates="watch_history")

    __table_args__ = (
        Index("ix_watch_history_user_movie", "user_id", "movie_id"),
    )


class MyList(Base):
    __tablename__ = "my_list"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="my_list")
    movie = relationship("Movie", back_populates="my_list_entries")

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_my_list_user_movie"),
    )


class Like(Base):
    __tablename__ = "likes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    movie_id = Column(BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="likes")
    movie = relationship("Movie", back_populates="likes")

    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_likes_user_movie"),
    )


class UserGenrePreference(Base):
    __tablename__ = "user_genre_preferences"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    genre = Column(String(100), nullable=False)
    score = Column(Float, default=0.0, nullable=False)

    user = relationship("User", back_populates="genre_preferences")

    __table_args__ = (
        UniqueConstraint("user_id", "genre", name="uq_user_genre"),
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    favorite_genres = Column(JSON, default=list, nullable=False)
    favorite_movie_ids = Column(JSON, default=list, nullable=False)
    onboarding_completed = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="preferences")


# ---------------------------------------------------------------------------
# GenAI Assistant — conversational session + message storage
# ---------------------------------------------------------------------------
class AssistantSession(Base):
    __tablename__ = "assistant_sessions"

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    active_intent = Column(JSON, default=dict, nullable=False)  # Accumulated filters
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow, nullable=False)

    user = relationship("User")
    messages = relationship("AssistantMessageRecord", back_populates="session", cascade="all, delete-orphan",
                            order_by="AssistantMessageRecord.created_at")


class AssistantMessageRecord(Base):
    __tablename__ = "assistant_messages"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    intent_snapshot = Column(JSON, nullable=True)
    movie_ids = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    session = relationship("AssistantSession", back_populates="messages")


class AssistantFeedback(Base):
    __tablename__ = "assistant_feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("assistant_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(Integer, nullable=True)
    movie_id = Column(BigInteger, nullable=True)
    feedback_type = Column(String(50), nullable=False)  # helpful | not_helpful | wrong_movie | bad_explanation
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class MovieTextEmbedding(Base):
    __tablename__ = "movie_text_embeddings"

    movie_id = Column(BigInteger, ForeignKey("movies.movie_id", ondelete="CASCADE"), primary_key=True)
    embedding_model = Column(String(100), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    document_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    indexed_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    movie = relationship("Movie")


