import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, cast, String

from app.config import settings
from app.ml.model_service import model_service
from app.models.db_models import (
    User,
    Movie,
    Interaction,
    WatchHistory,
    MyList,
    Like,
    UserGenrePreference,
    UserPreference,
)
from app.schemas.schemas import ScoredMovieSchema, MovieSchema
from app.ml.tmdb import ensure_movie_posters

def movie_model_to_schema(movie: Movie) -> MovieSchema:
    return MovieSchema(
        movieId=movie.movie_id,
        title=movie.title,
        genres=movie.genres if isinstance(movie.genres, list) else [],
        year=movie.year or 0,
        rating=float(movie.rating or 0.0),
        runtime=movie.runtime or 0,
        posterUrl=movie.poster_url or "",
        backdropUrl=movie.backdrop_url or "",
        description=movie.description or "",
    )

def build_user_genre_vector(user: User, db: Session) -> torch.Tensor:
    """Builds dynamic genre preference vector from static profile + short-term interactions"""
    num_genres = model_service.config.get("num_genres", 20)
    genre_vec = torch.zeros(num_genres, dtype=torch.float32, device=model_service.device)

    # Base profile from model if mapped
    if user.model_user_id is not None and model_service.user_genre_matrix is not None:
        user_idx = model_service.user2idx.get(user.model_user_id)
        if user_idx is not None and user_idx < len(model_service.user_genre_matrix):
            genre_vec = model_service.user_genre_matrix[user_idx].clone().to(model_service.device)

    # Add short-term real-time genre preferences from PostgreSQL
    genre_prefs = db.query(UserGenrePreference).filter(UserGenrePreference.user_id == user.id).all()
    for gp in genre_prefs:
        g_idx = model_service.genre2idx.get(gp.genre)
        if g_idx is not None and g_idx < num_genres:
            genre_vec[g_idx] += float(gp.score)

    # If vector is zero, populate with user favorite genres if any
    if genre_vec.sum() == 0:
        pref = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        if pref and pref.favorite_genres:
            for g in pref.favorite_genres:
                g_idx = model_service.genre2idx.get(g)
                if g_idx is not None and g_idx < num_genres:
                    genre_vec[g_idx] += 1.0

    # Normalize if not all zeros
    if genre_vec.sum() > 0:
        genre_vec = genre_vec / (torch.norm(genre_vec, p=2) + 1e-6)
    
    return genre_vec


def get_user_interacted_movie_ids(user: User, db: Session) -> Set[int]:
    """Collect all movieIds the user has watched, liked, or saved"""
    interacted: Set[int] = set()
    
    # 1. From database interactions
    plays = db.query(Interaction.movie_id).filter(Interaction.user_id == user.id).all()
    interacted.update(p[0] for p in plays)

    # 2. From watch history
    history = db.query(WatchHistory.movie_id).filter(WatchHistory.user_id == user.id).all()
    interacted.update(h[0] for h in history)

    # 3. From likes & my_list
    likes = db.query(Like.movie_id).filter(Like.user_id == user.id).all()
    interacted.update(l[0] for l in likes)
    
    my_list = db.query(MyList.movie_id).filter(MyList.user_id == user.id).all()
    interacted.update(m[0] for m in my_list)

    # 4. From training set history if user has model_user_id
    if user.model_user_id is not None and user.model_user_id in model_service.user_interacted:
        interacted.update(model_service.user_interacted[user.model_user_id])

    return interacted


def get_user_recent_sequence(user: User, db: Session, max_len: int = 20) -> List[int]:
    """Retrieves user's recent sequence of movie item indices (1-indexed for transformer)"""
    recent_interactions = (
        db.query(Interaction.movie_id)
        .filter(Interaction.user_id == user.id)
        .order_by(Interaction.timestamp.desc())
        .limit(max_len)
        .all()
    )
    
    item_indices = []
    for (m_id,) in reversed(recent_interactions):
        idx = model_service.movie2idx.get(m_id)
        if idx is not None:
            item_indices.append(idx + 1) # 1-indexed (0 is padding)

    # Fallback to training sequence if available and db sequence is empty
    if not item_indices and user.model_user_id is not None:
        raw_seq = model_service.user_sequences.get(user.model_user_id, [])
        for m_id in raw_seq[-max_len:]:
            idx = model_service.movie2idx.get(m_id)
            if idx is not None:
                item_indices.append(idx + 1)

    return item_indices


@torch.no_grad()
def get_personalized_recommendations(
    user: Optional[User],
    db: Session,
    candidate_k: int = 100,
    top_k: int = 10,
) -> Tuple[str, List[ScoredMovieSchema]]:
    """
    Full 3-Stage Personalized Recommendation Flow:
    1. Hybrid NCF Candidate Generation -> Top 100
    2. Sequential Transformer Scoring
    3. Fresh Genre Preference Re-ranking
    -> Final Top 10
    """
    model_service.load_all()

    # If anonymous user, fall back to popular
    if not user:
        recs = get_popular_movies(db, limit=top_k)
        return "popular", recs

    # Check interaction count threshold: new user uses cold start
    interaction_count = db.query(Interaction).filter(Interaction.user_id == user.id).count()
    if (user.model_user_id is None) and (interaction_count < settings.MIN_INTERACTIONS_FOR_PERSONALIZATION):
        pref = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        fav_genres = pref.favorite_genres if pref else []
        fav_movies = pref.favorite_movie_ids if pref else []
        return "cold_start", get_cold_start_recommendations(fav_genres, fav_movies, db, top_k=top_k)

    # Get model user index
    model_user_id = user.model_user_id if user.model_user_id is not None else 1
    user_idx = model_service.user2idx.get(model_user_id, 0)

    # Interacted movies to exclude
    interacted_movie_ids = get_user_interacted_movie_ids(user, db)

    # Build candidate pool of movie model indices
    num_items = model_service.config.get("num_items", 22836)
    all_item_indices = []
    for idx in range(num_items):
        m_id = model_service.idx2movie.get(idx)
        if m_id is not None and m_id not in interacted_movie_ids:
            all_item_indices.append(idx)

    if not all_item_indices:
        return "popular", get_popular_movies(db, limit=top_k)

    candidate_item_tensor = torch.tensor(all_item_indices, dtype=torch.long, device=model_service.device)

    # Dynamic user genre profile
    user_genre_vector = build_user_genre_vector(user, db)

    # ==========================================
    # STAGE 1: Hybrid NCF Candidate Generation (Top 100)
    # ==========================================
    if model_service.ncf_hybrid is not None and model_service.movie_genre_matrix is not None:
        ncf_scores = model_service.ncf_hybrid.score_candidate_items(
            user_idx_int=user_idx,
            candidate_item_indices=candidate_item_tensor,
            user_genre_vector=user_genre_vector,
            movie_genre_matrix=model_service.movie_genre_matrix,
            device=model_service.device,
        )
    elif model_service.ncf_baseline is not None:
        user_tensor = torch.full((len(candidate_item_tensor),), user_idx, dtype=torch.long, device=model_service.device)
        ncf_scores = model_service.ncf_baseline(user_tensor, candidate_item_tensor)
    else:
        ncf_scores = torch.rand(len(candidate_item_tensor), device=model_service.device)

    # Select Top Candidate_K (default 100)
    actual_candidate_k = min(candidate_k, len(candidate_item_tensor))
    top_cand_values, top_cand_indices_in_pool = torch.topk(ncf_scores, k=actual_candidate_k)
    
    top_100_item_indices = candidate_item_tensor[top_cand_indices_in_pool]
    top_100_ncf_scores = top_cand_values

    # ==========================================
    # STAGE 2: Sequential Transformer Scoring
    # ==========================================
    user_seq = get_user_recent_sequence(user, db, max_len=model_service.config.get("transformer_max_len", 20))
    if model_service.sequential_transformer is not None:
        transformer_scores = model_service.sequential_transformer.score_candidates_with_sequence(
            sequence_item_indices=user_seq,
            candidate_item_indices=top_100_item_indices,
            device=model_service.device,
        )
    else:
        transformer_scores = torch.zeros(actual_candidate_k, device=model_service.device)

    # ==========================================
    # STAGE 3: Fresh Genre Re-ranking Signal
    # ==========================================
    if model_service.movie_genre_matrix is not None:
        cand_genre_matrix = model_service.movie_genre_matrix[top_100_item_indices].to(model_service.device)
        genre_scores = torch.mv(cand_genre_matrix, user_genre_vector)
        # Normalize genre scores
        if genre_scores.max() > 0:
            genre_scores = genre_scores / genre_scores.max()
    else:
        genre_scores = torch.zeros(actual_candidate_k, device=model_service.device)

    # ==========================================
    # ENSEMBLE RE-RANKING
    # ==========================================
    w_ncf = settings.NCF_WEIGHT
    w_trans = settings.TRANSFORMER_WEIGHT
    w_genre = settings.GENRE_WEIGHT

    final_scores = (
        w_ncf * top_100_ncf_scores +
        w_trans * transformer_scores +
        w_genre * genre_scores
    )

    # Take Top_K (default 10)
    actual_top_k = min(top_k, actual_candidate_k)
    final_top_values, final_top_indices = torch.topk(final_scores, k=actual_top_k)

    selected_item_indices = top_100_item_indices[final_top_indices].cpu().tolist()
    selected_movie_ids = [model_service.idx2movie[idx] for idx in selected_item_indices if idx in model_service.idx2movie]

    # Hydrate movie details from PostgreSQL database
    db_movies = db.query(Movie).filter(Movie.movie_id.in_(selected_movie_ids)).all()
    ensure_movie_posters(db_movies, db)
    movie_dict = {m.movie_id: m for m in db_movies}

    results: List[ScoredMovieSchema] = []
    for i, m_id in enumerate(selected_movie_ids):
        if m_id in movie_dict:
            m = movie_dict[m_id]
            f_score = float(final_top_values[i].item())
            n_score = float(top_100_ncf_scores[final_top_indices[i]].item())
            t_score = float(transformer_scores[final_top_indices[i]].item())
            g_score = float(genre_scores[final_top_indices[i]].item())
            
            results.append(
                ScoredMovieSchema(
                    movieId=m.movie_id,
                    title=m.title,
                    genres=m.genres if isinstance(m.genres, list) else [],
                    year=m.year or 0,
                    rating=float(m.rating or 0.0),
                    runtime=m.runtime or 0,
                    posterUrl=m.poster_url or "",
                    backdropUrl=m.backdrop_url or "",
                    description=m.description or "",
                    score=round(f_score, 4),
                    ncfScore=round(n_score, 4),
                    transformerScore=round(t_score, 4),
                    genreScore=round(g_score, 4),
                )
            )

    return "personalized", results


@torch.no_grad()
def get_similar_movies(movie_id: int, db: Session, top_k: int = 14) -> List[ScoredMovieSchema]:
    """
    Dedicated Movie-to-Movie Recommendation:
    Calculates item representation embedding similarity + genre vector similarity.
    """
    model_service.load_all()
    
    target_idx = model_service.movie2idx.get(movie_id)
    if target_idx is None:
        target_movie = db.query(Movie).filter(Movie.movie_id == movie_id).first()
        if not target_movie:
            return []
        target_genres = set(target_movie.genres if isinstance(target_movie.genres, list) else [])
        all_movies = db.query(Movie).filter(Movie.movie_id != movie_id).order_by(desc(Movie.rating_count)).limit(100).all()
        scored = []
        for m in all_movies:
            m_genres = set(m.genres if isinstance(m.genres, list) else [])
            overlap = len(target_genres.intersection(m_genres))
            if overlap > 0:
                scored.append((overlap, m))
        scored.sort(key=lambda x: (x[0], x[1].rating or 0), reverse=True)
        return [
            ScoredMovieSchema(
                movieId=m.movie_id,
                title=m.title,
                genres=m.genres if isinstance(m.genres, list) else [],
                year=m.year or 0,
                rating=float(m.rating or 0.0),
                runtime=m.runtime or 0,
                posterUrl=m.poster_url or "",
                backdropUrl=m.backdrop_url or "",
                description=m.description or "",
                score=round(0.5 + (overlap * 0.1), 2),
            )
            for overlap, m in scored[:top_k]
        ]

    # Use learned embeddings from NCF / Transformer + Genre Matrix
    sim_scores = torch.zeros(len(model_service.movie2idx), device=model_service.device)

    # 1. Item embeddings from NCF Hybrid
    if model_service.ncf_hybrid is not None:
        item_embs = model_service.ncf_hybrid.item_embedding.weight # (22836, 8)
        target_emb = item_embs[target_idx].unsqueeze(0) # (1, 8)
        ncf_sim = F.cosine_similarity(item_embs, target_emb, dim=-1) # (22836,)
        sim_scores += 0.5 * ncf_sim

    # 2. Genre Matrix similarity
    if model_service.movie_genre_matrix is not None:
        target_genre = model_service.movie_genre_matrix[target_idx].unsqueeze(0).to(model_service.device)
        genre_sim = F.cosine_similarity(model_service.movie_genre_matrix.to(model_service.device), target_genre, dim=-1)
        sim_scores += 0.5 * genre_sim

    # Mask out target movie
    sim_scores[target_idx] = -1.0

    # Top K similar
    top_values, top_indices = torch.topk(sim_scores, k=top_k)
    top_indices_list = top_indices.cpu().tolist()
    
    similar_movie_ids = [model_service.idx2movie[idx] for idx in top_indices_list if idx in model_service.idx2movie]
    
    db_movies = db.query(Movie).filter(Movie.movie_id.in_(similar_movie_ids)).all()
    ensure_movie_posters(db_movies, db)
    movie_dict = {m.movie_id: m for m in db_movies}

    results = []
    for i, m_id in enumerate(similar_movie_ids):
        if m_id in movie_dict:
            m = movie_dict[m_id]
            s = float(top_values[i].item())
            norm_score = max(0.1, min(0.99, (s + 1.0) / 2.0))
            results.append(
                ScoredMovieSchema(
                    movieId=m.movie_id,
                    title=m.title,
                    genres=m.genres if isinstance(m.genres, list) else [],
                    year=m.year or 0,
                    rating=float(m.rating or 0.0),
                    runtime=m.runtime or 0,
                    posterUrl=m.poster_url or "",
                    backdropUrl=m.backdrop_url or "",
                    description=m.description or "",
                    score=round(norm_score, 2),
                )
            )
    return results


@torch.no_grad()
def get_cold_start_recommendations(
    favorite_genres: List[str],
    favorite_movie_ids: List[int],
    db: Session,
    top_k: int = 10,
) -> List[ScoredMovieSchema]:
    """
    Cold-Start Recommendation:
    Computes genre-aligned recommendation matching new user's preferences
    with similar MovieLens users and weighting candidate movies by genre overlap.
    """
    model_service.load_all()

    num_genres = model_service.config.get("num_genres", 20)
    target_vec = torch.zeros(num_genres, dtype=torch.float32, device=model_service.device)

    # Add favorite genre weights
    for g in favorite_genres:
        idx = model_service.genre2idx.get(g)
        if idx is not None and idx < num_genres:
            target_vec[idx] += 2.0

    # Add genres from favorite movies
    for m_id in favorite_movie_ids:
        m_idx = model_service.movie2idx.get(m_id)
        if m_idx is not None and model_service.movie_genre_matrix is not None:
            target_vec += model_service.movie_genre_matrix[m_idx].to(model_service.device)

    if target_vec.sum() > 0:
        target_vec = target_vec / (torch.norm(target_vec, p=2) + 1e-6)

    # 1. Find top similar users from the 41,547 user genre matrix
    if model_service.user_genre_matrix is not None and target_vec.sum() > 0:
        user_matrix = model_service.user_genre_matrix.to(model_service.device)
        user_sims = F.cosine_similarity(user_matrix, target_vec.unsqueeze(0), dim=-1) # (41547,)
        
        # Find Top 100 similar users
        top_user_values, top_user_indices = torch.topk(user_sims, k=min(100, len(user_sims)))
        top_user_indices_list = top_user_indices.cpu().tolist()

        # Collect candidate movies from these similar users and weight by GENRE ALIGNMENT
        movie_scores: Dict[int, float] = {}
        fav_set = set(favorite_movie_ids)

        for u_idx in top_user_indices_list:
            u_id = model_service.idx2user.get(u_idx)
            if u_id and u_id in model_service.user_interacted:
                u_sim = float(user_sims[u_idx].item())
                for m_id in model_service.user_interacted[u_id]:
                    if m_id not in fav_set:
                        m_idx = model_service.movie2idx.get(m_id)
                        if m_idx is not None and model_service.movie_genre_matrix is not None:
                            m_genre_vec = model_service.movie_genre_matrix[m_idx].to(model_service.device)
                            genre_match = float(torch.dot(m_genre_vec, target_vec).item())
                            if genre_match > 0:
                                movie_scores[m_id] = movie_scores.get(m_id, 0.0) + (u_sim * (1.0 + 4.0 * genre_match))

        if movie_scores:
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            movie_ids = [m_id for m_id, _ in sorted_movies]
            
            db_movies = db.query(Movie).filter(Movie.movie_id.in_(movie_ids)).all()
            ensure_movie_posters(db_movies, db)
            movie_dict = {m.movie_id: m for m in db_movies}
            
            results = []
            max_score = sorted_movies[0][1] if sorted_movies else 1.0
            for m_id, raw_s in sorted_movies:
                if m_id in movie_dict:
                    m = movie_dict[m_id]
                    results.append(
                        ScoredMovieSchema(
                            movieId=m.movie_id,
                            title=m.title,
                            genres=m.genres if isinstance(m.genres, list) else [],
                            year=m.year or 0,
                            rating=float(m.rating or 0.0),
                            runtime=m.runtime or 0,
                            posterUrl=m.poster_url or "",
                            backdropUrl=m.backdrop_url or "",
                            description=m.description or "",
                            score=round(min(0.98, 0.70 + (raw_s / (max_score + 1e-6)) * 0.28), 2),
                        )
                    )
            if results:
                return results

    # Fallback to genre-filtered movies
    if favorite_genres:
        first_genre = favorite_genres[0]
        return get_movies_by_genre(first_genre, db, limit=top_k)

    return get_popular_movies(db, limit=top_k)


def get_popular_movies(db: Session, limit: int = 20) -> List[ScoredMovieSchema]:
    """Top rated movies with significant rating count (> 1000)"""
    movies = (
        db.query(Movie)
        .filter(Movie.rating_count >= 1000)
        .order_by(desc(Movie.rating), desc(Movie.rating_count))
        .limit(limit)
        .all()
    )
    ensure_movie_posters(movies, db)
    return [
        ScoredMovieSchema(
            movieId=m.movie_id,
            title=m.title,
            genres=m.genres if isinstance(m.genres, list) else [],
            year=m.year or 0,
            rating=float(m.rating or 0.0),
            runtime=m.runtime or 0,
            posterUrl=m.poster_url or "",
            backdropUrl=m.backdrop_url or "",
            description=m.description or "",
            score=round(min(0.99, float(m.rating or 3.5) / 5.0), 2),
        )
        for m in movies
    ]


def get_trending_movies(db: Session, limit: int = 20) -> List[ScoredMovieSchema]:
    """Trending movies based on interactions / popularity"""
    trending_db = (
        db.query(Movie)
        .order_by(desc(Movie.rating_count))
        .limit(limit)
        .all()
    )
    ensure_movie_posters(trending_db, db)
    return [
        ScoredMovieSchema(
            movieId=m.movie_id,
            title=m.title,
            genres=m.genres if isinstance(m.genres, list) else [],
            year=m.year or 0,
            rating=float(m.rating or 0.0),
            runtime=m.runtime or 0,
            posterUrl=m.poster_url or "",
            backdropUrl=m.backdrop_url or "",
            description=m.description or "",
            score=round(min(0.99, 0.75 + (i * -0.01)), 2),
        )
        for i, m in enumerate(trending_db)
    ]


def get_movies_by_genre(genre: str, db: Session, limit: int = 20) -> List[ScoredMovieSchema]:
    """Retrieves top movies for a given genre rail"""
    movies = (
        db.query(Movie)
        .filter(cast(Movie.genres, String).ilike(f"%{genre}%"))
        .order_by(desc(Movie.rating_count))
        .limit(limit)
        .all()
    )
    ensure_movie_posters(movies, db)

    return [
        ScoredMovieSchema(
            movieId=m.movie_id,
            title=m.title,
            genres=m.genres if isinstance(m.genres, list) else [],
            year=m.year or 0,
            rating=float(m.rating or 0.0),
            runtime=m.runtime or 0,
            posterUrl=m.poster_url or "",
            backdropUrl=m.backdrop_url or "",
            description=m.description or "",
            score=round(min(0.99, float(m.rating or 3.5) / 5.0), 2),
        )
        for m in movies
    ]
