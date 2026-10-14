import datetime as _dt
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc, cast, String, or_

from app.config import settings
from app.ml.model_service import model_service
from app.ml.damr import (
    TasteProfile,
    adaptive_weights,
    damr_rerank,
    estimate_user_state,
    estimate_user_state_from_positions,
    switches_for_variant,
    intra_list_diversity,
)
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
from app.ml.ranking_policy import select_candidate_positions, rank_candidate_positions, choose_history_policy, score_neighbor_items
from app.services.recommendation_feedback import hidden_movie_ids, positive_history

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
        norm = 1 if model_service.config.get("genre_profile_normalization") == "l1" else 2
        genre_vec = F.normalize(genre_vec, p=norm, dim=0)
    
    return genre_vec


def get_user_interacted_movie_ids(user: User, db: Session) -> Set[int]:
    """Collect all movieIds the user has watched, liked, or saved"""
    interacted: Set[int] = set()
    interacted.update(hidden_movie_ids(user.id, db))
    
    # 1. From database interactions
    plays = db.query(Interaction.movie_id).filter(Interaction.user_id == user.id, Interaction.type == "play").all()
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
    if user.model_user_id is not None:
        user_idx = model_service.user2idx.get(user.model_user_id)
        for item_idx in model_service.user_interacted.get(user_idx, []):
            movie_id = model_service.idx2movie.get(int(item_idx))
            if movie_id is not None:
                interacted.add(int(movie_id))

    return interacted


def get_user_recent_sequence(user: User, db: Session, max_len: int = 20) -> List[int]:
    """Return zero-based model item indices, oldest to newest."""
    recent_interactions = reversed(positive_history(user.id, db))
    
    item_indices = []
    for m_id, _ in recent_interactions:
        idx = model_service.movie2idx.get(m_id)
        if idx is not None:
            item_indices.append(int(idx))
            if len(item_indices) == max_len:
                break
    item_indices.reverse()

    # Fallback to training sequence if available and db sequence is empty
    if not item_indices and user.model_user_id is not None:
        user_idx = model_service.user2idx.get(user.model_user_id)
        hidden = hidden_movie_ids(user.id, db)
        raw_seq = [idx for idx in model_service.user_sequences.get(user_idx, [])
                   if model_service.idx2movie.get(int(idx)) not in hidden]
        item_indices.extend(int(idx) for idx in raw_seq[-max_len:])

    return item_indices


def build_taste_profile(user: User, db: Session) -> TasteProfile:
    """
    Build the DAMR user state (drift / focus / maturity / freshness) and the
    momentum vector from the user's timestamped interaction history.

    History source, in order of preference:
      1. `interactions` rows of type play/like (real, timestamped).
      2. Mapped training users: positional decay, without invented timestamps.
    """
    history: List[Tuple[int, _dt.datetime]] = []

    rows = positive_history(user.id, db)
    history = [
        (model_service.movie2idx[m_id], ts)
        for m_id, ts in rows
        if m_id in model_service.movie2idx
    ]

    if not history and user.model_user_id is not None:
        user_idx = model_service.user2idx.get(user.model_user_id)
        hidden = hidden_movie_ids(user.id, db)
        seq = [idx for idx in (model_service.user_sequences.get(user_idx, []) or [])
               if model_service.idx2movie.get(int(idx)) not in hidden]
        if seq:
            return estimate_user_state_from_positions([int(i) for i in seq])

    base_profile = None
    if user.model_user_id is not None and model_service.user_genre_matrix is not None:
        u_idx = model_service.user2idx.get(user.model_user_id)
        if u_idx is not None and u_idx < len(model_service.user_genre_matrix):
            base_profile = model_service.user_genre_matrix[u_idx]

    return estimate_user_state(history, base_profile=base_profile)


def get_taste_state(user: User, db: Session) -> Dict[str, Any]:
    """Full taste-state payload for the profile page / demo (radar chart data)."""
    profile = build_taste_profile(user, db)
    # Run the gate explicitly so the UI shows the weights DAMR would use right now
    # (state.weights is otherwise only filled during actual re-ranking).
    w = adaptive_weights(profile.state)
    profile.state.weights = (float(w[0]), float(w[1]), float(w[2]))
    return {
        "userId": str(user.id),
        "modelMapped": user.model_user_id is not None,
        "state": profile.state.to_dict(),
        "vectors": profile.vectors_to_lists(),
        "genres": [model_service.idx2genre.get(i, str(i)) for i in range(len(profile.g_long))],
        "weights": profile.state.to_dict()["weights"],
    }


@torch.no_grad()
def _get_sequence_recommendations(
    user: User, db: Session, sequence: List[int], top_k: int, candidate_k: Optional[int], policy: Dict[str, Any],
) -> List[ScoredMovieSchema]:
    """Recommend for a live account using item history, without a learned user ID."""
    sequence = sequence[-policy["min_history"]:]
    budget = max(top_k, candidate_k or policy.get("candidate_k", 300))
    weights = policy.get("weights", [0.0, 0.75, 0.25])
    strategy = policy.get("strategy", "transformer")
    excluded = get_user_interacted_movie_ids(user, db)
    # Restrict the scored universe before retrieval. Missing database titles and
    # already watched/saved/liked titles cannot occupy the candidate budget.
    catalogue_ids = db.query(Movie.movie_id).order_by(Movie.movie_id).all()
    eligible_ids = [int(row[0]) for row in catalogue_ids
                    if row[0] in model_service.movie2idx and row[0] not in excluded]
    if not eligible_ids:
        return []
    item_indices = torch.tensor([model_service.movie2idx[mid] for mid in eligible_ids],
                                dtype=torch.long, device=model_service.device)
    transformer = model_service.sequential_transformer.score_candidates_with_sequence(
        sequence, item_indices, model_service.device)
    genre_profile = model_service.movie_genre_matrix[sequence].sum(0).to(model_service.device)
    genre_profile = genre_profile / genre_profile.sum().clamp_min(1e-8)
    genre = (model_service.movie_genre_matrix[item_indices] @ genre_profile
             if model_service.movie_genre_matrix is not None else torch.zeros_like(transformer))
    collaborative = score_neighbor_items(sequence, genre_profile, model_service)[item_indices]
    pool = select_candidate_positions(transformer, collaborative, genre, budget, strategy)
    ranking, scores = rank_candidate_positions(pool, transformer, collaborative, genre, weights, top_k)
    movie_ids = [eligible_ids[position] for position in ranking.tolist()]
    movies = db.query(Movie).filter(Movie.movie_id.in_(movie_ids)).all()
    ensure_movie_posters(movies, db)
    by_id = {movie.movie_id:movie for movie in movies}
    return [ScoredMovieSchema(
        **movie_model_to_schema(by_id[mid]).model_dump(), score=round(float(score),4),
        transformerScore=round(float(transformer[position]),4),
        neighborScore=round(float(collaborative[position]),4),
        genreScore=round(float(genre[position]),4),
        expertWeights={"ncf":0.0,"neighbor":weights[0],"transformer":weights[1],"genre":weights[2]},
        variant="sequence_only", rank=rank+1,
    ) for rank,(mid,position,score) in enumerate(zip(movie_ids,ranking.tolist(),scores)) if mid in by_id]


@torch.no_grad()
def get_personalized_recommendations(
    user: Optional[User],
    db: Session,
    candidate_k: Optional[int] = None,
    top_k: int = 10,
    variant: str = "auto",
) -> Tuple[str, List[ScoredMovieSchema]]:
    """
    Full 4-Stage Personalized Recommendation Flow:
    1. Hybrid NCF Candidate Generation      -> Top 100 candidates + s_NCF
    2. Sequential Transformer Scoring       -> s_TR
    3. Fresh Genre Preference Signal        -> s_GEN
    4. DAMR Re-Ranker (Drift-Aware Momentum):
         drift-adaptive expert fusion + taste-momentum bonus
         + expert-agreement confidence + Bayesian quality prior + MMR diversity
       -> Final Top 10
    Set variant="mmr" for static fusion + quality + diversity, or
    variant="static" for the original fixed-weight ensemble (legacy).
    """
    model_service.load_all()

    # If anonymous user, fall back to popular
    if not user:
        recs = get_popular_movies(db, limit=top_k)
        return "popular", recs

    # A learned NCF user ID is not required by the sequence model.
    if user.model_user_id not in model_service.user2idx:
        sequence = get_user_recent_sequence(user, db, max_len=model_service.config.get("transformer_max_len", 20))
        policy = choose_history_policy(model_service.serving_policy, len(sequence))
        if (model_service.sequential_transformer is not None
            and model_service.movie_genre_matrix is not None
            and len(set(sequence)) >= settings.MIN_INTERACTIONS_FOR_PERSONALIZATION
            and policy["mode"] == "sequence_only"):
            return "sequence_personalized", _get_sequence_recommendations(user, db, sequence, top_k, candidate_k, policy)
        pref = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
        fav_genres = list(pref.favorite_genres or []) if pref else []
        dynamic = build_user_genre_vector(user, db)
        if dynamic.sum() > 0:
            recent_genres = [model_service.idx2genre[int(i)] for i in torch.topk(dynamic, k=min(3, len(dynamic))).indices if float(dynamic[i]) > 0]
            # Explicit preferences are never dropped by the top-three dynamic
            # signal cap. Add recent interests while retaining every chosen genre.
            fav_genres = list(dict.fromkeys(fav_genres + recent_genres))
        fav_movies = list(pref.favorite_movie_ids or []) if pref else []
        hidden = hidden_movie_ids(user.id, db)
        # Rejections are exclusions, never positive cold-start seed movies.
        positive_ids = {movie_id for movie_id, _ in positive_history(user.id, db)}
        positive_ids.update(row[0] for row in db.query(MyList.movie_id).filter(MyList.user_id == user.id))
        fav_movies = sorted((set(fav_movies) | positive_ids) - hidden)
        return "cold_start", get_cold_start_recommendations(
            fav_genres, fav_movies, db, top_k=top_k,
            exclude_ids=get_user_interacted_movie_ids(user, db),
        )

    # Get model user index
    user_idx = model_service.user2idx[user.model_user_id]
    candidate_k = max(top_k, candidate_k or model_service.config.get("candidate_k", settings.CANDIDATE_K))

    # Interacted movies to exclude
    interacted_movie_ids = get_user_interacted_movie_ids(user, db)

    # Build candidate pool of movie model indices
    num_items = model_service.config.get("num_items", 22836)
    catalogue_ids = {int(row[0]) for row in db.query(Movie.movie_id).all()}
    all_item_indices = []
    for idx in range(num_items):
        m_id = model_service.idx2movie.get(idx)
        if m_id in catalogue_ids and m_id not in interacted_movie_ids:
            all_item_indices.append(idx)

    if not all_item_indices:
        return "personalized", []

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
        if model_service.config.get("preferred_retriever") == "baseline" and model_service.ncf_baseline is not None:
            user_tensor = torch.full_like(candidate_item_tensor, user_idx)
            retrieval_scores = model_service.ncf_baseline(user_tensor, candidate_item_tensor)
        elif model_service.config.get("preferred_retriever") == "transformer" and model_service.sequential_transformer is not None:
            retrieval_scores = model_service.sequential_transformer.score_candidates_with_sequence(
                get_user_recent_sequence(user, db), candidate_item_tensor, model_service.device
            )
        else:
            retrieval_scores = ncf_scores
    elif model_service.ncf_baseline is not None:
        user_tensor = torch.full((len(candidate_item_tensor),), user_idx, dtype=torch.long, device=model_service.device)
        ncf_scores = model_service.ncf_baseline(user_tensor, candidate_item_tensor)
        retrieval_scores = ncf_scores
    else:
        return "popular", get_popular_movies(db, limit=top_k, exclude_ids=interacted_movie_ids)

    # Select Top Candidate_K (default 100)
    actual_candidate_k = min(candidate_k, len(candidate_item_tensor))
    _, top_cand_indices_in_pool = torch.topk(retrieval_scores, k=actual_candidate_k)
    
    top_100_item_indices = candidate_item_tensor[top_cand_indices_in_pool]
    top_100_ncf_scores = ncf_scores[top_cand_indices_in_pool]

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
    # FUSION: Static ensemble of the three expert scores (Stage 1-3)
    # ==========================================
    w_ncf = settings.NCF_WEIGHT
    w_trans = settings.TRANSFORMER_WEIGHT
    w_genre = settings.GENRE_WEIGHT

    final_scores = (
        w_ncf * top_100_ncf_scores +
        w_trans * transformer_scores +
        w_genre * genre_scores
    )

    # ==========================================
    # STAGE 4: DAMR — Drift-Aware Momentum Re-Ranker
    # ==========================================
    # variant = "damr" (proposed) | "mmr" (static + quality + diversity) |
    #           "static" (pure 0.55/0.25/0.20 ensemble, legacy behaviour)
    variant = (variant or "auto").lower()
    if variant == "auto":
        variant = model_service.config.get("serving_variant", settings.RERANK_VARIANT)
    actual_top_k = min(top_k, actual_candidate_k)

    if variant == "static":
        # Legacy path: raw ensemble scores -> straight top-k. No Stage 4.
        final_top_values, final_top_indices = torch.topk(final_scores, k=actual_top_k)
        selected_item_indices = top_100_item_indices[final_top_indices].cpu().tolist()
        # int(): idx2movie values are numpy.int64, which some DB drivers cannot bind
        selected_movie_ids = [int(model_service.idx2movie[idx]) for idx in selected_item_indices if idx in model_service.idx2movie]

        db_movies = db.query(Movie).filter(Movie.movie_id.in_(selected_movie_ids)).all()
        ensure_movie_posters(db_movies, db)
        movie_dict = {m.movie_id: m for m in db_movies}

        results: List[ScoredMovieSchema] = []
        for i, m_id in enumerate(selected_movie_ids):
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
                        score=round(float(final_top_values[i].item()), 4),
                        ncfScore=round(float(top_100_ncf_scores[final_top_indices[i]].item()), 4),
                        transformerScore=round(float(transformer_scores[final_top_indices[i]].item()), 4),
                        genreScore=round(float(genre_scores[final_top_indices[i]].item()), 4),
                        variant="static",
                    )
                )
        return "personalized", results

    # --- Build the Stage 4 candidate pool from the ensemble scores ---------
    pool_k = min(settings.RERANK_POOL_SIZE, actual_candidate_k)
    pool_values, pool_positions = torch.topk(final_scores, k=pool_k)
    pool_item_indices = top_100_item_indices[pool_positions]  # model item indices
    pool_ncf = top_100_ncf_scores[pool_positions]
    pool_tr = transformer_scores[pool_positions]
    pool_gen = genre_scores[pool_positions]

    # int(): idx2movie values are numpy.int64, which some DB drivers cannot bind
    pool_movie_ids = [int(model_service.idx2movie[int(i)]) for i in pool_item_indices.tolist() if int(i) in model_service.idx2movie]
    if not pool_movie_ids:
        return "popular", get_popular_movies(db, limit=top_k, exclude_ids=interacted_movie_ids)

    db_movies = db.query(Movie).filter(Movie.movie_id.in_(pool_movie_ids)).all()
    movie_by_id = {m.movie_id: m for m in db_movies}
    # NOTE: the movies table stores ratings on a 0-10 scale; the Bayesian prior
    # and the schema expose the original 0-5 MovieLens scale, so halve it here.
    pool_ratings = [
        float(movie_by_id[m].rating) / 2.0 if (m in movie_by_id and movie_by_id[m].rating is not None) else None
        for m in pool_movie_ids
    ]
    pool_counts = [
        int(movie_by_id[m].rating_count) if (m in movie_by_id and movie_by_id[m].rating_count is not None) else None
        for m in pool_movie_ids
    ]

    # --- Estimate the user's taste state + momentum (Step 1) ---------------
    profile = build_taste_profile(user, db)

    # --- DAMR re-ranking (Steps 2-5) ----------------------------------------
    ranked = damr_rerank(
        cand_idxs=pool_item_indices,
        s_ncf=pool_ncf,
        s_tr=pool_tr,
        s_gen=pool_gen,
        profile=profile,
        ratings=pool_ratings,
        counts=pool_counts,
        top_k=actual_top_k,
        **switches_for_variant(variant),
    )

    selected_movie_ids = [int(model_service.idx2movie[entry["modelIdx"]]) for entry in ranked if entry["modelIdx"] in model_service.idx2movie]
    ensure_movie_posters(db_movies, db)
    movie_dict = movie_by_id

    results = []
    for entry in ranked:
        m = movie_dict.get(model_service.idx2movie.get(entry["modelIdx"]))
        if m is None:
            continue
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
                score=entry["score"],
                ncfScore=entry["ncfScore"],
                transformerScore=entry["transformerScore"],
                genreScore=entry["genreScore"],
                momentumScore=entry["momentumScore"],
                agreementScore=entry["agreementScore"],
                qualityScore=entry["qualityScore"],
                diversityPenalty=entry["diversityPenalty"],
                expertWeights=entry["expertWeights"],
                userState=entry["userState"],
                variant=variant,
                rank=len(results) + 1,
                listDiversity=round(intra_list_diversity([e["modelIdx"] for e in ranked]), 4) if len(results) == 0 else None,
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
    sim_scores[target_idx] = float("-inf")
    top_values, top_indices = torch.topk(sim_scores, k=min(top_k, max(0, len(sim_scores) - 1)))
    top_indices_list = top_indices.cpu().tolist()
    
    similar_movie_ids = [int(model_service.idx2movie[idx]) for idx in top_indices_list if idx in model_service.idx2movie]
    
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
    exclude_ids: Optional[Set[int]] = None,
) -> List[ScoredMovieSchema]:
    """
    Cold-Start Recommendation:
    Computes genre-aligned recommendation matching new user's preferences
    with similar MovieLens users and weighting candidate movies by genre overlap.
    """
    model_service.load_all()
    excluded_ids = set(favorite_movie_ids) | (exclude_ids or set())

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
        fav_set = excluded_ids

        for u_idx in top_user_indices_list:
            if u_idx in model_service.user_interacted:
                u_sim = float(user_sims[u_idx].item())
                for item_idx in model_service.user_interacted[u_idx]:
                    m_id = model_service.idx2movie.get(int(item_idx))
                    if m_id is None:
                        continue
                    m_id = int(m_id)
                    if m_id not in fav_set:
                        m_idx = model_service.movie2idx.get(m_id)
                        if m_idx is not None and model_service.movie_genre_matrix is not None:
                            m_genre_vec = model_service.movie_genre_matrix[m_idx].to(model_service.device)
                            genre_match = float(torch.dot(m_genre_vec, target_vec).item())
                            if genre_match > 0:
                                movie_scores[m_id] = movie_scores.get(m_id, 0.0) + (u_sim * (1.0 + 4.0 * genre_match))

        if movie_scores:
            existing_ids = {int(row[0]) for row in db.query(Movie.movie_id).all()}
            sorted_movies = sorted(
                ((mid, score) for mid, score in movie_scores.items() if mid in existing_ids),
                key=lambda x: (-x[1], x[0]),
            )[:top_k]
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

    # Keep previously supplied movies excluded even when neighbour retrieval fails.
    fallback = db.query(Movie).filter(~Movie.movie_id.in_(excluded_ids))
    if favorite_genres:
        fallback = fallback.filter(or_(*[
            cast(Movie.genres, String).contains('"' + genre + '"', autoescape=True)
            for genre in favorite_genres
        ]))
    movies = fallback.order_by(
        desc(Movie.rating_count).nullslast(), desc(Movie.rating).nullslast(), Movie.movie_id
    ).limit(top_k).all()
    ensure_movie_posters(movies, db)
    return [ScoredMovieSchema(
        **movie_model_to_schema(movie).model_dump(),
        score=round(min(1.0, max(0.0, float(movie.rating or 0.0) / 10.0)), 2),
    ) for movie in movies]


def get_popular_movies(db: Session, limit: int = 20, exclude_ids: Optional[Set[int]] = None) -> List[ScoredMovieSchema]:
    """Top rated movies with significant rating count (> 1000)"""
    movies = (
        db.query(Movie)
        .filter(Movie.rating_count >= 1000)
        .filter(~Movie.movie_id.in_(exclude_ids or set()))
        .order_by(desc(Movie.rating).nullslast(), desc(Movie.rating_count).nullslast(), Movie.movie_id)
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
            score=round(min(1.0, max(0.0, float(m.rating or 0.0) / 10.0)), 2),
        )
        for m in movies
    ]


def get_trending_movies(db: Session, limit: int = 20) -> List[ScoredMovieSchema]:
    """Trending movies based on interactions / popularity"""
    trending_db = (
        db.query(Movie)
        .order_by(desc(Movie.rating_count).nullslast(), Movie.movie_id)
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
            score=round(min(1.0, max(0.0, float(m.rating or 0.0) / 10.0)), 2),
        )
        for i, m in enumerate(trending_db)
    ]


def get_movies_by_genre(genre: str, db: Session, limit: int = 20) -> List[ScoredMovieSchema]:
    """Retrieves top movies for a given genre rail"""
    movies = (
        db.query(Movie)
        .filter(cast(Movie.genres, String).contains('"' + genre + '"', autoescape=True))
        .order_by(desc(Movie.rating_count).nullslast(), Movie.movie_id)
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
            score=round(min(1.0, max(0.0, float(m.rating or 0.0) / 10.0)), 2),
        )
        for m in movies
    ]


@torch.no_grad()
def rerank_candidate_pool_with_damr(
    candidate_movies: List[Movie],
    user: Optional[User],
    db: Session,
    top_k: int = 10,
    variant: str = "damr",
) -> Tuple[List[ScoredMovieSchema], Dict[str, Any]]:
    """
    Pass an arbitrary candidate pool (e.g. from semantic or metadata search) through
    the true 4-Stage Personalization & DAMR Re-Ranking pipeline:
      1. NCF Hybrid scoring for candidate items (or Bayesian prior for unmapped)
      2. Sequential Transformer scoring based on user recent sequence
      3. Dynamic user genre-preference alignment scoring
      4. DAMR Re-Ranking (drift-adaptive expert fusion + momentum + agreement + Bayesian prior + MMR)

    Returns:
      (scored_movies, audit_metrics)
    """
    model_service.load_all()
    if user is not None:
        hidden = hidden_movie_ids(user.id, db)
        candidate_movies = [movie for movie in candidate_movies if movie.movie_id not in hidden]
    if not candidate_movies:
        return [], {
            "total_candidates": 0,
            "mapped_candidates": 0,
            "unmapped_candidates": 0,
            "variant": variant,
            "expert_weights": {},
            "user_state": {},
            "final_movie_ids": [],
        }

    variant = (variant or settings.RERANK_VARIANT).lower()
    total_cands = len(candidate_movies)

    # 1. Resolve model item indices for candidates
    mapped_count = 0
    item_indices = []
    for m in candidate_movies:
        midx = model_service.movie2idx.get(m.movie_id)
        if midx is not None:
            item_indices.append(int(midx))
            mapped_count += 1
        else:
            item_indices.append(0)

    cand_item_tensor = torch.tensor(item_indices, dtype=torch.long, device=model_service.device)
    unmapped_count = total_cands - mapped_count

    # 2. Stage 1: NCF Hybrid Scoring
    user_idx = model_service.user2idx.get(user.model_user_id) if user else None

    user_genre_vector = build_user_genre_vector(user, db) if user else torch.zeros(
        model_service.config.get("num_genres", 20), device=model_service.device
    )

    if user_idx is not None and model_service.ncf_hybrid is not None and model_service.movie_genre_matrix is not None:
        ncf_scores = model_service.ncf_hybrid.score_candidate_items(
            user_idx_int=user_idx,
            candidate_item_indices=cand_item_tensor,
            user_genre_vector=user_genre_vector,
            movie_genre_matrix=model_service.movie_genre_matrix,
            device=model_service.device,
        )
    elif user_idx is not None and model_service.ncf_baseline is not None:
        user_tensor = torch.full((total_cands,), user_idx, dtype=torch.long, device=model_service.device)
        ncf_scores = model_service.ncf_baseline(user_tensor, cand_item_tensor)
    else:
        ncf_scores = torch.tensor([float(m.rating or 0.0) / 10.0 for m in candidate_movies], device=model_service.device)

    # Unmapped items use normalized rating prior as NCF proxy
    for i, m in enumerate(candidate_movies):
        if m.movie_id not in model_service.movie2idx:
            ncf_scores[i] = float(m.rating or 3.5) / 10.0

    # 3. Stage 2: Sequential Transformer Scoring
    user_seq = get_user_recent_sequence(user, db, max_len=model_service.config.get("transformer_max_len", 20)) if user else []
    if model_service.sequential_transformer is not None and user_seq:
        transformer_scores = model_service.sequential_transformer.score_candidates_with_sequence(
            sequence_item_indices=user_seq,
            candidate_item_indices=cand_item_tensor,
            device=model_service.device,
        )
    else:
        transformer_scores = torch.full((total_cands,), 0.3, device=model_service.device)

    # 4. Stage 3: Fresh Genre Re-ranking Signal
    cand_genre_matrix = torch.zeros(total_cands, model_service.config.get("num_genres", 20), device=model_service.device)
    for pos, movie in enumerate(candidate_movies):
        midx = model_service.movie2idx.get(movie.movie_id)
        if midx is not None and model_service.movie_genre_matrix is not None:
            cand_genre_matrix[pos] = model_service.movie_genre_matrix[midx].to(model_service.device)
        else:
            for genre in movie.genres or []:
                gidx = model_service.genre2idx.get(genre)
                if gidx is not None:
                    cand_genre_matrix[pos, gidx] = 1.0
            transformer_scores[pos] = 0.5
    if model_service.movie_genre_matrix is not None:
        genre_scores = torch.mv(cand_genre_matrix, user_genre_vector)
        if genre_scores.max() > 0:
            genre_scores = genre_scores / genre_scores.max()
    else:
        genre_scores = torch.full((total_cands,), 0.5, device=model_service.device)

    # 5. Build user profile and inputs for DAMR
    profile = build_taste_profile(user, db) if user else estimate_user_state_from_positions([])

    pool_ratings = [
        float(m.rating) / 2.0 if m.rating is not None else None
        for m in candidate_movies
    ]
    pool_counts = [
        int(m.rating_count) if m.rating_count is not None else None
        for m in candidate_movies
    ]

    actual_top_k = min(top_k, total_cands)
    # Catalogue positions are NOT model indices. Supply actual pool features
    # to DAMR so unmapped titles never inherit model item zero's genres/embedding.
    gen_norm = F.normalize(cand_genre_matrix, dim=1)
    candidate_similarity = 0.5 * (gen_norm @ gen_norm.T)
    if model_service.ncf_hybrid is not None:
        emb = model_service.ncf_hybrid.item_embedding(cand_item_tensor).clone()
        for pos, movie in enumerate(candidate_movies):
            if movie.movie_id not in model_service.movie2idx:
                emb[pos] = 0.0
        emb = F.normalize(emb, dim=1)
        candidate_similarity += 0.5 * (emb @ emb.T)

    ranked = damr_rerank(
        cand_idxs=cand_item_tensor,
        s_ncf=ncf_scores,
        s_tr=transformer_scores,
        s_gen=genre_scores,
        profile=profile,
        ratings=pool_ratings,
        counts=pool_counts,
        top_k=actual_top_k,
        candidate_genres=cand_genre_matrix,
        candidate_similarity=candidate_similarity.clamp(-1, 1),
        **switches_for_variant(variant),
    )

    ensure_movie_posters(candidate_movies, db)

    results: List[ScoredMovieSchema] = []
    for entry in ranked:
        cand_idx = entry["pos"]
        if 0 <= cand_idx < len(candidate_movies):
            m = candidate_movies[cand_idx]
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
                    score=entry["score"],
                    ncfScore=entry["ncfScore"],
                    transformerScore=entry["transformerScore"],
                    genreScore=entry["genreScore"],
                    momentumScore=entry["momentumScore"],
                    agreementScore=entry["agreementScore"],
                    qualityScore=entry["qualityScore"],
                    diversityPenalty=entry["diversityPenalty"],
                    expertWeights=entry["expertWeights"],
                    userState=entry["userState"],
                    variant=variant,
                    rank=len(results) + 1,
                    listDiversity=_pool_diversity(candidate_similarity, [e["pos"] for e in ranked]) if len(results) == 0 else None,
                )
            )

    audit_metrics = {
        "total_candidates": total_cands,
        "mapped_candidates": mapped_count,
        "unmapped_candidates": unmapped_count,
        "variant": variant,
        "expert_weights": ranked[0]["expertWeights"] if ranked else {},
        "user_state": ranked[0]["userState"] if ranked else {},
        "final_movie_ids": [r.movieId for r in results],
    }

    return results, audit_metrics


def _pool_diversity(similarity: torch.Tensor, positions: List[int]) -> float:
    if len(positions) < 2:
        return 0.0
    indices = torch.tensor(positions, device=similarity.device)
    distances = 1.0 - similarity.index_select(0, indices).index_select(1, indices)
    distances.fill_diagonal_(0.0)
    return round(float(distances.sum() / (len(positions)*(len(positions)-1))), 4)

