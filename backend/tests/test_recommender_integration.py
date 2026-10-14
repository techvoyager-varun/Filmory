import datetime
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import Base
from app.models.db_models import User, Movie, Interaction
from app.ml.architectures import NCFHybrid, SequentialTransformer
from app.ml.model_service import model_service
from app.ml.recommender import (get_user_interacted_movie_ids, get_user_recent_sequence,
                               build_taste_profile, rerank_candidate_pool_with_damr,
                               get_personalized_recommendations)
from app.schemas.schemas import RegisterRequest
from app.core.security import create_access_token, decode_access_token


@pytest.fixture
def toy_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(model_service, "load_all", lambda: None)
    monkeypatch.setattr(model_service, "device", torch.device("cpu"))
    monkeypatch.setattr(model_service, "config", {"num_genres":2,"num_items":12,"transformer_max_len":5})
    monkeypatch.setattr(model_service, "user2idx", {999:0})
    monkeypatch.setattr(model_service, "idx2user", {0:999})
    monkeypatch.setattr(model_service, "movie2idx", {100+i:i for i in range(12)})
    monkeypatch.setattr(model_service, "idx2movie", {i:100+i for i in range(12)})
    monkeypatch.setattr(model_service, "genre2idx", {"Action":0,"Comedy":1})
    monkeypatch.setattr(model_service, "idx2genre", {0:"Action",1:"Comedy"})
    monkeypatch.setattr(model_service, "movie_genre_matrix", torch.tensor([[1.,0.],[0.,1.]]*6))
    monkeypatch.setattr(model_service, "user_genre_matrix", torch.tensor([[.8,.2]]))
    monkeypatch.setattr(model_service, "user_sequences", {0:[0,2,4,6,8]})
    monkeypatch.setattr(model_service, "user_interacted", {0:[0,2,4,6,8]})
    monkeypatch.setattr(model_service, "ncf_hybrid", NCFHybrid(1,12,2).eval())
    monkeypatch.setattr(model_service, "ncf_baseline", None)
    monkeypatch.setattr(model_service, "sequential_transformer", SequentialTransformer(12,max_len=5,pooling="last",dim_feedforward=64).eval())
    monkeypatch.setattr("app.ml.recommender.ensure_movie_posters", lambda *args: None)
    with Session(engine) as db:
        user=User(id=1,name="Test",email="test@example.invalid",password_hash="unused",model_user_id=999)
        db.add(user)
        db.add_all([Movie(movie_id=100+i,title=f"Movie {i}",genres=["Action" if i%2==0 else "Comedy"],year=2000,rating=7,rating_count=1500) for i in range(12)])
        db.add(Movie(movie_id=9999,title="Unmapped Comedy",genres=["Comedy"],year=2020,rating=8,rating_count=1000))
        db.commit()
        yield db,user
    engine.dispose()


def test_history_conversion_uses_user_index_and_canonical_movie_ids(toy_db):
    db,user=toy_db
    assert get_user_interacted_movie_ids(user,db)=={100,102,104,106,108}
    assert get_user_recent_sequence(user,db)==[0,2,4,6,8]
    profile=build_taste_profile(user,db)
    assert profile.state.freshness==0.0
    assert profile.state.n_interactions==5


def test_live_hybrid_genre_input_matches_training_normalization(toy_db,monkeypatch):
    from app.ml.recommender import build_user_genre_vector
    from app.models.db_models import UserGenrePreference
    db,user=toy_db
    monkeypatch.setitem(model_service.config,'genre_profile_normalization','l1')
    actual=build_user_genre_vector(user,db)
    trained=model_service.user_genre_matrix[0]
    assert torch.allclose(actual,trained)
    items=torch.arange(12)
    serving=model_service.ncf_hybrid.score_candidate_items(0,items,actual,model_service.movie_genre_matrix,torch.device('cpu'))
    reference=model_service.ncf_hybrid.score_candidate_items(0,items,trained,model_service.movie_genre_matrix,torch.device('cpu'))
    assert torch.allclose(serving,reference)
    db.add(UserGenrePreference(user_id=user.id,genre='Comedy',score=3.0));db.commit()
    assert float(build_user_genre_vector(user,db).sum()) == pytest.approx(1.0)


def test_database_history_keeps_item_zero_as_real_movie(toy_db):
    db,user=toy_db
    db.add(Interaction(user_id=user.id,movie_id=100,type="play",timestamp=datetime.datetime(2020,1,1)))
    db.commit()
    assert get_user_recent_sequence(user,db)==[0]


def test_arbitrary_catalogue_pool_preserves_positions_and_unmapped_metadata(toy_db):
    db,user=toy_db
    pool=[db.get(Movie,109),db.get(Movie,105),db.get(Movie,9999)]
    from app.ml import recommender
    actual=recommender.damr_rerank
    seen={}
    def observe(*args,**kwargs):
        seen.update(kwargs)
        return actual(*args,**kwargs)
    with patch.object(recommender,"damr_rerank",side_effect=observe):
        result,audit=rerank_candidate_pool_with_damr(pool,user,db,top_k=3)
    assert {m.movieId for m in result}=={109,105,9999}
    assert audit["mapped_candidates"]==2 and audit["unmapped_candidates"]==1
    assert seen["cand_idxs"].tolist()==[9,5,0]
    assert seen["candidate_genres"][2].tolist()==[0.,1.]
    assert all(m.score is not None and torch.isfinite(torch.tensor(m.score)) for m in result)


def test_anonymous_pool_has_valid_neutral_profile(toy_db):
    db,_=toy_db
    result,_=rerank_candidate_pool_with_damr([db.get(Movie,109),db.get(Movie,9999)],None,db)
    assert len(result)==2


def test_personalized_lists_exclude_training_history(toy_db):
    db,user=toy_db
    for variant in ["static","damr","mmr"]:
        _,results=get_personalized_recommendations(user,db,candidate_k=12,top_k=5,variant=variant)
        assert len(results)==5
        assert not ({m.movieId for m in results} & {100,102,104,106,108})


def test_cold_start_fallback_excludes_favorites_and_uses_all_genres(toy_db, monkeypatch):
    from app.ml.recommender import get_cold_start_recommendations
    db, _ = toy_db
    monkeypatch.setattr(model_service, "user_genre_matrix", None)
    results = get_cold_start_recommendations(["Action", "Comedy"], [100, 101, 9999], db, top_k=20)
    assert len(results) == 10
    assert not ({m.movieId for m in results} & {100, 101, 9999})
    assert {g for m in results for g in m.genres} == {"Action", "Comedy"}


def test_similar_movies_handles_catalogue_smaller_than_requested_count(toy_db):
    from app.ml.recommender import get_similar_movies
    db, _ = toy_db
    results = get_similar_movies(100, db, top_k=20)
    assert len(results) == 11
    assert 100 not in {m.movieId for m in results}


def test_genre_discovery_does_not_require_genre_words_in_titles_or_synopses(toy_db):
    from app.schemas.assistant import MovieIntent
    from app.services.semantic_search import search_movies_by_intent
    from app.services.llm_client import fallback_extract_intent
    db, _ = toy_db
    for movie in db.query(Movie).all():
        movie.runtime = 100
    db.commit()
    delta = fallback_extract_intent("comedy movies without horror or romance under 120 minutes")
    intent = MovieIntent(**{k:v for k,v in delta.model_dump().items() if k in MovieIntent.model_fields})
    results, _ = search_movies_by_intent(intent, db)
    assert len(results) == 7
    assert all("Comedy" in movie.genres for movie in results)
    # Removing a chip must not leave its old words acting as hidden filters.
    db.get(Movie, 9999).runtime = 150
    db.commit()
    intent.max_runtime_minutes = None
    results, _ = search_movies_by_intent(intent, db)
    assert 9999 in {movie.movie_id for movie in results}
    intent.preferred_genres = []
    intent.excluded_genres = []
    results, _ = search_movies_by_intent(intent, db)
    assert len(results) == 13


def test_unmapped_user_never_borrows_an_existing_embedding(toy_db):
    db,user=toy_db
    user.model_user_id=None
    db.add_all([Interaction(user_id=user.id,movie_id=100+i,type="play") for i in range(3)])
    db.commit()
    with patch.object(model_service.ncf_hybrid,"score_candidate_items",side_effect=AssertionError("Existing user embedding used")):
        _,results=get_personalized_recommendations(user,db)
    assert results


def test_new_account_transitions_from_cold_start_to_sequence_ranking(toy_db, monkeypatch):
    db,user=toy_db
    user.model_user_id=None
    monkeypatch.setattr(model_service,'serving_policy',{'history_bins':[
        {'min_history':2,'mode':'sequence_only','candidate_k':300,'strategy':'transformer','weights':[.65,.25,.1]}]})
    kind,_=get_personalized_recommendations(user,db,top_k=5)
    assert kind == 'cold_start'
    db.add(Interaction(user_id=user.id,movie_id=100,type='play',timestamp=datetime.datetime(2026,1,1)))
    db.commit()
    assert get_personalized_recommendations(user,db,top_k=5)[0] == 'cold_start'
    db.add(Interaction(user_id=user.id,movie_id=103,type='play',timestamp=datetime.datetime(2026,1,2)))
    db.commit()
    with patch.object(model_service.ncf_hybrid,'score_candidate_items',side_effect=AssertionError('Borrowed user embedding')):
        with patch.object(model_service.sequential_transformer,'score_candidates_with_sequence',wraps=model_service.sequential_transformer.score_candidates_with_sequence) as transformer:
            kind,movies=get_personalized_recommendations(user,db,top_k=5)
    assert kind == 'sequence_personalized'
    assert transformer.call_args.args[0] == [0,3]
    assert len(movies)==5 and len({m.movieId for m in movies})==5
    assert not {100,103,9999} & {m.movieId for m in movies}
    assert all(m.variant=='sequence_only' and m.ncfScore is None and m.expertWeights['ncf']==0 for m in movies)
    assert all(torch.isfinite(torch.tensor(m.score)) for m in movies)


def test_sequence_candidates_exclude_missing_and_saved_catalogue_items(toy_db, monkeypatch):
    from app.models.db_models import MyList
    db,user=toy_db
    user.model_user_id=None
    monkeypatch.setattr(model_service,'serving_policy',{'history_bins':[
        {'min_history':2,'mode':'sequence_only','candidate_k':300,'strategy':'transformer','weights':[0.,1.,0.]}]})
    db.add_all([Interaction(user_id=user.id,movie_id=100+i,type='play') for i in (0,1)])
    db.add(MyList(user_id=user.id,movie_id=102))
    db.delete(db.get(Movie,103));db.commit()
    _,movies=get_personalized_recommendations(user,db,top_k=50,candidate_k=1000)
    assert {m.movieId for m in movies} == set(range(104,112))


def test_new_user_without_matching_policy_preserves_legacy_fallback(toy_db,monkeypatch):
    db,user=toy_db
    user.model_user_id=None
    monkeypatch.setattr(model_service,'serving_policy',{})
    db.add_all([Interaction(user_id=user.id,movie_id=100+i,type='play') for i in range(3)])
    db.commit()
    assert get_personalized_recommendations(user,db)[0] == 'cold_start'


def test_hidden_movies_are_reversible_and_excluded_from_every_ranker(toy_db):
    from app.services.recommendation_feedback import hidden_movie_ids
    db, user = toy_db
    db.add(Interaction(user_id=user.id, movie_id=109, type='not_interested'))
    db.commit()
    assert hidden_movie_ids(user.id, db) == {109}
    for variant in ['static', 'normalized_fixed', 'damr', 'mmr']:
        _, movies = get_personalized_recommendations(user, db, top_k=50, candidate_k=1000, variant=variant)
        assert movies and 109 not in {movie.movieId for movie in movies}
    movies, _ = rerank_candidate_pool_with_damr([db.get(Movie, 109), db.get(Movie, 105)], user, db)
    assert [movie.movieId for movie in movies] == [105]
    db.add(Interaction(user_id=user.id, movie_id=109, type='show_again'))
    db.commit()
    assert hidden_movie_ids(user.id, db) == set()
    assert 109 in {movie.movieId for movie in get_personalized_recommendations(user, db, top_k=50, candidate_k=1000)[1]}


def test_live_history_ignores_removed_likes_hidden_titles_and_repeat_clicks(toy_db):
    from app.models.db_models import Like
    from app.services.preference_signals import refresh_genre_signals
    from app.models.db_models import UserGenrePreference
    db, user = toy_db
    user.model_user_id = None
    # Same timestamp: interaction IDs still give a deterministic order.
    when = datetime.datetime(2026, 1, 1)
    for movie_id, kind in [(100, 'play'), (101, 'like'), (101, 'unlike'), (100, 'play'), (103, 'like')]:
        db.add(Interaction(user_id=user.id, movie_id=movie_id, type=kind, timestamp=when))
    db.add(Like(user_id=user.id, movie_id=103))
    db.commit()
    assert get_user_recent_sequence(user, db) == [0, 3]
    assert build_taste_profile(user, db).state.n_interactions == 2
    refresh_genre_signals(user.id, db)
    db.commit()
    scores = {row.genre: row.score for row in db.query(UserGenrePreference).all()}
    assert scores == {'Action': 1.0, 'Comedy': 1.5}
    db.add(Interaction(user_id=user.id, movie_id=100, type='not_interested'))
    db.commit()
    assert get_user_recent_sequence(user, db) == [3]
    refresh_genre_signals(user.id, db)
    db.commit()
    assert {row.genre: row.score for row in db.query(UserGenrePreference).all()} == {'Comedy': 1.5}


def test_mapped_candidates_never_spend_budget_on_missing_catalogue_titles(toy_db):
    db, user = toy_db
    db.delete(db.get(Movie, 109))
    db.commit()
    _, movies = get_personalized_recommendations(user, db, candidate_k=1000, top_k=50)
    assert {movie.movieId for movie in movies} == {101, 103, 105, 107, 110, 111}


def test_exhausted_mapped_catalogue_does_not_recommend_consumed_titles(toy_db):
    db, user = toy_db
    for mid in [101, 103, 105, 107, 109, 110, 111]:
        db.add(Interaction(user_id=user.id, movie_id=mid, type='not_interested'))
    db.commit()
    assert get_personalized_recommendations(user, db)[1] == []


def test_rejected_movie_does_not_become_a_cold_start_preference(toy_db):
    from app.ml import recommender
    db, user = toy_db
    user.model_user_id = None
    db.add(Interaction(user_id=user.id, movie_id=109, type='not_interested'))
    db.commit()
    with patch.object(recommender, 'get_cold_start_recommendations', wraps=recommender.get_cold_start_recommendations) as cold:
        kind, movies = get_personalized_recommendations(user, db)
    assert kind == 'cold_start'
    assert cold.call_args.args[:2] == ([], [])
    assert cold.call_args.kwargs['exclude_ids'] == {109}
    assert 109 not in {movie.movieId for movie in movies}


def test_candidate_pool_is_unique_and_empty_pool_is_safe():
    from app.ml.ranking_policy import select_candidate_positions,rank_candidate_positions,choose_history_policy
    signals=torch.tensor([.3,.5,.1,.9])
    for strategy in ['transformer','mixed','collaborative']:
        pool=select_candidate_positions(signals,signals.flip(0),torch.ones(4),1000,strategy)
        assert len(pool)==4 and len(set(pool.tolist()))==4
        positions,scores=rank_candidate_positions(pool,signals,signals,torch.ones(4),[.65,.25,.1],10)
        assert len(positions)==4 and bool(torch.isfinite(scores).all())
    pool=select_candidate_positions(torch.empty(0),torch.empty(0),torch.empty(0),100)
    assert len(rank_candidate_positions(pool,torch.empty(0),torch.empty(0),torch.empty(0),[0.,1.,0.],10)[0])==0
    assert choose_history_policy({'history_bins':[{'min_history':2,'mode':'sequence_only'},{'min_history':5,'mode':'legacy_cold_start'}]},4)['mode']=='sequence_only'
    assert choose_history_policy({'history_bins':[{'min_history':2,'mode':'sequence_only'},{'min_history':5,'mode':'legacy_cold_start'}]},5)['mode']=='legacy_cold_start'


def test_password_limit_counts_utf8_bytes_and_zero_expiry_is_respected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RegisterRequest(name="test",email="a@example.com",password="é"*37)
    token=create_access_token(1,datetime.timedelta(0))
    # jose may allow the exact current second, so inspect the expiry directly.
    from jose import jwt
    claims=jwt.get_unverified_claims(token)
    assert abs(claims["exp"]-datetime.datetime.now(datetime.timezone.utc).timestamp())<2


def test_unlike_removes_genre_boost_and_missing_delete_is_404(toy_db):
    from fastapi import HTTPException
    from app.models.db_models import UserGenrePreference
    from app.routers.interactions import toggle_my_like, unlike_movie, remove_from_my_list
    db,user = toy_db
    toggle_my_like(100,user,db)
    assert db.query(UserGenrePreference).filter_by(user_id=user.id,genre="Action").one().score == 1.5
    unlike_movie(100,user,db)
    assert db.query(UserGenrePreference).filter_by(user_id=user.id,genre="Action").count() == 0
    for fn in [unlike_movie,remove_from_my_list]:
        with pytest.raises(HTTPException) as exc: fn(987654321,user,db)
        assert exc.value.status_code == 404


def test_preference_change_removes_previous_genre_signal(toy_db):
    from app.routers.auth import update_preferences
    from app.schemas.schemas import UserPreferencesUpdate
    from app.models.db_models import UserGenrePreference
    db,user=toy_db
    update_preferences(UserPreferencesUpdate(favoriteGenres=["Action"],favoriteMovieIds=[]),user,db)
    update_preferences(UserPreferencesUpdate(favoriteGenres=["Comedy"],favoriteMovieIds=[]),user,db)
    signals={r.genre:r.score for r in db.query(UserGenrePreference).filter_by(user_id=user.id)}
    assert signals == {"Comedy":2.0}
