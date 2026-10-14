from fastapi.testclient import TestClient
from app.main import app
from app.schemas.assistant import IntentDelta
from app.services.assistant_orchestrator import _merge_intents

def register(client, name):
    result=client.post('/api/auth/register',json={'name':name,'email':name+'@example.com','password':'test-password-123'})
    assert result.status_code == 200
    return {'Authorization':'Bearer '+result.json()['accessToken']}

def test_filter_chips_remove_semantic_and_reference_constraints():
    previous={'semantic_query':'alien','reference_titles':['Alien','Aliens'],'preferred_genres':['Sci-Fi']}
    intent=_merge_intents(previous,IntentDelta(clear_fields=['semantic_query','reference_titles:Alien']))
    assert intent.semantic_query == '' and intent.reference_titles == ['Aliens']
    assert intent.preferred_genres == ['Sci-Fi']

def test_session_ownership_and_feedback_validation():
    with TestClient(app) as client:
        alice=register(client,'alice'); bob=register(client,'bob')
        response=client.post('/api/assistant/chat',headers=alice,json={'message':'all movies'})
        assert response.status_code == 200
        sid=response.json()['session_id']
        for method,path,kwargs in [
            ('get',f'/api/assistant/sessions/{sid}',{}),
            ('delete',f'/api/assistant/sessions/{sid}',{}),
            ('post','/api/assistant/chat',{'json':{'message':'comedy','session_id':sid}}),
            ('delete',f'/api/assistant/sessions/{sid}/filters/all',{}),
        ]:
            assert getattr(client,method)(path,headers=bob,**kwargs).status_code == 404
        assert client.post('/api/assistant/feedback',headers=alice,json={'session_id':sid,'message_id':999999,'feedback_type':'helpful'}).status_code == 400
        assert client.post('/api/assistant/feedback',headers=alice,json={'session_id':sid,'movie_id':999999999,'feedback_type':'helpful'}).status_code == 400
        assert client.delete(f'/api/assistant/sessions/{sid}/filters/unknown',headers=alice).status_code == 400
        assert client.delete(f'/api/assistant/sessions/{sid}',headers=alice).status_code == 204
        assert client.get(f'/api/assistant/sessions/{sid}',headers=alice).status_code == 404

def test_blank_chat_and_invalid_tokens_are_rejected():
    with TestClient(app) as client:
        headers=register(client,'blank')
        assert client.post('/api/assistant/chat',headers=headers,json={'message':'   '}).status_code == 422
        assert client.get('/api/auth/me',headers={'Authorization':'Bearer invalid'}).status_code == 401

def test_user_limit_rejections_do_not_exhaust_other_users_ip_quota(monkeypatch):
    import pytest
    from fastapi import HTTPException
    from app.routers import assistant
    from app.config import settings
    monkeypatch.setattr(settings,'ASSISTANT_RATE_LIMIT',1)
    monkeypatch.setattr(settings,'ASSISTANT_GLOBAL_RATE_LIMIT',2)
    assistant._enforce_chat_rate_limit(1,'same-ip')
    with pytest.raises(HTTPException): assistant._enforce_chat_rate_limit(1,'same-ip')
    assistant._enforce_chat_rate_limit(2,'same-ip')

def test_registration_and_preferences_validate_before_persisting():
    with TestClient(app) as client:
        assert client.post('/api/auth/register',json={'name':'  ','email':'blank@example.com','password':'test-password-123'}).status_code == 422
        headers=register(client,'preferences')
        assert client.post('/api/auth/register',json={'name':'Other','email':'PREFERENCES@example.com','password':'test-password-123'}).status_code == 409
        assert client.put('/api/auth/preferences',headers=headers,json={'favoriteGenres':['Unknown'],'favoriteMovieIds':[]}).status_code == 400
        assert client.put('/api/auth/preferences',headers=headers,json={'favoriteGenres':['Comedy'],'favoriteMovieIds':[999999999]}).status_code == 400
        assert client.get('/api/auth/me',headers=headers).json()['favoriteGenres'] == []
        response=client.post('/api/interactions',headers=headers,json={'movieId':109487,'type':'play'})
        assert response.status_code == 200
        assert response.json()['timestamp'].endswith('+00:00')

def test_readiness_fails_when_models_are_unavailable(monkeypatch):
    from app.ml.model_service import model_service
    with TestClient(app) as client:
        monkeypatch.setattr(model_service,'is_loaded',False)
        assert client.get('/health').status_code == 503


def test_assistant_fallback_explains_every_displayed_movie():
    with TestClient(app) as client:
        headers=register(client,'explanation')
        response=client.post('/api/assistant/chat',headers=headers,json={'message':'comedy movies without horror or romance under 120 minutes'})
        assert response.status_code == 200
        data=response.json()
        assert len(data['movies']) == 10
        assert f"{len(data['movies'])} films" in data['message']
        assert {m['movieId'] for m in data['movies']} == {m['movie_id'] for m in data['structured_explanation']['items']}
        assert all('Comedy' in m['genres'] and not {'Horror','Romance'} & set(m['genres']) for m in data['movies'])


def test_new_registered_account_reaches_sequence_recommendations():
    with TestClient(app) as client:
        headers=register(client,'sequence-account')
        before=client.get('/api/recommendations/me?top_k=12',headers=headers)
        assert before.status_code==200 and before.json()
        for movie_id in [1,2]:
            assert client.post('/api/interactions',headers=headers,json={'movieId':movie_id,'type':'play'}).status_code==200
        after=client.get('/api/recommendations/me?top_k=12',headers=headers)
        assert after.status_code==200 and len(after.json())==12
        assert all(movie['variant']=='sequence_only' and movie['ncfScore'] is None for movie in after.json())
        assert not {1,2} & {movie['movieId'] for movie in after.json()}


def test_not_interested_is_private_reversible_and_enforced_by_assistant():
    with TestClient(app) as client:
        alice = register(client, 'hidden-alice')
        bob = register(client, 'hidden-bob')
        assert client.get('/api/users/me/hidden-movies').status_code == 401
        assert client.post('/api/interactions', headers=alice, json={'movieId':999999999, 'type':'not_interested'}).status_code == 404
        assert client.post('/api/interactions', headers=alice, json={'movieId':1, 'type':'unknown'}).status_code == 422
        title = client.get('/api/movies/1').json()['title']
        assert client.post('/api/interactions', headers=alice, json={'movieId':1, 'type':'like'}).status_code == 200
        assert client.post('/api/interactions', headers=alice, json={'movieId':1, 'type':'not_interested'}).status_code == 200
        assert client.get('/api/users/me/hidden-movies', headers=alice).json() == [1]
        assert client.get('/api/users/me/hidden-movies', headers=bob).json() == []
        assert 1 not in client.get('/api/users/me/likes', headers=alice).json()
        assert 1 not in {movie['movieId'] for movie in client.get('/api/recommendations/me?top_k=50', headers=alice).json()}
        chat = client.post('/api/assistant/chat', headers=alice, json={'message':title})
        assert chat.status_code == 200
        assert 1 not in {movie['movieId'] for movie in chat.json()['movies']}
        assert client.post('/api/interactions', headers=alice, json={'movieId':1, 'type':'show_again'}).status_code == 200
        assert client.get('/api/users/me/hidden-movies', headers=alice).json() == []
        # Restoring visibility does not invent a new like.
        assert 1 not in client.get('/api/users/me/likes', headers=alice).json()


def test_recommendation_study_requires_consent_and_preserves_owned_slate():
    with TestClient(app) as client:
        alice=register(client,'study-alice')
        bob=register(client,'study-bob')
        assert client.post('/api/recommendation-study',json={'consent':True}).status_code==401
        assert client.post('/api/recommendation-study',headers=alice,json={'consent':False}).status_code==422
        started=client.post('/api/recommendation-study',headers=alice,json={'consent':True})
        assert started.status_code==200
        study=started.json()
        assert study['movies'] and study['answers']=={}
        path='/api/recommendation-study/'+study['id']
        movie_id=study['movies'][0]['movieId']
        assert client.get(path,headers=bob).status_code==404
        assert client.put(path+'/answers',headers=bob,json={'movieId':movie_id,'choice':'interesting'}).status_code==404
        assert client.put(path+'/answers',headers=alice,json={'movieId':999999999,'choice':'interesting'}).status_code==400
        assert client.put(path+'/answers',headers=alice,json={'movieId':movie_id,'choice':'invalid'}).status_code==422
        for choice in ['interesting','not_for_me','unsure']:
            updated=client.put(path+'/answers',headers=alice,json={'movieId':movie_id,'choice':choice})
            assert updated.status_code==200
            assert updated.json()['answers']=={str(movie_id):choice}
            assert updated.json()['movies']==study['movies']
        assert client.get('/api/users/me/likes',headers=alice).json()==[]
        assert client.get('/api/users/me/history',headers=alice).json()==[]
        assert client.get(path,headers=alice).json()['answers']=={str(movie_id):'unsure'}


def test_cold_start_preserves_all_five_explicit_genres():
    from unittest.mock import patch
    from app.ml import recommender
    with TestClient(app) as client:
        headers=register(client,'five-genres')
        genres=['Sci-Fi','Comedy','Thriller','Animation','Documentary']
        saved=client.put('/api/auth/preferences',headers=headers,json={'favoriteGenres':genres,'favoriteMovieIds':[]})
        assert saved.status_code==200
        with patch.object(recommender,'get_cold_start_recommendations',wraps=recommender.get_cold_start_recommendations) as cold:
            response=client.get('/api/recommendations/me?top_k=12',headers=headers)
        assert response.status_code==200 and len(response.json())==12
        assert set(cold.call_args.args[0])==set(genres)
        assert all(set(movie['genres']) & set(genres) for movie in response.json())
