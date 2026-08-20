import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] is True

def test_get_genres():
    with TestClient(app) as client:
        res = client.get("/api/genres")
        assert res.status_code == 200
        genres = res.json()
        assert "Sci-Fi" in genres
        assert "Action" in genres

def test_get_movies_browse():
    with TestClient(app) as client:
        res = client.get("/api/movies?limit=10&genre=Sci-Fi&sort=popular")
        assert res.status_code == 200
        data = res.json()
        assert "movies" in data
        assert len(data["movies"]) == 10
        assert data["total"] > 0

def test_search_movies():
    with TestClient(app) as client:
        res = client.get("/api/search?query=interstellar")
        assert res.status_code == 200
        movies = res.json()
        assert len(movies) > 0
        assert any("Interstellar" in m["title"] for m in movies)

def test_demo_auth():
    with TestClient(app) as client:
        res = client.post("/api/auth/demo")
        assert res.status_code == 200
        data = res.json()
        assert "accessToken" in data
        assert data["user"]["email"] == "demo@filmory.app"

def test_similar_movies():
    with TestClient(app) as client:
        # 109487 is Interstellar
        res = client.get("/api/similar/109487?top_k=10")
        assert res.status_code == 200
        similar = res.json()
        assert len(similar) > 0
        assert all("movieId" in m for m in similar)

def test_trending_movies():
    with TestClient(app) as client:
        res = client.get("/api/trending?limit=10")
        assert res.status_code == 200
        trending = res.json()
        assert len(trending) == 10

def test_popular_movies():
    with TestClient(app) as client:
        res = client.get("/api/popular?limit=10")
        assert res.status_code == 200
        popular = res.json()
        assert len(popular) == 10

def test_cold_start():
    with TestClient(app) as client:
        payload = {
            "favoriteGenres": ["Sci-Fi", "Drama"],
            "favoriteMovieIds": [109487, 79132]
        }
        res = client.post("/api/cold-start?top_k=10", json=payload)
        assert res.status_code == 200
        recs = res.json()
        assert len(recs) > 0

def test_authenticated_recommendations_and_interactions():
    with TestClient(app) as client:
        # 1. Register a dedicated test user
        test_email = "test_eval_user@filmory.app"
        reg_res = client.post(
            "/api/auth/register",
            json={"name": "Test Eval", "email": test_email, "password": "password123"},
        )
        if reg_res.status_code == 409:
            login_res = client.post(
                "/api/auth/login",
                json={"email": test_email, "password": "password123"},
            )
            token = login_res.json()["accessToken"]
        else:
            token = reg_res.json()["accessToken"]

        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get recommendations (cold start for new user)
        rec_res = client.get("/api/recommendations/me?candidate_k=50&top_k=10", headers=headers)
        assert rec_res.status_code == 200
        recs = rec_res.json()
        assert len(recs) == 10

        # 3. Record play interaction
        play_res = client.post(
            "/api/interactions",
            json={"movieId": 109487, "type": "play"},
            headers=headers,
        )
        assert play_res.status_code == 200

        # 4. Check history
        hist_res = client.get("/api/users/me/history", headers=headers)
        assert hist_res.status_code == 200
        history = hist_res.json()
        assert any(h["movie"]["movieId"] == 109487 for h in history)

        # 5. Toggle Like
        like_res = client.post("/api/users/me/likes/109487", headers=headers)
        assert like_res.status_code == 200
        likes = client.get("/api/users/me/likes", headers=headers).json()
        assert 109487 in likes

        # 6. Add / Remove from My List
        add_list = client.post("/api/users/me/my-list/109487", headers=headers)
        assert add_list.status_code == 200
        my_list = client.get("/api/users/me/my-list", headers=headers).json()
        assert any(m["movieId"] == 109487 for m in my_list)

        del_list = client.delete("/api/users/me/my-list/109487", headers=headers)
        assert del_list.status_code == 200
