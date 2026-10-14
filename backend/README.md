# Filmory — Backend & Deep Learning Recommendation Engine

Filmory is a full-stack AI movie discovery platform powered by deep learning recommendation models trained on the MovieLens catalog.

Ask Filmory defaults to local parsing and catalogue-evidence explanations:
`ASSISTANT_LOCAL_ONLY=true`. It does not make Gemini requests, even when a key is
configured. No paid service or new model download is required. Complex language
may need rephrasing into explicit titles, genres, years and runtime constraints.

Movie feedback uses the existing interaction log: `not_interested` hides a title
from personalized/assistant recommendations; `show_again` reverses that choice.
`GET /api/users/me/hidden-movies` returns the current account's hidden IDs. The
frontend profile offers undo without deleting saved titles or watch history.
See `docs/verification/recommendation-improvements-2026-10-14.md` at the repository
root for scope and verification details.

The signed-in homepage also offers an optional **Help evaluate Filmory** pilot.
Its three owned study endpoints freeze the movie list/policy and save relevance
judgments separately from recommendation history. Fresh migrations include
`0003_recommendation_study`; local startup creates the missing study table.
Export human results with `scripts/export_recommendation_study.py --output
evaluation/classmate-pilot-01.json` after classmates participate. Dedicated QA
accounts are excluded and names/emails omitted. Instructions are in
`docs/verification/classmate-feedback-guide.md` at the repository root.

---

## System Architecture

```
React Frontend (TanStack Start / React 19)
       ↓  (JWT Bearer Auth + REST APIs)
FastAPI Backend
       ↓
PostgreSQL Database (Users, Movies, Interactions, Watch History, My List, Likes)
       +
PyTorch Recommendation Engine (NCF Hybrid + Sequential Transformer + Genre Matrix)
```

---

## ML Models & Artifacts

The active trained checkpoints remain immutable. `backend/ml/serving_policy.json`
binds the new-account serving policy to their SHA-256 hashes. Accounts without a
learned user embedding start with the existing cold-start recommendations; after
two distinct mapped movie events, a validation-selected 300-candidate path combines
Transformer scores with collaborative-neighbor votes. History groups of 2, 5, 10
and 20 events have separately validated weights. The first expert in this path is
neighbor voting, not another user's NCF embedding. Mapped accounts retain the
previous validated default.

See `docs/research/RECOMMENDATION_QUALITY_REVIEW_2026-10-13.md` for the measured
trade-offs. To roll back this serving change, remove the standalone policy overlay
and restart the backend; the original checkpoint bundle and old cold-start path
remain available. A changed bundle requires a new matching policy evaluation.

All models are loaded into memory once during FastAPI startup (`lifespan`) and run on CPU or CUDA GPU automatically:

1. **NCF Baseline (`ncf_baseline.pth`)**
   - 8-dim User Embedding × 8-dim Item Embedding → MLP (16 → 64 → 32 → 1)

2. **NCF Hybrid (`ncf_hybrid.pth`)**
   - 8-dim User Embedding + 8-dim Item Embedding + 8-dim User Genre Projection + 8-dim Item Genre Projection → MLP (32 → 64 → 32 → 1)

3. **Sequential Transformer (`sequential_transformer.pth`)**
   - 64-dim Item Embeddings + 20-step Position Embeddings + 2-layer 4-head Transformer Encoder → Sequence Representation

4. **Matrix Artifacts**
   - `movie_genre_matrix.pt`: 22,836 movies × 20 genres
   - `user_genre_matrix.pt`: 41,547 users × 20 genres
   - Mappings: `user2idx.pkl`, `idx2user.pkl`, `movie2idx.pkl`, `idx2movie.pkl`, `genre2idx.pkl`, `idx2genre.pkl`

---

## 3 Recommendation Flows

### 1. Personalized Recommendations (`GET /api/recommendations/me`)
- **Stage 1**: Candidate generation via Hybrid NCF → Top 100 candidate movies
- **Stage 2**: Sequential Transformer scoring on the Top 100 using the user's recent watch sequence
- **Stage 3**: Fresh dynamic genre preference signal from real-time interactions
- **Ensemble**: `0.55 * NCF + 0.25 * Transformer + 0.20 * Genre`
- **Output**: Top 10 ranked movies returned to frontend

### 2. Movie-to-Movie Similarity (`GET /api/similar/{movieId}`)
- Computes cosine similarity between learned NCF item embeddings and the 20-dim movie genre matrix.
- Powers the "You May Also Like" rail on movie details pages without requiring authentication.

### 3. New-User Cold Start (`POST /api/cold-start`)
- For users with fewer than 5 interactions:
  - Generates a target genre vector from onboarding choices
  - Calculates cosine similarity against 41,547 existing user genre profiles
  - Gathers and ranks movies watched by the most similar users
  - Filters out already selected favorites

---

## Quickstart & Setup

### 1. Prerequisites
- Python 3.10+
- PostgreSQL server running locally (or remote connection)

### 2. Configure Environment (`.env`)
Create `backend/.env` (or copy from `.env.example`):

```env
DATABASE_URL=postgresql+psycopg://postgres:your-url-encoded-password@localhost:5432/filmory
SECRET_KEY=generate-a-long-random-secret-before-running
ACCESS_TOKEN_EXPIRE_MINUTES=10080
FRONTEND_ORIGIN=http://localhost:5173
PORT=8000
HOST=0.0.0.0
```

> **Note:** If your PostgreSQL password contains special characters like `@`, URL-encode it (`@` → `%40`).

### 3. Create Virtual Environment & Install Dependencies
```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```

### 4. Run Migrations & Seed Catalog
```bash
# Apply database schema
alembic upgrade head

# Seed 27,278 MovieLens catalog titles and Demo user
python -m app.scripts.seed_movies
```

### 5. Start Backend Server
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 6. Run Automated Tests
```bash
python -m pytest -v tests
```

API tests use a disposable SQLite catalogue and disable external AI/poster calls;
they do not write test users into the configured PostgreSQL database. The active
model bundle must be present. On Windows with restricted temporary folders, create
`backend/.test-tmp` and pass a new `--basetemp=.test-tmp/<run-name>` from the backend directory.

Migration `0002_assistant_tables` completes the fresh-install schema. Existing
databases created by application startup may already contain these tables with no
Alembic stamp: back up and compare the schema before reconciling their migration
history. Do not blindly apply or stamp migrations over such a database.

---

## API Endpoints Summary

### Authentication (`/api/auth`)
- `POST /api/auth/register` — Create account with password hashing
- `POST /api/auth/login` — Login and receive JWT access token
- `POST /api/auth/demo` — Quick login as Demo Viewer (mapped to model user `2847`)
- `GET /api/auth/me` — Get current user profile and preferences
- `PUT /api/auth/preferences` — Update onboarding favorite genres & movies

### Movies & Catalog (`/api`)
- `GET /api/movies` — Browse with genre filter, sorting (`popular`, `rating`, `year`, `title`), and pagination
- `GET /api/movies/{movieId}` — Get movie details
- `GET /api/genres` — List of distinct genres
- `GET /api/search?query=...` — Search by title or genre

### Recommendations (`/api`)
- `GET /api/recommendations/{userId}` — Personalized recommendations (3-stage model)
- `GET /api/similar/{movieId}` — Movie-to-movie item embedding similarity
- `GET /api/trending` — Trending titles based on interaction popularity
- `GET /api/popular` — Top-rated movies with significant audience counts
- `GET /api/movies-by-genre?genre=...` — Genre rail
- `POST /api/cold-start` — Cold start recommendations from genre profile

### User Activity & Interactions (`/api`)
- `POST /api/interactions` — Record `play`, `like`, `unlike`, `list_add`, `list_remove`
- `GET /api/users/me/history` — User's chronological watch history
- `GET /api/users/me/my-list` — User's saved watchlist
- `POST /api/users/me/my-list/{movieId}` — Add to watchlist
- `DELETE /api/users/me/my-list/{movieId}` — Remove from watchlist
- `GET /api/users/me/likes` — Liked movie IDs
- `POST /api/users/me/likes/{movieId}` — Toggle like
- `DELETE /api/users/me/likes/{movieId}` — Unlike movie
