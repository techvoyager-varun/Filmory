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

The active pointer selects `training/runs/20261001_clean_extended`, not the legacy
checkpoints directly under `ml`. Restore the complete active bundle on a new
checkout: its large binaries/data are local and ignored by Git. Current setup,
measured results and artifact requirements are in the [root README](../README.md).

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
   - Active 16-dim user and item embeddings → MLP (32 → 64 → 32 → 1).

2. **NCF Hybrid (`ncf_hybrid.pth`)**
   - Active 16-dim user/item embeddings and 16-dim genre projections → MLP (64 → 64 → 32 → 1), with L1-normalized user genres and training-compatible feature order.

3. **Sequential Transformer (`sequential_transformer.pth`)**
   - 64-dim embeddings, 20-step position embeddings, 2-layer 4-head encoder, 256-dim feed-forward block, separate padding ID and last-valid-position pooling.

4. **Matrix Artifacts**
   - `movie_genre_matrix.pt`: 22,836 movies × 20 genres
   - `user_genre_matrix.pt`: 41,547 users × 20 genres
   - Mappings: `user2idx.pkl`, `idx2user.pkl`, `movie2idx.pkl`, `idx2movie.pkl`, `genre2idx.pkl`, `idx2genre.pkl`

---

## Recommendation Flows

### 1. Personalized Recommendations (`GET /api/recommendations/me`)
- `variant=auto` selects the active mapped-user bundle's Transformer retrieval and
  normalized fixed fusion on 100 candidates: Hybrid NCF / Transformer / genre
  weights 0.55 / 0.25 / 0.20.
- New accounts start with collaborative-neighbor cold start, then use the frozen
  sequence/neighbor history-bin policy after at least two distinct mapped positive
  movies. NCF baseline is a comparison model, not a borrowed requester identity.
- Personalized results exclude played, liked, saved and hidden titles. The
  homepage requests 12 results; the endpoint's default is 10.
- DAMR is a supported alternative and used by assistant discovery; it is not the
  current default homepage policy. Title lookup bypasses discovery reranking.

### 2. Movie-to-Movie Similarity (`GET /api/similar/{movieId}`)
- Combines learned NCF item-embedding and genre cosine similarity. Unmapped
  catalogue titles use a genre-overlap fallback.
- Powers the "You May Also Like" rail on movie details pages without requiring authentication.

### 3. New-User Cold Start (`POST /api/cold-start`)
- Explicit genres/favourite films and eligible activity define a target profile;
  similarity to training-user genre profiles yields neighbor votes for movies.
- All selected genres are preserved. The authenticated `auto` path uses this
  fallback until the mapped positive history permits the sequence policy.
- `/api/cold-start` accepts an explicit request and is separate from the automatic
  account/history branch.

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
FRONTEND_ORIGIN=http://127.0.0.1:5173
ASSISTANT_LOCAL_ONLY=true
GEMINI_API_KEY=
TMDB_API_KEY=
PORT=8100
HOST=127.0.0.1
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

These migration/seed instructions are for a fresh `filmory` database created
using PostgreSQL/pgAdmin. For an existing installation, review migration history
as described below before applying them. The complete active model bundle must
already be present.

### 5. Start Backend Server
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```
API Documentation will be available at:
- Swagger UI: `http://127.0.0.1:8100/docs`
- ReDoc: `http://127.0.0.1:8100/redoc`

Set the frontend's `VITE_API_BASE_URL` to `http://127.0.0.1:8100`; its fallback is
localhost:8000. The root README includes the corresponding frontend commands.

### 6. Run Automated Tests
```bash
python -m pytest -v tests
```

API tests use a disposable SQLite catalogue and disable external AI/poster calls;
they do not write test users into the configured PostgreSQL database. The active
model bundle must be present. On Windows with restricted temporary folders, create
`backend/.test-tmp` and pass a new `--basetemp=.test-tmp/<run-name>` from the backend directory.

Migrations `0002_assistant_tables` and `0003_recommendation_study` complete the fresh-install schema. Existing
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
- `GET /api/recommendations/{userId}?variant=auto` — Account/history-dependent recommendations
- `GET /api/similar/{movieId}` — Movie-to-movie item embedding similarity
- `GET /api/trending` — Most-rated catalogue titles (the homepage labels this Most-Rated Movies)
- `GET /api/popular` — Top-rated movies with significant audience counts
- `GET /api/movies-by-genre?genre=...` — Genre rail
- `POST /api/cold-start` — Cold start recommendations from genre profile

### User Activity & Interactions (`/api`)
- `POST /api/interactions` — Record `play`, `like`, `unlike`, `list_add`, `list_remove`, `not_interested`, `show_again`
- `GET /api/users/me/hidden-movies` — Current account's reversible exclusions
- `GET /api/users/me/history` — User's chronological watch history
- `GET /api/users/me/my-list` — User's saved watchlist
- `POST /api/users/me/my-list/{movieId}` — Add to watchlist
- `DELETE /api/users/me/my-list/{movieId}` — Remove from watchlist
- `GET /api/users/me/likes` — Liked movie IDs
- `POST /api/users/me/likes/{movieId}` — Toggle like
- `DELETE /api/users/me/likes/{movieId}` — Unlike movie

### Assistant and optional study
- Owned chat/session/feedback/filter routes under `/api/assistant`; local-only
  parsing/explanations are the default.
- `POST /api/recommendation-study` — Start a consented frozen review.
- `GET /api/recommendation-study/{study_id}` — Read an owned review.
- `PUT /api/recommendation-study/{study_id}/answers` — Save/replace a judgment.

For the exact request schemas, use the running `/docs` Swagger page. Playback is
simulated, and user relevance/genre balance are not guaranteed. See the root
README and dated verification reports for current checks and limitations.
