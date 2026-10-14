# Filmory — Local Movie Recommendation and Discovery

Filmory is a local academic project built with React/TypeScript, FastAPI,
PostgreSQL and PyTorch. It combines Hybrid Neural Collaborative Filtering (NCF),
a Sequential Transformer, genre signals and recommendation policies. An NCF
baseline supports comparisons. Ask Filmory handles explicit movie constraints;
DAMR is the project's proposed custom reranking method.

**Playback is simulated:** Play records an interaction; Filmory does not stream
movie video. Scores are ranking signals, not calibrated match percentages. The
project does not claim Netflix-level performance or perfect recommendations.

Updated **14 October 2026** against the current source and retained evidence.

## Features

- Home: Recommended For You, Most-Rated Movies, Popular Movies, five genre
  shelves and Recently Watched.
- Browse/search/sorting, movie details and You May Also Like.
- Registration, login, demo login, onboarding, saved list and history.
- Plays and likes refresh personalized recommendations. Likes are optional;
  playing is a possible-interest signal rather than proof of enjoyment.
- Not interested hides a title from personalized/assistant suggestions; Show
  again reverses that choice through the movie page or profile.
- Ask Filmory supports title lookup, similar-movie requests and genre, year,
  runtime and rating constraints. Local parsing and catalogue-evidence
  explanations are the default: `ASSISTANT_LOCAL_ONLY=true` disables cloud AI.
- Help evaluate Filmory is an optional review of a frozen recommendation slate,
  saved separately from likes/history. Classmate participation is pending.
- A separate React Flow architecture explorer offers source-backed nodes,
  search, filters, six walkthroughs and SVG/PNG/Mermaid/JSON exports.

The seed catalogue has **27,278 titles**. The trained model universe has
**22,836 items, 41,547 training users and 20 genre features**. Extra catalogue
titles can be browsed; model coverage depends on the recommendation path.

## Recommendation pipeline

| Account/request | Current behaviour |
| --- | --- |
| Guest | Popular fallback. |
| New account with little mapped activity | Explicit genres/favourite films and eligible positive activity feed collaborative-neighbor cold-start retrieval. All selected genres are preserved. |
| New account with at least two distinct mapped positive movies | A frozen policy retrieves Transformer candidates and combines neighbor/Transformer/genre scores. History bins at 2, 5, 10 and 20 events have checkpoint-bound weights; no other user's NCF embedding is borrowed. |
| Mapped account, including demo | 100 Transformer candidates and normalized fixed Hybrid NCF / Transformer / genre fusion with weights **0.55 / 0.25 / 0.20**. |
| Ask Filmory discovery | SQL constraints/exclusions restrict candidates; the supported discovery branch uses DAMR and grounded explanations. Title lookup bypasses this ranking branch. |
| Movie details | Item-embedding/genre similarity for mapped titles and a genre-overlap fallback for unmapped titles. |

Personalized suggestions exclude played, liked, saved and explicitly hidden
titles. Positive movie history is deduplicated; removed likes and hidden titles
are not retained as positive evidence. Feedback updates live signals without
retraining checkpoints. Popular/genre shelves and Recently Watched have different
purposes and can still show played titles.

DAMR combines drift-conditioned weights, taste momentum, agreement and optional
quality/diversity components. MMR and the neural families are established
methods; this combination is Filmory's proposed custom method, not a verified
worldwide novelty claim. **DAMR is not the default homepage policy.** See the
[pipeline evaluation/ablations](docs/research/PIPELINE_QUALITY_2026-10-14.md).

## Active models and results

[active_bundle.json](backend/ml/active_bundle.json) selects
`backend/training/runs/20261001_clean_extended`. Its
[configuration](backend/training/runs/20261001_clean_extended/model_config.json)
and [serving overlay](backend/ml/serving_policy.json) define inference.

| Component | Active architecture |
| --- | --- |
| NCF baseline | 16-dimensional user/item embeddings; MLP **32 → 64 → 32 → 1**. |
| Hybrid NCF | 16-dimensional user/item embeddings and two 16-dimensional genre projections; training-compatible concatenation; MLP **64 → 64 → 32 → 1**; L1-normalized user genres. |
| Transformer | 64-dimensional embeddings, 4 heads, 2 encoder layers, feed-forward width 256, history length 20, zero-based IDs, separate padding ID and last-valid-position pooling. |
| Genre/neighbor experts | Deterministic profile affinity and similarity-weighted collaborative votes. |

Fresh fitting removes validation/test targets before training. The objective is
BCE-with-logits plus a pairwise ranking term; checkpoint selection uses validation
NDCG. Stored ordinal histories and a static training prefix are used; this is
transductive replay rather than an unseen-person evaluation.

**Original clean extended experiment:** sampled test uses 400 users and one target
against 99 negatives; full-catalogue test uses 200 users and the mapped catalogue
after excluding training history.

| Model | Selected epoch | Training BCE | Total objective | Sampled HR@10 | Catalogue HR@10 | Catalogue NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NCF baseline | 10 | 0.2099 | 0.2392 | 82.25% | 2.50% | 0.008827 |
| Hybrid NCF | 9 | 0.1896 | 0.2143 | 86.00% | 3.00% | 0.012452 |
| Transformer | 5 | 0.1361 | 0.1514 | 90.75% | 3.50% | 0.018740 |

Sources: [active-run metrics](backend/training/runs/20261001_clean_extended/metrics.json)
and [verified evidence](docs/research/research_evidence_verified_20261014.json).
The hybrid's final epoch had lower BCE, but validation selected epoch nine.
**90.75% sampled HR@10 is not overall accuracy.** Lower loss alone does not
establish better recommendations.

Later, separate experiments:

- [Frozen-policy confirmation](docs/research/CONFIRMATION_EVALUATION_2026-10-14.md):
  1,000 additional users; mean HR@10 across four contexts increased from
  **3.425% to 4.300%**, with a paired interval excluding zero under that protocol.
- [Pipeline/ablation evaluation](docs/research/PIPELINE_QUALITY_2026-10-14.md):
  another 1,000-user cohort; current mapped ensemble HR@10 **6.60%**, NDCG@10
  **0.028675**. Wider validation-selected retrieval improved candidate recall
  but failed the held-out promotion rule, so serving defaults stayed unchanged.

These cohorts/protocols differ; do not substitute one result for another.
Older shipped-checkpoint diagnostics and legacy evaluation scripts are historical
records, not the clean current measurements.

## Architecture and files

```mermaid
flowchart LR
    UI[React / TanStack frontend] -->|JSON and JWT bearer| API[FastAPI]
    API --> DB[(PostgreSQL)]
    API --> REC[Recommendation policies]
    REC --> MODELS[Frozen PyTorch models and genre matrices]
    REC --> DB
    API --> ASSIST[Local intent parsing / constrained discovery]
    ASSIST --> REC
    ASSIST --> DB
    API --> STUDY[Owned optional feedback study]
    STUDY --> DB
    ASSIST -. explicit cloud opt-in .-> GEMINI[Gemini]
    API -. optional artwork enrichment .-> TMDB[TMDB]
```

| Location | Purpose |
| --- | --- |
| [frontend](frontend) | React 19, TypeScript, TanStack Start/Router/Query, Tailwind/Radix UI and typed API clients. |
| [backend/app](backend/app) | FastAPI routes, authentication, SQLAlchemy models, inference and assistant services. |
| [backend/app/ml](backend/app/ml) | Architectures, loader, recommender, ranking helpers, DAMR and artwork adapter. |
| [backend/migrations](backend/migrations) | Initial, assistant and study migrations. |
| [backend/scripts](backend/scripts) | Training, evaluation, evidence audit and study-export utilities. |
| [backend/training](backend/training/README.md) | Corrected/original notebooks and run metadata. |
| [backend/evaluation](backend/evaluation) | Protocols, validation/test rows, selections, summaries and immutable snapshots. |
| [docs/verification](docs/verification) | Frontend, recommendation, test and supplementary checks. |
| [tools/architecture-explorer](tools/architecture-explorer/README.md) | Independent React Flow viewer, dependencies and port. |

## Local setup

Use Python **3.10+** (verified here with 3.13.3), Node.js **22.17+** recommended
for both packages, and PostgreSQL. No paid cloud AI service is required for the
default local assistant. Artwork/fonts may load from external URLs; TMDB
enrichment is optional and needs a key.

### Clone and restore the trained bundle

```powershell
git clone https://github.com/techvoyager-varun/Filmory.git
cd Filmory
```

**A clone does not include the active trained binaries.** Large checkpoint,
mapping, tensor, candidate-array and data files under `backend/training/runs` are
ignored. Restore the complete `20261001_clean_extended` bundle from the local
project installation/backup into that same relative directory, matching its
[27-entry SHA-256 manifest](backend/training/runs/20261001_clean_extended/sha256_manifest.json).
The current local workspace already has it. Missing/mismatched artifacts prevent
healthy recommendation serving. New experiments are described in the
[training guide](backend/training/README.md); a new bundle needs its own validation
and matching policy hashes. Training is not part of ordinary startup.

### Backend and fresh database — PowerShell

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` using your database credentials and a newly generated secret. Generate
the secret with `.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))"`.
Replace placeholders and URL-encode the database password:

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:your-url-encoded-password@localhost:5432/filmory
SECRET_KEY=replace-with-your-generated-secret
FRONTEND_ORIGIN=http://127.0.0.1:5173
ASSISTANT_LOCAL_ONLY=true
GEMINI_API_KEY=
TMDB_API_KEY=
HOST=127.0.0.1
PORT=8100
```

Create an empty `filmory` database using PostgreSQL/pgAdmin. For a **fresh
database**, migrate, seed and start:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m app.scripts.seed_movies
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

An existing database may have startup-created tables without an Alembic stamp;
review its schema/migration history before using fresh-install commands. Startup
creates missing tables and loads models; health checks SQL access/model readiness.
Health: <http://127.0.0.1:8100/health>. Swagger:
<http://127.0.0.1:8100/docs>. ReDoc: <http://127.0.0.1:8100/redoc>.

### Frontend — second terminal, from repository root

```powershell
cd frontend
npm ci
$env:VITE_API_BASE_URL = 'http://127.0.0.1:8100'
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open <http://127.0.0.1:5173/>. Demo login uses the seeded mapped profile; register
separately to test a new account. The API client's fallback is localhost:8000,
so set `VITE_API_BASE_URL` for port 8100. Ctrl+C stops each server and frees its
port. On Linux/macOS, use `.venv/bin/python` and set
`VITE_API_BASE_URL=http://127.0.0.1:8100` before the frontend command.

### Optional architecture explorer — separate terminal

```powershell
cd tools/architecture-explorer
npm ci
npm run scan
npm run verify
npm run dev
```

Open <http://127.0.0.1:5191/>. Python must be on PATH for scan/verify. This viewer
needs no Filmory API/database or API keys. It maps **43 modules, 76 relationships,
12 routes, 37 endpoints and 13 tables**. Refresh the snapshot after relevant source
changes; see its [setup/export guide](tools/architecture-explorer/README.md).

## Verification

Backend, from `backend` with dependencies, environment and active bundle present:

```powershell
$run = '.test-tmp/check-' + [guid]::NewGuid().ToString('N')
.\.venv\Scripts\python.exe -m pytest tests --basetemp $run -q
```

API tests use disposable SQLite fixtures and disable external provider/poster
calls. Frontend checks, from `frontend`:

```powershell
npm run test
npm run typecheck
npm run lint
npm run build
```

On 14 October 2026, **100 backend, 9 frontend and 7 explorer tests passed**;
type checks/builds passed. Frontend lint had zero errors and eight existing Fast
Refresh warnings; library/build warnings were nonfatal. See
[application checks](docs/verification/commit-checks-2026-10-14.md) and
[supplementary checks](docs/verification/supplemental-files-2026-10-14.md).

The [five-genre browser check](docs/verification/five-genre-frontend-2026-10-14.md)
confirmed homepage refreshes, API agreement and played-title exclusions. Its
final mixed-history slate favoured Comedy/Animation and omitted Sci-Fi/Documentary.
Ideal genre balance and human relevance remain open quality questions.

## API guide

Swagger is the complete schema reference. Main endpoints:

| Area | Endpoints |
| --- | --- |
| Auth/preferences | `POST /api/auth/register`, `/api/auth/login`, `/api/auth/demo`; `GET /api/auth/me`; `PUT /api/auth/preferences` |
| Catalogue | `GET /api/movies`, `/api/movies/{movie_id}`, `/api/genres`, `/api/search?query=...` |
| Recommendations | `GET /api/recommendations/me?top_k=12&variant=auto`, `/api/similar/{movie_id}`, `/api/trending`, `/api/popular`, `/api/movies-by-genre`; `POST /api/cold-start` |
| Model transparency | `GET /api/metrics`, `/api/taste-state` |
| Activity/feedback | `POST /api/interactions`; `GET /api/users/me/history`, `/api/users/me/likes`, `/api/users/me/my-list`, `/api/users/me/hidden-movies` |
| Assistant | `POST /api/assistant/chat`; owned session, feedback and filter routes under `/api/assistant` |
| Optional study | `POST /api/recommendation-study`; `GET /api/recommendation-study/{study_id}`; `PUT /api/recommendation-study/{study_id}/answers` |

Accepted recommendation variants: `auto`, `normalized_fixed`, `damr`, `mmr` and
`static`. Account/history branches determine the applicable serving path.
Activity, chats and studies require authentication/ownership. See the
[backend guide](backend/README.md).

