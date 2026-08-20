# 🎬 Filmory — AI-Powered Movie Recommendation Platform

<p align="center">
  <strong>A cinematic movie discovery and recommendation platform powered by Deep Learning (Neural Collaborative Filtering + Sequential Transformer) trained on the MovieLens catalog.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React 19" />
  <img src="https://img.shields.io/badge/TypeScript-5.8-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Tailwind_CSS-4.2-38B2AC?style=flat-square&logo=tailwind-css&logoColor=white" alt="Tailwind CSS" />
  <img src="https://img.shields.io/badge/PostgreSQL-14+-4169E1?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/TanStack_Router-1.170-FF4154?style=flat-square&logo=react-query&logoColor=white" alt="TanStack Router" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License" />
</p>

---

## 🌟 Overview

**Filmory** is an end-to-end, production-grade movie streaming discovery platform designed to deliver hyper-personalized movie recommendations in real-time. It marries modern deep learning recommendation systems with a responsive, Netflix-style cinematic user interface.

Unlike simple popularity or tag-based recommendation engines, Filmory leverages a **3-stage hybrid deep learning pipeline**:
1. **Candidate Generation**: High-throughput candidate retrieval using a **Hybrid Neural Collaborative Filtering (NCF)** network over 22,800+ MovieLens catalog titles.
2. **Sequential Scoring**: Session-aware sequence prediction with a **Sequential Transformer** (SASRec-inspired) trained on user watch order.
3. **Dynamic Re-ranking**: Real-time preference weighting based on immediate user actions (*likes*, *watchlist saves*, *plays*).

---

## 🚀 Key Features

### 🧠 Deep Learning Recommendation Engine
- **3-Stage Ensemble Architecture**:
  $$\text{Final Score} = 0.55 \cdot \text{Score}_{\text{NCF}} + 0.25 \cdot \text{Score}_{\text{Transformer}} + 0.20 \cdot \text{Score}_{\text{Genre}}$$
- **Sequential Transformer Scoring**: Predicts the user's next movie based on their last 20 watch sequence steps.
- **Cold-Start Onboarding Engine**: Cosine similarity KNN against 41,547 user taste profiles for new users with fewer than 5 interactions.
- **Item-to-Item Similarity Rails**: Vector cosine similarity between learned NCF embeddings and 20-dimensional genre vectors.
- **Real-Time Feedback Loop**: Every play, like, or list addition dynamically updates user preference vectors in PostgreSQL without needing full model retraining.
- **AI Recommendation Breakdown**: Inspectable match confidence scores with detailed NCF, Transformer, and Genre affinity breakdowns on every recommended title.

### 🎭 Netflix-Grade Cinematic Frontend
- **Hero Spotlight**: Dynamic banner with trailers, synopsis, genre tags, and quick-action buttons.
- **Personalized Rails**: "Top AI Picks For You", "Trending Now", "Popular Titles", "Because You Watched...", and Genre-specific shelves.
- **Interactive Search & Filtering**: Instant search by title/genre with sorting (`Popular`, `Rating`, `Year`, `Title`).
- **Interactive Onboarding Wizard**: Beautiful multi-step genre and movie picker for new accounts.
- **Watchlist ("My List") & History**: Saved titles and chronological watch history with optimistic UI updates.
- **User Taste Profile**: Breakdown of favorite genres, interaction stats, and AI user persona.
- **One-Click Demo Mode**: Jump straight into a pre-configured profile mapped to MovieLens User `2847` for instant testing.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend (React 19 + TypeScript + Vite)"]
        UI["Cinematic UI / TanStack Start Router"]
        TQ["TanStack Query (Cache & Optimistic State)"]
        API_C["API Client (JWT Bearer Auth)"]
    end

    subgraph Backend["Backend (FastAPI + Python 3.10+)"]
        API["FastAPI REST Endpoints"]
        AUTH["JWT & Password Hash (Bcrypt)"]
        REC_ENG["3-Stage Recommendation Engine"]
    end

    subgraph Database["Database (PostgreSQL)"]
        PG[("PostgreSQL 14+")]
        TABLES["Users • Movies (27K+) • Interactions • History • Likes • MyList"]
    end

    subgraph ML["PyTorch Recommendation Core"]
        NCF["Hybrid NCF (8-dim Embeddings + Genre Projections)"]
        TRANS["Sequential Transformer (SASRec 2-layer 4-head)"]
        MATRICES["Movie-Genre Matrix (22.8K x 20) • User-Genre Matrix (41.5K x 20)"]
    end

    UI --> TQ --> API_C
    API_C <-->|REST API / JSON| API
    API --> AUTH
    API --> REC_ENG
    REC_ENG <--> PG
    REC_ENG <--> ML
    PG --- TABLES
    ML --- NCF
    ML --- TRANS
    ML --- MATRICES
```

---

## 📂 Project Structure

```
Filmory/
├── backend/
│   ├── app/
│   │   ├── config.py                 # Pydantic environment configuration
│   │   ├── database.py               # SQLAlchemy async/sync session management
│   │   ├── main.py                   # FastAPI app entry & lifespan loader
│   │   ├── ml/
│   │   │   ├── architectures.py      # PyTorch NCF & Transformer model classes
│   │   │   ├── model_service.py      # In-memory model artifact loader & cache
│   │   │   └── recommender.py        # 3-stage candidate retrieval & ranking logic
│   │   ├── models/
│   │   │   └── db_models.py          # SQLAlchemy ORM models
│   │   ├── routers/
│   │   │   ├── auth.py               # Registration, Login, Demo, Preferences
│   │   │   ├── movies.py             # Catalog, Search, Details, Genre queries
│   │   │   ├── recommendations.py    # Personalized, Similar, Trending, Cold-start
│   │   │   └── interactions.py       # Watch history, Likes, Watchlist, Interactions
│   │   ├── schemas/
│   │   │   └── schemas.py            # Pydantic v2 validation schemas
│   │   └── scripts/
│   │       └── seed_movies.py        # Seeds 27,278 MovieLens catalog & Demo user
│   ├── ml/                           # Trained PyTorch .pth models & mappings
│   │   ├── ncf_baseline.pth
│   │   ├── ncf_hybrid.pth
│   │   ├── sequential_transformer.pth
│   │   ├── movie_genre_matrix.pt
│   │   ├── user_genre_matrix.pt
│   │   └── *.pkl                     # User, movie, and genre ID index maps
│   ├── migrations/                   # Alembic schema migrations
│   ├── tests/                        # Automated Pytest suite
│   ├── requirements.txt              # Python package dependencies
│   └── .env.example                  # Backend environment template
│
├── frontend/
│   ├── src/
│   │   ├── api/                      # Typed backend API clients
│   │   ├── components/               # UI components (HeroBanner, MovieRow, Poster, etc.)
│   │   │   ├── onboarding/           # Onboarding genre/movie selector
│   │   │   └── ui/                   # Radix UI primitives & custom components
│   │   ├── context/                  # AuthContext & UserDataContext
│   │   ├── routes/                   # TanStack file-based routes
│   │   │   ├── index.tsx             # Home discovery feed
│   │   │   ├── movies.$movieId.tsx   # Movie detail & similar titles page
│   │   │   ├── movies.index.tsx      # Browse & filter catalog
│   │   │   ├── search.tsx            # Live movie search
│   │   │   ├── my-list.tsx           # Saved watchlist
│   │   │   ├── history.tsx           # Watch history
│   │   │   ├── profile.tsx           # User profile & taste stats
│   │   │   ├── login.tsx / register.tsx
│   │   │   └── onboarding.tsx        # Cold-start preference setup
│   │   ├── styles.css                # Tailwind CSS styling & animations
│   │   └── router.tsx                # TanStack Router initialization
│   ├── package.json                  # Frontend dependencies
│   └── vite.config.ts                # Vite build configuration
│
└── README.md
```

---

## 🛠️ Complete Setup Guide (From Scratch)

### Prerequisites
- **Python**: `3.10` or higher
- **Node.js**: `18.0` or higher (with `npm`)
- **PostgreSQL**: `14.0` or higher running locally or remotely

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/techvoyager-varun/Filmory.git
cd Filmory
```

---

### Step 2: Backend Setup (FastAPI + PyTorch + PostgreSQL)

Open a new terminal (**Terminal 1**):

```bash
cd backend

# 1. Create and activate a Python virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
# .\.venv\Scripts\activate.bat
# Linux / macOS:
# source .venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Create your .env file
# Windows PowerShell:
Copy-Item .env.example .env
# Linux / macOS:
# cp .env.example .env
```

#### Configure `.env`
Open `backend/.env` and configure your PostgreSQL database credentials:

```env
DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/filmory
SECRET_KEY=filmory_super_secret_jwt_key_2026_production_grade
ACCESS_TOKEN_EXPIRE_MINUTES=10080
FRONTEND_ORIGIN=http://localhost:5173
PORT=8000
HOST=0.0.0.0
```

> [!NOTE]
> **Special Characters in Database Password**: If your password contains characters like `@` or `:`, URL-encode them (e.g., `@` becomes `%40`).

#### Run Migrations & Seed Database

```bash
# Apply database schema migrations via Alembic
alembic upgrade head

# Seed the 27,278 MovieLens catalog and create the default Demo user
python -m app.scripts.seed_movies

# Start the FastAPI backend server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> **Backend Service Status:**
> - 🌐 **API Root**: [http://localhost:8000](http://localhost:8000)
> - 📖 **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
> - 📋 **ReDoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
> - 💓 **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### Step 3: Frontend Setup (React 19 + TypeScript + Vite)

Open a second terminal (**Terminal 2**):

```bash
cd frontend

# 1. Install npm packages
npm install

# 2. Start the Vite development server
npm run dev
```

> **Frontend Service Status:**
> - 🎬 **App URL**: [http://localhost:5173](http://localhost:5173) (or `http://localhost:8080`)

---

## 🧪 Running Automated Tests

Run the complete backend test suite covering authentication, movie catalog filtering, interaction recording, and recommendation flows:

```bash
cd backend
python -m pytest -v tests
```

---

## 📡 REST API Reference

| Category | Method | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **Auth** | `POST` | `/api/auth/register` | Register a new user account with email & password |
| **Auth** | `POST` | `/api/auth/login` | Authenticate with credentials and receive JWT access token |
| **Auth** | `POST` | `/api/auth/demo` | Instant login as Demo Viewer (mapped to model user `2847`) |
| **Auth** | `GET` | `/api/auth/me` | Retrieve current authenticated user profile & taste vectors |
| **Auth** | `PUT` | `/api/auth/preferences` | Update onboarding favorite genres and initial movie picks |
| **Movies** | `GET` | `/api/movies` | Browse catalog with genre filter, sorting, and pagination |
| **Movies** | `GET` | `/api/movies/{movieId}` | Retrieve single movie details, ratings, and backdrop metadata |
| **Movies** | `GET` | `/api/genres` | List all 20 distinct MovieLens genre categories |
| **Movies** | `GET` | `/api/search` | Search movies by title or genre keywords |
| **Recs** | `GET` | `/api/recommendations/{userId}` | Get personalized recommendations (3-Stage Hybrid Deep Learning) |
| **Recs** | `GET` | `/api/similar/{movieId}` | Get movie-to-movie embedding similarity recommendations |
| **Recs** | `GET` | `/api/trending` | Get trending titles based on recent interaction popularity |
| **Recs** | `GET` | `/api/popular` | Get top-rated movies with significant audience thresholds |
| **Recs** | `POST` | `/api/cold-start` | Generate recommendations for new users via genre profile KNN |
| **Activity**| `POST` | `/api/interactions` | Record user interaction (`play`, `like`, `unlike`, `list_add`, `list_remove`) |
| **Activity**| `GET` | `/api/users/me/history` | Chronological watch history with timestamps |
| **Activity**| `GET` | `/api/users/me/my-list` | User's saved watchlist |
| **Activity**| `POST` | `/api/users/me/my-list/{movieId}` | Add movie to watchlist |
| **Activity**| `DELETE`| `/api/users/me/my-list/{movieId}` | Remove movie from watchlist |
| **Activity**| `GET` | `/api/users/me/likes` | Retrieve list of all liked movie IDs |
| **Activity**| `POST` | `/api/users/me/likes/{movieId}` | Like a movie |
| **Activity**| `DELETE`| `/api/users/me/likes/{movieId}` | Remove like from movie |

---

## 🤖 Deep Learning Models & Technical Specifications

| Model / Matrix | Architecture Details | Purpose |
| :--- | :--- | :--- |
| **NCF Baseline** | 8-dim User Embedding + 8-dim Item Embedding $\to$ MLP (`16` $\to$ `64` $\to$ `32` $\to$ `1`) | Matrix Factorization + Non-linear interaction baseline |
| **NCF Hybrid** | User + Item Embeddings + User Genre Projection + Item Genre Projection $\to$ MLP (`32` $\to$ `64` $\to$ `32` $\to$ `1`) | Stage 1 candidate generation from catalog of 22,836 items |
| **Sequential Transformer** | 64-dim Item Embeddings + 20-step Position Embeddings + 2-layer 4-head Transformer Encoder | Stage 2 sequence-aware scoring of top 100 candidates |
| **Movie Genre Matrix** | $22,836 \times 20$ Binary Matrix | Item-item similarity & genre affinity scoring |
| **User Genre Matrix** | $41,547 \times 20$ Frequency Matrix | Cold-start user profile matching via Cosine Similarity KNN |

---

## 💻 Tech Stack

### Frontend
- **Framework**: React 19, TypeScript
- **Routing & State**: TanStack Router, TanStack Query
- **Styling**: Tailwind CSS v4, Radix UI Primitives, Lucide Icons
- **Tooling**: Vite 8, ESLint, Prettier

### Backend
- **Framework**: FastAPI (Python 3.10+), Uvicorn
- **ORM & Migrations**: SQLAlchemy 2.0, Alembic, psycopg3
- **Security**: JWT (`python-jose`), Passlib, Bcrypt
- **Testing**: Pytest, HTTPX

### Machine Learning
- **Core Library**: PyTorch (`torch>=2.0.0`)
- **Data & Linear Algebra**: NumPy, Pandas, Scikit-learn
- **Dataset**: MovieLens Catalog (27,278 movies, 41,547 user profiles, millions of ratings)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
