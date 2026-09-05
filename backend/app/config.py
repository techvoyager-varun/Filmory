import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote_plus, urlparse, urlunparse

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg://postgres:Varunalwar%40025@localhost:5432/filmory"
    SECRET_KEY: str = "filmory_super_secret_jwt_key_2026_production_grade"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    FRONTEND_ORIGIN: str = "http://localhost:5173"
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # ML parameters
    CANDIDATE_K: int = 100
    TOP_K: int = 10
    MIN_INTERACTIONS_FOR_PERSONALIZATION: int = 2
    NCF_WEIGHT: float = 0.55
    TRANSFORMER_WEIGHT: float = 0.25
    GENRE_WEIGHT: float = 0.20

    # ==========================================================
    # Stage 4 — DAMR: Drift-Aware Momentum Re-Ranker
    # ==========================================================
    # Pool of candidates handed from the Stage 1-3 ensemble to Stage 4.
    RERANK_POOL_SIZE: int = 100
    # Re-ranking variant served by default: "damr" | "mmr" | "static"
    RERANK_VARIANT: str = "damr"

    # --- User-state estimation (Step 1 of DAMR) ---
    DAMR_TAU_DAYS: float = 7.0         # half-life scale for the short-term (decayed) genre profile
    DAMR_TAU_SESSION_H: float = 48.0   # decay scale for the "active session" freshness feature
    DAMR_N_REF: int = 200              # history length at which maturity saturates

    # --- Fusion / re-ranking strengths ---
    DAMR_ETA: float = 0.30             # strength of the taste-momentum bonus (scaled by drift)
    DAMR_GAMMA: float = 0.20           # strength of the expert-agreement confidence factor
    QUALITY_WEIGHT: float = 0.15       # weight of the Bayesian quality prior in final relevance
    MMR_LAMBDA: float = 0.70           # MMR trade-off: 0.7 relevance / 0.3 diversity

    # --- Adaptive expert gate ---
    # Slopes of the per-expert logit functions over the user-state features.
    DAMR_SLOPES: dict = {
        "a1": 0.8,   # d(z_NCF)/d(maturity)   — more history -> trust NCF more
        "a2": 1.5,   # d(z_NCF)/d(-drift)     — more drift    -> trust NCF less
        "b1": 2.0,   # d(z_TR)/d(drift)       — more drift    -> trust sequence more
        "b2": 0.6,   # d(z_TR)/d(freshness)   — active session-> trust sequence more
        "c1": 0.8,   # d(z_GEN)/d(focus)      — focused taste -> trust genre signal more
        "c2": 1.0,   # d(z_GEN)/d(1-maturity) — new user      -> trust genre signal more
    }
    # Anchor user state at which the gate must reproduce the classic 0.55/0.25/0.20
    # fixed ensemble exactly. Intercepts are derived from this anchor (see damr.py),
    # which makes static fusion a provable special case of DAMR.
    DAMR_ANCHOR: dict = {"drift": 0.10, "focus": 0.40, "maturity": 0.60, "freshness": 0.50}

    # --- Bayesian quality prior (IMDb-style weighted rating) ---
    BAYES_M: int = 500                 # minimum-votes threshold
    BAYES_C: float = 3.5               # global mean rating prior

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def clean_database_url(self) -> str:
        url = self.DATABASE_URL
        if not url:
            return url
        # Ensure postgresql+psycopg driver prefix is used
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

settings = Settings()
