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
    MIN_INTERACTIONS_FOR_PERSONALIZATION: int = 5
    NCF_WEIGHT: float = 0.55
    TRANSFORMER_WEIGHT: float = 0.25
    GENRE_WEIGHT: float = 0.20

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
