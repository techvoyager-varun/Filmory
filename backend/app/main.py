import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.ml.model_service import model_service
from app.routers.auth import router as auth_router
from app.routers.movies import router as movies_router
from app.routers.recommendations import router as recommendations_router
from app.routers.interactions import router as interactions_router
from app.routers.assistant import router as assistant_router
from app.routers.recommendation_study import router as study_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("filmory")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Filmory Backend...")
    # 1. Ensure DB tables exist
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("PostgreSQL database tables verified/created.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

    # 2. Load ML models and matrices (once at startup)
    try:
        model_service.load_all()
        logger.info("PyTorch Recommendation Engine successfully loaded into memory.")
    except Exception as e:
        logger.error(f"Error loading ML models: {e}")

    yield

    logger.info("Shutting down Filmory Backend...")

app = FastAPI(
    title="Filmory AI Recommendation Backend",
    description="Production-grade API for deep learning movie recommendations using NCF, Sequential Transformer, and real-time user personalization.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: origins come from config so a deployed frontend can be allowed purely
# through environment variables (FRONTEND_ORIGIN / CORS_EXTRA_ORIGINS).
def _allowed_origins() -> list[str]:
    configured = [
        origin.strip()
        for origin in f"{settings.FRONTEND_ORIGIN},{settings.CORS_EXTRA_ORIGINS}".split(",")
        if origin.strip()
    ]
    dev_defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    # de-duplicate while preserving order
    return list(dict.fromkeys(configured + dev_defaults))


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount Routers
app.include_router(auth_router)
app.include_router(movies_router)
app.include_router(recommendations_router)
app.include_router(interactions_router)
app.include_router(assistant_router)
app.include_router(study_router)

@app.get("/health", tags=["Health"])
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Recommendation models unavailable")
    return {
        "status": "healthy",
        "service": "Filmory AI Backend",
        "device": str(model_service.device),
        "models_loaded": model_service.is_loaded,
    }

@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Welcome to Filmory AI Recommendation Engine API",
        "docs": "/docs",
        "health": "/health",
    }
