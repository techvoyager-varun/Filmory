import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.ml.model_service import model_service
from app.routers.auth import router as auth_router
from app.routers.movies import router as movies_router
from app.routers.recommendations import router as recommendations_router
from app.routers.interactions import router as interactions_router

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

# CORS Configuration allowing any local dev port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_ORIGIN,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-z0-9-]+\.e2b\.app)",
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

@app.get("/health", tags=["Health"])
def health_check():
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
