from app.routers.auth import router as auth_router
from app.routers.movies import router as movies_router
from app.routers.recommendations import router as recommendations_router
from app.routers.interactions import router as interactions_router

__all__ = [
    "auth_router",
    "movies_router",
    "recommendations_router",
    "interactions_router",
]
