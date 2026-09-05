from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, ConfigDict

class MovieSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    movieId: int
    title: str
    genres: List[str]
    year: int
    rating: float
    runtime: int = 0
    posterUrl: str = ""
    backdropUrl: str = ""
    description: str = ""

class ScoredMovieSchema(MovieSchema):
    score: Optional[float] = None
    ncfScore: Optional[float] = None
    transformerScore: Optional[float] = None
    genreScore: Optional[float] = None
    # ---- Stage 4 (DAMR) transparency fields ----
    momentumScore: Optional[float] = None       # taste-momentum alignment bonus component
    agreementScore: Optional[float] = None      # expert-agreement confidence factor
    qualityScore: Optional[float] = None        # Bayesian quality prior [0..1]
    diversityPenalty: Optional[float] = None    # max similarity to already-picked items
    expertWeights: Optional[Dict[str, float]] = None  # drift-adaptive gate output
    userState: Optional[Dict[str, Any]] = None  # drift / focus / maturity / freshness
    variant: Optional[str] = None               # which Stage-4 variant produced this list
    rank: Optional[int] = None                  # 1-based position in the final list
    listDiversity: Optional[float] = None       # intra-list diversity (ILD) of the final list

class MoviePageSchema(BaseModel):
    movies: List[MovieSchema]
    total: int

class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    userId: str
    name: str
    email: str
    favoriteGenres: List[str] = []
    favoriteMovieIds: List[int] = []
    onboardingCompleted: bool = False

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class AuthResponse(BaseModel):
    accessToken: str
    tokenType: str = "bearer"
    user: UserSchema

class UserPreferencesUpdate(BaseModel):
    favoriteGenres: List[str] = []
    favoriteMovieIds: List[int] = []

class ColdStartRequest(BaseModel):
    favoriteGenres: List[str] = []
    favoriteMovieIds: List[int] = []

class InteractionCreate(BaseModel):
    movieId: int
    type: str # play, like, unlike, list_add, list_remove

class InteractionResponse(BaseModel):
    userId: str
    movieId: int
    type: str
    timestamp: str

class HistoryEntrySchema(BaseModel):
    movie: MovieSchema
    watchedAt: str
    progress: int

class RecommendationsResponse(BaseModel):
    type: str
    candidateK: int = 100
    recommendations: List[ScoredMovieSchema]
