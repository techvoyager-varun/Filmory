from typing import List, Optional
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
