from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, ConfigDict, Field, field_validator

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
    neighborScore: Optional[float] = None       # item-only collaborative votes for live accounts
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
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("Name cannot be blank")
        return name

    @field_validator("password")
    @classmethod
    def validate_bcrypt_length(cls, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return password

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("password")
    @classmethod
    def validate_bcrypt_length(cls, password: str) -> str:
        return RegisterRequest.validate_bcrypt_length(password)

class AuthResponse(BaseModel):
    accessToken: str
    tokenType: str = "bearer"
    user: UserSchema

class UserPreferencesUpdate(BaseModel):
    favoriteGenres: List[str] = Field(default_factory=list, max_length=20)
    favoriteMovieIds: List[int] = Field(default_factory=list, max_length=100)

    @field_validator("favoriteGenres", "favoriteMovieIds")
    @classmethod
    def unique_values(cls, values):
        return list(dict.fromkeys(values))

class ColdStartRequest(BaseModel):
    favoriteGenres: List[str] = []
    favoriteMovieIds: List[int] = []

INTERACTION_TYPES = ("play", "like", "unlike", "list_add", "list_remove", "not_interested", "show_again")

class InteractionCreate(BaseModel):
    movieId: int
    type: str = Field(pattern="^(play|like|unlike|list_add|list_remove|not_interested|show_again)$")

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
