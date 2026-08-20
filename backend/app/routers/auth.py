from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.db_models import User, UserPreference, UserGenrePreference
from app.schemas.schemas import (
    RegisterRequest,
    LoginRequest,
    AuthResponse,
    UserSchema,
    UserPreferencesUpdate,
)
from app.core.security import verify_password, get_password_hash, create_access_token
from app.core.deps import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])

def user_to_schema(user: User, db: Session) -> UserSchema:
    pref = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    return UserSchema(
        userId=str(user.id),
        name=user.name,
        email=user.email,
        favoriteGenres=pref.favorite_genres if pref and pref.favorite_genres else [],
        favoriteMovieIds=pref.favorite_movie_ids if pref and pref.favorite_movie_ids else [],
        onboardingCompleted=pref.onboarding_completed if pref else False,
    )

@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )

    user = User(
        name=payload.name,
        email=payload.email.lower(),
        password_hash=get_password_hash(payload.password),
        model_user_id=None, # New user starts without model embedding
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Initialize empty preferences
    pref = UserPreference(
        user_id=user.id,
        favorite_genres=[],
        favorite_movie_ids=[],
        onboarding_completed=False,
    )
    db.add(pref)
    db.commit()

    token = create_access_token(user.id)
    return AuthResponse(
        accessToken=token,
        tokenType="bearer",
        user=user_to_schema(user, db),
    )

@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id)
    return AuthResponse(
        accessToken=token,
        tokenType="bearer",
        user=user_to_schema(user, db),
    )

@router.post("/demo", response_model=AuthResponse)
def login_as_demo(db: Session = Depends(get_db)):
    demo_email = "demo@filmory.app"
    user = db.query(User).filter(User.email == demo_email).first()

    if not user:
        # Create persistent demo user mapped to MovieLens user 2847 for testing personalized recs
        user = User(
            name="Demo Viewer",
            email=demo_email,
            password_hash=get_password_hash("demo123456"),
            model_user_id=2847,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        pref = UserPreference(
            user_id=user.id,
            favorite_genres=["Sci-Fi", "Drama", "Thriller", "Action"],
            favorite_movie_ids=[109487, 79132, 58559],
            onboarding_completed=True,
        )
        db.add(pref)
        db.commit()
    elif user.model_user_id is None:
        user.model_user_id = 2847
        db.commit()

    token = create_access_token(user.id)
    return AuthResponse(
        accessToken=token,
        tokenType="bearer",
        user=user_to_schema(user, db),
    )

@router.get("/me", response_model=UserSchema)
def get_me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return user_to_schema(user, db)

@router.put("/preferences", response_model=UserSchema)
def update_preferences(
    payload: UserPreferencesUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pref = db.query(UserPreference).filter(UserPreference.user_id == user.id).first()
    if not pref:
        pref = UserPreference(user_id=user.id)
        db.add(pref)

    pref.favorite_genres = payload.favoriteGenres
    pref.favorite_movie_ids = payload.favoriteMovieIds
    pref.onboarding_completed = True
    
    # Also initialize genre preference signals in user_genre_preferences table
    for g in payload.favoriteGenres:
        existing_gp = (
            db.query(UserGenrePreference)
            .filter(UserGenrePreference.user_id == user.id, UserGenrePreference.genre == g)
            .first()
        )
        if not existing_gp:
            db.add(UserGenrePreference(user_id=user.id, genre=g, score=2.0))

    db.commit()
    return user_to_schema(user, db)
