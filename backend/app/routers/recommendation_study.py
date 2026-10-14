"""Voluntary local relevance feedback. Never fabricate participant responses."""
import uuid
from pathlib import Path
import hashlib
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.core.time import utc_isoformat
from app.database import get_db
from app.models.db_models import RecommendationStudy, User
from app.ml.model_service import model_service
from app.ml.recommender import get_personalized_recommendations
from app.schemas.schemas import ScoredMovieSchema

router = APIRouter(prefix='/api/recommendation-study',tags=['Local Recommendation Study'])


class StartStudy(BaseModel):
    consent: Literal[True]


class StudyAnswer(BaseModel):
    movieId: int
    choice: Literal['interesting','not_for_me','unsure']


class StudyResponse(BaseModel):
    id: str
    movies: list[ScoredMovieSchema]
    answers: dict[str,str]
    createdAt: str


def response(study):
    return StudyResponse(id=study.id,movies=study.movies,answers=study.answers,
                         createdAt=utc_isoformat(study.created_at))


def owned(study_id, user, db, lock=False):
    query = db.query(RecommendationStudy).filter_by(id=study_id,user_id=user.id)
    if lock:
        query = query.with_for_update()
    study = query.first()
    if study is None:
        raise HTTPException(status_code=404,detail='Review not found')
    return study


@router.post('',response_model=StudyResponse)
def start_review(payload: StartStudy,user: User=Depends(get_current_user),db: Session=Depends(get_db)):
    kind, movies = get_personalized_recommendations(user,db,top_k=10)
    if not movies:
        raise HTTPException(status_code=409,detail='There are no recommendations available to review. Update your preferences and try again.')
    backend = Path(__file__).resolve().parents[2]
    snapshot = {'recommendation_type':kind,'model_mapped':user.model_user_id is not None,
                'model_config':model_service.config,'serving_policy':model_service.serving_policy,
                'serving_source_sha256':hashlib.sha256((backend/'app/ml/recommender.py').read_bytes()).hexdigest(),
                'interpretation':'Judged interest in a displayed slate, not actual viewing satisfaction'}
    study = RecommendationStudy(id=str(uuid.uuid4()),user_id=user.id,
        movies=[movie.model_dump(mode='json') for movie in movies],policy_snapshot=snapshot,answers={})
    db.add(study)
    db.commit()
    db.refresh(study)
    return response(study)


@router.get('/{study_id}',response_model=StudyResponse)
def get_review(study_id: str,user: User=Depends(get_current_user),db: Session=Depends(get_db)):
    return response(owned(study_id,user,db))


@router.put('/{study_id}/answers',response_model=StudyResponse)
def save_answer(study_id: str,payload: StudyAnswer,user: User=Depends(get_current_user),db: Session=Depends(get_db)):
    study = owned(study_id,user,db,lock=True)
    if payload.movieId not in {movie['movieId'] for movie in study.movies}:
        raise HTTPException(status_code=400,detail='This movie is not part of the displayed review')
    # Assign a fresh object so SQLAlchemy tracks JSON changes. Changing an answer
    # replaces it; it never creates duplicate votes or alters recommendation history.
    study.answers = {**(study.answers or {}),str(payload.movieId):payload.choice}
    db.commit()
    db.refresh(study)
    return response(study)
