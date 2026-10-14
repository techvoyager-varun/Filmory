"""API regressions run against a disposable catalogue, never the user's DB."""
import csv
import re
import shutil
import sys
from pathlib import Path
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

@pytest.fixture(scope="session")
def catalogue_template(tmp_path_factory):
    import torch
    from app.database import Base
    from app.models.db_models import Movie
    from app.ml.model_service import model_service
    torch.set_num_threads(4)
    model_service.load_all()
    path = tmp_path_factory.mktemp('catalogue')/'template.sqlite'
    engine = create_engine(f'sqlite:///{path.as_posix()}')
    Base.metadata.create_all(engine)
    rows = []
    with (Path(model_service.ml_dir)/'movies_metadata.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            year = re.search(r'\((\d{4})\)', row['title'])
            rows.append(dict(movie_id=int(row['movieId']),title=row['title'],genres=row['genres'].split('|'),
                year=int(year.group(1)) if year else None, rating=7.5,rating_count=1000,runtime=100,
                description='Synthetic API-test metadata; not evaluation evidence.',poster_url='',backdrop_url=''))
    with Session(engine) as db:
        db.bulk_insert_mappings(Movie, rows); db.commit()
    engine.dispose()
    return path

@pytest.fixture(autouse=True)
def isolated_api(request, monkeypatch, tmp_path):
    if request.module.__name__.split('.')[-1] not in {'test_api','test_assistant_acceptance','test_api_regressions'}:
        yield
        return
    from app.main import app
    from app.database import get_db
    from app.config import settings
    from app.services.llm_client import fallback_extract_intent
    from app.routers import assistant
    path=tmp_path/'api.sqlite'
    shutil.copyfile(request.getfixturevalue('catalogue_template'), path)
    engine=create_engine(f'sqlite:///{path.as_posix()}',connect_args={'check_same_thread':False})
    @event.listens_for(engine,'connect')
    def foreign_keys(connection, _): connection.execute('PRAGMA foreign_keys=ON')
    def db_override():
        with Session(engine) as db: yield db
    monkeypatch.setattr('app.main.engine',engine)
    app.dependency_overrides[get_db]=db_override
    monkeypatch.setattr(settings,'ASSISTANT_RATE_LIMIT',0)
    monkeypatch.setattr(settings,'ASSISTANT_GLOBAL_RATE_LIMIT',0)
    assistant._user_timestamps.clear(); assistant._ip_timestamps.clear()
    monkeypatch.setattr('app.services.assistant_orchestrator.extract_intent',lambda text,*args: fallback_extract_intent(text))
    def offline_client(): raise RuntimeError('Provider disabled in isolated API tests')
    monkeypatch.setattr('app.services.llm_client._get_client',offline_client)
    for module in ['app.routers.movies','app.ml.recommender','app.services.assistant_orchestrator']:
        monkeypatch.setattr(module+'.ensure_movie_posters',lambda *args:None)
    yield engine
    app.dependency_overrides.pop(get_db,None)
    engine.dispose()
