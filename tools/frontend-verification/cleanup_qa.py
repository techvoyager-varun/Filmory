"""Remove only the dedicated account created during this frontend verification."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.database import SessionLocal
from app.models.db_models import (User,UserPreference,UserGenrePreference,Interaction,WatchHistory,
                                 MyList,Like,AssistantSession,AssistantMessageRecord,AssistantFeedback,RecommendationStudy)

parser=argparse.ArgumentParser()
parser.add_argument('--email',required=True)
parser.add_argument('--report', type=Path, help='Optional separate cleanup report for a later QA run')
args=parser.parse_args()
if not args.email.startswith('filmory.frontend.qa.') or not args.email.endswith('@example.com'):
    raise SystemExit('Refusing to touch any account outside this QA naming pattern.')
with SessionLocal() as db:
    user=db.query(User).filter(User.email==args.email).one_or_none()
    if user is None:
        raise SystemExit('No QA account found; nothing removed.')
    if user.name!='Filmory Frontend QA' or user.model_user_id is not None:
        raise SystemExit('Refusing cleanup: account does not match the created QA fixture.')
    uid=user.id
    sessions=[s.id for s in db.query(AssistantSession).filter(AssistantSession.user_id==uid)]
    removed={}
    for model in [AssistantFeedback,AssistantMessageRecord]:
        removed[model.__tablename__]=db.query(model).filter(model.session_id.in_(sessions)).delete(synchronize_session=False)
    for model in [RecommendationStudy,AssistantSession,Interaction,WatchHistory,MyList,Like,UserGenrePreference,UserPreference]:
        removed[model.__tablename__]=db.query(model).filter(model.user_id==uid).delete(synchronize_session=False)
    removed['users']=db.query(User).filter(User.id==uid,User.email==args.email).delete(synchronize_session=False)
    db.commit()
target=args.report or Path(__file__).resolve().parent/'qa-cleanup.json'
target.write_text(json.dumps({'status':'completed','scope':'Only the newly created frontend QA account and its child rows. Catalogue, models, demo account and other users untouched.','removed_test_rows':removed},indent=2)+'\n',encoding='utf-8')
print('Dedicated QA account and owned test rows removed. Report:',target)
