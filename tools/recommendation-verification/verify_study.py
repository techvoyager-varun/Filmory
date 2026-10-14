"""Read-only live QA study checks; keep all test data out of human evidence."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
sys.path.insert(0,str(ROOT/'backend/scripts'))
from app.database import SessionLocal
from app.models.db_models import User,RecommendationStudy,Like,WatchHistory
from export_recommendation_study import build_report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--email',required=True)
    args=parser.parse_args()
    if not (args.email.startswith('filmory.frontend.qa.') and args.email.endswith('@example.com')):
        raise ValueError('Only the dedicated QA fixture can be inspected')
    with SessionLocal() as db:
        user=db.query(User).filter_by(email=args.email,name='Filmory Frontend QA').one()
        studies=db.query(RecommendationStudy).filter_by(user_id=user.id).all()
        assert len(studies)==1
        study=studies[0]
        assert len(study.movies)==10 and len({movie['movieId'] for movie in study.movies})==10
        assert study.answers=={'296':'interesting','356':'not_for_me','318':'unsure'}
        assert study.policy_snapshot['recommendation_type']=='cold_start'
        assert study.policy_snapshot['model_mapped'] is False
        assert study.policy_snapshot['serving_source_sha256']
        assert db.query(Like).filter_by(user_id=user.id).count()==0
        assert db.query(WatchHistory).filter_by(user_id=user.id).count()==0
        exported=build_report(studies,{user.id:user},'qa-only-secret')
        assert exported['participants']==0 and exported['rows']==[]
    target=ROOT/'docs/verification/recommendation-study-live-20261014.json'
    target.write_text(json.dumps({'scope':'Dedicated QA fixture, not human participation',
        'checks':['One consent-triggered study','Ten unique frozen titles','Three distinct ratings saved',
                  'Policy and serving source snapshot retained','No likes/watch history side effects',
                  'QA ratings excluded from participant export'],'status':'passed'},indent=2)+'\n')
    print('Six live QA study checks passed:',target)


if __name__=='__main__':main()
